"""
The API contract for hw1/api.py — and an autograded deliverable.

This file is the SPECIFICATION of what `insert` / `retrieve` / `compare` must do.
It is not a suggestion and it is not a sample: your `api.py`, your
`ontology/hw1.ttl` and your `queries/*.rq` are correct when this passes.

WHAT IT COVERS
    1. The measurers, against fixtures whose values are CLOSED FORM — a synthetic
       frame has an exact clip fraction, an exact mean and an exact valid-depth
       fraction, so these assertions hold for any correct implementation.
       `depth_roughness` is checked differently and more strongly: it is handed a
       frame carrying Gaussian noise of a KNOWN sigma under a fixed seed, and it
       must RECOVER that sigma. An estimator validated by recovering the noise it
       was given cannot be satisfied by a constant that happens to fit.
    2. A real, black-box round trip against a live Apache Jena Fuseki server:
       spin up `fuseki-server.jar --mem /ds` on a free port, synthesise a batch on
       disk, `insert` it (Graph Store HTTP PUT, one named graph), `retrieve` it
       through queries/valid_frames.rq, and assert the CSV holds exactly the frames
       inside the band.
    3. The provenance and staleness rules: tau on the Batch node, re-insert
       replacing (not appending to) a batch, and a changed tau changing the answer.

EVERY CASE PASSES tau EXPLICITLY
    tau_lo and tau_hi are yours to derive, so no default may be assumed anywhere —
    not in a measurer signature, not in this file. Each fixture states the tau it
    measures at, which is exactly why the measurers stay autogradable even though
    two students' stored observables are not the same quantity.

    The bands are passed explicitly too, via a thresholds.json written into the
    test's temp directory. Your own hw1/thresholds.json is never read here and its
    contents can never make this suite pass or fail.

FIVE FIXTURES ARE LOAD-BEARING
    These are not extra coverage — each is the executable argument for a design
    decision, and each is the case that a plausible-looking wrong implementation
    fails. Read them before "simplifying" a measurer:

      test_half_black_half_white_*     why mean(V) is not a quality factor
      test_solid_pure_red_*            why no colorimetric weighting is allowed
      test_depth_roughness_recovers_*  the noise estimator must recover a KNOWN
                                       sigma, not merely return a plausible number
      test_depth_roughness_ignores_a_hard_step_edge
                                       why the median, not the literature's mean
      test_depth_roughness_only_reads_fully_valid_windows
                                       why the nine-pixel validity mask is
                                       required — without it factor 2 degenerates
                                       into a restatement of factor 1

RUN
    pixi run -e habitat python -m pytest hw1/test_e2e.py -v      # from repo root
    pixi run -e habitat python hw1/test_e2e.py                   # direct

    HW1_API=instructor/solution/api.py pixi run -e habitat python -m pytest hw1/test_e2e.py -v
        # instructor: run the same contract against the reference solution

REQUIREMENTS
    A JVM (`java` on PATH) and the Fuseki jar under hw1/fuseki_bin/ (the `fuseki`
    pixi task downloads it). If either is missing the whole module SKIPS rather
    than fails — the round trip is an integration test, not a unit test.
"""
import csv
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import types
import unittest
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))

# The module under test defaults to hw1/api.py; HW1_API points it elsewhere so the
# same contract can be run against a reference implementation.
_API_PATH = os.environ.get("HW1_API") or os.path.join(_HERE, "api.py")
if not os.path.isabs(_API_PATH):
    _API_PATH = os.path.join(os.path.dirname(_HERE), _API_PATH)
_spec = importlib.util.spec_from_file_location("hw1_api_under_test", _API_PATH)
api = importlib.util.module_from_spec(_spec)
sys.modules["hw1_api_under_test"] = api
_spec.loader.exec_module(api)

_JAR = os.path.join(_HERE, "fuseki_bin", "apache-jena-fuseki-6.1.0", "fuseki-server.jar")

# Module-level handles populated by setUpModule / torn down by tearDownModule.
_PROC = None
_ENDPOINT = None
_LOG = None
_TMP = None

FLOOR = 1
BATCH_DIRNAME = "e2e_batch"
BATCH = f"floor{FLOOR}_{BATCH_DIRNAME}"          # what hw1:batchName must hold
BATCH_B_DIRNAME = "e2e_batch_b"
BATCH_B = f"floor{FLOOR}_{BATCH_B_DIRNAME}"
BATCH_DEGEN_DIRNAME = "e2e_batch_degenerate"
BATCH_DEGEN = f"floor{FLOOR}_{BATCH_DEGEN_DIRNAME}"

# ---------------------------------------------------------------------------
# The tau this suite measures at. Passed explicitly everywhere; never defaulted.
# ---------------------------------------------------------------------------
TAU_LO = 16.0
TAU_HI = 250.0

# ---------------------------------------------------------------------------
# Synthetic frames and their CLOSED-FORM observables.
#
# RGB frames are solid or half/half, so the clip fractions and the mean of
# V = max(R,G,B) are exact. Depth frames are constant (exactly one depth, so
# valid_depth_fraction is exactly 1.0 and depth_roughness is exactly 0), constant
# over a strip with the rest dropped out (an exact fraction of no-return pixels),
# or constant plus Gaussian noise at a known sigma (so depth_roughness is that
# sigma).
#
#   stem  rgb          depth                              fails on
#   0     grey 128     clean 3 m                          -- nothing
#   1     black 0      clean 3 m                          clip_lo = 1.0
#   2     white 255    clean 3 m                          clip_hi = 1.0
#   3     grey 128     3 m over 13 of 64 rows, rest 0     vdf = 0.203
#   4     grey 128     3 m + noise, sigma = 0.05 m        depth_roughness
#
# Each of the four bands is failed by exactly ONE frame, and each failing frame
# clears the other three — frame 3 in particular is clean where it has depth at
# all, so its roughness is 0 and only its validity is bad. That separation is what
# lets `test_each_band_excludes_its_own_frame` attribute an exclusion to a band.
# Frame 0 is the only frame that clears all four.
# ---------------------------------------------------------------------------
IMG = 64
_FRAMES = [
    (0, "grey", "clean"),
    (1, "black", "clean"),
    (2, "white", "clean"),
    (3, "grey", "sparse"),
    (4, "grey", "noisy"),
]
PASS_ALL_BANDS = [0]

# Depth of the synthetic frames, and the noise the "noisy" frame carries.
DEPTH_Z_M = 3.0
BATCH_NOISE_SIGMA_M = 0.05
SPARSE_ROWS = 13                                 # of IMG: vdf = 13/64 = 0.203125

# Bands used by the round-trip cases: loose enough that only the intended failure
# mode excludes a frame. Written to a thresholds.json in the temp dir, never read
# from the student's own file. Note that only the FIRST is a lower bound.
BANDS = {
    "valid_depth_fraction_min": 0.5,
    "depth_roughness_max": 0.02,                 # metres; frame 4 sits at 0.05
    "clip_hi_fraction_max": 0.25,
    "clip_lo_fraction_max": 0.25,
}
# A roughness bound that gates nothing, for the "all bands wide open" cases. It
# has to be a number: the band is `<= max`, so wide open means LARGE, not 0.
ROUGHNESS_WIDE_OPEN = 10.0

# The sigma-recovery fixture is measured on a larger raster than the batch frames:
# the estimator is a median over (N-2)^2 overlapping windows, and a bigger N buys
# a tighter sampling error for free. The seed is fixed so the raster is identical
# on every run.
NOISE_IMG = 128
NOISE_SEED = 7


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _ping(endpoint_base, timeout_s=40):
    """Poll Fuseki's /$/ping until it answers 200 or timeout; True iff it came up."""
    url = endpoint_base + "/$/ping"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Fixture synthesis
# ---------------------------------------------------------------------------
def _rgb_solid(rgb):
    return Image.new("RGB", (IMG, IMG), rgb)


def _rgb_half_black_half_white():
    """Top half (0,0,0), bottom half (255,255,255) — exactly half of each."""
    arr = np.zeros((IMG, IMG, 3), dtype=np.uint8)
    arr[IMG // 2:] = 255
    return Image.fromarray(arr)


def _mm(depth_m):
    """Metres -> the uint16 millimetre raster api.py actually reads."""
    return np.round(np.asarray(depth_m) * 1000.0).astype(np.uint16)


def _depth_clean(z_m=DEPTH_Z_M, size=IMG):
    """Constant depth: every pixel valid, and no high-frequency content at all."""
    return _mm(np.full((size, size), float(z_m)))


def _depth_noisy(sigma_m, z_m=DEPTH_Z_M, size=IMG, seed=0):
    """Constant depth plus i.i.d. Gaussian noise of a KNOWN standard deviation.

    The seed is fixed, so the raster — and therefore every observable measured on
    it — is identical on every run and on every machine.
    """
    rng = np.random.default_rng(seed)
    return _mm(z_m + rng.normal(0.0, float(sigma_m), size=(size, size)))


def _depth_sparse(z_m=DEPTH_Z_M, rows=SPARSE_ROWS):
    """Clean depth over `rows` rows, sensor no-return (raw == 0) everywhere else.

    valid_depth_fraction is exactly rows/IMG. The depth that IS returned is
    perfectly clean, so this frame fails the validity band and nothing else — the
    two depth factors have to disagree about it or they are not two factors.
    """
    arr = np.zeros((IMG, IMG), dtype=np.uint16)
    arr[:rows] = int(round(z_m * 1000))
    return arr


_RGB_MAKERS = {"grey": lambda: _rgb_solid((128, 128, 128)),
               "black": lambda: _rgb_solid((0, 0, 0)),
               "white": lambda: _rgb_solid((255, 255, 255))}
_DEPTH_MAKERS = {"clean": _depth_clean,
                 "sparse": _depth_sparse,
                 "noisy": lambda: _depth_noisy(BATCH_NOISE_SIGMA_M, seed=1)}


def _write_batch(root, rgb_shift=0):
    """Materialise _FRAMES under root/rgb and root/depth.

    `rgb_shift` darkens every RGB frame by that many levels, so a second batch on
    the same trajectory differs from the first only in exposure — which is what the
    cross-batch delta query is for.
    """
    rgb_dir, depth_dir = os.path.join(root, "rgb"), os.path.join(root, "depth")
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    for stem, rgb_kind, depth_kind in _FRAMES:
        img = _RGB_MAKERS[rgb_kind]()
        if rgb_shift:
            img = Image.fromarray(
                np.clip(np.asarray(img, dtype=np.int16) - rgb_shift, 0, 255).astype(np.uint8))
        img.save(os.path.join(rgb_dir, f"{stem}.png"))
        Image.fromarray(_DEPTH_MAKERS[depth_kind]()).save(
            os.path.join(depth_dir, f"{stem}.png"))          # uint16 -> I;16


def _write_degenerate_batch(root):
    """Two frames, both mid-grey: one with clean depth, one with NO depth at all.

    Frame 1's depth raster is entirely sensor no-return, so it has no fully-valid
    3x3 window and depth_roughness is inf. It lives in its own batch so the main
    batch's "all bands wide open returns every frame" case stays honest.
    """
    rgb_dir, depth_dir = os.path.join(root, "rgb"), os.path.join(root, "depth")
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    for stem, depth in ((0, _depth_clean()),
                        (1, np.zeros((IMG, IMG), dtype=np.uint16))):
        _rgb_solid((128, 128, 128)).save(os.path.join(rgb_dir, f"{stem}.png"))
        Image.fromarray(depth).save(os.path.join(depth_dir, f"{stem}.png"))


def _write_thresholds(path, tau_lo=TAU_LO, tau_hi=TAU_HI, **band_overrides):
    bands = dict(BANDS)
    bands.update(band_overrides)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tau_lo": tau_lo, "tau_hi": tau_hi, "bands": bands}, fh, indent=2)
    return path


def _extra_rgb(name, image):
    """Write a one-off RGB fixture into the temp dir and return its path."""
    path = os.path.join(_TMP, f"{name}.png")
    image.save(path)
    return path


def _extra_depth(name, arr):
    path = os.path.join(_TMP, f"{name}.png")
    Image.fromarray(arr).save(path)
    return path


def setUpModule():
    global _PROC, _ENDPOINT, _LOG, _TMP
    # Checked before anything is written, so a skip leaves no temp directory
    # behind (unittest does not call tearDownModule after a module-level skip).
    if shutil.which("java") is None:
        raise unittest.SkipTest("java not on PATH — skipping Fuseki e2e test")
    if not os.path.exists(_JAR):
        raise unittest.SkipTest(f"Fuseki jar missing ({_JAR}); run `pixi run -e fuseki fuseki` once")

    _TMP = tempfile.mkdtemp(prefix="hw1_e2e_")
    _write_batch(os.path.join(_TMP, BATCH_DIRNAME))
    _write_batch(os.path.join(_TMP, BATCH_B_DIRNAME), rgb_shift=40)
    _write_degenerate_batch(os.path.join(_TMP, BATCH_DEGEN_DIRNAME))
    _write_thresholds(os.path.join(_TMP, "thresholds.json"))

    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    _ENDPOINT = f"{base}/ds"
    _LOG = open(os.path.join(_TMP, "fuseki.log"), "w")
    _PROC = subprocess.Popen(
        ["java", "-Xmx1g", "-jar", _JAR, "--port", str(port), "--mem", "/ds"],
        stdout=_LOG, stderr=subprocess.STDOUT, cwd=_TMP)

    if not _ping(base):
        _teardown_proc()
        tail = ""
        try:
            with open(os.path.join(_TMP, "fuseki.log")) as fh:
                tail = "".join(fh.readlines()[-15:])
        except OSError:
            pass
        raise unittest.SkipTest(f"Fuseki did not come up on {base}\n{tail}")


def _teardown_proc():
    global _PROC, _LOG
    if _PROC is not None:
        _PROC.terminate()
        try:
            _PROC.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _PROC.kill()
            _PROC.wait()
        _PROC = None
    if _LOG is not None:
        _LOG.close()
        _LOG = None


def tearDownModule():
    _teardown_proc()
    if _TMP and os.path.isdir(_TMP):
        shutil.rmtree(_TMP, ignore_errors=True)


def _insert_args(data_dir, thresholds, floor=FLOOR):
    return types.SimpleNamespace(data_dir=data_dir, floor=floor, endpoint=_ENDPOINT,
                                 thresholds=thresholds)


def _retrieve_args(out, thresholds, batch=BATCH, **overrides):
    ns = types.SimpleNamespace(
        batch=batch, out=out, endpoint=_ENDPOINT, thresholds=thresholds,
        valid_depth_min=None, roughness_max=None, clip_hi_max=None, clip_lo_max=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _read_frames(csv_path):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return sorted(int(r["frame"]) for r in rows), rows


def _thresholds_path():
    return os.path.join(_TMP, "thresholds.json")


# =============================================================================
# 1. The measurers — closed-form fixtures, tau always explicit
# =============================================================================
class Measurers(unittest.TestCase):
    """Pure checks. No server needed; these are the autograded measurer contract."""

    # ---- the baseline ----------------------------------------------------
    def test_mean_value_of_solid_frame_is_the_value(self):
        p = _extra_rgb("solid_mid_grey", _rgb_solid((128, 128, 128)))
        self.assertAlmostEqual(api.frame_mean_value(p), 128.0, places=6)

    def test_mean_value_uses_max_channel_not_a_weighted_luma(self):
        # (255,0,0): max(R,G,B) = 255 everywhere. Rec.601 luma would be 76.245.
        p = _extra_rgb("solid_red_mean", _rgb_solid((255, 0, 0)))
        self.assertAlmostEqual(api.frame_mean_value(p), 255.0, places=6)

    # ---- clip fractions, solid mid-grey ---------------------------------
    def test_solid_mid_grey_clips_neither_way(self):
        """Any 5 < tau_lo < 128 < tau_hi < 250 must give clip_hi = clip_lo = 0."""
        p = _extra_rgb("mid_grey", _rgb_solid((128, 128, 128)))
        for tau_lo, tau_hi in ((16.0, 250.0), (6.0, 249.0), (127.0, 129.0)):
            self.assertEqual(api.frame_clip_hi_fraction(p, tau_hi), 0.0)
            self.assertEqual(api.frame_clip_lo_fraction(p, tau_lo), 0.0)
        self.assertAlmostEqual(api.frame_mean_value(p), 128.0, places=6)

    # ---- LOAD-BEARING FIXTURE 1 -----------------------------------------
    def test_half_black_half_white_is_why_the_mean_is_excluded(self):
        """The executable proof that mean(V) cannot be a quality factor.

        Half the pixels are crushed to black and half are railed to white: not one
        pixel of this frame carries recoverable information. clip_lo and clip_hi
        both report exactly 0.5 and say so. The mean reports 127.5 — dead centre of
        [0,255], indistinguishable from a perfectly exposed mid-grey frame.

        A first moment is blind to distribution shape. That is the whole argument,
        and this is where it is checked.
        """
        p = _extra_rgb("half_black_half_white", _rgb_half_black_half_white())
        self.assertAlmostEqual(api.frame_clip_hi_fraction(p, TAU_HI), 0.5, places=9)
        self.assertAlmostEqual(api.frame_clip_lo_fraction(p, TAU_LO), 0.5, places=9)
        self.assertAlmostEqual(api.frame_mean_value(p), 127.5, places=6)

        # ... and the mean is indistinguishable from an actually-fine frame.
        ok = _extra_rgb("even_grey_127", _rgb_solid((128, 128, 128)))
        self.assertLess(abs(api.frame_mean_value(p) - api.frame_mean_value(ok)), 1.0)
        self.assertNotEqual(api.frame_clip_hi_fraction(p, TAU_HI),
                            api.frame_clip_hi_fraction(ok, TAU_HI))

    # ---- LOAD-BEARING FIXTURE 2 -----------------------------------------
    def test_solid_pure_red_is_why_luma_is_excluded(self):
        """The executable proof that no colorimetric weighting may be used.

        Pure red (255,0,0) has its red channel fully railed in every pixel, so
        V = max(R,G,B) = 255 and clip_hi = 1.0 — correct, the frame is clipped.
        Rec.601 luma of the same pixel is 0.299*255 = 76.245: an ordinary mid-tone.
        A weighted reduction HIDES single-channel saturation, and clipping is a
        per-channel phenomenon, so the per-channel test is the correct one.

        It is also the counter-example against the "symmetric" shadow test: pure
        red has G = B = 0, so min(R,G,B) <= tau_lo would call this fully-saturated
        frame crushed. clip_lo on V correctly reports 0.
        """
        p = _extra_rgb("solid_pure_red", _rgb_solid((255, 0, 0)))
        self.assertEqual(api.frame_clip_hi_fraction(p, TAU_HI), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, TAU_LO), 0.0)

        rec601 = 0.299 * 255.0
        self.assertLess(rec601, 128.0)                       # would read mid-tone
        self.assertEqual(api.frame_mean_value(p), 255.0)     # V does not

    # ---- degenerate ends -------------------------------------------------
    def test_all_black_and_all_white_frames(self):
        black = _extra_rgb("all_black", _rgb_solid((0, 0, 0)))
        white = _extra_rgb("all_white", _rgb_solid((255, 255, 255)))
        self.assertEqual(api.frame_clip_lo_fraction(black, TAU_LO), 1.0)
        self.assertEqual(api.frame_clip_hi_fraction(black, TAU_HI), 0.0)
        self.assertEqual(api.frame_clip_hi_fraction(white, TAU_HI), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(white, TAU_LO), 0.0)

    def test_clip_fractions_are_inclusive_at_tau(self):
        """The comparisons are >= and <=, so a pixel exactly at tau counts."""
        p = _extra_rgb("exactly_tau", _rgb_solid((200, 200, 200)))
        self.assertEqual(api.frame_clip_hi_fraction(p, 200.0), 1.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, 200.0), 1.0)
        self.assertEqual(api.frame_clip_hi_fraction(p, 201.0), 0.0)
        self.assertEqual(api.frame_clip_lo_fraction(p, 199.0), 0.0)

    def test_clip_fraction_is_closed_form_at_any_tau(self):
        """A synthetic frame has an exact clip fraction for ANY tau — which is what
        keeps the measurers autogradable when the stored observables, measured at
        each student's own tau, are not comparable at all.

        The fixture is a vertical ramp: column j has V = 4j, so
            clip_hi(tau) = |{j : 4j >= tau}| / 64   exactly.
        """
        values = (np.arange(IMG, dtype=np.uint8) * 4)         # 0, 4, ..., 252
        arr = np.repeat(values[None, :, None], IMG, axis=0).repeat(3, axis=2)
        p = _extra_rgb("value_ramp", Image.fromarray(arr))
        for tau in (0.0, 1.0, 100.0, 200.0, 252.0, 253.0):
            with self.subTest(tau=tau):
                self.assertAlmostEqual(api.frame_clip_hi_fraction(p, tau),
                                       float((values >= tau).sum()) / IMG, places=12)
                self.assertAlmostEqual(api.frame_clip_lo_fraction(p, tau),
                                       float((values <= tau).sum()) / IMG, places=12)

    # ---- depth validity ---------------------------------------------------
    def test_valid_fraction_constant_depth(self):
        full = _extra_depth("depth_full", np.full((IMG, IMG), 2000, dtype=np.uint16))
        empty = _extra_depth("depth_zero", np.zeros((IMG, IMG), dtype=np.uint16))
        self.assertEqual(api.frame_valid_fraction(full, 0.0, 10.0), 1.0)
        self.assertEqual(api.frame_valid_fraction(empty, 0.0, 10.0), 0.0)

    def test_valid_fraction_is_a_range_test_not_a_nonzero_test(self):
        """12 m is a real reading and still invalid: it is outside [0, 10] m."""
        far = _extra_depth("depth_far", np.full((IMG, IMG), 12000, dtype=np.uint16))
        self.assertEqual(api.frame_valid_fraction(far, 0.0, 10.0), 0.0)
        self.assertEqual(api.frame_valid_fraction(far, 0.0, 15.0), 1.0)

    def test_valid_fraction_exact_on_a_half_invalid_frame(self):
        arr = np.full((IMG, IMG), 2000, dtype=np.uint16)
        arr[: IMG // 2] = 0
        p = _extra_depth("depth_half", arr)
        self.assertAlmostEqual(api.frame_valid_fraction(p, 0.0, 10.0), 0.5, places=9)

    # ---- depth roughness --------------------------------------------------
    # ---- LOAD-BEARING FIXTURE 3 -----------------------------------------
    def test_depth_roughness_recovers_a_known_noise_sigma(self):
        """The estimator is validated by RECOVERING THE NOISE IT WAS GIVEN.

        This is a stronger fixture than any closed-form constant. The frame is
        constant depth plus i.i.d. Gaussian noise of a known sigma under a fixed
        seed, and depth_roughness is defined to be an estimate of exactly that
        sigma, in metres. A wrong normalisation cannot survive it: the Immerkaer
        mask amplifies an i.i.d. sigma by ||M|| = 6, and 0.6745 is the median of
        |N(0,1)|, so any implementation that drops either constant — or that uses
        mean|R| with the median's constant, which runs about 18% high — lands
        outside the tolerance.

        TOLERANCE. 10% relative. The estimator is a median over the (IMG-2)^2
        overlapping windows of one frame, so it carries a sampling error of its
        own — about 1.5% (1 s.d.) at IMG=128, measured — and the depth raster is
        quantised to whole millimetres, which granulates the median at roughly
        1/(1000*sigma). 10% is several times the worst deviation observed across
        seeds and still far tighter than any wrong-constant variant.
        """
        for sigma in (0.02, 0.05):
            with self.subTest(sigma=sigma):
                p = _extra_depth(f"rough_noise_{sigma}",
                                 _depth_noisy(sigma, size=NOISE_IMG, seed=NOISE_SEED))
                self.assertAlmostEqual(api.frame_depth_roughness(p), sigma,
                                       delta=0.10 * sigma)

    def test_depth_roughness_of_a_clean_frame_is_zero(self):
        """No noise, no roughness. Exactly 0.0, not merely small."""
        p = _extra_depth("rough_clean", _depth_clean())
        self.assertEqual(api.frame_depth_roughness(p), 0.0)

    def test_depth_roughness_annihilates_a_tilted_plane(self):
        """The mask is the difference of two Laplacians, so it kills any locally
        LINEAR depth surface — not just a constant one. A slanted wall is not
        noisy, and a metric that called it noisy would fire on half the capture.

        The ramp steps by whole millimetres so the uint16 raster is exactly affine
        and the expected residual is exactly zero.
        """
        u = np.arange(IMG, dtype=np.int64)
        arr = (2000 + 10 * u)[None, :].repeat(IMG, axis=0).astype(np.uint16)
        p = _extra_depth("rough_ramp", arr)
        self.assertLess(api.frame_depth_roughness(p), 1e-9)

    # ---- LOAD-BEARING FIXTURE 4 -----------------------------------------
    def test_depth_roughness_ignores_a_hard_step_edge(self):
        """The executable proof that the MEDIAN, not the mean, is required.

        A real depth frame contains real discontinuities — an object boundary, a
        doorway — and each one drives a large Laplacian response that is signal,
        not noise. This frame is noiseless: 3 m on one side of a hard step, 6 m on
        the other, every pixel valid. The correct answer is 0. The literature's
        mean form, sqrt(pi/2)/6 * mean|R|, reports about 0.08 m — an invented
        80 mm of "noise" in a frame that has none.

        THE EDGE IS DIAGONAL ON PURPOSE. M is separable, M = v v^T with
        v = [1,-2,1], so an AXIS-ALIGNED step is annihilated by the pass along the
        other axis and produces R = 0 everywhere. A vertical or horizontal step
        would therefore be passed by every implementation, correct or not, and
        would test nothing. Do not "simplify" this fixture.
        """
        u, v = np.meshgrid(np.arange(IMG), np.arange(IMG))
        arr = np.where(u + v < IMG, 3000, 6000).astype(np.uint16)
        p = _extra_depth("rough_step", arr)
        self.assertLess(api.frame_depth_roughness(p), 1e-9)

    # ---- LOAD-BEARING FIXTURE 5 -----------------------------------------
    def test_depth_roughness_only_reads_fully_valid_windows(self):
        """The executable proof that the nine-pixel validity mask is required.

        Dropout is encoded as raw == 0, i.e. as a 0 m reading pressed against a
        3 m one. Every window that straddles the boundary therefore looks like an
        enormous edge. Here 20% of the pixels are dropped at random, so about 86%
        of the 3x3 windows touch dropout — enough to move a median, which a single
        edge never could.

        The depth that survives is perfectly clean, so the answer is exactly 0.
        An implementation that skips the mask reports roughly 0.7 m and has
        stopped measuring noise: it is measuring dropout, which is factor 1's job.
        That is the failure this fixture exists to catch — two "independent"
        factors that are really one.
        """
        rng = np.random.default_rng(7)
        arr = _depth_clean()
        arr[rng.random((IMG, IMG)) < 0.20] = 0
        p = _extra_depth("rough_dropout", arr)
        self.assertEqual(api.frame_depth_roughness(p), 0.0)
        self.assertLess(api.frame_valid_fraction(p, 0.0, 10.0), 0.9)   # really dropped

    def test_depth_roughness_of_an_unmeasurable_frame_is_inf(self):
        """No fully-valid window anywhere -> inf, and inf FAILS every band.

        The band is `<= max`, so the two candidate "nothing to report" values are
        not symmetric: 0.0 would mean BEST and would let a frame carrying no
        usable depth at all sail through the roughness gate. When a measurement is
        impossible a quality filter must fail closed.
        """
        empty = _extra_depth("rough_empty", np.zeros((IMG, IMG), dtype=np.uint16))
        rough = api.frame_depth_roughness(empty)
        self.assertEqual(rough, float("inf"))
        self.assertFalse(rough <= BANDS["depth_roughness_max"])   # gate rejects it
        self.assertFalse(rough <= ROUGHNESS_WIDE_OPEN)            # any finite bound

    def test_depth_roughness_is_deterministic_and_a_float(self):
        p = _extra_depth("rough_repeat", _depth_noisy(0.02, seed=3))
        first = api.frame_depth_roughness(p)
        self.assertIsInstance(first, float)
        self.assertGreater(first, 0.0)
        self.assertEqual(first, api.frame_depth_roughness(p))

    def test_depth_roughness_is_unaffected_by_the_standoff_distance(self):
        """It is a noise level, not a depth: the same sigma at 2 m and at 8 m must
        read the same. A metric that drifted with range would gate on where the
        camera happened to be standing.
        """
        near = _extra_depth("rough_near", _depth_noisy(0.02, z_m=2.0, seed=5))
        far = _extra_depth("rough_far", _depth_noisy(0.02, z_m=8.0, seed=5))
        near_r = api.frame_depth_roughness(near)
        far_r = api.frame_depth_roughness(far)
        # Each is pinned to the sigma it was given first: comparing the two to each
        # other alone would be satisfied by any implementation that returns the
        # same wrong number twice.
        self.assertAlmostEqual(near_r, 0.02, delta=0.10 * 0.02)
        self.assertAlmostEqual(far_r, 0.02, delta=0.10 * 0.02)
        self.assertAlmostEqual(near_r, far_r, delta=0.10 * 0.02)


# =============================================================================
# 2. thresholds.json — the loader, and the absence of defaults
# =============================================================================
class ThresholdsFile(unittest.TestCase):

    def test_loads_a_well_formed_file(self):
        th = api.load_thresholds(_thresholds_path())
        self.assertEqual(th["tau_lo"], TAU_LO)
        self.assertEqual(th["tau_hi"], TAU_HI)
        for key, value in BANDS.items():
            self.assertAlmostEqual(th["bands"][key], value)

    def test_missing_file_raises_rather_than_defaulting(self):
        with self.assertRaises(FileNotFoundError):
            api.load_thresholds(os.path.join(_TMP, "definitely_not_here.json"))

    def test_missing_key_raises(self):
        path = os.path.join(_TMP, "incomplete.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"tau_lo": 16, "tau_hi": 250, "bands": {}}, fh)
        with self.assertRaises(ValueError):
            api.load_thresholds(path)

    def test_inverted_tau_raises(self):
        path = os.path.join(_TMP, "inverted.json")
        _write_thresholds(path, tau_lo=200.0, tau_hi=20.0)
        with self.assertRaises(ValueError):
            api.load_thresholds(path)

    def test_resolve_band_order_and_cli_override(self):
        """The 4-tuple order is the order of the query's band tokens
        @@DMIN@@ @@RMAX@@ @@CLIPHI@@ @@CLIPLO@@, and the CLI override names are
        the SHORT spellings, not the thresholds.json keys."""
        th = api.load_thresholds(_thresholds_path())
        band = api.resolve_band(th, None)
        self.assertEqual(tuple(band), (BANDS["valid_depth_fraction_min"],
                                       BANDS["depth_roughness_max"],
                                       BANDS["clip_hi_fraction_max"],
                                       BANDS["clip_lo_fraction_max"]))
        args = types.SimpleNamespace(valid_depth_min=0.9, roughness_max=None,
                                     clip_hi_max=None, clip_lo_max=0.01)
        overridden = api.resolve_band(th, args)
        self.assertEqual(overridden[0], 0.9)                       # override wins
        self.assertEqual(overridden[1], BANDS["depth_roughness_max"])
        self.assertEqual(overridden[3], 0.01)


# =============================================================================
# 3. Graph construction — the ontology contract, no server needed
# =============================================================================
class BatchGraph(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.th = api.load_thresholds(_thresholds_path())
        cls.data_dir = os.path.join(_TMP, BATCH_DIRNAME)
        cls.g, cls.name, cls.b = api.build_batch_graph(cls.data_dir, FLOOR, cls.th)

    def test_batch_name_is_floor_qualified(self):
        """floor1_x and floor2_x are different batches, and must not collide."""
        self.assertEqual(self.name, BATCH)
        self.assertEqual(str(self.b), f"{api.NS}batch/{BATCH}")
        other = api.build_batch_graph(self.data_dir, 2, self.th)[1]
        self.assertNotEqual(other, self.name)

    def test_every_frame_carries_all_five_observables(self):
        from rdflib import RDF
        frames = list(self.g.objects(self.b, api.HW1.hasFrame))
        self.assertEqual(len(frames), len(_FRAMES))
        for f in frames:
            rc = next(self.g.objects(f, api.HW1.hasRGBImage))
            dc = next(self.g.objects(f, api.HW1.hasDepthImage))
            self.assertIn((rc, RDF.type, api.HW1.RGBImage), self.g)
            self.assertIn((dc, RDF.type, api.HW1.DepthImage), self.g)
            for prop in (api.HW1.meanValue, api.HW1.clipHiFraction, api.HW1.clipLoFraction):
                self.assertIsNotNone(next(self.g.objects(rc, prop), None),
                                     f"{prop} missing on the RGBImage node")
            for prop in (api.HW1.validDepthFraction, api.HW1.depthRoughness):
                self.assertIsNotNone(next(self.g.objects(dc, prop), None),
                                     f"{prop} missing on the DepthImage node")

    def test_tau_is_recorded_on_the_batch_node(self):
        """Without tau, a stored clip fraction has no recoverable meaning."""
        self.assertEqual(float(next(self.g.objects(self.b, api.HW1.tauHi))), TAU_HI)
        self.assertEqual(float(next(self.g.objects(self.b, api.HW1.tauLo))), TAU_LO)

    def test_stored_depth_observables_are_the_measurers_output(self):
        """The store must hold what the measurers actually returned — including
        for frame 4, whose roughness is the one non-trivial depth number in the
        batch. A graph built from re-derived or rounded values is a graph that
        answers a different question than the one the files support."""
        for stem in (0, 4):
            with self.subTest(frame=stem):
                dc = next(self.g.objects(api._frame_iri(BATCH, stem),
                                         api.HW1.hasDepthImage))
                depth = os.path.join(self.data_dir, "depth", f"{stem}.png")
                self.assertAlmostEqual(
                    float(next(self.g.objects(dc, api.HW1.depthRoughness))),
                    api.frame_depth_roughness(depth), places=12)
                self.assertAlmostEqual(
                    float(next(self.g.objects(dc, api.HW1.validDepthFraction))),
                    api.frame_valid_fraction(depth), places=12)

    def test_the_two_depth_factors_disagree_about_frame_three(self):
        """Frame 3 is mostly dropout but perfectly clean where it has depth, so it
        fails the validity band with a roughness of 0. If a roughness
        implementation lets dropout leak in, this frame's two depth observables
        move together and the second factor has stopped carrying its own evidence.
        """
        dc = next(self.g.objects(api._frame_iri(BATCH, 3), api.HW1.hasDepthImage))
        self.assertAlmostEqual(float(next(self.g.objects(dc, api.HW1.validDepthFraction))),
                               SPARSE_ROWS / IMG, places=9)
        self.assertEqual(float(next(self.g.objects(dc, api.HW1.depthRoughness))), 0.0)

    def test_stored_clip_fractions_are_the_measurers_output_at_this_tau(self):
        f0 = api._frame_iri(BATCH, 0)
        rc = next(self.g.objects(f0, api.HW1.hasRGBImage))
        rgb0 = os.path.join(self.data_dir, "rgb", "0.png")
        self.assertAlmostEqual(float(next(self.g.objects(rc, api.HW1.clipHiFraction))),
                               api.frame_clip_hi_fraction(rgb0, TAU_HI), places=12)
        self.assertAlmostEqual(float(next(self.g.objects(rc, api.HW1.meanValue))),
                               api.frame_mean_value(rgb0), places=12)

    def test_changing_tau_changes_the_stored_fractions(self):
        """tau is a measurement parameter: the same frames measured at a different
        tau are different numbers. This is why changing tau demands a re-insert."""
        path = _write_thresholds(os.path.join(_TMP, "tau_wide.json"),
                                 tau_lo=200.0, tau_hi=201.0)
        g2 = api.build_batch_graph(self.data_dir, FLOOR, api.load_thresholds(path))[0]
        rc1 = next(self.g.objects(api._frame_iri(BATCH, 0), api.HW1.hasRGBImage))
        rc2 = next(g2.objects(api._frame_iri(BATCH, 0), api.HW1.hasRGBImage))
        self.assertNotEqual(float(next(self.g.objects(rc1, api.HW1.clipLoFraction))),
                            float(next(g2.objects(rc2, api.HW1.clipLoFraction))))


# =============================================================================
# 4. Query templates
# =============================================================================
class QueryTemplates(unittest.TestCase):

    def test_build_select_fully_substituted(self):
        q = api.build_select(BATCH, 0.5, 0.02, 0.25, 0.25)
        for token in ("@@NS@@", "@@BATCH@@", "@@DMIN@@", "@@RMAX@@",
                      "@@CLIPHI@@", "@@CLIPLO@@"):
            self.assertNotIn(token, q, f"{token} was never substituted")
        self.assertIn(f'hw1:batchName "{BATCH}"', q)
        self.assertIn("GRAPH ?g", q)
        for var in ("?idx", "?rgb", "?depth", "?vdf", "?rough", "?cliphi", "?cliplo"):
            self.assertIn(var, q)

    def test_build_select_escapes_and_substitutes_the_batch_name_last(self):
        """A batch name is a command-line string, not a trusted one.

        Escaping the quote keeps it inside the literal; substituting it LAST keeps a
        token that happens to appear inside it inert. Substitute the name first and
        `@@CLIPHI@@` in the name becomes a band value — a query that quietly matches
        nothing instead of failing.
        """
        q = api.build_select('weird"name @@CLIPHI@@', 0.5, 0.02, 0.25, 0.25)
        self.assertIn(r'weird\"name', q)                     # quote escaped
        self.assertIn("@@CLIPHI@@", q)                       # inert, not expanded
        self.assertNotIn('weird\\"name 0.25', q)             # would mean order is wrong
        self.assertEqual(q.count("0.25"), 2)                 # only the two real bounds

    def test_build_compare_fully_substituted(self):
        q = api.build_compare(BATCH, BATCH_B)
        for token in ("@@NS@@", "@@BATCH_A@@", "@@BATCH_B@@"):
            self.assertNotIn(token, q, f"{token} was never substituted")
        self.assertIn(f'"{BATCH}"', q)
        self.assertIn(f'"{BATCH_B}"', q)


# =============================================================================
# 5. Live round trip against Fuseki
# =============================================================================
class InsertRetrieveE2E(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(_TMP, BATCH_DIRNAME)
        cls.th_path = _thresholds_path()
        assert api.cmd_insert(_insert_args(cls.data_dir, cls.th_path)) == 0

    def test_retrieve_applies_all_four_bands(self):
        out = os.path.join(_TMP, "all_bands.csv")
        self.assertEqual(api.cmd_retrieve(_retrieve_args(out, self.th_path)), 0)
        frames, rows = _read_frames(out)
        self.assertEqual(frames, PASS_ALL_BANDS)
        self.assertEqual(set(rows[0].keys()), set(api.CSV_COLUMNS))
        for r in rows:
            self.assertTrue(r["rgb_path"].endswith(f"{r['frame']}.png"))
            self.assertTrue(os.path.exists(r["depth_path"]))

    def test_each_band_excludes_its_own_frame(self):
        """Relax three bands at a time: the frame that fails the fourth stays out.

        Note that "relaxed" is not one direction for all four. valid_depth_min is
        a lower bound, so relaxing it means 0.0; the other three are upper bounds,
        so relaxing them means LARGE. A comparator carried over from a
        higher-is-better factor shows up here as a band that excludes everything
        or nothing.
        """
        loose = {"valid_depth_min": 0.0, "roughness_max": ROUGHNESS_WIDE_OPEN,
                 "clip_hi_max": 1.0, "clip_lo_max": 1.0}
        for tightened, excluded in (("valid_depth_min", 3),
                                    ("roughness_max", 4),
                                    ("clip_hi_max", 2),
                                    ("clip_lo_max", 1)):
            with self.subTest(band=tightened):
                overrides = dict(loose)
                overrides[tightened] = BANDS[{
                    "valid_depth_min": "valid_depth_fraction_min",
                    "roughness_max": "depth_roughness_max",
                    "clip_hi_max": "clip_hi_fraction_max",
                    "clip_lo_max": "clip_lo_fraction_max"}[tightened]]
                out = os.path.join(_TMP, f"band_{tightened}.csv")
                api.cmd_retrieve(_retrieve_args(out, self.th_path, **overrides))
                frames, _ = _read_frames(out)
                self.assertNotIn(excluded, frames)
                self.assertIn(0, frames)

    def test_all_bands_wide_open_returns_every_frame(self):
        out = os.path.join(_TMP, "wide.csv")
        api.cmd_retrieve(_retrieve_args(out, self.th_path, valid_depth_min=0.0,
                                        roughness_max=ROUGHNESS_WIDE_OPEN,
                                        clip_hi_max=1.0, clip_lo_max=1.0))
        frames, _ = _read_frames(out)
        self.assertEqual(frames, [s for s, _, _ in _FRAMES])

    def test_an_unmeasurable_frame_is_rejected_by_any_roughness_band(self):
        """The fail-closed rule, end to end and through the store.

        A frame with no fully-valid window scores inf. rdflib writes that as
        "INF"^^xsd:double, Fuseki accepts it, and a SPARQL FILTER(?r <= x) is false
        against it for EVERY finite x — so the frame is gated out even with the
        band wide open, which is the point. Had the measurer returned 0.0 for
        "nothing measurable", this frame would instead be the best-scoring frame in
        the batch on the roughness factor, and the gate would wave through the one
        frame carrying no usable depth at all.
        """
        api.cmd_insert(_insert_args(os.path.join(_TMP, BATCH_DEGEN_DIRNAME),
                                    self.th_path))
        out = os.path.join(_TMP, "degenerate.csv")
        api.cmd_retrieve(_retrieve_args(out, self.th_path, batch=BATCH_DEGEN,
                                        valid_depth_min=0.0,
                                        roughness_max=ROUGHNESS_WIDE_OPEN,
                                        clip_hi_max=1.0, clip_lo_max=1.0))
        frames, _ = _read_frames(out)
        self.assertEqual(frames, [0])            # the clean frame, and only it

    def test_impossible_band_is_empty_not_an_error(self):
        out = os.path.join(_TMP, "empty.csv")
        self.assertEqual(
            api.cmd_retrieve(_retrieve_args(out, self.th_path, valid_depth_min=1.1)), 0)
        frames, _ = _read_frames(out)
        self.assertEqual(frames, [])

    def test_reinsert_replaces_the_batch(self):
        """Re-inserting REPLACES the named graph. That is what makes a tau change
        safe: no stale triple can survive next to a fresh one."""
        api.cmd_insert(_insert_args(self.data_dir, self.th_path))
        out = os.path.join(_TMP, "reinsert.csv")
        api.cmd_retrieve(_retrieve_args(out, self.th_path))
        frames, rows = _read_frames(out)
        self.assertEqual(frames, PASS_ALL_BANDS)
        self.assertEqual(len(rows), len(PASS_ALL_BANDS))       # not duplicated

    def test_changed_tau_needs_a_reinsert_to_take_effect(self):
        """The staleness rule, end to end.

        Re-measure the same batch at a tau that calls mid-grey crushed. Until the
        re-insert the store still answers at the OLD tau; after it, the new one.
        """
        strict = _write_thresholds(os.path.join(_TMP, "tau_strict.json"),
                                   tau_lo=200.0, tau_hi=250.0)
        out_stale = os.path.join(_TMP, "tau_stale.csv")
        api.cmd_retrieve(_retrieve_args(out_stale, strict))
        stale_frames, _ = _read_frames(out_stale)
        self.assertEqual(stale_frames, PASS_ALL_BANDS)         # stored at TAU_LO

        api.cmd_insert(_insert_args(self.data_dir, strict))    # re-measure
        out_fresh = os.path.join(_TMP, "tau_fresh.csv")
        api.cmd_retrieve(_retrieve_args(out_fresh, strict))
        fresh_frames, _ = _read_frames(out_fresh)
        self.assertEqual(fresh_frames, [])                     # grey now crushed

        api.cmd_insert(_insert_args(self.data_dir, self.th_path))   # restore

    def test_floor_qualified_batches_do_not_collide(self):
        """The same directory inserted as floor 2 is a DIFFERENT batch."""
        api.cmd_insert(_insert_args(self.data_dir, self.th_path, floor=2))
        out1 = os.path.join(_TMP, "floor1.csv")
        out2 = os.path.join(_TMP, "floor2.csv")
        api.cmd_retrieve(_retrieve_args(out1, self.th_path, batch=BATCH))
        api.cmd_retrieve(_retrieve_args(out2, self.th_path,
                                        batch=f"floor2_{BATCH_DIRNAME}"))
        self.assertEqual(_read_frames(out1)[0], PASS_ALL_BANDS)
        self.assertEqual(_read_frames(out2)[0], PASS_ALL_BANDS)

    def test_unknown_batch_returns_nothing(self):
        out = os.path.join(_TMP, "unknown.csv")
        api.cmd_retrieve(_retrieve_args(out, self.th_path, batch="floor9_no_such_batch"))
        self.assertEqual(_read_frames(out)[0], [])


# =============================================================================
# 6. The cross-batch join — the thing a dataframe filter is not
# =============================================================================
class CompareBatches(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.th_path = _thresholds_path()
        assert api.cmd_insert(_insert_args(os.path.join(_TMP, BATCH_DIRNAME),
                                           cls.th_path)) == 0
        assert api.cmd_insert(_insert_args(os.path.join(_TMP, BATCH_B_DIRNAME),
                                           cls.th_path)) == 0

    def test_compare_joins_the_two_batches_on_frame_index(self):
        out = os.path.join(_TMP, "compare.csv")
        args = types.SimpleNamespace(batch_a=BATCH, batch_b=BATCH_B,
                                     out=out, endpoint=_ENDPOINT)
        self.assertEqual(api.cmd_compare(args), 0)
        with open(out, newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), len(_FRAMES))              # joined on ?idx
        self.assertIn("idx", rows[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)

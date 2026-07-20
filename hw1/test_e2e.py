"""
End-to-end test for the HW1 ontology data-quality pipeline (hw1/api.py + Fuseki).

WHAT IT COVERS
    A real, black-box round trip against a live Apache Jena Fuseki server:
      1. spin up `fuseki-server.jar --mem /ds` on a free port,
      2. synthesise a tiny batch on disk (rgb/ + depth/ PNGs) with KNOWN per-frame
         luma and valid-depth so PASS/FAIL is deterministic,
      3. `api.cmd_insert` it into the store (Graph Store HTTP PUT, one named graph),
      4. `api.cmd_retrieve` it back through the SPARQL SELECT (queries/valid_frames.rq)
         and assert the CSV holds exactly the frames inside the band.

    Both the default floor-1 band and a CLI override are exercised, plus the pure
    per-frame measurers and the .rq template substitution.

RUN
    pixi run -e habitat python -m unittest hw1.test_e2e      # from repo root
    pixi run -e habitat python hw1/test_e2e.py               # direct

REQUIREMENTS
    A JVM (`java` on PATH) and the Fuseki jar under hw1/fuseki_bin/ (the `fuseki`
    pixi task downloads it). If either is missing the whole module SKIPS rather
    than fails — this is an integration test, not a unit test.
"""
import csv
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
sys.path.insert(0, _HERE)
import api  # noqa: E402  (hw1/api.py — the module under test)

_JAR = os.path.join(_HERE, "fuseki_bin", "apache-jena-fuseki-6.1.0", "fuseki-server.jar")

# Module-level handles populated by setUpModule / torn down by tearDownModule.
_PROC = None
_ENDPOINT = None
_LOG = None
_TMP = None
BATCH_NAME = "e2e_batch"

# Synthetic frames: (stem, luma_value, depth_mm). A solid RGB(v,v,v) frame has
# Rec.601 luma == v exactly; a constant depth of D mm (0 < D/1000 <= 10) is 100%
# valid, D == 0 is 0% valid. Floor-1 band = luma in [146.35, 230.87], vdf >= 0.570.
_FRAMES = [
    (0, 190, 2000),   # PASS  (bright ok, depth ok)
    (1,  50, 2000),   # FAIL  (too dark)
    (2, 200, 2000),   # PASS
    (3, 190,    0),   # FAIL  (no valid depth)
]
_PASS_DEFAULT = [0, 2]          # inside the floor-1 default band
_PASS_BMIN_195 = [2]            # with --brightness-min 195, frame 0 (luma 190) drops


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


def _write_batch(root):
    """Materialise _FRAMES under root/rgb and root/depth as 16x16 PNGs."""
    rgb_dir = os.path.join(root, "rgb")
    depth_dir = os.path.join(root, "depth")
    os.makedirs(rgb_dir)
    os.makedirs(depth_dir)
    for stem, luma, depth_mm in _FRAMES:
        Image.new("RGB", (16, 16), (luma, luma, luma)).save(
            os.path.join(rgb_dir, f"{stem}.png"))
        arr = np.full((16, 16), depth_mm, dtype=np.uint16)
        Image.fromarray(arr).save(os.path.join(depth_dir, f"{stem}.png"))  # uint16 -> I;16


def setUpModule():
    global _PROC, _ENDPOINT, _LOG, _TMP
    if shutil.which("java") is None:
        raise unittest.SkipTest("java not on PATH — skipping Fuseki e2e test")
    if not os.path.exists(_JAR):
        raise unittest.SkipTest(f"Fuseki jar missing ({_JAR}); run `pixi run -e fuseki fuseki` once")

    _TMP = tempfile.mkdtemp(prefix="hw1_e2e_")
    _write_batch(os.path.join(_TMP, BATCH_NAME))

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
            with open(os.path.join(_TMP, "fuseki.log")) as f:
                tail = "".join(f.readlines()[-15:])
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


def _insert_args(data_dir, floor=1):
    return types.SimpleNamespace(data_dir=data_dir, floor=floor, endpoint=_ENDPOINT)


def _retrieve_args(out, floor=1, bmin=None, bmax=None, dmin=None):
    return types.SimpleNamespace(
        batch=BATCH_NAME, out=out, endpoint=_ENDPOINT, floor=floor,
        brightness_min=bmin, brightness_max=bmax, valid_depth_min=dmin)


def _read_frames(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(int(r["frame"]) for r in rows), rows


class MeasurersAndTemplate(unittest.TestCase):
    """Pure checks — no server needed, but they pin the fixture's assumptions."""

    def test_frame_luma_matches_solid_value(self):
        p = os.path.join(_TMP, BATCH_NAME, "rgb", "0.png")
        self.assertAlmostEqual(api.frame_luma(p), 190.0, places=3)

    def test_valid_fraction_full_and_empty(self):
        full = os.path.join(_TMP, BATCH_NAME, "depth", "0.png")   # 2000 mm
        empty = os.path.join(_TMP, BATCH_NAME, "depth", "3.png")  # 0 mm
        self.assertAlmostEqual(api.frame_valid_fraction(full), 1.0, places=6)
        self.assertAlmostEqual(api.frame_valid_fraction(empty), 0.0, places=6)

    def test_build_select_fully_substituted(self):
        q = api.build_select(BATCH_NAME, 146.35, 230.87, 0.570)
        self.assertNotIn("@@", q)
        self.assertIn(f'hw1:batchName "{BATCH_NAME}"', q)
        self.assertIn("GRAPH ?g", q)


class InsertRetrieveE2E(unittest.TestCase):
    """Live round trip against the running Fuseki server."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(_TMP, BATCH_NAME)
        rc = api.cmd_insert(_insert_args(cls.data_dir))
        assert rc == 0

    def test_retrieve_default_band(self):
        out = os.path.join(_TMP, "default.csv")
        rc = api.cmd_retrieve(_retrieve_args(out))
        self.assertEqual(rc, 0)
        frames, rows = _read_frames(out)
        self.assertEqual(frames, _PASS_DEFAULT)
        # header + path columns present and point at the real files
        self.assertEqual(set(rows[0].keys()),
                         {"frame", "rgb_path", "depth_path", "luma", "valid_fraction"})
        for r in rows:
            self.assertTrue(r["rgb_path"].endswith(f"{r['frame']}.png"))
            self.assertTrue(os.path.exists(r["depth_path"]))

    def test_retrieve_brightness_override_narrows_set(self):
        out = os.path.join(_TMP, "override.csv")
        rc = api.cmd_retrieve(_retrieve_args(out, bmin=195.0))
        self.assertEqual(rc, 0)
        frames, _ = _read_frames(out)
        self.assertEqual(frames, _PASS_BMIN_195)

    def test_retrieve_impossible_band_is_empty(self):
        out = os.path.join(_TMP, "empty.csv")
        rc = api.cmd_retrieve(_retrieve_args(out, dmin=1.1))  # vdf can never exceed 1.0
        self.assertEqual(rc, 0)
        frames, _ = _read_frames(out)
        self.assertEqual(frames, [])

    def test_reinsert_is_idempotent(self):
        # Re-inserting the same batch replaces its named graph, not appends to it.
        api.cmd_insert(_insert_args(self.data_dir))
        out = os.path.join(_TMP, "reinsert.csv")
        api.cmd_retrieve(_retrieve_args(out))
        frames, _ = _read_frames(out)
        self.assertEqual(frames, _PASS_DEFAULT)


if __name__ == "__main__":
    unittest.main(verbosity=2)

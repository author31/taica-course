# HW1 description logic

This document describes the semantic layer in `hw1/ontology/hw1.ttl`.
`hw1:` abbreviates the ontology namespace and `schema:` abbreviates
`https://schema.org/`. Instance data uses `http://taica.course/hw1/data/`,
with `batch:`, `frame:`, `rgb:`, `depth:`, `exp:`, `factor:`, `setting:`,
and `run:` prefixes in serialized Turtle.

The ontology provides vocabulary, domains, ranges, subclass relationships,
and documentation. RDFLib does not run a complete OWL reasoner and RDF is
open world, so exact cardinality, uniqueness, capture adjacency,
expected-measurement coverage, and completion are enforced by
`hw1/semantic_model.py`. The axioms below describe the intended model;
parsing Turtle alone does not validate it.

## Terminology

Two levels share the word "factor". This document keeps them distinct:

- `FactorDefinition` — reusable definition of what a number means and what
  counts as good. Ontology IRI: `hw1:QualityFactor`.
- `FactorMeasurement` — one experiment-scoped measurement of one definition
  on one image or image pair. Ontology IRI: `hw1:Factor`.
- `hasDefinition` — link from a measurement to its definition.
  Ontology IRIs: `hw1:hasDefinition` (primary), `hw1:factorType` (compatible
  alias). New writers emit `hw1:hasDefinition`; readers accept both.

IRIs are unchanged for compatibility; the new names are documentation
aliases used consistently below and in diagnostics.

The identity shapes are:

```text
batch       = <http://taica.course/hw1/data/batch>/<batch-name>
frame       = <http://taica.course/hw1/data/batch>/<batch-name>/frame/<index>
rgb image   = <http://taica.course/hw1/data/batch>/<batch-name>/rgb/<index>
depth image = <http://taica.course/hw1/data/batch>/<batch-name>/depth/<index>
experiment  = <http://taica.course/hw1/data/experiment>/<experiment-name>
measurement = <…/experiment>/<name>/factor/<definition>_<index>
pair meas.  = <…/experiment>/<name>/factor/<definition>_<previous>_<current>
setting     = <…/experiment>/<name>/setting/<factor>_<parameter>
run         = <…/experiment>/<name>/run/<mode>
```

## Capture structure

```text
Batch ──hasFrame──> Frame ──hasRGBImage────> RGBImage
                         └─hasDepthImage──> DepthImage
```

### Batch

`Batch` is one capture collection and the shared structural identity for
experiments over the same pixels.

```text
Batch ⊑ ∃hasFrame.Frame
Batch ⊑ ∀hasFrame.Frame
Batch ⊑ (≥ 1 hasFrame.Frame)
hasFrame : Batch → Frame
batchName : Batch → xsd:string
batchPath : Batch → xsd:string
hasGenerationSetting : Batch → FactorSetting
```

The validator requires a nonempty Batch. `batchName` identifies the capture
condition; `batchPath` locates local pixels. Generation settings describe
how the pixels were produced.

### Frame and images

```text
Frame ⊑ (= 1 hasRGBImage.RGBImage)
Frame ⊑ (= 1 hasDepthImage.DepthImage)
Frame ⊑ (= 1 frameIndex.xsd:integer)
RGBImage ⊑ schema:ImageObject
DepthImage ⊑ schema:ImageObject
RGBImage ⊓ DepthImage ⊑ ⊥
```

`frameIndex` is unique within a Batch and determines numeric order. Adjacent
Frames are consecutive members of that sorted order, even if source stems
have gaps; integer `+1` is not required. Each image has its
`schema:contentUrl`, belongs to one Frame, and carries no
experiment-specific result.

## Experiment

An `Experiment` is a sealed assessment declaration over one Batch. It
selects reusable definitions and records its effective settings.

```text
Experiment ⊑ (= 1 onBatch.Batch)
Experiment ⊑ (≥ 1 evaluatesFactor.FactorDefinition)
Experiment ⊑ ∀evaluatesFactor.FactorDefinition
Experiment ⊑ ∀hasFactorSetting.FactorSetting
onBatch : Experiment → Batch
batchFile : Experiment → xsd:string
declarationDigest : Experiment → xsd:string
evaluatesFactor : Experiment → FactorDefinition
hasFactorSetting : Experiment → FactorSetting
```

`onBatch` is the RDF join; `batchFile` is the local capture-directory path
used by tools. `declarationDigest` seals the student-authored section above
the machine marker. `evaluatesFactor` names definitions, never individual
measurements. An assessed file embeds the complete Batch/Frame/image
snapshot below the machine marker so it is independently queryable; the
declaration section itself contains only the Experiment, batch path/join,
selection, settings, and human-authored label/prediction text.

## Factor definitions

A `FactorDefinition` is reusable and ownerless. It says what a number means,
which direction is better, which parameter holds the threshold, and which
images it applies to.

```text
FactorDefinition ⊑ (= 1 polarity.Polarity)
FactorDefinition ⊑ (= 1 qualifiedBy.Parameter)
FactorDefinition ⊑ (= 1 appliesTo.rdfs:Class)
FactorDefinition ⊑ (= 1 targetKind.TargetKind)
FactorDefinition ⊑ (= 1 evaluationPhase.EvaluationPhase)
polarity : FactorDefinition → Polarity
qualifiedBy : FactorDefinition → Parameter
appliesTo : FactorDefinition → rdfs:Class
targetKind : FactorDefinition → TargetKind
evaluationPhase : FactorDefinition → EvaluationPhase
```

| Definition | Target | Measurement shape |
| --- | --- | --- |
| `HighlightClipping`, `ShadowClipping` | `RGBTarget` | one RGBImage |
| `HighFrequencyDepthResidual`, `FlyingPixelRatio`, `ValidTileCoverage` | `DepthTarget` | one DepthImage |
| `IdentityMedianDepthChange`, `JointValidDepthRatio`, `PriorWarpDepthResidual` | `DepthPairTarget` | ordered adjacent DepthImages |
| `ReconstructionAccuracy`, `Coverage` | run outcome | `ReconstructionRun` |

`HigherIsBetter` grades `value ≥ threshold`; `LowerIsBetter` grades
`value ≤ threshold`. Equality passes. Thresholds come from the same
Experiment's effective `QualificationSetting`.

## Factor measurements

A `FactorMeasurement` is one experiment-scoped occurrence. It owns its
value, status, masks, and support counts; shared Frame and image nodes never
carry experiment-specific measurements.

```text
FactorMeasurement ⊑ (= 1 inExperiment.Experiment)
FactorMeasurement ⊑ (= 1 hasDefinition.FactorDefinition)
FactorMeasurement ⊑ (= 1 hasCurrentFrame.schema:ImageObject)
FactorMeasurement ⊑ (= 1 evaluationState.EvaluationState)
inExperiment : FactorMeasurement → Experiment
hasDefinition : FactorMeasurement → FactorDefinition
hasCurrentFrame : FactorMeasurement → schema:ImageObject
value : FactorMeasurement → xsd:double
status : FactorMeasurement → Status
evaluationState : FactorMeasurement → EvaluationState
supportCount : FactorMeasurement → xsd:integer
maskFile : FactorMeasurement → xsd:string
```

Name warning: `hasCurrentFrame` points to an image, not a Frame, despite its
name. `hasPrevious` likewise points to the earlier DepthImage. Both names
are kept unchanged for compatibility.

The closed-world measurement keys are:

```text
single image = (Experiment, definition, current image)
depth pair   = (Experiment, definition, previous image, current image)
```

Duplicate keys are invalid even when values agree. A measurement must belong
to its Experiment and use an image from that Experiment's Batch.

### Single-image measurements

```text
SingleImageFactor ⊑ FactorMeasurement
SingleImageFactor ⊑ ¬DepthPairFactor
SingleImageFactor ⊑ (= 0 hasPrevious.DepthImage)
```

Its current image must match `targetKind`: RGB definitions use `RGBImage`;
depth definitions use `DepthImage`. For N Frames and U selected single-image
definitions, exactly N×U keys are expected.

### Depth-pair measurements

```text
DepthPairFactor ⊑ FactorMeasurement
DepthPairFactor ⊑ ¬SingleImageFactor
DepthPairFactor ⊑ (= 1 hasPrevious.DepthImage)
DepthPairFactor ⊑ (= 1 hasCurrentFrame.DepthImage)
```

`hasPrevious` is earlier and `hasCurrentFrame` is later. Endpoints must be
distinct, depth images from adjacent Batch Frames, and ordered correctly.
RGB or mixed pairs are invalid. For N Frames and P selected pair
definitions, exactly (N−1)×P keys are expected.

## Evaluation states and status

```text
EvaluationState = {Pending, Measured, MeasurementError}
Status = {Pass, Fail}
Measured ⊑ (= 1 value.xsd:double)
Measured ⊑ (= 1 status.Status)
Pending ⊑ (= 0 value.xsd:double)
Pending ⊑ (= 0 status.Status)
MeasurementError ⊑ (= 1 status.Fail)
```

`Pending` represents deferred evidence, especially prior-warp depth
residuals before baseline reconstruction. `MeasurementError` records a
failed attempt and is not completion. A missing measurement must never be
inferred as Pass. A successful non-finite metric follows that metric's
fail-closed grading rule and remains distinguishable from missing evidence.

## FullEvaluatedFrames

```text
FullEvaluatedFrames ⊑ Experiment
```

This generated class means **complete evaluation**, not all measurements
passing. For ordered Batch frames F and selected definitions S:

```text
R = {(s, image(f)) | s is single-image, f ∈ F}
  ∪ {(s, depth(fᵢ), depth(fᵢ₊₁)) |
       s is pair and (fᵢ, fᵢ₊₁) are adjacent in F}
```

An Experiment is a `FullEvaluatedFrames` individual iff structure and
settings are valid, actual measurement keys equal R with no duplicates or
foreign targets, and every expected occurrence is `Measured` with one
numeric value and one correctly graded status. All values may Fail. Pending,
missing, errored, wrong-modality, unexpected, or duplicate measurements
prevent completion. A one-frame Batch has zero pair obligations; empty
Batches and empty selections are rejected.

Completion is bound to the Experiment because the requirement R depends on
that Experiment's selection S: two experiments over one Batch have different
R sets and disjoint measurements. Asserting completion on shared Batch or
Frame nodes would conflate them.

The validator materializes or removes this type and rejects stale/forged
assertions. Inspection may read incomplete graphs to report missing
evidence; production qualification and reconstruction selection require
completion. Deferred pair evidence is written as Pending first and updated
after baseline reconstruction; absent returned evidence remains Pending.

## Settings and parameters

```text
Parameter ⊑ (= 1 paramRole.SettingRole)
Parameter ⊑ (= 1 paramValueKind.xsd:string)
Parameter ⊑ (= 1 paramPrimaryFactor.FactorDefinition)
paramRole : Parameter → SettingRole
paramValueKind : Parameter → xsd:string
paramPrimaryFactor : Parameter → FactorDefinition
paramAffectsFactor : Parameter → FactorDefinition
paramDefault : Parameter → literal

FactorSetting ⊑ (= 1 settingParameter.Parameter)
FactorSetting ⊑ (= 1 settingRole.SettingRole)
FactorSetting ⊑ (= 1 settingForFactor.FactorDefinition)
FactorSetting ⊑ (= 1 settingValue.literal)
settingParameter : FactorSetting → Parameter
settingRole : FactorSetting → SettingRole
settingForFactor : FactorSetting → FactorDefinition
settingValue : FactorSetting → literal
```

```text
SettingRole = {GenerationSetting, MeasurementSetting, QualificationSetting}
```

Generation settings attach to a Batch; measurement and qualification
settings attach to an Experiment. The application resolves one effective
value for every required Parameter. An override wins over a TBox default,
and a setting affecting no selected definition is invalid.

## ReconstructionRun

`ReconstructionRun` is an outcome of reconstruction, not an input
measurement.

```text
ReconstructionRun ⊑ (= 1 selectionMode.SelectionMode)
ReconstructionRun ⊑ ∀usedFrame.Frame
selectionMode : ReconstructionRun → SelectionMode
usedFrame : ReconstructionRun → Frame
runFrameCount : ReconstructionRun → xsd:integer
maskFactor : ReconstructionRun → FactorDefinition
diagnosticFile : ReconstructionRun → xsd:string
mapMeanL2 : ReconstructionRun → xsd:double
mapMeanL2Status : ReconstructionRun → Status
coverageF : ReconstructionRun → xsd:double
coverageFStatus : ReconstructionRun → Status
gatedSteps : ReconstructionRun → xsd:integer
spliceCount : ReconstructionRun → xsd:integer
maxGapLength : ReconstructionRun → xsd:integer
```

`SelectionMode` is `{FullBatch, GoodSegments, MaskFiltered}`. Selected runs
record consumed Frames and segment diagnostics; full-batch runs record their
count.

## Closed vocabularies

```text
Polarity = {HigherIsBetter, LowerIsBetter}
Status = {Pass, Fail}
TargetKind = {RGBTarget, DepthTarget, DepthPairTarget}
EvaluationPhase = {CaptureAssessment, BaselineReconstruction}
```

These are vocabulary individuals, not classes of measurements. Target kind
and evaluation phase belong to reusable definitions; a measurement reaches
them through `hasDefinition`.

## Qualification semantics

Qualification is a query policy over measurement occurrences. It is not a
type asserted on shared Frames and is not equivalent to completeness. Every
production query must bind one Experiment, enumerate Frames through its
Batch, and join measurements by exact image identity.

Supported policy forms include a single definition passing, AND/OR
combinations, explicit Pass-and-Fail combinations, measured Fail as a safe
"not pass", and an incoming depth-pair pass. Unrestricted absence of a Pass
triple is invalid because it conflates Fail, Pending, and missing evidence.
Production selection requires `FullEvaluatedFrames`; inspection may query
incomplete graphs.

## Global invariants

1. Every Frame has exactly one RGBImage, one DepthImage, and one unique integer index within its Batch.
2. Images are not shared between Frames and RGB/Depth modality is consistent.
3. Every measurement has one owner, one definition, one current image, and one evaluation state.
4. Single measurements have no previous image; pair measurements have exactly two adjacent depth endpoints.
5. Measurement keys are neither duplicated, missing, unexpected, nor foreign to the Experiment's Batch.
6. Measured occurrences have one numeric value and one status consistent with threshold and polarity.
7. Pending/error evidence never qualifies a frame or completes an Experiment.
8. A stale or forged `FullEvaluatedFrames` assertion is removed or rejected.
9. Query results must use exact experiment Batch membership and valid numeric frame indices.

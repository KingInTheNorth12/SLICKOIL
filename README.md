# Oil Spill Detection, Source Tracing & Vessel Attribution

An automated, modular pipeline for detecting marine oil spills from satellite imagery, tracing their likely origin using oceanographic and meteorological data, correlating the reconstructed source with historical AIS vessel traffic, and forecasting future slick movement.

The project is designed as a research and benchmarking platform: individual segmentation models, drift models, attribution algorithms, and data providers can be replaced without rewriting the end-to-end pipeline.

---

## Problem Statement

### Title

**Leveraging Satellite Imagery to Determine Oil Spills at Sea Along with AIS Data Correlations to Identify the Vessel Responsible for the Spill**

### Background

Marine oil spills can cause severe damage to marine ecosystems, coastal environments, fisheries, and maritime infrastructure.

In many cases, the vessel responsible for a spill cannot be immediately identified.

Satellite remote-sensing imagery provides a way to detect possible oil slicks over large ocean areas, while historical Automatic Identification System (AIS) data can provide information about vessel movements around the affected location.

Combining satellite observations, AIS vessel trajectories, and environmental data can therefore support both:

* detection of marine oil spills; and
* attribution of a detected spill to potential responsible vessels.

### Challenge

The core challenge is to develop an intelligent automated pipeline capable of using remote-sensing satellite data, including Synthetic Aperture Radar (SAR) and optionally Earth Observation imagery, together with AIS, oceanographic, meteorological and geospatial data.

The system should perform three major tasks.

#### A. Oil-spill detection and characterization

The pipeline should detect possible oil slicks from satellite imagery and characterize the detected slick.

Where feasible, derived properties may include:

* spill mask;
* confidence score;
* spill class or morphology;
* polygon;
* area;
* centroid;
* length;
* width;
* orientation;
* centerline or skeleton;
* approximate spill-age or discharge-time window.

#### B. Source tracing and future prediction

Using environmental information such as:

* ocean currents;
* wind;
* coastline data;
* acquisition timestamp;
* slick geometry;

the system should estimate how the slick may have moved.

This includes:

* backward or hindcast tracing toward a possible source region;
* estimation of a plausible source time;
* forward simulation from candidate source locations;
* future slick forecasting;
* uncertainty estimation;
* possible coastal-impact prediction.

#### C. Vessel attribution

Historical AIS data should be used to reconstruct vessel traffic around the estimated source region and time window.

Irrelevant traffic should be filtered.

Remaining candidate vessels should then be ranked using evidence such as:

* spatial proximity;
* temporal proximity;
* vessel trajectory;
* trajectory consistency;
* direction of movement;
* distance from reconstructed source;
* forward drift agreement;
* backward drift agreement;
* behavioural features or anomalies where supported.

The final result should provide a ranked list of candidate vessels together with the evidence contributing to each score.

---

## Expected Solution

The expected system is an automated oil-spill detection, hindcasting, vessel-attribution and forecasting pipeline.

Conceptually:

```text
Satellite Imagery
        │
        ▼
SAR / EO Preprocessing
        │
        ▼
Oil-Spill Detection
        │
        ▼
Slick Characterization
        │
        ▼
Spill Age / Source-Time Estimation
        │
        ├───────────────────────┐
        ▼                       ▼
Historical AIS          Wind / Ocean Current
        │                       │
        ▼                       ▼
Candidate Vessels      Environmental Forcing
        │                       │
        └───────────┬───────────┘
                    ▼
             Drift Simulation
             ┌──────┼──────┐
             ▼      ▼      ▼
          Forward Backward Future
           Trace   Trace   Forecast
             │      │      │
             └──┬───┘      │
                ▼          │
         Vessel Attribution │
                │          │
                ▼          ▼
         Suspected Vessel  Ensemble Forecast
                │          │
                │          ▼
                │     Coastal Impact
                │          │
                └────┬─────┘
                     ▼
                  Dashboard
```

---

# Project Goals

The project has five primary goals:

1. **Detect oil slicks** from satellite imagery.
2. **Characterize detected slicks** geometrically and geospatially.
3. **Hindcast slick movement** to estimate possible origin locations and times.
4. **Rank candidate vessels** using historical AIS and drift consistency.
5. **Forecast future slick movement** and potential coastal impact.

A sixth goal is equally important from an engineering perspective:

6. **Provide a modular benchmarking framework** so alternative models and algorithms can be compared under identical conditions.

---

# Design Principles

## 1. Modular by default

No major pipeline component should depend directly on a particular implementation.

For example, pipeline code should depend on:

```python
DetectorModel
```

rather than:

```python
YOLOv8
```

Likewise, the pipeline should depend on:

```python
DriftModel
```

rather than directly depending on OpenOil throughout the codebase.

---

## 2. Models should be interchangeable

The initial detector may be YOLOv8-Seg.

However, alternative models should be usable through the same interface, for example:

* YOLOv8-Seg;
* U-Net;
* DeepLabV3+;
* Mask R-CNN;
* SegFormer;
* future segmentation models.

Switching models should require changing configuration rather than pipeline source code.

Example:

```yaml
detector:
  name: yolov8_seg
  config: configs/models/yolov8_seg.yaml
```

can become:

```yaml
detector:
  name: deeplabv3plus
  config: configs/models/deeplabv3plus.yaml
```

without changing the pipeline.

---

## 3. Scientific components should also be interchangeable

Modularity is not limited to machine-learning models.

The following components should have stable interfaces:

```text
SARPreprocessor
DetectorModel
SpillAgeEstimator
AISProvider
CandidateGenerator
EnvironmentalProvider
DriftModel
AttributionModel
EnsembleForecaster
CoastalImpactAnalyzer
```

This allows experiments such as:

```text
Experiment A
YOLOv8-Seg
+ OpenOil
+ Weighted Attribution V1
```

versus:

```text
Experiment B
DeepLabV3+
+ OpenOil
+ Weighted Attribution V1
```

or:

```text
Experiment C
YOLOv8-Seg
+ OpenOil
+ Attribution V2
```

without rebuilding the entire application.

---

# Proposed Architecture

```text
                         CONFIGURATION
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      DetectorRegistry   DriftRegistry  AttributionRegistry
             │                │                │
       ┌─────┼─────┐          │          ┌─────┼─────┐
       ▼     ▼     ▼          ▼          ▼           ▼
     YOLO  U-Net  Other    OpenOil    Weighted     Future
                                       Scorer       Models
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     Pipeline Orchestrator
                              │
                              ▼
                     Standardized Results
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
              Benchmark                Dashboard
```

The orchestrator coordinates components.

It should contain very little scientific logic itself.

---

# Pipeline

## 1. Data Sources

Initial data sources may include:

### Satellite imagery

* Sentinel-1 GRD;
* VV polarization;
* VH polarization;
* optionally other SAR/EO products.

### Vessel information

* historical AIS.

### Environmental information

* wind;
* ocean currents;
* potentially wave information.

### GIS information

* coastline;
* land/sea masks;
* administrative or response zones where required.

---

## 2. SAR Preprocessing

The preprocessing stage converts raw satellite products into data suitable for detection.

A typical Sentinel-1 GRD workflow may include:

```text
GRD
 ↓
Calibration
 ↓
Speckle Filtering
 ↓
Terrain Correction
 ↓
Land / Sea Mask
 ↓
Georeferenced VV / VH
```

Possible libraries/tools include:

* SNAP;
* pyroSAR;
* Rasterio;
* GDAL.

Exact preprocessing parameters should remain configurable.

---

## 3. Oil-Spill Detection

The initial baseline detector is:

**YOLOv8-Seg**

The detector should output a standardized representation such as:

```text
SpillDetection
├── mask
├── probability/confidence
├── class
├── raster metadata
├── acquisition time
└── model metadata
```

Model-specific tensors or objects should not be exposed to downstream modules.

---

# Detector Interface

Conceptually:

```python
class DetectorModel(Protocol):

    def load(self) -> None:
        ...

    def predict(self, scene: SARScene) -> SpillDetection:
        ...

    def predict_batch(
        self,
        scenes: Sequence[SARScene],
    ) -> list[SpillDetection]:
        ...

    def model_metadata(self) -> ModelMetadata:
        ...
```

Models are instantiated through a registry.

Example:

```python
detector = detector_registry.create(
    config.detector.name,
    config.detector,
)

prediction = detector.predict(scene)
```

The pipeline must not contain logic such as:

```python
if detector == "yolo":
    ...
elif detector == "unet":
    ...
```

---

# 4. Slick Characterization

Detected masks are converted into geospatial slick representations.

Potential properties include:

```text
Mask
 ↓
Polygon
 ↓
Area
 ↓
Centroid
 ↓
Length / Width
 ↓
Orientation
 ↓
Skeleton / Centerline
```

Potential libraries:

* OpenCV;
* Rasterio;
* Shapely;
* GeoPandas;
* scikit-image.

Geographic CRS coordinates should not be treated directly as physical metre/area measurements.

---

# 5. Spill-Time Estimation

Satellite acquisition time provides the observed slick time:

```text
T0 = satellite acquisition timestamp
```

The system may estimate a plausible discharge-time interval:

```text
T0 - Δt_max  →  T0
```

The exact estimator should remain replaceable through a:

```text
SpillAgeEstimator
```

interface.

If scientifically validated age estimation is unavailable, this component should return a configurable plausible search window rather than claiming an unsupported exact spill age.

---

# 6. AIS Candidate Generation

Historical AIS tracks are searched within a configurable:

```text
spatial window × temporal window
```

Candidate generation may perform:

* temporal filtering;
* spatial filtering;
* track reconstruction;
* interpolation;
* direction consistency calculations;
* vessel metadata lookup where available.

Candidate generation and final vessel attribution must remain separate stages.

---

# 7. Environmental Forcing

The pipeline ingests historical and forecast environmental data.

Examples:

```text
Wind
Ocean Current
```

Environmental data should be represented using a common interface regardless of provider.

Possible representations include:

* NetCDF;
* GRIB;
* xarray datasets;
* provider APIs.

---

# 8. Oil Drift Simulation

The initial physics engine is expected to use:

**OpenDrift / OpenOil**

The software should expose a generic:

```python
DriftModel
```

interface.

Conceptually it should support:

```python
forward_source_trace(...)
backward_source_trace(...)
forecast(...)
```

OpenOil-specific logic must remain inside its adapter.

---

# 9. Forward Source Trace

For each AIS candidate:

```text
Historical vessel position
        ↓
Hypothetical release
        ↓
Forward drift simulation
        ↓
Predicted slick at satellite time T0
        ↓
Compare against observed slick
```

A strong spatial match provides evidence supporting that vessel-source hypothesis.

---

# 10. Backward Source Trace

Starting from the detected slick:

```text
Observed slick at T0
        ↓
Backward / hindcast simulation
        ↓
Possible source region
        ↓
Possible source-time window
```

The resulting source region can be compared with historical AIS vessel trajectories.

---

# 11. Vessel Attribution

Potential vessel-ranking evidence includes:

```text
Forward-fit score
+
Backward-fit score
+
AIS trajectory consistency
+
Source distance
+
Temporal consistency
+
Directional consistency
+
Spatial overlap
+
Hausdorff distance
```

The initial implementation may use a configurable weighted scoring model.

Example:

```text
score(vessel) =
    w1 × forward_fit
  + w2 × backward_fit
  + w3 × trajectory_consistency
  + w4 × temporal_consistency
  + w5 × spatial_overlap
```

Weights should live in configuration.

They should not be represented as scientifically validated unless validation data supports that conclusion.

Each ranked candidate should expose its individual evidence rather than only an opaque final score.

---

# 12. Future Forecast

Starting from the observed slick at `T0`:

```text
Observed slick
      ↓
Multiple OpenOil runs
      ↓
Different environmental /
uncertainty conditions
      ↓
Particle ensemble
      ↓
Probability / uncertainty field
```

Target forecast horizons may initially include:

```text
+6 hours
+12 hours
+24 hours
+48 hours
```

---

# 13. Coastal Impact

Forecast trajectories can be compared with coastline geometries to estimate:

* potential impacted coastline segments;
* probability of impact;
* approximate arrival time where supported;
* forecast uncertainty.

---

# 14. Dashboard

The final visual interface should be able to display:

* satellite scene;
* detected slick;
* slick geometry;
* reconstructed source region;
* candidate vessel tracks;
* suspected vessel ranking;
* attribution evidence;
* forward/backward simulations;
* future slick forecasts;
* probability fields;
* possible coastal impact.

The dashboard should consume standardized pipeline output rather than directly executing scientific algorithms.

---

# Repository Structure

```text
oil-spill-tracer/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
├── .env.example
│
├── configs/
│   ├── pipeline.yaml
│   │
│   ├── models/
│   │   ├── yolov8_seg.yaml
│   │   ├── unet.yaml
│   │   └── deeplabv3plus.yaml
│   │
│   ├── drift/
│   │   └── openoil.yaml
│   │
│   └── benchmarks/
│       ├── detection.yaml
│       └── end_to_end.yaml
│
├── src/
│   └── oilspill/
│       ├── core/
│       ├── data/
│       ├── preprocessing/
│       ├── detection/
│       ├── characterization/
│       ├── traceability/
│       ├── drift/
│       ├── attribution/
│       ├── forecasting/
│       ├── pipeline/
│       ├── benchmark/
│       └── cli.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── benchmarks/
│   ├── manifests/
│   └── results/
│
├── scripts/
│
├── notebooks/
│
└── data/
    ├── raw/
    ├── interim/
    └── processed/
```

---

# Test Fixtures

Development and automated tests should use small fixtures.

These fixtures **do not need to represent a real oil-spill incident**.

Synthetic data is acceptable as long as its schema and geospatial relationships resemble the actual data used by the pipeline.

Example:

```text
tests/fixtures/
├── sar/
│   ├── example_vv.tif
│   ├── example_vh.tif
│   └── expected_mask.tif
│
├── ais/
│   └── tracks.csv
│
├── metocean/
│   └── forcing.nc
│
└── gis/
    └── coastline.geojson
```

For example, an AIS test dataset could contain several synthetic vessels:

```csv
vessel_id,timestamp,latitude,longitude,sog,cog
TEST_001,2026-01-01T10:00:00Z,12.400,74.800,12.2,90
TEST_001,2026-01-01T10:10:00Z,12.400,74.830,12.1,91
TEST_002,2026-01-01T10:00:00Z,12.650,74.600,8.4,180
```

Test data should preferably include deliberate cases such as:

* one vessel passing close to the reconstructed source;
* one vessel far away;
* one vessel close spatially but outside the time window;
* missing AIS observations;
* duplicated AIS points;
* irregular observation intervals.

This makes the tests useful without requiring large operational datasets.

Real validation and benchmarking must use suitable real-world or scientifically validated datasets.

---

# Configuration

The pipeline is controlled through configuration.

Example:

```yaml
detector:
  name: yolov8_seg
  config: configs/models/yolov8_seg.yaml

drift:
  name: openoil
  config: configs/drift/openoil.yaml

spill_age:
  name: configurable_window

attribution:
  name: weighted_trajectory_v1

forecast:
  ensemble_size: 50
  horizons_hours:
    - 6
    - 12
    - 24
    - 48
```

---

# Benchmarking

Benchmarking is treated as a core project feature.

The same model interface used by the production pipeline is also used by the benchmark system.

Example:

```yaml
experiment:
  name: segmentation_comparison
  seed: 42

models:
  - name: yolov8_seg
    config: configs/models/yolov8_seg.yaml

  - name: unet
    config: configs/models/unet.yaml

  - name: deeplabv3plus
    config: configs/models/deeplabv3plus.yaml
```

This allows models to be compared under identical preprocessing, datasets and metrics.

---

## Detection Metrics

Where ground truth permits, measurements may include:

* Intersection over Union;
* Dice/F1;
* precision;
* recall;
* segmentation mAP;
* inference latency;
* throughput;
* peak GPU memory.

---

## Attribution Metrics

Where vessel ground truth exists:

* top-1 vessel attribution accuracy;
* top-k accuracy;
* source-location error;
* discharge-time error;
* candidate recall;
* ranking metrics.

---

## Drift / Forecast Metrics

Where observational ground truth exists:

* trajectory distance;
* Hausdorff distance;
* spatial overlap;
* forecast displacement error;
* probability calibration;
* coastline-impact accuracy.

---

# Experiment Reproducibility

Each benchmark run should generate a new experiment directory:

```text
benchmarks/results/<experiment-id>/
├── config.yaml
├── metrics.json
├── summary.csv
├── per_sample.parquet
├── environment.json
└── report.md
```

Experiment metadata should capture:

* model;
* checkpoint/version;
* dataset version;
* preprocessing configuration;
* random seed;
* hardware;
* software versions;
* Git commit;
* runtime;
* metrics.

Benchmark results must never be silently overwritten.

---

# Command-line interface

Install the package into the active environment (`pip install -e .`) to expose the
`oilspill` executable. The CLI is a thin composition layer: it validates arguments and
delegates to application services, registries, benchmark runners, or the pipeline
orchestrator.

Run the configured pipeline:

```bash
oilspill run \
    --config configs/pipeline.yaml
```

## Hybrid Eulerian-Lagrangian MVP

The optional `hybrid_mvp` mode uses the NumPy finite-volume Eulerian model only as a
fast surface-transport screen. It samples physics-first source states independently of
AIS, retains the strongest coarse matches, and then evaluates that subset through the
configured high-fidelity `DriftModel`. OpenOil therefore remains the intended
high-fidelity physical model; the Eulerian solver does not replace its particle,
weathering, vertical-process, or oil-property capabilities.

AIS is queried only after an uncalibrated source posterior exists. The posterior credible
region and release-time interval define the AIS search window. Vessel rankings are
investigative prioritization, not proof of discharge. Forecasting is separately initialized
from the observed slick probability raster or mask, so it does not depend on AIS,
attribution, or an inferred source.

Run the deterministic orchestration demonstration with:

```bash
pytest -q tests/test_hybrid_pipeline_orchestrator.py
```

The configuration template is `configs/pipeline.hybrid.synthetic.yaml`; its fixture
artifact placeholders must be replaced with paths produced by the synthetic fixture
generator before invoking `oilspill run --config configs/pipeline.hybrid.synthetic.yaml`
directly. It selects `numpy_fvm`, `fake_drift`, small Sobol designs, and values explicitly
marked `DATASET_VALIDATION_REQUIRED`. Select `openoil` in the drift component for
high-fidelity operational experiments; that requires the optional OpenDrift/OpenOil
dependency and validated local forcing.

Scientifically unvalidated MVP elements include transport ranges, mismatch and attribution
weights, support thresholds, east/north vector-basis approximation, source-posterior
temperature, ensemble design weights, and convergence tolerances. Stokes drift, vertical
transport, oil weathering in the Eulerian screen, remote downloads, multi-SAR assimilation,
adaptive meshes, calibrated posteriors, learned surrogates, and distributed execution are
deferred.

With fixture paths and a known-source manifest populated, the reproducible A/B/C source
search comparison is run through the existing benchmark command:

```bash
oilspill benchmark pipeline \
    --config configs/benchmarks/pipeline.hybrid.synthetic.yaml
```

It writes append-only JSON plus `source_search_comparison.csv` containing measured
legacy, Eulerian-screening, and hybrid evaluation counts and runtimes. Missing ground truth
is recorded as unavailable rather than replaced with zero, and no speedup is reported
unless the corresponding executions were measured.

Detection benchmarking:

```bash
oilspill benchmark detection \
    --config configs/benchmarks/detection.yaml
```

End-to-end benchmarking:

```bash
oilspill benchmark pipeline \
    --config configs/benchmarks/pipeline.yaml
```

Inspect a typed configuration, list registered adapters, and validate a canonical dataset
manifest without running scientific processing:

```bash
oilspill inspect-config --config configs/pipeline.yaml
oilspill list-components
oilspill validate-dataset --config configs/datasets/validation.yaml
```

Generate a comparison report. Omitting `--config` uses
`configs/benchmarks/report.yaml`:

```bash
oilspill benchmark report --config configs/benchmarks/report.yaml
```

Run tests:

```bash
pytest
```

---

# Development Strategy

Recommended implementation order:

### Phase 1 — Software foundation

* domain models;
* interfaces;
* registries;
* configuration;
* test fixtures;
* CLI skeleton.

### Phase 2 — Satellite pipeline

* SAR loading;
* SAR preprocessing;
* detector abstraction;
* baseline segmentation model;
* slick characterization.

### Phase 3 — Traceability

* AIS ingestion;
* track reconstruction;
* source-time search window;
* candidate generation;
* environmental forcing.

### Phase 4 — Physics

* OpenDrift/OpenOil adapter;
* forward simulations;
* backward simulations.

### Phase 5 — Attribution

* candidate scoring;
* vessel ranking;
* attribution evidence.

### Phase 6 — Forecasting

* ensembles;
* uncertainty fields;
* coastline impact.

### Phase 7 — Benchmarking

* detector comparison;
* attribution evaluation;
* end-to-end experiments;
* reproducible reports.

### Phase 8 — Visualization

* interactive map;
* detection overlay;
* AIS trajectories;
* source tracing;
* vessel ranking;
* future forecasts;
* coastal impact.

---

# Research vs Engineering

This repository distinguishes between:

**Engineering implementation**

and

**scientifically validated methodology**.

A method being implemented in software does not automatically mean that its thresholds, model weights, uncertainty assumptions or physical parameters have been scientifically validated.

Parameters not established from supplied literature or experimental validation should:

1. remain configurable;
2. be documented;
3. be included in experiment metadata;
4. not be presented as authoritative scientific values.

---

# Current Status

The project is under active development.

Initial priorities are:

* [ ] Core interfaces and domain models
* [ ] Configuration system
* [ ] Model registry
* [ ] Synthetic test fixtures
* [ ] Sentinel-1 preprocessing
* [ ] YOLOv8-Seg baseline
* [ ] Slick characterization
* [ ] Historical AIS ingestion
* [ ] AIS candidate generation
* [ ] Environmental-data ingestion
* [ ] OpenOil/OpenDrift integration
* [ ] Forward source tracing
* [ ] Backward source tracing
* [ ] Vessel-attribution scoring
* [ ] Ensemble future forecast
* [ ] Coastal-impact estimation
* [ ] Benchmark runner
* [ ] Visualization/dashboard

## Hackathon incident UI

The small demonstration UI in `ui/` presents one synthetic cached hybrid incident from SAR
detection through source posterior, AIS candidate ranking, +6/+12/+24/+48 hour forecast, and
weighted coastal impact. It begins with only the selected example incident and SAR scene. Each
stage action reveals a precomputed result after a brief, explicitly cached loading transition.
"Run full demo" advances the same stages automatically; it can be paused or cancelled, and reset
returns to the selected scene. It never starts scientific models from browser actions.

Serve it from the repository root:

```bash
python3 -m http.server 8080 --directory ui
```

Then open `http://localhost:8080`. The bundled `ui/data/incident-demo.json` is explicitly marked
`DATASET_VALIDATION_REQUIRED`; its scores are uncalibrated demonstration values, not operational
claims or proof of vessel discharge. Replace that cache with a presentation export from the stable
dashboard contract when real validated pipeline output is available.

The optional SAR, probability, mask, and thumbnail PNGs in `ui/assets/incidents/os-2026-0915/`
are illustrative synthetic visuals, not Sentinel-1 measurements or output from a live checkpoint.
The first stage uses them for the cached comparison viewer; missing assets fall back to the cached
SVG slick polygon after analysis.

---

# Disclaimer

The system is intended as a research and decision-support platform.

A vessel receiving a high attribution score indicates consistency with the available satellite, AIS and environmental evidence. It should not by itself be interpreted as definitive proof of responsibility.

Operational or legal attribution would require appropriate validation, review of source-data quality, uncertainty analysis and supporting evidence.

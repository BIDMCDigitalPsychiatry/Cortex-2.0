# Cortex 2.0

**Cortex 2.0 — Reliability-Aware Personalized Health Phenotyping Engine**

Cortex 2.0 is an open-source framework for building reliable and personalized digital phenotyping pipelines from longitudinal smartphone and wearable data.

Cortex 2.0 is designed around the following processing chain:

```text
Raw sensing data
        ↓
Input adapters
        ↓
Canonical data representation
        ↓
Source / export availability
        ↓
Sensor observability and data quality
        ↓
Feature extraction
        ↓
Feature reliability
        ↓
Personalized baselines and deviations
        ↓
Clinical alignment
        ↓
Degradation and robustness analysis
        ↓
Personalized and transdiagnostic modeling
```

## Cortex 2.0 and legacy Cortex

Cortex 2.0 is a substantial redesign and extension of the original Cortex digital phenotyping framework.

The original Cortex implementation remains available as a legacy reference and comparison pipeline. Cortex 2.0 extends the framework with standardized ingestion, source availability, sensor observability, temporal missingness characterization, reliability-aware feature extraction, personalization, clinical alignment, degradation testing, and cross-pipeline validation.

Where appropriate, legacy Cortex, FOREST, RAPIDS, and other digital phenotyping pipelines can be evaluated alongside Cortex 2.0 to compare feature semantics, reliability, and transportability.

## Core principles

- Platform-agnostic architecture
- Standardized canonical sensing representation
- Separation of source availability from sensor observability
- Episode-aware participant identity
- Explicit distinction between missing data and behavioral zero
- Reliability-aware digital biomarker interpretation
- Participant- and feature-specific baselines
- Empirical evaluation of sensing degradation and feature robustness

## Current development status

The current implementation includes:

- canonical schemas
- generic CSV ingestion
- mindLAMP adapter
- GPS, accelerometer, and screen canonicalization
- episode-aware source identity
- virtual canonical streaming
- source-file manifests
- temporal file indexing
- source-file integrity checking
- source/export boundary representation
- source availability classification
- GPS and accelerometer temporal data quality
- screen observation summaries
- temporal missingness metrics
- automated validation tests

The current test suite has 156 passing tests.

Cortex 2.0 is under active development and interfaces may change as the reliability, personalization, clinical alignment, degradation, and external-pipeline validation components are expanded.

## Installation

Install in editable mode:

```bash
pip install -e .
```

For development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest -q
```

## Command line

```bash
cortex2 version
cortex2 info
```

## Repository structure

```text
src/cortex2_engine/
    adapters/
    canonical/
    quality/
    engines/
    features/
    baseline/
    clinical/
    degradation/
    transportability/
    validation/
    visualization/

workflows/
tests/
configs/
feature_registry/
docs/
examples/
```

## Project status

Cortex 2.0 is currently an early research and development version.

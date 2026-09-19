# LCT 2026 Street Falcon Vehicle ReID

Private Engineering Team repository for the Street Falcon vehicle
re-identification challenge. The system must retrieve the same vehicle across
different cameras without using licence-plate information and must be able to
refuse a match when the gallery has no valid candidate.

## Scope

- reproducible train/validation protocol with identity-disjoint splits;
- embedding baselines and metric-learning experiments;
- top-10 retrieval and open-set refusal calibration;
- Python inference API, browser demo and offline Docker delivery;
- organizer-compatible submission artifacts.

Expected implementation layout:

```text
backend/        Python API, inference and tests
frontend/       Browser demo and tests
training/       Training and validation entry points
configs/        Versioned experiment configuration
docs/           Runbook, data contract and decisions
```

## Data boundary

The organizer dataset is restricted hackathon material. It must never be
committed, attached to issues, copied into CI artifacts or included in container
images. Store it outside the checkout and pass its location through
`LCT_DATA_DIR`. A recommended local path is:

```text
~/datasets/lct26-street-falcon-reid/source/dataset.zip
```

The repository ignores archives, datasets, model weights, credentials and local
environment files. Dataset provenance, checksum and access conditions must be
recorded locally before training begins.

## Current state

Repository bootstrap only. No model result, runtime service or deployment is
claimed yet.

## Engineering contract

- profile: `web-product`;
- Engineering Standards pin: `0.2.9`;
- source: private GitHub repository `ikanam-ai/lct26-street-falcon-reid`;
- changes: isolated branch and reviewed Pull Request;
- large artifacts: outside Git, with checksum and provenance.

GitHub cannot execute the laboratory's GitLab CI components directly. The
repository therefore does not claim that the GitLab policy gate is active. A
compatible checked CI route or a canonical GitLab mirror must be adopted before
the first runnable release.

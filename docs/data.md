# Data contract

## Source and access

The source is a password-protected organizer share. Access is restricted to the
challenge team. Passwords, cookies and expiring download links must not enter
Git, issue text, CI logs or manifests.

## Observed package

The source archive contains 11,416 JPEG images plus `train.csv`,
`test_query.csv`, `test_gallery.csv` and `README.md`.

| Split | Images |
|---|---:|
| train | 9,556 |
| test query | 1,110 |
| test gallery | 750 |

The training table contains 1,541 vehicle identities and 96 cameras. Each
annotated row identifies one target vehicle and its bounding box; a full frame
may contain other vehicles. Test identities and ground truth are hidden.

## Local storage

Keep source material outside the repository:

```text
~/datasets/lct26-street-falcon-reid/
├── source/dataset.zip
├── source/evaluate.py
├── source/example_submission.zip
├── extracted/
│   └── .prepared.json
└── runs/
```

The local manifest must record the acquisition date, exact byte size, SHA-256,
provenance and access restriction before extraction or training. Do not upload
the dataset to shared S3 without a separate data-governance decision.

The verified source archive has SHA-256
`a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613`.
The preparation command refuses an archive with another checksum and records the
validated table and image counts in `extracted/.prepared.json`.

## Validation rule

Split validation by `vehicle_id`, not by image. Build query/gallery pairs across
different cameras and simulate unmatched queries for open-set evaluation.
`camera_id` may guide sampling and validation but must not become a model input.

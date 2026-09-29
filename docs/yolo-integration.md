# YOLO online integration

Branch: maxim_yolo_integration, based on maxim_frontend (441e3db).
Model source: yolo-finetune-metric-learning at 711294d0033421a5c345e0374acd4d2b478ecc44.
Only reviewed inference behavior was ported; the research branch was not merged.

## Behavior

The final 60-epoch full-train YOLO26-L classification backbone produces 1280-D
Re-ID embeddings. This is not a detector. Client x/y/w/h are validated and cropped
with zero margin, resized to 224x224 (squash), converted to [0,1] without ImageNet
normalization. Original and horizontal flip are processed together, pre-BN features
summed and L2-normalized. Model and gallery are loaded once.

A query is compared with 750 gallery vectors. The cosine top-100 is re-ranked
using k-reciprocal k1=10, k2=3, lambda=0.5. Gallery-to-gallery distances are cached.
The kernel follows the source, adding zero-distance guards and self-reciprocal
membership for degenerate tied vectors. Uploaded frames are never detected.

Response score remains cosine. Optional rerank_score supplies the ranking score.
decision_score is the maximum cosine across the gallery; accepted uses that
value, not the first re-ranked item's score. The UI explains that card cosine
scores need not decrease with rank.

The refusal threshold is frozen at 0.9402066469192505 from the final test run's
25% quantile. It is not a per-request quantile or a validated online threshold.
New data requires separate calibration. No same-frame exclusion is implemented;
that earlier request still needs a precise definition/source-frame metadata.

## Metrics and provenance

GET /api/v1/metrics returns the model hash and an optional checked report.
The UI shows mAP@10 66.1%, Rank-1 62.6%, Rank-5 81.6%, candidate F1 88.6%.
These rounded values are reported in the source README for the recipe's
identity-disjoint hold-out seeds 7 and 2026, not measured on deployed full-train
weights. F1 uses cosine candidates and a batch 25% refusal rule, not the online
fixed threshold. The UI includes separate split rows, limitations and source link.
No hidden test metrics or training curves were invented.

configs/yolo_metrics.json is a curated aggregate report with source revision and
README checksum. No per-image results or organizer data are included. The runtime
copy metrics.json is checksum-checked with the model/gallery bundle.

## Runtime and dependencies

A separate CPU service binds 127.0.0.1:27816. Other services at 27812 and 27814
remain running. Dockerfile.yolo uses Python 3.12.3 and copies only /usr/local
dependencies from the audited team image, not /app code or weights.
The dependency image ID is:
sha256:96b97299f5140629e70e3e4c98075a9396a49bf52be3cc91054f167977f43173
It is tagged locally falcon-reid-inference:integration-96b97299.
This build currently requires that local dependency image; it is not a portable
CI build. Baseline pyproject dependency pins remain unchanged.
Observed ML versions: torch 2.6.0+cu124, torchvision 0.21.0+cu124,
ultralytics 8.4.163, numpy 2.5.2, pillow 12.3.0.
HTTP dependencies come from requirements-api.lock. Inference runs on CPU,
non-root, with a read-only filesystem and mounts, two CPUs and 3 GiB RAM.

Runtime files live outside Git:
- artifacts/street-falcon/yolo-integration: checkpoint.pt, base.pt, source
  embeddings.npy, run_meta.json, source_revision.txt, expected.csv, metrics.json,
  plus prepared gallery.npy, ids.json and manifest.json.
- artifacts/street-falcon/yolo.env: bundle/thumbnail paths, port, UID/GID.
- artifacts/street-falcon/gallery-thumbnails-v1: shared read-only preview crops.

To reproduce the bundle, export checkpoint and base weights from the pinned
source revision, final full-rerank embeddings/run_meta/submission (as expected.csv),
write source_revision.txt, and copy configs/yolo_metrics.json as metrics.json.
Verify both weight checksums against the source SHA256SUMS files before use.
Use a new directory and the original extracted dataset:
~~~sh
python scripts/prepare_yolo_bundle.py --directory /path/to/new-bundle --data-dir /path/to/extracted
~~~
Then mount the directory and run scripts/smoke_yolo.py in the YOLO image before
serving it. The original combined embeddings contain 1110 query rows followed by
750 gallery rows; only gallery.npy is loaded for search.

Current server launch from the repository:
~~~sh
docker build -f Dockerfile.yolo -t lct26-street-falcon-reid:yolo-maxim .
docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/yolo.env -p maxim-reid-yolo -f compose.yolo.yml up -d --no-build
~~~

Windows access (keep the terminal open):
~~~powershell
ssh -N -L 8786:127.0.0.1:27816 hackathon-lunopopicks
~~~
Open http://127.0.0.1:8786/. No public proxy or production deployment was performed.

## Verification and postflight

research-python / Engineering Standards 0.2.9; no registered policy exceptions.
- Ruff and 79 tests passed under YOLO dependencies; model smoke passed.
- All 1110 saved query embeddings reproduced the saved submission top-10 exactly.
- 12 decoded real queries: top-1 matched in all 12, exact top-10 in 11;
  the remaining query had 9/10 overlap at the cutoff. CPU vs saved GPU max absolute
  embedding error was 0.00009513 (also checked three gallery vectors).
- Median embedding plus search time in that small sample was 0.0723 s.
  This is integration evidence, not a capacity or quality benchmark.
- Chromium browser smoke: exact manual coordinates, real HTTP search, ten decoded
  photos, enlargement, four metric cards/two split rows, errors and mobile layout.
- Desktop screenshot inspected; screenshots are outside Git because they show data.
- Main and upstream model/frontend branches were not merged or modified.
- No claim of GitLab policy gates or production acceptance.

Bundle checksums:
{
  "schema_version": 1,
  "epoch": 60,
  "threshold": 0.9402066469192505,
  "source_revision": "711294d0033421a5c345e0374acd4d2b478ecc44",
  "sha256": {
    "checkpoint.pt": "44c94833c1ec49a25f7dea0e487a5887de7de7b6fd362ffb310d7401e23ee638",
    "base.pt": "7a3aebb7de6f5d79f0153da627a5f68bd3d7df30e648c0fdb7f9217986ee129e",
    "gallery.npy": "e721cc6b957ca86f5b4b3b0dd8680ddc58854857eaca56edcadef5aa89935136",
    "ids.json": "59864111724cc45ed3d66c32b546cecfe0f4d9dedfd8763f0abb01e2fafae3a2",
    "metrics.json": "2f59d460b83f3d16de15573cbb28ffa7c7ef8e9e68ed0e290441bc0c27a17c5e"
  },
  "source_embeddings_sha256": "d86ae87548b68c2d4b444759a86143b2251195ebf4e2021638cc154b03ad445c",
  "gallery_csv_sha256": "a64ed21fa39bbfd8c7a172468d043415800500b6ed695cdb8162188cf9005496",
  "query_csv_sha256": "97e1ed21942bae9c95b1ce2e5d339d9f635bf49fa484367e6b19349789bb9b4c",
  "threshold_source": "fixed test-derived quantile; online calibration pending"
}

BBox input now supports both manual coordinates and mouse/touch rectangle selection; see frontend.md.

# Exact full-frame exclusion

All gallery entries with identical decoded RGB pixels and dimensions to the
uploaded full frame are removed before cosine top-100 selection and re-ranking.
Different IDs/bbox targets from the same frame are removed together. The eligible
pool refills top_k before applying the existing cosine threshold.
No eligible rows: HTTP 200, accepted=false, matches=[], decision_score=null.
excluded_same_frame counts removed rows; the UI displays that count.
GET /api/v1/info reports same_frame_filter=exact_rgb (off for legacy bundles).

Matching ignores filenames and metadata, but requires identical pixels.
No resize, EXIF rotation, detection, camera classifier or plate features are used.
Recompressed, resized, cropped or edited copies are NOT guaranteed to match.
Other frames from the same camera remain eligible. Uploads/hashes are not saved.

## Preparation and rollback

frames.json contains schema_version=1, algorithm=sha256-rgb-pixels-v1, gallery IDs
in exact order and per-row fingerprints. Hash input is the algorithm ASCII label,
big-endian unsigned 32-bit width/height, then RGB bytes. The manifest checksum and
complete row mapping are validated at startup; the lookup loads once.
New demo builds create this file. Existing YOLO bundles can be migrated:
~~~sh
python scripts/add_frame_index.py --source-bundle /path/to/old-bundle --data-dir /path/to/extracted --output /path/to/new-bundle
~~~
Source hashes/CSVs are verified, fingerprints built, model/gallery copied, and
a new manifest published last. The original bundle is untouched.
Change LCT_YOLO_BUNDLE only; reuse existing LCT_THUMBNAIL_DIR.
The service does not mount source data. All artifacts stay outside Git/images.

Active bundle: yolo-demo-frames-20260929; 11,416 rows, 11,405 unique RGB frames;
116.2 seconds preparation. Original bundle yolo-demo-all-20260929 is retained.
yolo-before-frame-filter.env selects the old bundle and before-frame-filter
image tag for rollback through the same maxim-reid-yolo Compose project.

## Verification and standards postflight

research-python / Engineering Standards 0.2.9; no registered exceptions.
Ruff, 101 tests and model smoke passed. Tests cover renamed PNG uploads/metadata,
different bbox targets, pre-top-100 exclusion, candidate refill, zero/one/two
eligible rows, threshold refusal and invalid index mappings/fingerprints.
Existing Starlette/httpx deprecation warning remains.
Candidate HTTP checks: three real inputs, all source-frame matches excluded,
30 decoded thumbnails; sample times 0.1099, 0.0853, 0.0830 seconds.
Chromium verified exclusion notice, real photos, refusal/recovery, manual and
mouse/touch bbox, errors and mobile layout.
Use check_demo_gallery.py --thumbnail-dir for reused thumbnails.
Browser smoke: QUERY_FILE=/query.jpg uses the existing sample bbox
1202,270,588,474; EXPECT_FRAME_FILTER=1 checks the exclusion notice.
Data boundary: private source data read-only, fingerprints outside Git.
No model-quality improvement, GitLab policy-gate or production acceptance claim.

Final deployed service verified: three real queries, 30 decoded thumbnails,
same_frame_filter=exact_rgb at localhost:8786. Temporary preview removed.
Final diff review: source, tests and docs only; no private artifacts or secrets.
Delivery stays in maxim_yolo_integration through PR #6; main is unchanged.

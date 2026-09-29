# Full demonstration gallery

Requested scope: all 11,416 supplied photographs, explicitly for demonstration.
Source partitions are disjoint by image ID: train 9,556, test_query 1,110,
test_gallery 750. Each CSV supplies the vehicle bbox; no detection is added.

This is not a new held-out evaluation set. The final model was trained on the
train partition, and query images are now also searchable candidates. Exact
self-matches and related frames can appear. The prior hold-out metric cards
describe the training recipe, not this gallery. A visible UI notice states this.
The original 0.9402 decision threshold is retained only as a demonstration
setting; it has not been calibrated for the larger candidate pool.

## Preparation

scripts/build_demo_gallery.py accepts --source-bundle, --data-dir, --output,
and optional --batch-size (1..32; default 16). The output directory must be new.
It verifies/loads the existing model bundle once, then encodes original+flipped
bbox crops with the same online predictor in batches. The first three vectors
are compared with single-image inference (max error <=1e-4 required).
Generated artifacts stay outside Git and the Docker image:
- gallery.npy: float32, L2-normalized, [11416, 1280];
- ids.json: unique IDs in train/query/gallery CSV order;
- thumbnails/: one hashed JPEG filename per gallery ID;
- model weights, curated metrics report plus demo-gallery provenance;
- build-evidence.json: elapsed time and batch/single numerical error;
- manifest.json: written last after checksums and finite/norm checks pass.

Source images and CSVs are read-only. A failed build leaves an unready directory;
the active service is not pointed to it. No data, thumbnails or weight files are
committed. Building is offline in a separate two-CPU/3-GiB container.

## Search memory

At up to 2,048 gallery rows, the existing distance cache is retained. Above that,
the service calculates gallery distances only among each query's cosine top-100.
This avoids a persistent quadratic 11,416-by-11,416 matrix while preserving
k-reciprocal ranking parameters. Embeddings and thumbnails remain prepared once.

## Preview, switch and rollback

Build a separate image tag:
~~~sh
docker build -f Dockerfile.yolo -t lct26-street-falcon-reid:yolo-demo-maxim .
~~~

The Compose image can be set with LCT_YOLO_IMAGE; its default stays yolo-maxim.
A candidate uses an independent Compose project and port 27818, the new bundle
and its thumbnails. Run scripts/check_demo_gallery.py with --directory,
--data-dir and --url against it, then browser smoke scripts with:
BASE_URL=http://127.0.0.1:27818/ EXPECT_YOLO=1 EXPECT_GALLERY_COUNT=11416.

For the final switch, the existing maxim-reid-yolo project still serves local
port 27816. Its env points to the new bundle, thumbnail directory and image.
A short container replacement reconnects subsequent requests. SSH local port
8786 and the prepared authenticated IP gateway keep their addresses.

The original bundle yolo-integration and gallery-thumbnails-v1, image yolo-maxim,
and backup yolo-750.env remain available outside the checkout. To roll back,
run the same project with the old environment:
~~~sh
docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/yolo-750.env -p maxim-reid-yolo -f compose.yolo.yml up -d --no-build
~~~

## Verification

- Ruff and 81 Python tests passed, including bounded large-gallery memory and
  equivalence of candidate-only versus cached ranking.
- Original-bundle model smoke retained the previously documented CPU/GPU
  tolerance and 12/12 top-1 agreement.
- Full gallery built in 559.2 seconds; first three batch/single vectors agreed exactly.
- Manifest checksum: a43048397b7fdbe411bcd0b37bc4329c28f27b067f0000e4bee6f53d80a0178b.
- Candidate and final APIs verified: 11,416 unique IDs, [11416,1280] normalized
  vectors, all thumbnail files present, and three real requests (one per split)
  returned valid top-10 with all 30 result photos decoded over HTTP.
- Final sample search times: 0.1066, 0.0781, 0.0785 seconds, not a load benchmark.
- Both frontend and mouse/touch bbox browser smokes passed on the candidate.
- Desktop screenshot verified the 11,416 count and explicit demonstration notice.
- Active maxim-reid-yolo now uses the expanded bundle and yolo-demo-maxim image.
  Temporary candidate service was stopped after successful verification.
- Original image/bundle and yolo-750.env preserved for rollback.

Standards: research-python / Engineering Standards 0.2.9. Same data boundary,
no new policy exceptions. This work makes no model-quality improvement,
production acceptance or public-network availability claim.

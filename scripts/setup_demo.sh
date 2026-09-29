#!/usr/bin/env bash
# Prepare the web demo from this repository and the dataset, offline, in the demo image:
#
#   scripts/setup_demo.sh DATASET_DIR OUT_DIR [--full]
#
# DATASET_DIR: test_query.csv, test_gallery.csv, images/ (and train.csv for --full).
# OUT_DIR (must not exist) receives:
#   bundle/      model bundle: the repository's weights and final embeddings, gallery = test_gallery (750 photos)
#   thumbnails/  gallery preview crops
#   full/        with --full: gallery of all 11,416 photos train + test_query + test_gallery (~15 min on CPU),
#                with its own thumbnails/; the demo then serves this gallery
#   yolo.env     settings for: docker compose --env-file OUT_DIR/yolo.env -f compose.yolo.yml up -d
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != --full ) ]]; then
    sed -n '2,13p' "$0" >&2
    exit 2
fi
repo=$(cd "$(dirname "$0")/.." && pwd)
data=$(realpath "$1")
out=$(realpath -m "$2")
image=${LCT_YOLO_IMAGE:-lct26-street-falcon-reid:yolo}
if [[ -e $out ]]; then
    echo "$out already exists; use a new directory" >&2
    exit 1
fi

# the weights must be the ones the submission was made with (as in Dockerfile)
(cd "$repo/models/yolo_finetune/weights" && sha256sum -c --quiet SHA256SUMS)
(cd "$repo/models/yolo_embedding/weights" && grep ' yolo26l-cls.pt$' SHA256SUMS | sha256sum -c --quiet -)

docker build -f "$repo/Dockerfile.yolo" -t "$image" "$repo"

mkdir -p "$out/bundle"
cp "$repo/models/yolo_finetune/weights/yolo26l-cls-reid.pt" "$out/bundle/checkpoint.pt"
cp "$repo/models/yolo_embedding/weights/yolo26l-cls.pt" "$out/bundle/base.pt"
cp "$repo/embeddings.npy" "$out/bundle/embeddings.npy"
cp "$repo/submission.csv" "$out/bundle/expected.csv"
cp "$repo/run_meta.json" "$out/bundle/run_meta.json"
cp "$repo/configs/yolo_metrics.json" "$out/bundle/metrics.json"
# revision of the team repository the model artifacts come from (docs/yolo-integration.md)
echo 711294d0033421a5c345e0374acd4d2b478ecc44 > "$out/bundle/source_revision.txt"

run() {
    docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint python \
        -v "$repo/scripts:/workspace/scripts:ro" -v "$data:/data:ro" -v "$out:/out" "$image" "$@"
}
run scripts/prepare_yolo_bundle.py --directory /out/bundle --data-dir /data
# the online model must reproduce the saved submission on real test images
run scripts/smoke_yolo.py --bundle /out/bundle --data-dir /data

if [[ ${3:-} == --full ]]; then
    run scripts/build_demo_gallery.py --source-bundle /out/bundle --data-dir /data --output /out/full
    bundle=$out/full
    thumbnails=$out/full/thumbnails
else
    run scripts/export_thumbnails.py --data-dir /data --gallery-dir /out/bundle --output-dir /out/thumbnails
    bundle=$out/bundle
    thumbnails=$out/thumbnails
fi

cat > "$out/yolo.env" <<EOF
LCT_YOLO_IMAGE=$image
LCT_YOLO_BUNDLE=$bundle
LCT_THUMBNAIL_DIR=$thumbnails
LCT_API_PORT=${LCT_API_PORT:-27816}
LOCAL_UID=$(id -u)
LOCAL_GID=$(id -g)
EOF
echo "Demo ready. Start: docker compose --env-file $out/yolo.env -f compose.yolo.yml up -d"
echo "Open: http://127.0.0.1:${LCT_API_PORT:-27816}/"

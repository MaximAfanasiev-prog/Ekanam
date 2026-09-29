# Offline inference image: produces submission.csv, embeddings.npy and candidates.csv for the test split.
# Everything the run needs (code, weights, Python packages) is baked in at build time; the container itself
# runs without network (docker-compose.yml: network_mode: none).
FROM python:3.12.3-slim

# libGL / glib are runtime dependencies of opencv-python, which ultralytics imports
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    FALCON_DATASET_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models ./models
# fail the build if a weight file is missing or differs from the one the submission was made with
RUN cd models/yolo_finetune/weights && sha256sum -c SHA256SUMS \
    && cd ../../yolo_embedding/weights && grep ' yolo26l-cls.pt$' SHA256SUMS | sha256sum -c -

# /data   - the dataset (test_query.csv, test_gallery.csv, images/), mounted read-only
# /output - where the three artifacts and run_meta.json are written
ENTRYPOINT ["python", "-m", "models.yolo_finetune.predict", \
            "--checkpoint", "models/yolo_finetune/weights/yolo26l-cls-reid.pt", \
            "--dataset-dir", "/data", "--out", "/output"]
# refusal rule for candidates.csv and re-ranking of submission.csv, see the module docstring of
# models/yolo_finetune/predict.py
CMD ["--refuse-rate", "0.25", "--rerank"]

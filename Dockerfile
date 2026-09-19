FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml LICENSE README.md ./
COPY src ./src
COPY configs ./configs

RUN python -m pip install --no-cache-dir \
      numpy==2.1.3 \
      pillow==11.0.0 \
      torchvision==0.20.1 && \
    python -m pip install --no-cache-dir --no-deps .

ENTRYPOINT ["street-falcon-reid"]
CMD ["--help"]


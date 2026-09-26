# Backend readiness — 2026-09-26

Scope: internal hackathon API on maxim_backend; no merge to main and no public deployment.
Profile: research-python, Engineering Standards 0.2.9; no registered exceptions.

## Evidence

- Ruff: passed.
- Pytest: 72 tests passed using generated images/checkpoints; CPU model smoke passed.
- API Docker image built from pinned PyTorch base digest with constrained HTTP dependencies.
- Real baseline: resnet50-baseline-93f4287, checkpoint SHA-256
  64879053616e87e0ea8d7af7ae6e5417048a91e5d5ad54ee70236e718deeb3f5.
- Gallery: 750 images, 512-dimensional embeddings; bundle manifest SHA-256
  121405a74893302c33858a68f39a04451b9da4d706a59b75e657cd66639d4c74.
- Bundle export recorded source_csv_checksum_verified=false because this legacy run
  predates the gallery CSV checksum. The original dataset CSV was used.
  Conversion preserves this limitation instead of claiming historical verification.
- Candidate image: sha256:73f1c23a6f9d1a3795b300764da2f824f4c194e21778c48d7812fa562b1ea6f4.
- Actual HTTP upload using one dataset query and its bbox: top-10 matched the saved run.
- Invalid bbox: HTTP 422.
- Burst smoke: 12 requests / 4 clients; 1 HTTP 200, 11 HTTP 429, no HTTP 500.
- Sequential smoke: 12 requests / 1 client; 12 HTTP 200.
  Successful-response median 79.75 ms, nearest-rank p95 81.77 ms.
- Candidate container became healthy and recovered after an explicit restart.
- Observed candidate memory after the burst: 542.4 MiB under a 3 GiB limit.
- Candidate was replaced by the final Compose service on loopback port 27812.
  The earlier temporary preview and candidate containers were removed.
- Existing dashboard on port 27810 was not restarted.

These timings are from a tiny repeated-query smoke sample on a shared CPU server.
They are not throughput, full-dataset quality, GPU or production acceptance claims.
GitHub CI status should be read from PR #4; this document does not assert GitLab gates passed.

## Remaining acceptance

The ML team must supply a compatible model/gallery pair and validate its threshold
and quality. A new architecture requires an adapter. Frontend/proxy integration
and any public access controls require a joint check. main remains unchanged.

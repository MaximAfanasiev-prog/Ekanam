# Search frontend

Branch maxim_frontend starts from maxim_backend commit 72bf1d4.
FastAPI serves / and packaged CSS/JavaScript under /ui-assets.
No frontend build step or external CDN is required. API requests use the same origin.

## User flow

Upload a static JPEG or PNG, then enter integer x, y, w, h in original-image
pixels. x/y are the top-left corner; w/h are width and height. Fields start
empty. The preview only draws the entered rectangle; clicking the image does
not select a box. No detector or automatic bbox calculation is implemented.
The server retains its existing crop/preprocessing contract.

Client checks size (10 MiB), dimensions (20 million pixels), and bbox bounds.
The server remains authoritative. JPEG EXIF rotation and PNG EXIF metadata
are rejected in the UI to prevent a mismatch between displayed and raw coordinates.
Save a copy without orientation metadata before uploading such a file.

Select top-k (1, 5, 10) and search. Results show image IDs, cosine similarity,
threshold decision and request ID. Scores are not probabilities. Gallery
thumbnails are not yet available in the API.
Inputs lock during the request; changing inputs clears previous results.
Busy, invalid input and network errors are shown without automatic retries.

## Isolated preview

Build the image from this branch:

~~~sh
docker build -f Dockerfile.api -t lct26-street-falcon-reid:frontend-maxim .
~~~

Start a separate Compose project using the existing trusted checkpoint and
gallery settings (see server-api.md). On the current server:

~~~sh
LCT_API_PORT=27814 docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/api.env -p maxim-reid-frontend -f compose.api.yml -f compose.frontend.yml up -d --no-build api
~~~

This binds only 127.0.0.1:27814. The existing backend at 27812 stays separate.
From Windows PowerShell, keep this tunnel open:

~~~powershell
ssh -N -L 8784:127.0.0.1:27814 hackathon-lunopopicks
~~~

Open http://127.0.0.1:8784/. A shared-domain reverse proxy is a later deployment
step; it has not been configured by this change.

## Verification

- Ruff: passed; pytest: 73 passed; model smoke: passed.
- Chromium / Playwright 1.63.0: upload, initially empty manual bbox, bounds
  validation, unchanged coordinates in outgoing FormData, real search returning
  ten rows, controlled 429 and invalid-file messages, and 390px mobile layout.
- Desktop screenshot inspected. Browser smoke uses a generated synthetic image,
  not organizer images. This is an integration check, not a model quality claim.
- Docker image includes static assets and starts with the existing gallery.
- Run scripts/smoke_frontend.cjs with Node and playwright@1.63.0 installed;
  BASE_URL defaults to http://127.0.0.1:27814/. Optional SCREENSHOTS must point
  outside the repository. Install matching Chromium with Playwright or use
  mcr.microsoft.com/playwright:v1.63.0-noble.

Standards postflight: research-python / Engineering Standards 0.2.9.
No new policy exceptions. Source-only changes; no datasets, weights, embeddings
or credentials committed. No model-quality, production or GitLab gate claim.

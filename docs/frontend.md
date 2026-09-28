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
threshold decision and request ID. Scores are not probabilities. Gallery thumbnails appear in ranked cards; click a photo to enlarge it.
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
LCT_API_PORT=27814 docker compose --env-file /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/frontend.env -p maxim-reid-frontend -f compose.api.yml -f compose.frontend.yml up -d --no-build api
~~~

This binds only 127.0.0.1:27814. The existing backend at 27812 stays separate.
From Windows PowerShell, keep this tunnel open:

~~~powershell
ssh -N -L 8784:127.0.0.1:27814 hackathon-lunopopicks
~~~

Open http://127.0.0.1:8784/. A shared-domain reverse proxy is a later deployment
step; it has not been configured by this change.

## Verification

- Ruff: passed; pytest: 75 passed; model smoke: passed.
- Chromium / Playwright 1.63.0: upload, initially empty manual bbox, bounds
  validation, unchanged coordinates in outgoing FormData, real search returning
  ten photo cards, controlled 429 and invalid-file messages, and 390px mobile layout.
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

## Gallery photographs

The preview reads LCT_THUMBNAIL_DIR through a separate read-only mount.
GET /api/v1/gallery/{image_id}/thumbnail only serves IDs in the loaded gallery;
unknown IDs, missing files and symlinks return 404. Paths use SHA-256 of the ID,
not user-provided filenames. Without the optional directory, search still works.

Export cropped previews once with scripts/export_thumbnails.py:
~~~sh
python scripts/export_thumbnails.py --data-dir /path/to/extracted --gallery-dir /path/to/bundle --output-dir /path/outside/git/new-thumbnails
~~~
The exporter uses supplied test_gallery.csv bboxes, not detection; crops are for
display only and do not alter inference. Existing output directories are refused.
Set LCT_THUMBNAIL_DIR to this new directory in frontend.env, alongside the
checkpoint, gallery, UID/GID and port settings. Re-export for a new gallery.
Current preview contains 750 thumbnails. Original images stay read-only.
The thumbnail mount and generated photos must never be included in Git or images.

Additional verification: 75 Python tests, image endpoint access restrictions,
all ten browser photos decoded, enlargement dialog and desktop/mobile cards.
Screenshot evidence remains outside Git and CI because it contains gallery data.

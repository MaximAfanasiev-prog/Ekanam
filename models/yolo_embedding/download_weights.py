"""Downloads the pretrained Ultralytics *-cls weights (release v8.4.0 of ultralytics/assets) and checks SHA256.

    python -m models.yolo_embedding.download_weights                 # all 10 models
    python -m models.yolo_embedding.download_weights yolo26l-cls     # only selected ones
"""

import hashlib
import sys
import urllib.request

from .extractor import MODELS, WEIGHTS_DIR, weights_path

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/{}.pt"


def main():
    expected = dict(line.split()[::-1] for line in (WEIGHTS_DIR / "SHA256SUMS").read_text().splitlines())
    for name in sys.argv[1:] or MODELS:
        path = weights_path(name)
        if not path.exists():
            urllib.request.urlretrieve(URL.format(name), path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected[path.name]:
            raise SystemExit(f"{path.name}: sha256 mismatch ({digest})")
        print(f"{path.name}: ok")


if __name__ == "__main__":
    main()

"""Download the public upstream dataset; verify a frozen archive hash when available."""

import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=URL)
    args = parser.parse_args()
    archive = ROOT / "data/raw/construction-site-safety.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(args.url, timeout=120) as response, partial.open("wb") as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        partial.replace(archive)
    with archive.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = ROOT / "data/provenance.json"
    if provenance.exists():
        expected = json.loads(provenance.read_text())["archive_sha256"]
        if checksum != expected:
            raise ValueError("Dataset archive changed; obtain the recorded version before continuing")
    destination = ROOT / "data/raw/source"
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        z.extractall(destination)
    print("Archive SHA256:", checksum)


if __name__ == "__main__":
    main()

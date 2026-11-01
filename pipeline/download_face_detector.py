"""Fetch the S3FD checkpoint also used by upstream MuseTalk."""

import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from setup_assets import hf_files, model_download, ROOT, HERE


def main():
    entries = hf_files("camenduru/Wav2Lip", ["checkpoints/s3fd-619a316812.pth"],
                       ROOT / "models/face_detection")
    entry = entries[0]
    if not entry["sha256"].startswith("619a316812"):
        raise RuntimeError("Mirror does not match the upstream S3FD hash prefix")
    model_download(entry)
    target = HERE / "MuseTalk/musetalk/utils/face_detection/detection/sfd/s3fd.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry["target"], target)
    (HERE / "s3fd-manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print("S3FD VERIFIED AND INSTALLED", flush=True)


if __name__ == "__main__":
    main()

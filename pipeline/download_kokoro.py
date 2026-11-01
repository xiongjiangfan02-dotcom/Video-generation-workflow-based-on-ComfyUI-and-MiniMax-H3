"""Install the upstream multilingual Kokoro export for Chinese dialogue."""

import concurrent.futures
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from setup_assets import get_json, hf_files, model_download, ROOT, HERE


def main():
    repo = "csukuangfj/kokoro-multi-lang-v1_0"
    info = get_json(f"https://hf-mirror.com/api/models/{repo}")
    names = [
        item["rfilename"] for item in info["siblings"]
        if item["rfilename"].startswith(("dict/", "espeak-ng-data/"))
        or item["rfilename"] in (
            "model.onnx", "voices.bin", "tokens.txt", "lexicon-zh.txt",
            "date-zh.fst", "number-zh.fst", "phone-zh.fst", "README.md", "LICENSE")
    ]
    entries = hf_files(repo, names, ROOT / "models/tts/kokoro-multi-lang-v1_0")
    (HERE / "kokoro-manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        list(pool.map(model_download, entries))
    print("KOKORO ASSETS VERIFIED", flush=True)


if __name__ == "__main__":
    main()

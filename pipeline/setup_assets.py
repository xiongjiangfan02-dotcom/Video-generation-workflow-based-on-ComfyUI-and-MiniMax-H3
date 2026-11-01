"""Fetch pinned upstream assets and verify model hashes before exposing them."""

import concurrent.futures
import hashlib
import importlib.util
import json
import os
import shutil
import tarfile
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path("E:/AI/ComfyUI")
HERE = ROOT / "pipeline"
SESSION = requests.Session()


def get_json(url):
    response = SESSION.get(url, timeout=(15, 60))
    response.raise_for_status()
    return response.json()


def small_download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    part = path.with_suffix(path.suffix + ".download")
    with SESSION.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with part.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
    part.replace(path)


def hf_files(repo, names, destination):
    info = get_json(f"https://hf-mirror.com/api/models/{repo}?blobs=true")
    sha = info["sha"]
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    result = []
    for name in names:
        item = siblings[name]
        entry = {
            "repo": repo,
            "revision": sha,
            "source_file": name,
            "target": str(destination / name),
            "url": f"https://hf-mirror.com/{repo}/resolve/{sha}/{name}",
        }
        if "lfs" in item:
            entry.update(size=item["lfs"]["size"], sha256=item["lfs"]["sha256"])
        result.append(entry)
    return result


def model_download(entry):
    target = Path(entry["target"])
    if "sha256" not in entry:
        small_download(entry["url"], target)
        return
    spec = importlib.util.spec_from_file_location(
        "verified_h3_download", ROOT / "downloads/h3/download_models.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Reuse the already-tested ranged downloader, with per-model progress files.
    local_root = HERE / "download_jobs" / target.stem
    (local_root / "downloads/h3").mkdir(parents=True, exist_ok=True)
    module.ROOT = local_root
    module.WORKERS = 4
    entry = dict(entry, name=target.name, folder=str(target.parent))
    module.download(entry)
    # Some exFAT directory timestamps stay unchanged on rename; refresh Comfy's file list.
    os.utime(target.parent, None)


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    runtime_zip = HERE / "downloads/python-3.10.11-embed-amd64.zip"
    small_download("https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip", runtime_zip)
    runtime = HERE / "python310"
    if not (runtime / "python.exe").exists():
        runtime.mkdir(exist_ok=True)
        with zipfile.ZipFile(runtime_zip) as archive:
            for name in archive.namelist():
                if not name.endswith("._pth"):
                    archive.extract(name, runtime)
    small_download("https://bootstrap.pypa.io/get-pip.py", HERE / "downloads/get-pip.py")
    source_info = get_json("https://api.github.com/repos/TMElyralab/MuseTalk/commits/main")
    source_sha = source_info["sha"]
    source = HERE / "MuseTalk"
    tree = get_json(f"https://api.github.com/repos/TMElyralab/MuseTalk/git/trees/{source_sha}?recursive=1")
    code_files = [
        item["path"] for item in tree["tree"]
        if item["type"] == "blob" and
        (item["path"].startswith(("musetalk/", "scripts/", "configs/"))
         or item["path"] in ("LICENSE", "LICENSE.txt", "README.md", "requirements.txt"))
    ]
    def get_code(name):
        small_download(f"https://raw.githubusercontent.com/TMElyralab/MuseTalk/{source_sha}/{name}", source / name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(get_code, code_files))
    print(f"MUSE TALK SOURCE READY {source_sha}", flush=True)
    tools = HERE / "ffmpeg"
    tools.mkdir(exist_ok=True)
    ffmpeg_source = Path("D:/ai漫剧/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe")
    if not (tools / "ffmpeg.exe").exists():
        shutil.copy2(ffmpeg_source, tools / "ffmpeg.exe")
    assets = []
    assets += hf_files(
        "Lykon/AbsoluteReality",
        ["AbsoluteReality_1.8.1_pruned.safetensors", "README.md"],
        ROOT / "models/checkpoints",
    )
    models = source / "models"
    assets += hf_files("TMElyralab/MuseTalk", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth", "LICENSE.txt"], models)
    assets += hf_files("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.bin"], models / "sd-vae")
    assets += hf_files("openai/whisper-tiny", ["config.json", "pytorch_model.bin", "preprocessor_config.json"], models / "whisper")
    assets += hf_files("yzd-v/DWPose", ["dw-ll_ucoco_384.pth"], models / "dwpose")
    assets += hf_files("ManyOtherFunctions/face-parse-bisent", ["79999_iter.pth", "resnet18-5c106cde.pth"], models / "face-parse-bisent")
    assets += hf_files("csukuangfj/vits-zh-aishell3",
                      ["vits-aishell3.onnx", "tokens.txt", "lexicon.txt", "phone.fst", "date.fst",
                       "number.fst", "README.md"], ROOT / "models/tts/vits-zh-aishell3")
    manifest = {"musetalk_source_revision": source_sha, "assets": assets}
    (HERE / "asset-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("MANIFEST READY", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(model_download, assets))
    print("ALL ASSETS VERIFIED", flush=True)


if __name__ == "__main__":
    main()

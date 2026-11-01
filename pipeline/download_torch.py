"""Download official pinned Windows CUDA wheels with SHA256 verification."""

import importlib.util
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

import requests

ROOT = Path("E:/AI/ComfyUI")
HERE = ROOT / "pipeline"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links += [value for key, value in attrs if key == "href"]


def main():
    for package, version in [("torch", "2.0.1"), ("torchvision", "0.15.2"), ("torchaudio", "2.0.2")]:
        response = requests.get(f"https://download.pytorch.org/whl/cu118/{package}/", timeout=(15, 45))
        response.raise_for_status()
        parser = Links()
        parser.feed(response.text)
        filename = f"{package}-{version}+cu118-cp310-cp310-win_amd64.whl"
        url = next(link for link in parser.links if unquote(urlparse(link).path).endswith("/" + filename))
        sha = parse_qs(urlparse(url).fragment)["sha256"][0]
        url = url.split("#", 1)[0]
        head = requests.head(url, allow_redirects=True, timeout=(15, 45))
        head.raise_for_status()
        size = int(head.headers["Content-Length"])
        spec = importlib.util.spec_from_file_location("ranged", ROOT / "downloads/h3/download_models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = HERE / "wheel_progress" / package
        (module.ROOT / "downloads/h3").mkdir(parents=True, exist_ok=True)
        module.WORKERS = 12
        module.CHUNK = 16 * 1024 * 1024
        module.download({
            "name": filename, "folder": str(HERE / "wheels"),
            "size": size, "sha256": sha, "url": url,
        })


if __name__ == "__main__":
    main()

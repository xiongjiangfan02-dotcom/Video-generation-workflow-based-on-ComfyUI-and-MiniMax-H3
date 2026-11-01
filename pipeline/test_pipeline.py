"""Submit isolated stage tests and retain a verifiable local report."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import wave

import av
import numpy as np
from PIL import Image
import requests

ROOT = Path("E:/AI/ComfyUI")
HERE = ROOT / "pipeline"
BASE = "http://127.0.0.1:8190"


def execute(name, prompt):
    queue = requests.get(BASE + "/queue", timeout=20).json()
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("The complete-drama server is busy; not adding a test")
    graph = json.loads((ROOT / "workflows" / (name + ".json")).read_text(encoding="utf-8"))
    response = requests.post(BASE + "/prompt", json={
        "prompt": prompt, "extra_data": {"extra_pnginfo": {"workflow": graph}},
        "client_id": "short-drama-installer",
    }, timeout=60)
    if not response.ok:
        raise RuntimeError(response.text)
    prompt_id = response.json()["prompt_id"]
    started = time.monotonic()
    print("TEST " + name + " " + prompt_id, flush=True)
    while True:
        history = requests.get(BASE + "/history/" + prompt_id, timeout=30).json()
        if prompt_id in history:
            result = history[prompt_id]
            break
        if time.monotonic() - started > 1800:
            queue = requests.get(BASE + "/queue", timeout=20).json()
            if any(item[1] == prompt_id for item in queue["queue_running"]):
                requests.post(BASE + "/interrupt", json={}, timeout=20).raise_for_status()
            raise TimeoutError("Test exceeded 30 minutes")
        time.sleep(5)
    output = {"name": name, "prompt_id": prompt_id,
              "elapsed_seconds": round(time.monotonic() - started, 2), "history": result}
    report = HERE / "tests"
    report.mkdir(exist_ok=True)
    (report / (name + ".result.json")).write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["status"]["status_str"] != "success":
        raise RuntimeError(json.dumps(result["status"], ensure_ascii=False))
    print("PASS " + name + f" {output['elapsed_seconds']}s", flush=True)
    return result


def verify_video(path):
    with av.open(str(path)) as container:
        audio = len(container.streams.audio)
        video = container.streams.video[0]
        fps = float(video.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    if not frames or any(float(frame.std()) < 2 for frame in (frames[0], frames[len(frames)//2], frames[-1])):
        raise RuntimeError("Video is empty or blank")
    if not audio:
        raise RuntimeError("Dubbed video is missing its audio stream")
    with av.open(str(path)) as container:
        decoded_audio = [frame.to_ndarray().astype(np.float32) for frame in container.decode(audio=0)]
    if not decoded_audio or max(float(np.abs(frame).max()) for frame in decoded_audio) < 0.001:
        raise RuntimeError("The dubbed audio stream is silent")
    selected = [frames[0], frames[len(frames)//2], frames[-1]]
    sheet = Image.new("RGB", (frames[0].shape[1] * 3, frames[0].shape[0]))
    for index, frame in enumerate(selected):
        sheet.paste(Image.fromarray(frame), (index * frames[0].shape[1], 0))
    sheet.save(path.with_suffix(".contact.jpg"), quality=92)
    return {"path": str(path), "frames": len(frames), "fps": fps, "audio_streams": audio,
            "width": frames[0].shape[1], "height": frames[0].shape[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["image", "reference", "tts", "lipsync", "full"])
    args = parser.parse_args()
    names = {"image": "Drama_01_FirstFrame", "reference": "Drama_01B_ReferenceFirstFrame", "tts": "Drama_02_ChineseVoice",
             "lipsync": "Drama_03_VideoLipSync", "full": "Drama_04_FullPipeline"}
    name = names[args.stage]
    prompt = json.loads((HERE / (name + ".api.json")).read_text(encoding="utf-8"))
    if args.stage == "full":
        prompt["108"]["inputs"]["length"] = 124
        prompt["201"]["inputs"]["filename_prefix"] = "EP01_full_test_voice"
        prompt["203"]["inputs"]["filename_prefix"] = "short_drama/tests/FullPipeline"
    result = execute(name, prompt)
    if args.stage in ("image", "reference"):
        item = result["outputs"]["7"]["images"][0]
        path = ROOT / "output" / item["subfolder"] / item["filename"]
        image = Image.open(path).convert("RGB")
        if float(np.asarray(image).std()) < 2:
            raise RuntimeError("First-frame image is blank")
        image.save(ROOT / "input" / (
            "pipeline_first_frame.png" if args.stage == "image" else "pipeline_reference_frame.png"))
        print(str(path), flush=True)
    if args.stage == "tts":
        item = result["outputs"]["201"]["audio"][0]
        path = ROOT / "output" / item["subfolder"] / item["filename"]
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype(np.float32) / 32768
        summary = {"path": str(path), "sample_rate": rate,
                   "duration": len(samples) / rate, "rms": float(np.sqrt(np.mean(samples**2)))}
        if rate < 16000 or summary["duration"] <= 0 or summary["rms"] < 0.001:
            raise RuntimeError("TTS did not produce the expected non-silent, wideband voice")
        (HERE / "tests/tts-media.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    if args.stage in ("lipsync", "full"):
        node = "204" if args.stage == "lipsync" else "203"
        outputs = result["outputs"].get(node, {})
        print(json.dumps(outputs, ensure_ascii=False), flush=True)
        prefix = "EP01_dialogue" if args.stage == "lipsync" else "FullPipeline"
        paths = sorted((ROOT / "output/short_drama").rglob(prefix + "*.mp4"),
                       key=lambda path: path.stat().st_mtime)
        summary = verify_video(paths[-1])
        (HERE / "tests" / (args.stage + "-media.json")).write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
    requests.post(BASE + "/free", json={"unload_models": True, "free_memory": True}, timeout=30).raise_for_status()


if __name__ == "__main__":
    main()

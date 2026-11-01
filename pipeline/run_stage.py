"""Isolated adapters for upstream Chinese TTS and MuseTalk inference."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("E:/AI/ComfyUI")
HERE = ROOT / "pipeline"
os.environ["HF_HOME"] = str(ROOT / "cache/huggingface")
os.environ["TORCH_HOME"] = str(ROOT / "cache/torch")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TEMP"] = str(ROOT / "cache/temp")
os.environ["TMP"] = os.environ["TEMP"]
os.environ["MPLCONFIGDIR"] = str(ROOT / "cache/matplotlib")
os.environ["NUMBA_CACHE_DIR"] = str(ROOT / "cache/numba")
os.environ["XDG_CACHE_HOME"] = str(ROOT / "cache")
os.environ["PATH"] = str(HERE / "ffmpeg") + os.pathsep + os.environ["PATH"]


def tts(job):
    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    kokoro = ROOT / "models/tts/kokoro-multi-lang-v1_0"
    use_kokoro = (kokoro / "model.onnx.verified.json").exists() and (kokoro / "voices.bin.verified.json").exists()
    model = kokoro if use_kokoro else ROOT / "models/tts/vits-zh-aishell3"
    rules = [model / name for name in (
        ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
        if use_kokoro else ("phone.fst", "date.fst", "number.fst")
    )]
    if use_kokoro:
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model / "model.onnx"), voices=str(model / "voices.bin"),
                tokens=str(model / "tokens.txt"), lexicon=str(model / "lexicon-zh.txt"),
                data_dir=str(model / "espeak-ng-data"), dict_dir=str(model / "dict"), lang="zh",
            ),
            num_threads=4, provider="cpu",
        )
    else:
        model_config = sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(next(model.glob("*.onnx"))), lexicon=str(model / "lexicon.txt"),
                tokens=str(model / "tokens.txt"),
            ),
            num_threads=4, provider="cpu",
        )
    config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        rule_fsts=",".join(str(path) for path in rules if path.exists()),
        max_num_sentences=1,
    )
    if not config.validate():
        raise RuntimeError("Chinese TTS configuration validation failed")
    engine = sherpa_onnx.OfflineTts(config)
    speaker = int(job["speaker"])
    if not 0 <= speaker < engine.num_speakers:
        raise ValueError(f"Speaker must be in 0..{engine.num_speakers - 1}")
    audio = engine.generate(job["text"], sid=speaker, speed=float(job["speed"]))
    samples = np.asarray(audio.samples, dtype=np.float32)
    if not samples.size or np.max(np.abs(samples)) < 0.001:
        raise RuntimeError("TTS returned empty or silent audio")
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, samples, audio.sample_rate, subtype="PCM_16")
    return {
        "audio": str(output),
        "sample_rate": audio.sample_rate,
        "duration": samples.size / audio.sample_rate,
        "speaker": speaker,
        "num_speakers": engine.num_speakers,
        "engine": "Kokoro" if use_kokoro else "AISHELL3/VITS",
        "rms": float(np.sqrt(np.mean(samples**2))),
    }


def lipsync(job):
    import cv2
    import numpy as np
    import pickle
    import shutil
    import soundfile as sf

    work = Path(job["work"])
    work.mkdir(parents=True, exist_ok=True)
    source = Path(job["video"])
    audio = Path(job["audio"])
    if not source.is_file() or not audio.is_file():
        raise FileNotFoundError("Input video and voice WAV must both exist")
    # Upstream CLI constructs shell commands, so only give it controlled ASCII paths.
    local_audio = work / "voice.wav"
    shutil.copy2(audio, local_audio)
    audio_info = sf.info(local_audio)
    if audio_info.duration <= 0 or audio_info.duration > 30:
        raise ValueError("Use one dialogue line per clip (maximum 30 seconds)")
    local_video = work / "source.mp4"
    ffmpeg = str(HERE / "ffmpeg/ffmpeg.exe")
    subprocess.run(
        [ffmpeg, "-y", "-v", "warning", "-stream_loop", "-1", "-i", str(source),
         "-t", str(audio_info.duration), "-an", "-vf", "fps=25",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(local_video)],
        check=True,
    )
    config = work / "inference.json"
    config.write_text(json.dumps({"shot": {
        "video_path": local_video.as_posix(),
        "audio_path": local_audio.as_posix(),
        "result_name": "dubbed.mp4",
    }}), encoding="utf-8")
    output_dir = work / "result"
    command = [
        sys.executable, "-m", "scripts.inference",
        "--inference_config", str(config), "--result_dir", str(output_dir),
        "--unet_model_path", "models/musetalkV15/unet.pth",
        "--unet_config", "models/musetalkV15/musetalk.json",
        "--version", "v15", "--use_float16", "--batch_size", "1", "--saved_coord",
        "--ffmpeg_path", str(HERE / "ffmpeg"),
    ]
    subprocess.run(command, cwd=HERE / "MuseTalk", check=True)
    result = output_dir / "v15/dubbed.mp4"
    if not result.is_file() or result.stat().st_size < 1024:
        raise RuntimeError("MuseTalk did not produce a video; inspect the job log")
    # Check the lower face actually changed, not merely that audio was muxed.
    coordinates = pickle.loads((work / "source.pkl").read_bytes())
    before = cv2.VideoCapture(str(local_video))
    after = cv2.VideoCapture(str(result))
    differences = []
    count = 0
    try:
        while True:
            ok1, original = before.read()
            ok2, dubbed = after.read()
            if not ok1 or not ok2:
                break
            x1, y1, x2, y2 = map(int, coordinates[count % len(coordinates)])
            y1 = y1 + (y2 - y1) // 2
            if x2 > x1 and y2 > y1 and count % 5 == 0:
                a = original[y1:y2, x1:x2].astype(np.float32)
                b = dubbed[y1:y2, x1:x2].astype(np.float32)
                differences.append(float(np.abs(a - b).mean()))
            count += 1
    finally:
        before.release()
        after.release()
    if not differences or float(np.mean(differences)) < 1.5:
        raise RuntimeError("No meaningful mouth-region change detected")
    output = Path(job["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result, output)
    return {"video": str(output), "audio_duration": audio_info.duration,
            "checked_frames": count, "mouth_region_change_mae": float(np.mean(differences))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["tts", "lipsync"])
    parser.add_argument("job")
    args = parser.parse_args()
    job_path = Path(args.job)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    result = {"tts": tts, "lipsync": lipsync}[args.stage](job)
    job_path.with_suffix(".result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

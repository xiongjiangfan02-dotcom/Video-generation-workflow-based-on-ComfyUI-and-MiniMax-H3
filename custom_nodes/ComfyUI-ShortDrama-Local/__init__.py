"""Small ComfyUI connectors; synthesis is done by upstream isolated engines."""

import gc
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import wave

import numpy as np
import torch
import folder_paths
import comfy.model_management
from comfy_api.latest import InputImpl

ROOT = Path("E:/AI/ComfyUI")
PIPELINE = ROOT / "pipeline"
PYTHON = PIPELINE / "python310/python.exe"


def new_job(stage):
    job = PIPELINE / "jobs" / f"{stage}_{uuid.uuid4().hex}"
    job.mkdir(parents=True)
    return job


def output_path(prefix, extension):
    prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", prefix).strip("._") or "shot"
    output = Path(folder_paths.get_output_directory()) / "short_drama" / "complete"
    output.mkdir(parents=True, exist_ok=True)
    return output / f"{prefix}_{time.time_ns()}{extension}"


def run_stage(stage, job_dir, payload):
    job = job_dir / "job.json"
    job.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    environment = os.environ.copy()
    environment.update(
        HF_HOME=str(ROOT / "cache/huggingface"),
        TORCH_HOME=str(ROOT / "cache/torch"),
        TEMP=str(ROOT / "cache/temp"),
        TMP=str(ROOT / "cache/temp"),
        PYTHONUTF8="1",
    )
    log = job_dir / "run.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [str(PYTHON), str(PIPELINE / "run_stage.py"), stage, str(job)],
            stdout=stream, stderr=subprocess.STDOUT, env=environment,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        try:
            while process.poll() is None:
                comfy.model_management.throw_exception_if_processing_interrupted()
                time.sleep(0.5)
        except BaseException:
            # Kill this job's child tree, never unrelated ComfyUI processes.
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            raise
    result = job.with_suffix(".result.json")
    if process.returncode != 0 or not result.exists():
        tail = log.read_text(encoding="utf-8", errors="replace")[-5000:]
        raise RuntimeError(f"{stage} failed. Log: {log}\n{tail}")
    return json.loads(result.read_text(encoding="utf-8"))


class ShortDramaChineseTTS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": "小满，你终于回来了。"}),
            "speaker_id": ("INT", {"default": 0, "min": 0, "max": 173}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.6, "max": 1.6, "step": 0.05}),
            "filename_prefix": ("STRING", {"default": "EP01_voice"}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "wav_path")
    FUNCTION = "generate"
    CATEGORY = "Short Drama/Local"
    OUTPUT_NODE = True

    def generate(self, text, speaker_id, speed, filename_prefix):
        if not text.strip():
            raise ValueError("Enter one line of dialogue")
        result = run_stage("tts", new_job("tts"), {
            "text": text, "speaker": speaker_id, "speed": speed,
            "output": str(output_path(filename_prefix, ".wav")),
        })
        path = Path(result["audio"])
        with wave.open(str(path), "rb") as stream:
            rate = stream.getframerate()
            samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").copy()
        audio = {"waveform": torch.from_numpy(samples.astype(np.float32) / 32768.0)[None, None, :],
                 "sample_rate": rate}
        subfolder = path.parent.relative_to(folder_paths.get_output_directory()).as_posix()
        return {"ui": {"audio": [{"filename": path.name, "subfolder": subfolder, "type": "output"}]},
                "result": (audio, str(path))}


class ShortDramaMuseTalk:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "wav_path": ("STRING", {"forceInput": True}),
            "filename_prefix": ("STRING", {"default": "EP01_dubbed"}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "mp4_path")
    FUNCTION = "generate"
    CATEGORY = "Short Drama/Local"

    def generate(self, video, wav_path, filename_prefix):
        job = new_job("lipsync")
        source = job / "input.mp4"
        video.save_to(str(source), format="mp4", codec="h264")
        comfy.model_management.unload_all_models()
        gc.collect()
        comfy.model_management.soft_empty_cache()
        result = run_stage("lipsync", job, {
            "video": str(source), "audio": wav_path, "work": str(job / "work"),
            "output": str(output_path(filename_prefix, ".mp4")),
        })
        return (InputImpl.VideoFromFile(result["video"]), result["video"])


class ShortDramaLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video_path": ("STRING", {"default": "E:/AI/ComfyUI/input/dialogue_source.mp4"}),
        }}

    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load"
    CATEGORY = "Short Drama/Local"

    def load(self, video_path):
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return (InputImpl.VideoFromFile(str(path)),)


NODE_CLASS_MAPPINGS = {
    "ShortDramaChineseTTS": ShortDramaChineseTTS,
    "ShortDramaMuseTalk": ShortDramaMuseTalk,
    "ShortDramaLoadVideo": ShortDramaLoadVideo,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ShortDramaChineseTTS": "本地中文配音",
    "ShortDramaMuseTalk": "MuseTalk 单人口型",
    "ShortDramaLoadVideo": "读取待配音视频",
}

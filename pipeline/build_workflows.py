"""Build visible ComfyUI workflows and matching API copies."""

from copy import deepcopy
import json
from pathlib import Path
import uuid

import requests

ROOT = Path("E:/AI/ComfyUI")
HERE = ROOT / "pipeline"
BASE = "http://127.0.0.1:8190"
CHECKPOINT = "AbsoluteReality_1.8.1_pruned.safetensors"
FIRST_FRAME = (
    "cinematic live action film still, photorealistic, one adult East Asian woman, "
    "23 years old, shoulder length straight black hair, gray blue loose shirt, "
    "medium close up, front facing, face fully visible, natural closed mouth, "
    "standing inside a modest apartment at night, warm practical lamp, "
    "natural skin texture, realistic eyes, soft cinematic lighting, 50mm lens"
)
NEGATIVE = (
    "anime, illustration, cartoon, 3d render, painting, text, watermark, "
    "multiple people, duplicate face, blurry, deformed, bad anatomy, "
    "open mouth, extra fingers, plastic skin"
)
MOTION = (
    "真人影视近景，同一名成年女子，保持首帧的脸、发型、灰蓝色衬衫和室内背景。"
    "自然呼吸，轻轻眨眼，头部仅有微小动作，表情平静。固定镜头，嘴部自然放松，"
    "面部始终清晰可见，不遮挡嘴，不切换场景。"
)


def first_frame():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CHECKPOINT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": FIRST_FRAME, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 432, "height": 768, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
            "latent_image": ["4", 0], "seed": 42, "steps": 24, "cfg": 6.5,
            "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {
            "images": ["6", 0], "filename_prefix": "short_drama/first_frames/EP01"}},
    }


def tts(node_id="201"):
    return {node_id: {"class_type": "ShortDramaChineseTTS", "inputs": {
        "text": "有你在，好像也不错。", "speaker_id": 45, "speed": 1.0,
        "filename_prefix": "EP01_Xiaoman"}}}


def add_h3(api):
    source = json.loads((ROOT / "downloads/h3/H3_03_Shot01_Fixed.api.json").read_text(encoding="utf-8"))
    for key, node in source.items():
        if key == "17":
            continue
        node = deepcopy(node)
        for name, value in node["inputs"].items():
            if isinstance(value, list):
                node["inputs"][name] = (
                    ["6", 0] if value[0] == "17" else [str(int(value[0]) + 100), value[1]]
                )
        api[str(int(key) + 100)] = node
    api["108"]["inputs"].update(prompt=MOTION, width=288, height=512, length=124)
    api["116"]["inputs"]["filename_prefix"] = "short_drama/silent/EP01"
    api["201"] = tts()["201"]
    api["202"] = {"class_type": "ShortDramaMuseTalk", "inputs": {
        "video": ["115", 0], "wav_path": ["201", 1], "filename_prefix": "EP01_dubbed"}}
    api["203"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["202", 0], "filename_prefix": "short_drama/final/EP01",
        "format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}}
    return api


TITLES = {
    "CheckpointLoaderSimple": "首帧：真人摄影模型",
    "EmptyLatentImage": "首帧尺寸：竖屏 9:16",
    "KSampler": "首帧采样",
    "SaveImage": "保存首帧到 E 盘",
    "ShortDramaChineseTTS": "中文台词与声音编号",
    "ShortDramaMuseTalk": "单人对口型（嘴部需清晰）",
    "ShortDramaLoadVideo": "已有视频的完整路径",
    "LoadImage": "参考人物图片",
}


def workflow(api, name, info):
    graph = {"id": str(uuid.uuid4()), "revision": 0, "nodes": [], "links": [],
             "groups": [], "config": {}, "extra": {"ds": {"scale": 0.65, "offset": [40, 40]}},
             "version": 0.4}
    h3_source = json.loads((ROOT / "workflows/H3_03_Shot01_Fixed.json").read_text(encoding="utf-8"))
    h3_templates = {str(node["id"] + 100): node for node in h3_source["nodes"] if node["id"] != 17}
    by_id = {}
    for index, (node_id, entry) in enumerate(api.items()):
        schema = info[entry["class_type"]]
        if node_id in h3_templates and entry["class_type"] == h3_templates[node_id]["type"]:
            node = deepcopy(h3_templates[node_id])
            node.update(id=int(node_id), order=index)
            node["pos"] = [node["pos"][0] + 1700, node["pos"][1]]
            for value in node.get("inputs", []):
                value["link"] = None
            for value in node.get("outputs", []):
                value["links"] = []
            if node_id == "108":
                node["widgets_values"] = [entry["inputs"]["prompt"], entry["inputs"]["width"],
                                         entry["inputs"]["height"], entry["inputs"]["length"]]
            if node_id == "116":
                node["widgets_values"] = [entry["inputs"]["filename_prefix"], "mp4", "h264", "auto"]
                node["widgets_values_named"] = {key: value for key, value in entry["inputs"].items()
                                                if not isinstance(value, list)}
        else:
            node = {
                "id": int(node_id), "type": entry["class_type"], "pos": [index % 4 * 400, index // 4 * 430],
                "size": [350, 300], "flags": {}, "order": index, "mode": 0,
                "inputs": [], "outputs": [], "properties": {"Node name for S&R": entry["class_type"]},
                "widgets_values": [], "title": TITLES.get(entry["class_type"], entry["class_type"]),
            }
            definitions = {**schema["input"].get("required", {}), **schema["input"].get("optional", {})}
            for key, definition in definitions.items():
                if key not in entry["inputs"]:
                    continue
                value = entry["inputs"][key]
                kind = definition[0]
                options = definition[1] if len(definition) > 1 else {}
                if isinstance(value, list):
                    node["inputs"].append({"name": key, "type": kind if isinstance(kind, str) else "COMBO", "link": None})
                    if kind in ("STRING", "INT", "FLOAT") and not options.get("forceInput"):
                        node["inputs"][-1]["widget"] = {"name": key}
                else:
                    node["widgets_values"].append(value)
                    if key in ("seed", "noise_seed"):
                        node["widgets_values"].append("fixed")
            if entry["class_type"] == "LoadImage":
                node["widgets_values"].append("image")
            if entry["class_type"] == "SaveVideo":
                node["widgets_values"] = [entry["inputs"]["filename_prefix"], "mp4", "h264", "auto"]
                node["widgets_values_named"] = {key: value for key, value in entry["inputs"].items()
                                                if not isinstance(value, list)}
            for slot, kind in enumerate(schema["output"]):
                node["outputs"].append({
                    "name": schema.get("output_name", schema["output"])[slot],
                    "type": kind, "links": [], "slot_index": slot,
                })
            if node_id == "2":
                node["title"] = "首帧画面描述（英文）"
                node["size"] = [390, 320]
            elif node_id == "3":
                node["title"] = "不希望出现的画面"
        graph["nodes"].append(node)
        by_id[node_id] = node
    for target, entry in api.items():
        for name, value in entry["inputs"].items():
            if not isinstance(value, list):
                continue
            source, output_slot = value
            node = by_id[target]
            input_slot = next(i for i, item in enumerate(node["inputs"]) if item["name"] == name)
            link_id = len(graph["links"]) + 1
            kind = by_id[source]["outputs"][output_slot]["type"]
            graph["links"].append([link_id, int(source), output_slot, int(target), input_slot, kind])
            node["inputs"][input_slot]["link"] = link_id
            by_id[source]["outputs"][output_slot]["links"].append(link_id)
    graph["last_node_id"] = max(int(key) for key in api)
    graph["last_link_id"] = len(graph["links"])
    if len(api) > 20:
        by_id["201"]["pos"] = [4350, -160]
        by_id["202"]["pos"] = [4800, 300]
        by_id["203"]["pos"] = [5200, 300]
        graph["extra"]["ds"] = {"scale": 0.25, "offset": [80, 100]}
    return graph


def main():
    info = requests.get(BASE + "/object_info", timeout=60).json()
    standalone = tts()
    standalone["202"] = {"class_type": "ShortDramaLoadVideo", "inputs": {
        "video_path": "E:/AI/ComfyUI/input/dialogue_source.mp4"}}
    standalone["203"] = {"class_type": "ShortDramaMuseTalk", "inputs": {
        "video": ["202", 0], "wav_path": ["201", 1], "filename_prefix": "EP01_dialogue"}}
    standalone["204"] = {"class_type": "SaveVideo", "inputs": {
        "video": ["203", 0], "filename_prefix": "short_drama/final/EP01_dialogue",
        "format": "mp4", "format.codec": "h264", "format.codec.encoding": "auto"}}
    reference = first_frame()
    reference["8"] = {"class_type": "LoadImage", "inputs": {"image": "h3_shot01_corridor.png"}}
    reference["9"] = {"class_type": "ImageScale", "inputs": {
        "image": ["8", 0], "upscale_method": "lanczos", "width": 432, "height": 768, "crop": "center"}}
    reference["4"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["1", 2]}}
    reference["5"]["inputs"]["denoise"] = 0.45
    reference["7"]["inputs"]["filename_prefix"] = "short_drama/first_frames/reference"
    workflows = {
        "Drama_01_FirstFrame": first_frame(),
        "Drama_01B_ReferenceFirstFrame": reference,
        "Drama_02_ChineseVoice": tts(),
        "Drama_03_VideoLipSync": standalone,
        "Drama_04_FullPipeline": add_h3(first_frame()),
    }
    for name, api in workflows.items():
        graph = workflow(api, name, info)
        for folder in (ROOT / "workflows", ROOT / "user/default/workflows"):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / (name + ".json")).write_text(
                json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
        (HERE / (name + ".api.json")).write_text(json.dumps(api, indent=2, ensure_ascii=False), encoding="utf-8")
        print("CREATED " + name)


if __name__ == "__main__":
    main()

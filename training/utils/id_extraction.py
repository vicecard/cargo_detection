import json
from pathlib import Path

json_path = Path("/home/trainer/cargo_detection/tmp/merged/annotations/instances_default.json")

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

image_ids = sorted({ann["image_id"] for ann in data.get("annotations", []) if "image_id" in ann})

print(image_ids)
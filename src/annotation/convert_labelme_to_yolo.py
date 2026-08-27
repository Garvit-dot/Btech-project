import json
from pathlib import Path

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELME_DIR = PROJECT_ROOT / "data" / "annotations" / "labelme_json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "annotations" / "yolo_labels"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Find JSON Files
# ============================================================

json_files = sorted(LABELME_DIR.glob("*.json"))

if len(json_files) == 0:
    print("No JSON files found.")
    exit()

print(f"\nFound {len(json_files)} JSON files.\n")

# ============================================================
# Discover Classes
# ============================================================

classes = set()

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):

        label = shape["label"].strip()

        classes.add(label)

classes = sorted(classes)

class_to_id = {cls: idx for idx, cls in enumerate(classes)}

# ============================================================
# Save classes.txt
# ============================================================

classes_path = OUTPUT_DIR / "classes.txt"

with open(classes_path, "w", encoding="utf-8") as f:

    for cls in classes:
        f.write(cls + "\n")

print("Classes Found:\n")

for cls, idx in class_to_id.items():
    print(f"{idx} -> {cls}")

print("\nclasses.txt created.\n")

# ============================================================
# Convert each JSON
# ============================================================

converted = 0

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    output_lines = []

    for shape in data.get("shapes", []):

        if shape["shape_type"] != "oriented_rectangle":
            continue

        label = shape["label"]

        class_id = class_to_id[label]

        points = shape["points"]

        normalized = []

        for x, y in points:

            normalized.append(x / image_width)
            normalized.append(y / image_height)

        line = (
            str(class_id)
            + " "
            + " ".join(f"{v:.6f}" for v in normalized)
        )

        output_lines.append(line)

    txt_path = OUTPUT_DIR / (json_file.stem + ".txt")

    with open(txt_path, "w", encoding="utf-8") as f:

        for line in output_lines:
            f.write(line + "\n")

    converted += 1

# ============================================================
# Summary
# ============================================================

print("=" * 50)
print("Conversion Complete")
print("=" * 50)
print(f"JSON files processed : {converted}")
print(f"Classes discovered   : {len(classes)}")
print(f"Output directory     : {OUTPUT_DIR}")
print("=" * 50)
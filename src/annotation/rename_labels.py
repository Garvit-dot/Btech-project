import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELME_DIR = PROJECT_ROOT / "data" / "annotations" / "labelme_json"


LABEL_MAPPING = {
    "healingsocket": "healing_socket",
    "workinglenghthforrct": "working_length_rct",
    "periaple_legion": "periapical_lesion",
}


json_files = sorted(LABELME_DIR.glob("*.json"))

if not json_files:
    print("❌ No JSON files found.")
    exit()

files_modified = 0
labels_changed = 0

print(f"\nFound {len(json_files)} JSON files.\n")

for json_file in json_files:

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    modified = False

    for shape in data.get("shapes", []):

        old_label = shape["label"]

        if old_label in LABEL_MAPPING:

            new_label = LABEL_MAPPING[old_label]

            shape["label"] = new_label

            print(f"{json_file.name} : {old_label}  -->  {new_label}")

            modified = True
            labels_changed += 1

    if modified:

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        files_modified += 1

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 50)
print("Rename Complete")
print("=" * 50)
print(f"Files modified : {files_modified}")
print(f"Labels renamed : {labels_changed}")
print("=" * 50)
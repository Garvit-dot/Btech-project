import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABELME_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "labelme_json"
)


def main():

    json_files = sorted(
        LABELME_DIR.glob("*.json")
    )

    print(f"Found {len(json_files)} JSON files.")

    fixed = 0
    missing_images = 0

    for json_file in json_files:

        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        image_name = Path(
            data["imagePath"]
        ).name

        actual_image = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "images"
            / image_name
        )

        if not actual_image.exists():

            print(
                f"WARNING: Image not found: "
                f"{image_name}"
            )

            missing_images += 1
            continue

        data["imagePath"] = (
            "../../raw/images/"
            + image_name
        )

        with open(
            json_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2
            )

        fixed += 1

    print("\n" + "=" * 50)
    print("IMAGE PATH FIX COMPLETE")
    print("=" * 50)

    print(
        f"JSON files found   : {len(json_files)}"
    )

    print(
        f"JSON files fixed   : {fixed}"
    )

    print(
        f"Missing images     : {missing_images}"
    )

    print("=" * 50)


if __name__ == "__main__":
    main()
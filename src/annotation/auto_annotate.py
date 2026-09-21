import os
import json
import random
from pathlib import Path

from PIL import Image
from ultralytics import YOLO



BATCH_SIZE = 206

CONFIDENCE_THRESHOLD = 0.25

RANDOM_SEED = 42

DEVICE = "cuda"

MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "yolov8n_obb"
    / "training"
    / "weights"
    / "best.pt"
)



PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
)

MANUAL_ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "labelme_json"
)

AUTO_ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "auto_annotations"
)

YOLO_LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "yolo_labels"
)

CLASSES_FILE = (
    YOLO_LABELS_DIR
    / "classes.txt"
)


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png"
}


# ============================================================
# PRINT HEADER
# ============================================================

def print_header(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_requirements():

    print_header("Checking Requirements")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{MODEL_PATH}"
        )

    if not RAW_IMAGES_DIR.exists():

        raise FileNotFoundError(
            f"\nImage directory not found:\n"
            f"{RAW_IMAGES_DIR}"
        )

    if not MANUAL_ANNOTATIONS_DIR.exists():

        raise FileNotFoundError(
            f"\nManual annotation directory not found:\n"
            f"{MANUAL_ANNOTATIONS_DIR}"
        )

    if not CLASSES_FILE.exists():

        raise FileNotFoundError(
            f"\nclasses.txt not found:\n"
            f"{CLASSES_FILE}"
        )

    print("Model found.")
    print("Image directory found.")
    print("Manual annotation directory found.")
    print("classes.txt found.")


# ============================================================
# READ CLASS NAMES
# ============================================================

def read_classes():

    with open(
        CLASSES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if not classes:

        raise RuntimeError(
            "classes.txt is empty."
        )

    print_header("Classes")

    for index, class_name in enumerate(classes):

        print(
            f"{index:2d} -> {class_name}"
        )

    return classes


# ============================================================
# FIND MANUALLY LABELED IMAGES
# ============================================================

def get_manual_annotation_stems():

    manual_stems = set()

    for json_file in MANUAL_ANNOTATIONS_DIR.glob("*.json"):

        manual_stems.add(
            json_file.stem
        )

    return manual_stems


# ============================================================
# FIND PREVIOUSLY AUTO-ANNOTATED IMAGES
# ============================================================

def get_previous_auto_annotation_stems():

    previous_stems = set()

    if not AUTO_ANNOTATIONS_DIR.exists():

        return previous_stems

    for batch_dir in AUTO_ANNOTATIONS_DIR.iterdir():

        if not batch_dir.is_dir():
            continue

        for json_file in batch_dir.glob("*.json"):

            previous_stems.add(
                json_file.stem
            )

    return previous_stems


# ============================================================
# FIND UNLABELED IMAGES
# ============================================================

def find_unlabeled_images():

    print_header("Finding Unlabeled Images")

    manual_stems = (
        get_manual_annotation_stems()
    )

    previous_auto_stems = (
        get_previous_auto_annotation_stems()
    )

    excluded_stems = (
        manual_stems
        | previous_auto_stems
    )

    unlabeled_images = []

    for image_path in sorted(
        RAW_IMAGES_DIR.iterdir()
    ):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if image_path.stem in excluded_stems:
            continue

        unlabeled_images.append(
            image_path
        )

    print(
        f"Total raw images              : "
        f"{len(list(RAW_IMAGES_DIR.iterdir()))}"
    )

    print(
        f"Manually labeled images       : "
        f"{len(manual_stems)}"
    )

    print(
        f"Previously auto-annotated     : "
        f"{len(previous_auto_stems)}"
    )

    print(
        f"Currently available unlabeled: "
        f"{len(unlabeled_images)}"
    )

    return unlabeled_images


# ============================================================
# DETERMINE NEXT BATCH NUMBER
# ============================================================

def get_next_batch_directory():

    AUTO_ANNOTATIONS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_batches = []

    for directory in AUTO_ANNOTATIONS_DIR.iterdir():

        if not directory.is_dir():
            continue

        name = directory.name.lower()

        if not name.startswith("batch_"):
            continue

        try:

            number = int(
                name.replace("batch_", "")
            )

            existing_batches.append(number)

        except ValueError:

            continue

    if not existing_batches:

        next_number = 1

    else:

        next_number = max(
            existing_batches
        ) + 1

    batch_directory = (
        AUTO_ANNOTATIONS_DIR
        / f"batch_{next_number:02d}"
    )

    batch_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    return batch_directory


# ============================================================
# SELECT NEXT 50 IMAGES
# ============================================================

def select_batch(unlabeled_images):

    print_header("Selecting Next Batch")

    if len(unlabeled_images) == 0:

        raise RuntimeError(
            "There are no unlabeled images remaining."
        )

    random.seed(RANDOM_SEED)

    number_to_select = min(
        BATCH_SIZE,
        len(unlabeled_images)
    )

    selected_images = random.sample(
        unlabeled_images,
        number_to_select
    )

    print(
        f"Selected {len(selected_images)} images."
    )

    return selected_images


# ============================================================
# CREATE LABELME JSON
# ============================================================

def create_labelme_json(
    image_path,
    result,
    classes,
    batch_directory
):

    # --------------------------------------------------------
    # Get image dimensions
    # --------------------------------------------------------

    with Image.open(image_path) as image:

        image_width, image_height = image.size

    # --------------------------------------------------------
    # Basic LabelMe structure
    # --------------------------------------------------------
    relative_image_path = os.path.relpath(
        image_path.resolve(),
        batch_directory.resolve()
    )

    relative_image_path = relative_image_path.replace("\\", "/")
    labelme_data = {

        "version": "6.3.1",

        "flags": {},

        "shapes": [],

        "imagePath": relative_image_path,

        "imageData": None,

        "imageHeight": image_height,

        "imageWidth": image_width
    }

    # --------------------------------------------------------
    # Check whether detections exist
    # --------------------------------------------------------

    if result.obb is None:

        return labelme_data

    # --------------------------------------------------------
    # Extract OBB predictions
    # --------------------------------------------------------

    points = (
        result.obb.xyxyxyxy
        .cpu()
        .numpy()
    )

    class_ids = (
        result.obb.cls
        .cpu()
        .numpy()
    )

    confidences = (
        result.obb.conf
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Convert every detection
    # --------------------------------------------------------

    for box_points, class_id, confidence in zip(
        points,
        class_ids,
        confidences
    ):

        class_id = int(class_id)

        confidence = float(
            confidence
        )

        # Safety check
        if class_id >= len(classes):
            continue

        class_name = classes[class_id]

        # Convert numpy points to normal Python floats
        formatted_points = []

        for point in box_points:

            x = float(point[0])
            y = float(point[1])

            formatted_points.append(
                [x, y]
            )

        # ----------------------------------------------------
        # LabelMe oriented rectangle
        # ----------------------------------------------------

        shape = {

            "label": class_name,

            "points": formatted_points,

            "group_id": None,

            "description": (
                f"Auto-generated "
                f"confidence={confidence:.4f}"
            ),

            "shape_type": "oriented_rectangle",

            "flags": {},

            "mask": None
        }

        labelme_data["shapes"].append(
            shape
        )

    return labelme_data


# ============================================================
# RUN MODEL ON BATCH
# ============================================================

def generate_predictions(
    selected_images,
    classes,
    batch_directory
):

    print_header(
        "Running YOLOv8n-OBB Predictions"
    )

    print(
        f"Images to process: "
        f"{len(selected_images)}"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"Device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = YOLO(
        str(MODEL_PATH)
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Run predictions
    # --------------------------------------------------------

    results = model.predict(

        source=[
            str(image)
            for image in selected_images
        ],

        conf=CONFIDENCE_THRESHOLD,

        device=DEVICE,

        imgsz=640,

        save=False,

        verbose=True
    )

    # --------------------------------------------------------
    # Create JSON files
    # --------------------------------------------------------

    total_detections = 0

    images_with_detections = 0

    images_without_detections = 0

    for image_path, result in zip(
        selected_images,
        results
    ):

        labelme_data = create_labelme_json(
            image_path,
            result,
            classes,
            batch_directory
        )

        detection_count = len(
            labelme_data["shapes"]
        )

        total_detections += (
            detection_count
        )

        if detection_count > 0:

            images_with_detections += 1

        else:

            images_without_detections += 1

        # ----------------------------------------------------
        # Save JSON
        # ----------------------------------------------------

        output_json = (
            batch_directory
            / f"{image_path.stem}.json"
        )

        with open(
            output_json,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                labelme_data,
                file,
                indent=2
            )

        print(
            f"Saved: {output_json.name} "
            f"({detection_count} detections)"
        )

    return (
        total_detections,
        images_with_detections,
        images_without_detections
    )


# ============================================================
# SAVE SELECTED IMAGE LIST
# ============================================================

def save_batch_list(
    selected_images,
    batch_directory
):

    list_file = (
        batch_directory
        / "selected_images.txt"
    )

    with open(
        list_file,
        "w",
        encoding="utf-8"
    ) as file:

        for image_path in selected_images:

            file.write(
                image_path.name + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "AUTOMATIC DENTAL IMAGE ANNOTATION"
    )

    print(
        "YOLOv8n-OBB → LabelMe"
    )

    # --------------------------------------------------------
    # 1. Check requirements
    # --------------------------------------------------------

    check_requirements()

    # --------------------------------------------------------
    # 2. Load classes
    # --------------------------------------------------------

    classes = read_classes()

    # --------------------------------------------------------
    # 3. Find unlabeled images
    # --------------------------------------------------------

    unlabeled_images = (
        find_unlabeled_images()
    )

    # --------------------------------------------------------
    # 4. Select 50
    # --------------------------------------------------------

    selected_images = select_batch(
        unlabeled_images
    )

    # --------------------------------------------------------
    # 5. Create batch directory
    # --------------------------------------------------------

    batch_directory = (
        get_next_batch_directory()
    )

    print_header("Batch Created")

    print(
        f"Output directory:\n"
        f"{batch_directory}"
    )

    # --------------------------------------------------------
    # 6. Save selected image list
    # --------------------------------------------------------

    save_batch_list(
        selected_images,
        batch_directory
    )

    # --------------------------------------------------------
    # 7. Run predictions
    # --------------------------------------------------------

    (
        total_detections,
        images_with_detections,
        images_without_detections
    ) = generate_predictions(
        selected_images,
        classes,
        batch_directory
    )

    # --------------------------------------------------------
    # 8. Final summary
    # --------------------------------------------------------

    print_header(
        "AUTO-ANNOTATION COMPLETE"
    )

    print(
        f"Images processed       : "
        f"{len(selected_images)}"
    )

    print(
        f"Images with detections : "
        f"{images_with_detections}"
    )

    print(
        f"Images without         : "
        f"{images_without_detections}"
    )

    print(
        f"Total detections       : "
        f"{total_detections}"
    )

    print(
        f"\nPredicted JSON files:"
    )

    print(
        f"{batch_directory}"
    )

    print("\nNEXT STEP:")

    print(
        "Open these JSON files in LabelMe "
        "and manually verify/correct the predictions."
    )

    print(
        "\nDO NOT move them into labelme_json "
        "until they have been verified."
    )


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
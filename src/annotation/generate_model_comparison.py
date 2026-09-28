import os
import random
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ultralytics import YOLO

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)
from torchvision.transforms import functional as F


# ============================================================
# PROJECT ROOT
# ============================================================

# This file is:
#
# Btech-project/
# └── src/
#     └── annotation/
#         └── generate_model_comparison.py
#
# Therefore parents[2] = Btech-project

SCRIPT_DIR = Path(__file__).resolve()

PROJECT_ROOT = SCRIPT_DIR.parents[2]

print("=" * 70)
print("DENTAL MODEL VISUAL COMPARISON")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# MODEL PATHS
# ============================================================

YOLOV8_MODEL = (
    PROJECT_ROOT
    / "models"
    / "yolov8n_obb"
    / "training"
    / "weights"
    / "best.pt"
)

YOLO11_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "obb"
    / "models"
    / "yolo11n_obb"
    / "training-2"
    / "weights"
    / "best.pt"
)

FASTER_RCNN_MODEL = (
    PROJECT_ROOT
    / "models"
    / "faster_rcnn"
    / "best.pt"
)


# ============================================================
# VALIDATION IMAGES
# ============================================================

VAL_IMAGES = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "images"
    / "val"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "model_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 5

CONFIDENCE_THRESHOLD = 0.50

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "GP",
    "boneloss",
    "caries",
    "crown",
    "dental_stone",
    "erupting_tooth",
    "filled",
    "healing_socket",
    "missing",
    "opturation",
    "periapical_lesion",
    "pulp_stone",
    "rootcanal",
    "working_length_rct"
]


# ============================================================
# VERIFY PATHS
# ============================================================

print("\nChecking required files...")
print("-" * 70)


def check_path(
    path,
    name
):

    print(
        f"{name}:"
    )

    print(
        f"  {path}"
    )

    if not path.exists():

        print(
            "  ❌ NOT FOUND"
        )

        return False

    print(
        "  ✓ Found"
    )

    return True


yolov8_exists = check_path(
    YOLOV8_MODEL,
    "YOLOv8"
)

yolo11_exists = check_path(
    YOLO11_MODEL,
    "YOLO11"
)

faster_exists = check_path(
    FASTER_RCNN_MODEL,
    "Faster R-CNN"
)

images_exist = check_path(
    VAL_IMAGES,
    "Validation images"
)


if not (
    yolov8_exists
    and yolo11_exists
    and faster_exists
    and images_exist
):

    print("\n")
    print("=" * 70)
    print("ERROR: One or more required paths were not found.")
    print("=" * 70)

    print(
        "\nThis script is intentionally stopping here "
        "so we don't get a confusing model-loading error."
    )

    raise SystemExit(1)


# ============================================================
# LOAD YOLOv8
# ============================================================

print("\n")
print("=" * 70)
print("LOADING YOLOv8")
print("=" * 70)

yolov8 = YOLO(
    str(YOLOV8_MODEL)
)

print(
    "YOLOv8 loaded successfully."
)


# ============================================================
# LOAD YOLO11
# ============================================================

print("\n")
print("=" * 70)
print("LOADING YOLO11")
print("=" * 70)

yolo11 = YOLO(
    str(YOLO11_MODEL)
)

print(
    "YOLO11 loaded successfully."
)


# ============================================================
# LOAD FASTER R-CNN
# ============================================================

print("\n")
print("=" * 70)
print("LOADING FASTER R-CNN")
print("=" * 70)

faster_rcnn = fasterrcnn_resnet50_fpn(
    weights=None
)

in_features = (
    faster_rcnn
    .roi_heads
    .box_predictor
    .cls_score
    .in_features
)

faster_rcnn.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        15
    )
)


checkpoint = torch.load(
    str(FASTER_RCNN_MODEL),
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    faster_rcnn.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    faster_rcnn.load_state_dict(
        checkpoint
    )


faster_rcnn.to(DEVICE)

faster_rcnn.eval()

print(
    "Faster R-CNN loaded successfully."
)


# ============================================================
# SELECT VALIDATION IMAGES
# ============================================================

all_images = sorted(
    [
        f
        for f in VAL_IMAGES.iterdir()
        if f.suffix.lower()
        in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]
    ]
)


print("\n")
print("=" * 70)
print("VALIDATION DATASET")
print("=" * 70)

print(
    f"\nTotal validation images: "
    f"{len(all_images)}"
)


if len(all_images) == 0:

    raise RuntimeError(
        "No validation images were found."
    )


random.seed(42)

selected_images = random.sample(
    all_images,
    min(
        NUM_IMAGES,
        len(all_images)
    )
)


print("\nSelected images:")

for image_path in selected_images:

    print(
        f"  - {image_path.name}"
    )


# ============================================================
# FASTER R-CNN DRAWING FUNCTION
# ============================================================

def draw_faster_rcnn(
    image,
    prediction
):

    output = image.copy()

    draw = ImageDraw.Draw(
        output
    )

    boxes = prediction[
        "boxes"
    ]

    labels = prediction[
        "labels"
    ]

    scores = prediction[
        "scores"
    ]

    for box, label, score in zip(
        boxes,
        labels,
        scores
    ):

        confidence = float(
            score.item()
        )

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = (
            box.tolist()
        )

        # Faster R-CNN labels:
        # 0 = background
        # 1-14 = dental classes

        class_id = (
            int(label.item()) - 1
        )

        if (
            class_id >= 0
            and class_id < len(
                CLASS_NAMES
            )
        ):

            class_name = (
                CLASS_NAMES[class_id]
            )

        else:

            class_name = (
                f"class_{class_id}"
            )

        label_text = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2
            ],
            outline="red",
            width=3
        )

        draw.text(
            (
                x1,
                max(
                    0,
                    y1 - 18
                )
            ),
            label_text,
            fill="red"
        )

    return output


# ============================================================
# ADD TITLE
# ============================================================

def add_title(
    image,
    title
):

    title_height = 45

    canvas = Image.new(
        "RGB",
        (
            image.width,
            image.height
            + title_height
        ),
        "white"
    )

    canvas.paste(
        image,
        (
            0,
            title_height
        )
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            10,
            10
        ),
        title,
        fill="black"
    )

    return canvas


# ============================================================
# CREATE COMPARISON SHEET
# ============================================================

def create_comparison_sheet(
    original,
    yolov8_image,
    yolo11_image,
    faster_image,
    image_name
):

    # --------------------------------------------------------
    # Resize images
    # --------------------------------------------------------

    target_width = 600

    def resize_image(
        image
    ):

        ratio = (
            target_width
            / image.width
        )

        new_height = int(
            image.height
            * ratio
        )

        return image.resize(
            (
                target_width,
                new_height
            )
        )

    original = resize_image(
        original
    )

    yolov8_image = resize_image(
        yolov8_image
    )

    yolo11_image = resize_image(
        yolo11_image
    )

    faster_image = resize_image(
        faster_image
    )

    # --------------------------------------------------------
    # Add titles
    # --------------------------------------------------------

    original = add_title(
        original,
        "Original"
    )

    yolov8_image = add_title(
        yolov8_image,
        "YOLOv8n-OBB"
    )

    yolo11_image = add_title(
        yolo11_image,
        "YOLO11n-OBB"
    )

    faster_image = add_title(
        faster_image,
        "Faster R-CNN"
    )

    # --------------------------------------------------------
    # Determine dimensions
    # --------------------------------------------------------

    cell_width = target_width

    cell_height = max(
        original.height,
        yolov8_image.height,
        yolo11_image.height,
        faster_image.height
    )

    sheet_width = (
        cell_width * 2
    )

    sheet_height = (
        cell_height * 2
    )

    # --------------------------------------------------------
    # Create sheet
    # --------------------------------------------------------

    sheet = Image.new(
        "RGB",
        (
            sheet_width,
            sheet_height
        ),
        "white"
    )

    # --------------------------------------------------------
    # Paste images
    # --------------------------------------------------------

    sheet.paste(
        original,
        (
            0,
            0
        )
    )

    sheet.paste(
        yolov8_image,
        (
            cell_width,
            0
        )
    )

    sheet.paste(
        yolo11_image,
        (
            0,
            cell_height
        )
    )

    sheet.paste(
        faster_image,
        (
            cell_width,
            cell_height
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_name = (
        image_name.stem
        + "_comparison.jpg"
    )

    output_path = (
        OUTPUT_DIR
        / output_name
    )

    sheet.save(
        output_path,
        quality=95
    )

    print(
        f"Saved: {output_path}"
    )


# ============================================================
# RUN MODEL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("GENERATING MODEL COMPARISONS")
print("=" * 70)


for index, image_path in enumerate(
    selected_images,
    start=1
):

    print("\n")
    print(
        f"[{index}/{len(selected_images)}] "
        f"{image_path.name}"
    )

    # --------------------------------------------------------
    # Load original
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # YOLOv8 prediction
    # --------------------------------------------------------

    print(
        "  Running YOLOv8..."
    )

    yolo8_result = (
        yolov8.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=0
            if DEVICE == "cuda"
            else "cpu"
        )[0]
    )

    yolo8_plot = (
        yolo8_result.plot(
            conf=True,
            labels=True,
            boxes=True
        )
    )

    yolo8_image = Image.fromarray(
        yolo8_plot
    )

    # --------------------------------------------------------
    # YOLO11 prediction
    # --------------------------------------------------------

    print(
        "  Running YOLO11..."
    )

    yolo11_result = (
        yolo11.predict(
            source=str(image_path),
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            device=0
            if DEVICE == "cuda"
            else "cpu"
        )[0]
    )

    yolo11_plot = (
        yolo11_result.plot(
            conf=True,
            labels=True,
            boxes=True
        )
    )

    yolo11_image = Image.fromarray(
        yolo11_plot
    )

    # --------------------------------------------------------
    # Faster R-CNN prediction
    # --------------------------------------------------------

    print(
        "  Running Faster R-CNN..."
    )

    image_tensor = (
        F.to_tensor(image)
        .to(DEVICE)
    )

    with torch.no_grad():

        faster_prediction = (
            faster_rcnn(
                [image_tensor]
            )[0]
        )

    faster_image = (
        draw_faster_rcnn(
            image,
            faster_prediction
        )
    )

    # --------------------------------------------------------
    # Create sheet
    # --------------------------------------------------------

    create_comparison_sheet(
        original=image,
        yolov8_image=yolo8_image,
        yolo11_image=yolo11_image,
        faster_image=faster_image,
        image_name=image_path
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nOpen the generated *_comparison.jpg files "
    "to view the results."
)
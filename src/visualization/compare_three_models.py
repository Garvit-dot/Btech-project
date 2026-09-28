import cv2
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

YOLO_PYTHON = (
    PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
)

MMROTATE_PYTHON = (
    PROJECT_ROOT / ".venv_mmrotate" / "Scripts" / "python.exe"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
    / "images"
    / "val"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "visual_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECKPOINTS
# ============================================================

YOLO_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "obb"
    / "models"
    / "yolo11n_obb"
    / "training-2"
    / "weights"
    / "best.pt"
)

RTMDET_DIR = (
    PROJECT_ROOT
    / "runs"
    / "mmrotate"
    / "rtmdet_tiny_r_iopa"
)

RTMDET_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "rtmdet"
    / "rtmdet_tiny_r_iopa.py"
)

RETINANET_DIR = (
    PROJECT_ROOT
    / "runs"
    / "mmrotate"
    / "rotated_retinanet_iopa"
)

RETINANET_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "rotated_retinanet"
    / "rotated_retinanet_iopa.py"
)


# ============================================================
# SETTINGS
# ============================================================

NUM_IMAGES = 6
CONF_THRESHOLD = 0.25


# ============================================================
# CLASSES
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
    "working_length_rct",
]


# ============================================================
# CHECK PATHS
# ============================================================

def check_paths():

    paths = {
        "YOLO Python": YOLO_PYTHON,
        "MMRotate Python": MMROTATE_PYTHON,
        "Validation directory": VAL_DIR,
        "YOLO checkpoint": YOLO_MODEL,
        "RTMDet config": RTMDET_CONFIG,
        "RTMDet directory": RTMDET_DIR,
        "RetinaNet config": RETINANET_CONFIG,
        "RetinaNet directory": RETINANET_DIR,
    }

    print("\n" + "=" * 70)
    print("CHECKING PATHS")
    print("=" * 70)

    for name, path in paths.items():

        if path.exists():
            print(f"[OK] {name}")
        else:
            print(f"[MISSING] {name}")
            print(f"         {path}")

            raise FileNotFoundError(
                f"Missing: {name}"
            )


# ============================================================
# FIND CHECKPOINT
# ============================================================

def find_checkpoint(directory):

    candidates = []

    candidates.extend(
        directory.rglob("best*.pth")
    )

    candidates.extend(
        directory.rglob("*best*.pth")
    )

    if candidates:

        candidates.sort(
            key=lambda p: p.stat().st_mtime
        )

        return candidates[-1]

    candidates = list(
        directory.rglob("epoch_*.pth")
    )

    if candidates:

        candidates.sort(
            key=lambda p: p.stat().st_mtime
        )

        return candidates[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {directory}"
    )


# ============================================================
# GET IMAGES
# ============================================================

def get_images():

    images = []

    for ext in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.bmp"
    ]:

        images.extend(
            VAL_DIR.glob(ext)
        )

    images.sort(
        key=lambda p: p.name
    )

    return images[:NUM_IMAGES]


# ============================================================
# TEMP IMAGE DIRECTORY
# ============================================================

def prepare_images(images):

    temp_dir = (
        OUTPUT_DIR / "_temp_images"
    )

    if temp_dir.exists():

        shutil.rmtree(
            temp_dir
        )

    temp_dir.mkdir(
        parents=True
    )

    for image in images:

        shutil.copy2(
            image,
            temp_dir / image.name
        )

    return temp_dir


# ============================================================
# YOLO RUNNER
# ============================================================

def create_yolo_runner():

    code = r'''
import sys
import json
from pathlib import Path

from ultralytics import YOLO


model_path = sys.argv[1]
image_dir = sys.argv[2]
output_json = sys.argv[3]


model = YOLO(model_path)


images = sorted(
    list(Path(image_dir).glob("*.jpg")) +
    list(Path(image_dir).glob("*.jpeg")) +
    list(Path(image_dir).glob("*.png")) +
    list(Path(image_dir).glob("*.bmp"))
)


results = {}


for image_path in images:

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )[0]


    detections = []


    if result.obb is not None:

        polygons = (
            result.obb.xyxyxyxy
            .cpu()
            .numpy()
        )

        scores = (
            result.obb.conf
            .cpu()
            .numpy()
        )

        labels = (
            result.obb.cls
            .cpu()
            .numpy()
        )


        for polygon, score, label in zip(
            polygons,
            scores,
            labels
        ):

            detections.append({

                "points":
                    polygon.tolist(),

                "confidence":
                    float(score),

                "class_id":
                    int(label)
            })


    results[
        image_path.name
    ] = detections


with open(
    output_json,
    "w"
) as f:

    json.dump(
        results,
        f
    )
'''

    path = (
        OUTPUT_DIR
        / "_run_yolo.py"
    )

    path.write_text(
        code,
        encoding="utf-8"
    )

    return path


# ============================================================
# RUN YOLO
# ============================================================

def run_yolo(temp_dir):

    print("\n" + "=" * 70)
    print("YOLO11n-OBB")
    print("=" * 70)

    runner = create_yolo_runner()

    output_json = (
        OUTPUT_DIR
        / "yolo11_predictions.json"
    )

    subprocess.run(
        [
            str(YOLO_PYTHON),
            str(runner),
            str(YOLO_MODEL),
            str(temp_dir),
            str(output_json)
        ],
        check=True
    )

    with open(
        output_json,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# MMROTATE RUNNER
# ============================================================

def create_mmrotate_runner():

    code = r'''
import sys
import json
from pathlib import Path

import numpy as np
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope

from mmdet.apis import init_detector
from mmdet.apis import inference_detector

import mmrotate.models


config_path = sys.argv[1]
checkpoint_path = sys.argv[2]
image_dir = sys.argv[3]
output_json = sys.argv[4]


# ============================================================
# CONFIG
# ============================================================

cfg = Config.fromfile(
    config_path
)


# ============================================================
# SCOPE
# ============================================================

init_default_scope(
    "mmrotate"
)


# ============================================================
# BUILD MODEL USING NATIVE MMDET API
# ============================================================

device = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


model = init_detector(
    cfg,
    checkpoint_path,
    device=device
)


# ============================================================
# IMAGES
# ============================================================

images = sorted(

    list(Path(image_dir).glob("*.jpg"))

    +

    list(Path(image_dir).glob("*.jpeg"))

    +

    list(Path(image_dir).glob("*.png"))

    +

    list(Path(image_dir).glob("*.bmp"))
)


results = {}


# ============================================================
# INFERENCE
# ============================================================

for image_path in images:

    result = inference_detector(
        model,
        str(image_path)
    )


    pred = result.pred_instances


    detections = []


    if len(pred) > 0:

        scores = (
            pred.scores
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            pred.labels
            .detach()
            .cpu()
            .numpy()
        )


        if hasattr(
            pred,
            "bboxes"
        ):

            boxes = (
                pred.bboxes
                .detach()
                .cpu()
                .numpy()
            )

        else:

            boxes = []


        for box, score, label in zip(
            boxes,
            scores,
            labels
        ):

            if score < 0.25:
                continue


            box = np.asarray(
                box
            ).reshape(-1)


            detections.append({

                "box":
                    box.tolist(),

                "confidence":
                    float(score),

                "class_id":
                    int(label)
            })


    results[
        image_path.name
    ] = detections


# ============================================================
# SAVE
# ============================================================

with open(
    output_json,
    "w"
) as f:

    json.dump(
        results,
        f
    )
'''

    path = (
        OUTPUT_DIR
        / "_run_mmrotate.py"
    )

    path.write_text(
        code,
        encoding="utf-8"
    )

    return path


# ============================================================
# RUN MMROTATE
# ============================================================

def run_mmrotate(
    name,
    config,
    checkpoint,
    temp_dir
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    runner = (
        create_mmrotate_runner()
    )

    filename = (
        name
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    output_json = (
        OUTPUT_DIR
        / f"{filename}_predictions.json"
    )

    subprocess.run(
        [
            str(MMROTATE_PYTHON),
            str(runner),
            str(config),
            str(checkpoint),
            str(temp_dir),
            str(output_json)
        ],
        check=True
    )

    with open(
        output_json,
        "r"
    ) as f:

        return json.load(f)


# ============================================================
# DRAW POLYGON
# ============================================================

def draw_polygon(
    image,
    points,
    label,
    confidence
):

    points = np.asarray(
        points,
        dtype=np.float32
    )

    points = points.reshape(
        -1,
        2
    )

    points = np.round(
        points
    ).astype(
        np.int32
    )

    cv2.polylines(
        image,
        [points],
        True,
        (0, 255, 0),
        2
    )

    x = int(
        points[:, 0].min()
    )

    y = int(
        points[:, 1].min()
    )

    cv2.putText(

        image,

        f"{label} {confidence:.2f}",

        (
            x,
            max(y - 5, 15)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (0, 255, 0),

        1,

        cv2.LINE_AA
    )


# ============================================================
# DRAW ROTATED BOX
# ============================================================

def draw_rotated_box(
    image,
    box,
    label,
    confidence
):

    box = np.asarray(
        box,
        dtype=np.float32
    ).reshape(-1)


    if len(box) != 5:
        return


    cx, cy, w, h, angle = box


    # MMRotate angle -> degrees
    angle = (
        angle
        * 180.0
        / np.pi
    )


    rectangle = (

        (
            float(cx),
            float(cy)
        ),

        (
            float(w),
            float(h)
        ),

        float(angle)
    )


    points = cv2.boxPoints(
        rectangle
    )

    points = np.round(
        points
    ).astype(
        np.int32
    )


    cv2.polylines(
        image,
        [points],
        True,
        (0, 255, 0),
        2
    )


    x = int(
        points[:, 0].min()
    )

    y = int(
        points[:, 1].min()
    )


    cv2.putText(

        image,

        f"{label} {confidence:.2f}",

        (
            x,
            max(y - 5, 15)
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.45,

        (0, 255, 0),

        1,

        cv2.LINE_AA
    )


# ============================================================
# PANEL
# ============================================================

def make_panel(
    image,
    title,
    width=500
):

    height, original_width = (
        image.shape[:2]
    )


    scale = (
        width
        / original_width
    )


    image = cv2.resize(

        image,

        (
            width,
            int(height * scale)
        )
    )


    header = np.zeros(

        (
            50,
            width,
            3
        ),

        dtype=np.uint8
    )


    cv2.putText(

        header,

        title,

        (
            15,
            34
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.75,

        (255, 255, 255),

        2,

        cv2.LINE_AA
    )


    return np.vstack(
        [
            header,
            image
        ]
    )


# ============================================================
# CREATE COMPARISON
# ============================================================

def create_comparisons(
    images,
    yolo,
    rtmdet,
    retinanet
):

    for image_path in images:

        original = cv2.imread(
            str(image_path)
        )


        if original is None:
            continue


        yolo_img = original.copy()

        rtmdet_img = original.copy()

        retinanet_img = original.copy()


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        for d in yolo.get(
            image_path.name,
            []
        ):

            if d["confidence"] < CONF_THRESHOLD:
                continue


            cls = d["class_id"]


            label = (
                CLASS_NAMES[cls]
                if cls < len(CLASS_NAMES)
                else str(cls)
            )


            draw_polygon(
                yolo_img,
                d["points"],
                label,
                d["confidence"]
            )


        # ----------------------------------------------------
        # RTMDET
        # ----------------------------------------------------

        for d in rtmdet.get(
            image_path.name,
            []
        ):

            if d["confidence"] < CONF_THRESHOLD:
                continue


            cls = d["class_id"]


            label = (
                CLASS_NAMES[cls]
                if cls < len(CLASS_NAMES)
                else str(cls)
            )


            draw_rotated_box(
                rtmdet_img,
                d["box"],
                label,
                d["confidence"]
            )


        # ----------------------------------------------------
        # RETINANET
        # ----------------------------------------------------

        for d in retinanet.get(
            image_path.name,
            []
        ):

            if d["confidence"] < CONF_THRESHOLD:
                continue


            cls = d["class_id"]


            label = (
                CLASS_NAMES[cls]
                if cls < len(CLASS_NAMES)
                else str(cls)
            )


            draw_rotated_box(
                retinanet_img,
                d["box"],
                label,
                d["confidence"]
            )


        # ----------------------------------------------------
        # PANELS
        # ----------------------------------------------------

        panels = [

            make_panel(
                yolo_img,
                "YOLO11n-OBB"
            ),

            make_panel(
                rtmdet_img,
                "RTMDet-tiny-R"
            ),

            make_panel(
                retinanet_img,
                "Rotated RetinaNet"
            )
        ]


        max_height = max(
            p.shape[0]
            for p in panels
        )


        padded = []


        for panel in panels:

            difference = (
                max_height
                - panel.shape[0]
            )


            if difference > 0:

                panel = np.vstack(

                    [
                        panel,

                        np.zeros(
                            (
                                difference,
                                panel.shape[1],
                                3
                            ),
                            dtype=np.uint8
                        )
                    ]
                )


            padded.append(
                panel
            )


        comparison = np.hstack(
            padded
        )


        output = (

            OUTPUT_DIR
            / f"comparison_{image_path.stem}.jpg"
        )


        cv2.imwrite(
            str(output),
            comparison
        )


        print(
            f"[SAVED] {output.name}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 70)

    print(
        "YOLO11 vs RTMDet vs Rotated RetinaNet"
    )

    print("=" * 70)


    check_paths()


    images = get_images()


    print(
        f"\nUsing {len(images)} images:"
    )

    for image in images:
        print(
            f"  {image.name}"
        )


    temp_dir = (
        prepare_images(
            images
        )
    )


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    rtmdet_checkpoint = (
        find_checkpoint(
            RTMDET_DIR
        )
    )


    retinanet_checkpoint = (
        find_checkpoint(
            RETINANET_DIR
        )
    )


    print(
        "\nRTMDet:"
    )

    print(
        rtmdet_checkpoint
    )


    print(
        "\nRotated RetinaNet:"
    )

    print(
        retinanet_checkpoint
    )


    # --------------------------------------------------------
    # YOLO11
    # --------------------------------------------------------

    yolo_predictions = run_yolo(
        temp_dir
    )


    # --------------------------------------------------------
    # RTMDet
    # --------------------------------------------------------

    rtmdet_predictions = run_mmrotate(

        "RTMDet-tiny-R",

        RTMDET_CONFIG,

        rtmdet_checkpoint,

        temp_dir
    )


    # --------------------------------------------------------
    # RetinaNet
    # --------------------------------------------------------

    retinanet_predictions = run_mmrotate(

        "Rotated RetinaNet",

        RETINANET_CONFIG,

        retinanet_checkpoint,

        temp_dir
    )


    # --------------------------------------------------------
    # Visuals
    # --------------------------------------------------------

    create_comparisons(

        images,

        yolo_predictions,

        rtmdet_predictions,

        retinanet_predictions
    )


    print("\n" + "=" * 70)

    print(
        "DONE"
    )

    print("=" * 70)

    print(
        f"\nResults:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()
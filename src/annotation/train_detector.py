import random
import shutil
from pathlib import Path

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
TRAIN_SPLIT = 0.80

# For the first bootstrap model, 50 epochs is enough.
EPOCHS = 50

IMAGE_SIZE = 640
BATCH_SIZE = 8

# IMPORTANT FOR WINDOWS
# Keep this at 0 initially.
WORKERS = 0

# Automatically use CUDA if available.
DEVICE = "cuda"

MODEL_NAME = "yolov8n-obb.pt"

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_IMAGES_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "images"
)

YOLO_LABELS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
    / "yolo_labels"
)

YOLO_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "yolo_dataset"
)

MODEL_OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "yolov8n_obb"
)

TRAIN_IMAGES_DIR = (
    YOLO_DATASET_DIR
    / "images"
    / "train"
)

VAL_IMAGES_DIR = (
    YOLO_DATASET_DIR
    / "images"
    / "val"
)

TRAIN_LABELS_DIR = (
    YOLO_DATASET_DIR
    / "labels"
    / "train"
)

VAL_LABELS_DIR = (
    YOLO_DATASET_DIR
    / "labels"
    / "val"
)

DATASET_YAML = YOLO_DATASET_DIR / "dataset.yaml"


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
# CREATE REQUIRED DIRECTORIES
# ============================================================

def create_directories():

    directories = [
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
        MODEL_OUTPUT_DIR
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CLEAN PREVIOUS YOLO DATASET
# ============================================================

def clean_dataset():

    print_header("Cleaning Previous YOLO Dataset")

    directories = [
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR
    ]

    for directory in directories:

        if not directory.exists():
            continue

        for file in directory.iterdir():

            if file.is_file():
                file.unlink()

    print("Previous train/validation files removed.")


# ============================================================
# FIND LABELED IMAGES
# ============================================================

def find_labeled_images():

    print_header("Finding Labeled Images")

    if not RAW_IMAGES_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{RAW_IMAGES_DIR}"
        )

    if not YOLO_LABELS_DIR.exists():

        raise FileNotFoundError(
            f"YOLO labels directory not found:\n{YOLO_LABELS_DIR}"
        )

    labeled_images = []
    unlabeled_images = []

    for image_path in sorted(RAW_IMAGES_DIR.iterdir()):

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = (
            YOLO_LABELS_DIR
            / f"{image_path.stem}.txt"
        )

        if label_path.exists():
            labeled_images.append(
                (image_path, label_path)
            )

        else:
            unlabeled_images.append(
                image_path
            )

    print(
        f"Labeled images found   : {len(labeled_images)}"
    )

    print(
        f"Unlabeled images found : {len(unlabeled_images)}"
    )

    if len(labeled_images) == 0:

        raise RuntimeError(
            "No labeled images were found."
        )

    return labeled_images, unlabeled_images
# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def split_dataset(labeled_images):

    print_header("Creating Train / Validation Split")

    random.seed(RANDOM_SEED)

    dataset = labeled_images.copy()

    random.shuffle(dataset)

    split_index = int(
        len(dataset) * TRAIN_SPLIT
    )

    train_set = dataset[:split_index]

    val_set = dataset[split_index:]

    print(
        f"Total labeled images : {len(dataset)}"
    )

    print(
        f"Training images      : {len(train_set)}"
    )

    print(
        f"Validation images    : {len(val_set)}"
    )

    print(
        f"Train percentage     : {TRAIN_SPLIT * 100:.0f}%"
    )

    print(
        f"Validation percentage: {(1 - TRAIN_SPLIT) * 100:.0f}%"
    )

    return train_set, val_set


# ============================================================
# COPY DATASET FILES
# ============================================================

def copy_dataset_files(
    samples,
    image_destination,
    label_destination
):

    for image_path, label_path in samples:

        destination_image = (
            image_destination
            / image_path.name
        )

        destination_label = (
            label_destination
            / label_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )


# ============================================================
# PREPARE YOLO DATASET
# ============================================================

def prepare_yolo_dataset(
    train_set,
    val_set
):

    print_header("Preparing YOLO-OBB Dataset")

    copy_dataset_files(
        train_set,
        TRAIN_IMAGES_DIR,
        TRAIN_LABELS_DIR
    )

    copy_dataset_files(
        val_set,
        VAL_IMAGES_DIR,
        VAL_LABELS_DIR
    )

    print(
        f"Training images copied   : {len(train_set)}"
    )

    print(
        f"Validation images copied : {len(val_set)}"
    )

    print(
        f"Dataset location:\n{YOLO_DATASET_DIR}"
    )


# ============================================================
# READ CLASS NAMES
# ============================================================

def read_classes():

    classes_file = (
        YOLO_LABELS_DIR
        / "classes.txt"
    )

    if not classes_file.exists():

        raise FileNotFoundError(
            f"classes.txt not found:\n{classes_file}"
        )

    with open(
        classes_file,
        "r",
        encoding="utf-8"
    ) as file:

        classes = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(classes) == 0:

        raise RuntimeError(
            "classes.txt is empty."
        )

    print_header("Classes")

    for index, class_name in enumerate(classes):

        print(
            f"{index:2d} -> {class_name}"
        )

    print(
        f"\nTotal classes: {len(classes)}"
    )

    return classes


# ============================================================
# CREATE DATASET YAML
# ============================================================

def create_dataset_yaml(classes):

    print_header("Creating dataset.yaml")

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            f"path: {YOLO_DATASET_DIR.as_posix()}\n"
        )

        file.write(
            "train: images/train\n"
        )

        file.write(
            "val: images/val\n\n"
        )

        file.write(
            f"nc: {len(classes)}\n"
        )

        file.write("names:\n")

        for index, class_name in enumerate(classes):

            file.write(
                f"  {index}: {class_name}\n"
            )

    print(
        f"dataset.yaml created:\n{DATASET_YAML}"
    )
# ============================================================
# CHECK DATASET
# ============================================================

def check_dataset():

    print_header("Checking Dataset")

    train_images = list(
        TRAIN_IMAGES_DIR.iterdir()
    )

    val_images = list(
        VAL_IMAGES_DIR.iterdir()
    )

    train_labels = list(
        TRAIN_LABELS_DIR.iterdir()
    )

    val_labels = list(
        VAL_LABELS_DIR.iterdir()
    )

    print(
        f"Train images : {len(train_images)}"
    )

    print(
        f"Train labels : {len(train_labels)}"
    )

    print(
        f"Val images   : {len(val_images)}"
    )

    print(
        f"Val labels   : {len(val_labels)}"
    )

    # --------------------------------------------------------
    # Check image-label correspondence
    # --------------------------------------------------------

    train_image_stems = {
        image.stem
        for image in train_images
    }

    train_label_stems = {
        label.stem
        for label in train_labels
    }

    val_image_stems = {
        image.stem
        for image in val_images
    }

    val_label_stems = {
        label.stem
        for label in val_labels
    }

    missing_train_labels = (
        train_image_stems
        - train_label_stems
    )

    missing_val_labels = (
        val_image_stems
        - val_label_stems
    )

    if missing_train_labels:

        print(
            "\nWARNING: Missing training labels:"
        )

        for name in sorted(missing_train_labels):
            print(f"  {name}")

    if missing_val_labels:

        print(
            "\nWARNING: Missing validation labels:"
        )

        for name in sorted(missing_val_labels):
            print(f"  {name}")

    if not missing_train_labels and not missing_val_labels:

        print(
            "\nAll images have corresponding labels."
        )


# ============================================================
# CHECK CUDA
# ============================================================

def check_cuda():

    print_header("Checking GPU")

    import torch

    print(
        f"PyTorch version : {torch.__version__}"
    )

    print(
        f"CUDA available  : {torch.cuda.is_available()}"
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            "\nCUDA is not available.\n"
            "Your RTX 3050 is not currently "
            "accessible through PyTorch.\n"
            "Please verify your CUDA-enabled "
            "PyTorch installation before training."
        )

    print(
        f"GPU             : "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA version    : "
        f"{torch.version.cuda}"
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print_header("Starting YOLOv8n-OBB Training")

    print(
        f"Model      : {MODEL_NAME}"
    )

    print(
        f"Epochs     : {EPOCHS}"
    )

    print(
        f"Image size : {IMAGE_SIZE}"
    )

    print(
        f"Batch size : {BATCH_SIZE}"
    )

    print(
        f"Workers    : {WORKERS}"
    )

    print(
        f"Device     : {DEVICE}"
    )

    print()

    # --------------------------------------------------------
    # Load pretrained YOLOv8n-OBB
    # --------------------------------------------------------

    model = YOLO(MODEL_NAME)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    results = model.train(

        data=str(DATASET_YAML),

        epochs=EPOCHS,

        imgsz=IMAGE_SIZE,

        batch=BATCH_SIZE,

        workers=WORKERS,

        device=DEVICE,

        project=str(MODEL_OUTPUT_DIR),

        name="training",

        exist_ok=True,

        pretrained=True,

        verbose=True
    )

    return results
# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print_header(
        "YOLOv8n-OBB Dental Annotation Pipeline"
    )

    print(
        f"Project root:\n{PROJECT_ROOT}"
    )

    # --------------------------------------------------------
    # 1. Create directories
    # --------------------------------------------------------

    create_directories()

    # --------------------------------------------------------
    # 2. Clean previous generated dataset
    # --------------------------------------------------------

    clean_dataset()

    # --------------------------------------------------------
    # 3. Check CUDA / GPU
    # --------------------------------------------------------

    check_cuda()

    # --------------------------------------------------------
    # 4. Find labeled images
    # --------------------------------------------------------

    labeled_images, unlabeled_images = (
        find_labeled_images()
    )

    # --------------------------------------------------------
    # 5. Create train / validation split
    # --------------------------------------------------------

    train_set, val_set = split_dataset(
        labeled_images
    )

    # --------------------------------------------------------
    # 6. Prepare YOLO dataset
    # --------------------------------------------------------

    prepare_yolo_dataset(
        train_set,
        val_set
    )

    # --------------------------------------------------------
    # 7. Read classes
    # --------------------------------------------------------

    classes = read_classes()

    # --------------------------------------------------------
    # 8. Create dataset.yaml
    # --------------------------------------------------------

    create_dataset_yaml(classes)

    # --------------------------------------------------------
    # 9. Check dataset
    # --------------------------------------------------------

    check_dataset()

    # --------------------------------------------------------
    # 10. Print summary
    # --------------------------------------------------------

    print_header("Dataset Summary")

    print(
        f"Labeled images   : {len(labeled_images)}"
    )

    print(
        f"Unlabeled images : {len(unlabeled_images)}"
    )

    print(
        f"Training images  : {len(train_set)}"
    )

    print(
        f"Validation images: {len(val_set)}"
    )

    print(
        f"Classes          : {len(classes)}"
    )

    # --------------------------------------------------------
    # 11. Start training
    # --------------------------------------------------------

    train_model()

    # --------------------------------------------------------
    # 12. Finished
    # --------------------------------------------------------

    print_header("TRAINING COMPLETE")

    print(
        "Your YOLOv8n-OBB model has finished training."
    )

    print(
        f"\nModel output:\n{MODEL_OUTPUT_DIR}"
    )

    print(
        "\nLook for:"
    )

    print(
        f"{MODEL_OUTPUT_DIR}\\training\\weights\\best.pt"
    )


# ============================================================
# WINDOWS-SAFE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
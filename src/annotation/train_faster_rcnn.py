import os
import json
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "data/faster_rcnn_dataset"

TRAIN_IMAGES = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_ANNOTATIONS = os.path.join(DATASET_ROOT, "annotations", "train")

VAL_IMAGES = os.path.join(DATASET_ROOT, "images", "val")
VAL_ANNOTATIONS = os.path.join(DATASET_ROOT, "annotations", "val")

MODEL_DIR = "models/faster_rcnn"

NUM_CLASSES = 15
# 14 dental classes + 1 background

BATCH_SIZE = 1
NUM_EPOCHS = 30

LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

class DentalDataset(Dataset):

    def __init__(self, image_dir, annotation_dir):

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        self.images.sort()

        print(
            f"Loaded {len(self.images)} images from "
            f"{image_dir}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # ----------------------------------------------------
        # Find annotation JSON
        # ----------------------------------------------------

        annotation_name = os.path.splitext(
            image_name
        )[0] + ".json"

        annotation_path = os.path.join(
            self.annotation_dir,
            annotation_name
        )

        boxes = []
        labels = []

        # ----------------------------------------------------
        # Read annotations
        # ----------------------------------------------------

        if os.path.exists(annotation_path):

            with open(
                annotation_path,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            annotations = data.get(
                "annotations",
                []
            )

            for annotation in annotations:

                bbox = annotation.get("bbox")

                class_id = annotation.get(
                    "class_id"
                )

                # --------------------------------------------
                # Validate annotation
                # --------------------------------------------

                if bbox is None:
                    continue

                if class_id is None:
                    continue

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox

                # --------------------------------------------
                # Make sure coordinates are valid
                # --------------------------------------------

                x1 = max(0.0, min(float(x1), width))
                y1 = max(0.0, min(float(y1), height))
                x2 = max(0.0, min(float(x2), width))
                y2 = max(0.0, min(float(y2), height))

                # --------------------------------------------
                # Correct reversed coordinates if necessary
                # --------------------------------------------

                if x2 < x1:
                    x1, x2 = x2, x1

                if y2 < y1:
                    y1, y2 = y2, y1

                # --------------------------------------------
                # Ignore invalid / zero-area boxes
                # --------------------------------------------

                if x2 <= x1 or y2 <= y1:
                    print(
                        f"WARNING: Invalid bbox in "
                        f"{annotation_name}: {bbox}"
                    )
                    continue

                boxes.append(
                    [x1, y1, x2, y2]
                )

                # Faster R-CNN labels:
                # 0 = background
                # Dental classes = 1-14
                labels.append(
                    int(class_id) + 1
                )

        else:

            print(
                f"WARNING: Annotation file missing for "
                f"{image_name}"
            )

        # ----------------------------------------------------
        # IMPORTANT FIX
        # ----------------------------------------------------
        # Always create boxes with shape [N, 4].
        #
        # Without this, an image with zero annotations
        # creates a 1-D tensor and causes:
        #
        # IndexError: too many indices for tensor
        # of dimension 1
        # ----------------------------------------------------

        if len(boxes) == 0:

            print(
                f"WARNING: No annotations found for "
                f"{image_name}"
            )

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        else:

            boxes = torch.as_tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.as_tensor(
                labels,
                dtype=torch.int64
            )

        # ----------------------------------------------------
        # Calculate area
        # ----------------------------------------------------

        if boxes.shape[0] == 0:

            areas = torch.zeros(
                (0,),
                dtype=torch.float32
            )

        else:

            areas = (
                (boxes[:, 2] - boxes[:, 0]) *
                (boxes[:, 3] - boxes[:, 1])
            )

        # ----------------------------------------------------
        # Iscrowd
        # ----------------------------------------------------

        iscrowd = torch.zeros(
            (boxes.shape[0],),
            dtype=torch.int64
        )

        # ----------------------------------------------------
        # Target dictionary
        # ----------------------------------------------------

        target = {

            "boxes": boxes,

            "labels": labels,

            "image_id": torch.tensor(
                [idx],
                dtype=torch.int64
            ),

            "area": areas,

            "iscrowd": iscrowd
        }

        # ----------------------------------------------------
        # Convert image to tensor
        # ----------------------------------------------------

        image = F.to_tensor(image)

        return image, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(
        zip(*batch)
    )


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print(
        "\nLoading Faster R-CNN "
        "ResNet-50 FPN..."
    )

    weights = (
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    model = fasterrcnn_resnet50_fpn(
        weights=weights
    )

    # --------------------------------------------------------
    # Replace the original COCO classifier
    # --------------------------------------------------------

    in_features = (
        model.roi_heads
        .box_predictor
        .cls_score
        .in_features
    )

    model.roi_heads.box_predictor = (
        FastRCNNPredictor(
            in_features,
            NUM_CLASSES
        )
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FASTER R-CNN - DENTAL DISEASE DETECTION")
    print("=" * 70)

    print(f"\nDevice: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA version: "
            f"{torch.version.cuda}"
        )

    else:

        print(
            "WARNING: CUDA is not available. "
            "Training will use CPU."
        )

    # ========================================================
    # DATASETS
    # ========================================================

    print("\nLoading datasets...")

    train_dataset = DentalDataset(
        TRAIN_IMAGES,
        TRAIN_ANNOTATIONS
    )

    val_dataset = DentalDataset(
        VAL_IMAGES,
        VAL_ANNOTATIONS
    )

    print(
        f"\nTraining images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(val_dataset)}"
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()

    model.to(DEVICE)

    print(
        "\nModel successfully loaded."
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.SGD(
        params,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # ========================================================
    # LEARNING RATE SCHEDULER
    # ========================================================

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # ========================================================
    # CREATE MODEL DIRECTORY
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    best_loss = float("inf")

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n")
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)

    for epoch in range(NUM_EPOCHS):

        model.train()

        epoch_loss = 0.0

        print(
            f"\nEpoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        print("-" * 70)

        for batch_idx, (images, targets) in enumerate(
            train_loader
        ):

            # ------------------------------------------------
            # Move images to GPU
            # ------------------------------------------------

            images = [
                image.to(DEVICE)
                for image in images
            ]

            # ------------------------------------------------
            # Move targets to GPU
            # ------------------------------------------------

            targets = [

                {
                    key: value.to(DEVICE)
                    for key, value in target.items()
                }

                for target in targets
            ]

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            loss_dict = model(
                images,
                targets
            )

            # ------------------------------------------------
            # Total loss
            # ------------------------------------------------

            losses = sum(
                loss
                for loss in loss_dict.values()
            )

            loss_value = losses.item()

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            losses.backward()

            optimizer.step()

            # ------------------------------------------------
            # Accumulate loss
            # ------------------------------------------------

            epoch_loss += loss_value

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_idx + 1
            ) % 50 == 0:

                print(
                    f"Batch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss_value:.4f}"
                )

        # ====================================================
        # EPOCH LOSS
        # ====================================================

        average_loss = (
            epoch_loss /
            len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"completed."
        )

        print(
            f"Average Training Loss: "
            f"{average_loss:.4f}"
        )

        # ====================================================
        # LEARNING RATE
        # ====================================================

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Learning Rate: "
            f"{current_lr:.6f}"
        )

        lr_scheduler.step()

        # ====================================================
        # SAVE LAST MODEL
        # ====================================================

        last_model_path = os.path.join(
            MODEL_DIR,
            "last.pt"
        )

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "loss":
                    average_loss,

                "num_classes":
                    NUM_CLASSES
            },
            last_model_path
        )

        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if average_loss < best_loss:

            best_loss = average_loss

            best_model_path = os.path.join(
                MODEL_DIR,
                "best.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "loss":
                        average_loss,

                    "num_classes":
                        NUM_CLASSES
                },
                best_model_path
            )

            print(
                f"New best model saved!"
            )

            print(
                f"Best training loss: "
                f"{best_loss:.4f}"
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nBest Training Loss: "
        f"{best_loss:.4f}"
    )

    print(
        f"\nBest model:"
        f"\n{os.path.join(MODEL_DIR, 'best.pt')}"
    )

    print(
        f"\nLast model:"
        f"\n{os.path.join(MODEL_DIR, 'last.pt')}"
    )

    print("\n")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
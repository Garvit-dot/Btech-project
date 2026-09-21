import os
import json
import torch
import torchvision

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "data/faster_rcnn_dataset"

VAL_IMAGES = os.path.join(
    DATASET_ROOT,
    "images",
    "val"
)

VAL_ANNOTATIONS = os.path.join(
    DATASET_ROOT,
    "annotations",
    "val"
)

MODEL_PATH = "models/faster_rcnn/best.pt"

NUM_CLASSES = 15
# 14 dental classes + background

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Detection confidence threshold
CONFIDENCE_THRESHOLD = 0.50

# IoU threshold for Precision / Recall / F1
IOU_THRESHOLD = 0.50


# ============================================================
# DATASET
# ============================================================

class DentalValidationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        annotation_dir
    ):

        self.image_dir = image_dir
        self.annotation_dir = annotation_dir

        self.images = sorted([
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ])

        print(
            f"Validation images found: "
            f"{len(self.images)}"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        width, height = image.size

        annotation_name = (
            os.path.splitext(image_name)[0]
            + ".json"
        )

        annotation_path = os.path.join(
            self.annotation_dir,
            annotation_name
        )

        boxes = []
        labels = []

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

                if bbox is None:
                    continue

                if class_id is None:
                    continue

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox

                x1 = max(
                    0.0,
                    min(float(x1), width)
                )

                y1 = max(
                    0.0,
                    min(float(y1), height)
                )

                x2 = max(
                    0.0,
                    min(float(x2), width)
                )

                y2 = max(
                    0.0,
                    min(float(y2), height)
                )

                if x2 < x1:
                    x1, x2 = x2, x1

                if y2 < y1:
                    y1, y2 = y2, y1

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append(
                    [x1, y1, x2, y2]
                )

                # Dataset class IDs are 0-13.
                # Faster R-CNN uses 1-14 because
                # 0 is reserved for background.
                labels.append(
                    int(class_id) + 1
                )

        # ----------------------------------------------------
        # Empty annotation handling
        # ----------------------------------------------------

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        else:

            boxes = torch.tensor(
                boxes,
                dtype=torch.float32
            )

            labels = torch.tensor(
                labels,
                dtype=torch.int64
            )

        image_tensor = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor(
                [idx],
                dtype=torch.int64
            )
        }

        return (
            image_tensor,
            target,
            image_name
        )


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

    model = fasterrcnn_resnet50_fpn(
        weights=None
    )

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
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    print(
        "\nLoading trained Faster R-CNN..."
    )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Our training script saved a dictionary
    # containing model_state_dict.
    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        # Fallback if a raw state_dict was saved
        model.load_state_dict(
            checkpoint
        )

    model.to(DEVICE)

    model.eval()

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# IoU FUNCTION
# ============================================================

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0.0,
        x2 - x1
    )

    intersection_height = max(
        0.0,
        y2 - y1
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area1 = (
        max(
            0.0,
            box1[2] - box1[0]
        )
        *
        max(
            0.0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0.0,
            box2[2] - box2[0]
        )
        *
        max(
            0.0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def calculate_detection_metrics(
    all_predictions,
    all_ground_truths
):

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for prediction, ground_truth in zip(
        all_predictions,
        all_ground_truths
    ):

        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]

        gt_boxes = ground_truth["boxes"]
        gt_labels = ground_truth["labels"]

        # ----------------------------------------------------
        # Filter predictions by confidence
        # ----------------------------------------------------

        keep = (
            pred_scores
            >= CONFIDENCE_THRESHOLD
        )

        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        matched_gt = set()

        # ----------------------------------------------------
        # Sort predictions by confidence
        # ----------------------------------------------------

        if len(pred_scores) > 0:

            order = torch.argsort(
                pred_scores,
                descending=True
            )

            pred_boxes = pred_boxes[
                order
            ]

            pred_labels = pred_labels[
                order
            ]

        # ----------------------------------------------------
        # Match predictions
        # ----------------------------------------------------

        for pred_box, pred_label in zip(
            pred_boxes,
            pred_labels
        ):

            best_iou = 0.0
            best_gt_index = -1

            for gt_index in range(
                len(gt_boxes)
            ):

                if gt_index in matched_gt:
                    continue

                # Class must match
                if (
                    int(pred_label)
                    !=
                    int(gt_labels[gt_index])
                ):
                    continue

                iou = calculate_iou(
                    pred_box.tolist(),
                    gt_boxes[
                        gt_index
                    ].tolist()
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_index = gt_index

            if (
                best_gt_index >= 0
                and best_iou >= IOU_THRESHOLD
            ):

                true_positives += 1

                matched_gt.add(
                    best_gt_index
                )

            else:

                false_positives += 1

        # ----------------------------------------------------
        # Unmatched ground truth boxes
        # ----------------------------------------------------

        false_negatives += (
            len(gt_boxes)
            - len(matched_gt)
        )

    return (
        true_positives,
        false_positives,
        false_negatives
    )


# ============================================================
# CALCULATE PRECISION / RECALL / F1
# ============================================================

def calculate_prf(
    tp,
    fp,
    fn
):

    if (
        tp + fp
    ) > 0:

        precision = (
            tp
            /
            (tp + fp)
        )

    else:

        precision = 0.0

    if (
        tp + fn
    ) > 0:

        recall = (
            tp
            /
            (tp + fn)
        )

    else:

        recall = 0.0

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1
    )


# ============================================================
# SIMPLE AP CALCULATION
# ============================================================

def calculate_map50(
    all_predictions,
    all_ground_truths
):

    # --------------------------------------------------------
    # Collect predictions globally
    # --------------------------------------------------------

    detections = []

    total_ground_truths = 0

    for image_idx, (
        prediction,
        ground_truth
    ) in enumerate(
        zip(
            all_predictions,
            all_ground_truths
        )
    ):

        gt_boxes = ground_truth["boxes"]
        gt_labels = ground_truth["labels"]

        total_ground_truths += len(
            gt_boxes
        )

        pred_boxes = prediction["boxes"]
        pred_labels = prediction["labels"]
        pred_scores = prediction["scores"]

        for box, label, score in zip(
            pred_boxes,
            pred_labels,
            pred_scores
        ):

            if (
                score.item()
                < CONFIDENCE_THRESHOLD
            ):
                continue

            detections.append(
                {
                    "image_idx": image_idx,
                    "box": box.tolist(),
                    "label": int(
                        label.item()
                    ),
                    "score": float(
                        score.item()
                    )
                }
            )

    # --------------------------------------------------------
    # Sort by confidence
    # --------------------------------------------------------

    detections.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {}

    tp = []
    fp = []

    for detection in detections:

        image_idx = detection[
            "image_idx"
        ]

        label = detection[
            "label"
        ]

        pred_box = detection[
            "box"
        ]

        ground_truth = (
            all_ground_truths[
                image_idx
            ]
        )

        gt_boxes = ground_truth[
            "boxes"
        ]

        gt_labels = ground_truth[
            "labels"
        ]

        best_iou = 0.0
        best_gt = -1

        for gt_idx in range(
            len(gt_boxes)
        ):

            if (
                int(
                    gt_labels[gt_idx]
                )
                != label
            ):
                continue

            key = (
                image_idx,
                gt_idx
            )

            if matched.get(
                key,
                False
            ):
                continue

            iou = calculate_iou(
                pred_box,
                gt_boxes[
                    gt_idx
                ].tolist()
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_idx

        if (
            best_gt >= 0
            and best_iou >= 0.50
        ):

            matched[
                (
                    image_idx,
                    best_gt
                )
            ] = True

            tp.append(1)
            fp.append(0)

        else:

            tp.append(0)
            fp.append(1)

    if len(tp) == 0:

        return 0.0

    # --------------------------------------------------------
    # Precision-recall curve
    # --------------------------------------------------------

    tp_cumulative = []
    fp_cumulative = []

    running_tp = 0
    running_fp = 0

    for t, f in zip(
        tp,
        fp
    ):

        running_tp += t
        running_fp += f

        tp_cumulative.append(
            running_tp
        )

        fp_cumulative.append(
            running_fp
        )

    precisions = []
    recalls = []

    for tpc, fpc in zip(
        tp_cumulative,
        fp_cumulative
    ):

        precision = (
            tpc
            /
            (tpc + fpc)
        )

        recall = (
            tpc
            /
            max(
                total_ground_truths,
                1
            )
        )

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

    # --------------------------------------------------------
    # 101-point interpolated AP
    # --------------------------------------------------------

    ap = 0.0

    for recall_level in [
        i / 100
        for i in range(101)
    ]:

        precision_at_recall = 0.0

        for precision, recall in zip(
            precisions,
            recalls
        ):

            if recall >= recall_level:

                precision_at_recall = max(
                    precision_at_recall,
                    precision
                )

        ap += precision_at_recall

    ap /= 101.0

    return ap


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print(
        "FASTER R-CNN VALIDATION EVALUATION"
    )
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU: "
            + torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = DentalValidationDataset(
        VAL_IMAGES,
        VAL_ANNOTATIONS
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print("\nRunning validation...")
    print("-" * 70)

    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():

        for batch_idx, (
            images,
            targets,
            image_names
        ) in enumerate(loader):

            images = [
                image.to(DEVICE)
                for image in images
            ]

            outputs = model(
                images
            )

            for output, target in zip(
                outputs,
                targets
            ):

                prediction = {

                    "boxes":
                        output["boxes"]
                        .detach()
                        .cpu(),

                    "labels":
                        output["labels"]
                        .detach()
                        .cpu(),

                    "scores":
                        output["scores"]
                        .detach()
                        .cpu()
                }

                ground_truth = {

                    "boxes":
                        target["boxes"]
                        .detach()
                        .cpu(),

                    "labels":
                        target["labels"]
                        .detach()
                        .cpu()
                }

                all_predictions.append(
                    prediction
                )

                all_ground_truths.append(
                    ground_truth
                )

            if (
                (batch_idx + 1) % 25
                == 0
            ):

                print(
                    f"Processed "
                    f"{batch_idx + 1}/"
                    f"{len(loader)} images"
                )

    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    tp, fp, fn = (
        calculate_detection_metrics(
            all_predictions,
            all_ground_truths
        )
    )

    precision, recall, f1 = (
        calculate_prf(
            tp,
            fp,
            fn
        )
    )

    # ========================================================
    # mAP@50
    # ========================================================

    map50 = calculate_map50(
        all_predictions,
        all_ground_truths
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FASTER R-CNN EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nValidation Images : "
        f"{len(dataset)}"
    )

    print(
        f"Confidence Threshold : "
        f"{CONFIDENCE_THRESHOLD}"
    )

    print(
        f"IoU Threshold : "
        f"{IOU_THRESHOLD}"
    )

    print("\nDetection Counts")
    print("-" * 70)

    print(
        f"True Positives  : {tp}"
    )

    print(
        f"False Positives : {fp}"
    )

    print(
        f"False Negatives : {fn}"
    )

    print("\nMetrics")
    print("-" * 70)

    print(
        f"Precision : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score  : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"mAP@50    : "
        f"{map50 * 100:.2f}%"
    )

    print(
        "\nmAP@50:95 : "
        "Requires COCO-style evaluation."
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
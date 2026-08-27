# B.Tech Project — Dental X-ray Analysis

## Overview

A computer vision pipeline for detecting dental conditions in X-ray images using **YOLOv8n-OBB**.

The dataset contains **1106 X-ray images** and is being labeled using a human-in-the-loop iterative annotation approach to reduce manual labeling effort.

## Workflow

Manual Annotation → LabelMe → YOLO-OBB Conversion → Model Training → Automatic Annotation → Manual Verification → Retraining

## Project Structure

```text
Btech-project/
├── data/
│   ├── raw/images/
│   ├── processed/
│   ├── annotations/
│   │   ├── labelme_json/
│   │   ├── auto_annotations/
│   │   └── yolo_labels/
│   └── yolo_dataset/
├── models/
├── outputs/
├── src/
│   └── annotation/
│       ├── rename_labels.py
│       ├── convert_labelme_to_yolo.py
│       ├── train_detector.py
│       ├── auto_annotate.py
│       └── fix_image_paths.py
├── .gitignore
└── README.md

## Annotation Process

- Initial images were manually annotated using **LabelMe**.
- Both rectangular and oriented bounding boxes are used.
- Label names were standardized before training.
- **YOLOv8n-OBB** generates annotations for previously unlabeled images.
- Generated annotations are manually verified and corrected.
- Verified annotations are added to the training dataset.
- The model is retrained and the process is repeated.

## Model

**YOLOv8n-OBB**

- Image size: `640 × 640`
- Epochs: `50`
- Batch size: `8`
- GPU: `NVIDIA RTX 3050`

## Main Scripts

```bash
python src/annotation/rename_labels.py
python src/annotation/convert_labelme_to_yolo.py
python src/annotation/train_detector.py
python src/annotation/auto_annotate.py
python src/annotation/fix_image_paths.py
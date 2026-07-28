# 🦷 Multimodal Deep Learning Framework for Automated Detection and Localization of Dental Diseases

## 📌 Project Overview

This project aims to develop a **multimodal deep learning framework** for the automated **detection and localization of dental diseases** from dental radiographs (X-ray images). The system is designed to assist clinicians by providing accurate and efficient disease identification while reducing manual diagnostic effort.

The project follows a modular machine learning pipeline consisting of:

- Dataset preprocessing
- Exploratory Data Analysis (EDA)
- Data standardization
- Model development
- Model evaluation
- Multimodal fusion
- Deployment

---

## 📂 Project Structure

```text
Btech-project/
│
├── data/
│   ├── raw/
│   │   └── images/
│   │
│   ├── processed/
│   │   └── images/
│   │
│   └── metadata/
│       └── dataset_summary.csv
│
├── outputs/
│   ├── sample_images.png
│   ├── pixel_histogram.png
│   └── preprocessing_report.txt
│
├── src/
│   └── preprocessing/
│       ├── inspect_dataset.py
│       ├── visualize_dataset.py
│       ├── analyze_pixels.py
│       └── preprocess_images.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The current dataset consists of:

- **1106 dental X-ray images**
- **1023 BMP images**
- **83 JPG images**
- Resolution: **795 × 1100 pixels**
- No corrupted images detected

---

## ✅ Preprocessing Pipeline

The preprocessing stage includes:

- Dataset inspection
- Metadata generation
- Random image visualization
- Pixel intensity analysis
- Histogram generation
- RGB to grayscale conversion
- Processed dataset generation

---

## 🛠 Technologies Used

- Python 3.12
- NumPy
- Pillow
- OpenCV
- Matplotlib
- Pandas
- Scikit-learn

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone <repository-url>
cd Btech-project
```

### Create virtual environment

```bash
python -m venv .venv
```

### Activate environment

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📅 Progress

### ✅ Completed

- Project setup
- Environment configuration
- Dataset inspection
- Metadata generation
- Dataset visualization
- Pixel intensity analysis
- Image standardization

### 🔄 In Progress

- Annotation analysis
- Multimodal data integration

### 📌 Upcoming

- Train/Validation/Test split
- DataLoader implementation
- Model development
- Disease localization
- Model evaluation
- Explainability
- Deployment

---

## 👨‍💻 Authors

**Garvit Kandpal**

B.Tech Artificial Intelligence & Machine Learning

Symbiosis Institute of Technology, Pune

---

## 📜 License

This project is developed for academic and research purposes.
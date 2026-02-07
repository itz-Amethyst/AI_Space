---
license: apache-2.0
language:
  - en
tags:
  - computer-vision
  - classical-machine-learning
  - object-classification
  - robotics
  - feature-engineering
  - opencv
  - scikit-learn
datasets:
  - coil-100
---

# COIL-100 Object Classification & Sorting Model (Classical ML Pipeline)

## Model Description

This repository contains a complete perception pipeline for **object classification** trained on the **COIL-100** dataset, designed to simulate industrial robotic sorting / pick-and-place tasks using **hand-crafted vision features** and classical machine learning classifiers.

The pipeline demonstrates that — in controlled environments with limited object categories, stable lighting, and single-object-per-image scenarios — carefully engineered features combined with interpretable models can achieve very high accuracy, often rivaling lightweight deep learning approaches while being much more explainable and computationally lightweight.

### Key Techniques

- **Feature Engineering** (OpenCV + scikit-image):
  - Color: HSV histogram (4×4×4 bins → 64 dims)
  - Shape: Hu moments (7 values) + normalized bounding box size & position (4 values)
  - Texture: Local Binary Patterns – uniform (59 bins)
  - Texture statistics: GLCM contrast, homogeneity, energy
  - Gradient structure: Histogram of Oriented Gradients (HOG) with 16×16 pixels/cell

- **Dimensionality Reduction** (hybrid approach):
  - Supervised feature selection: `SelectKBest` (ANOVA F-value) → top 500 features
  - Unsupervised compression: PCA preserving 95% variance on the selected features
  → Final feature vector: typically 150–350 dimensions

- **Classifiers** compared via RandomizedSearchCV + cross-validation:
  - Support Vector Machine (RBF & linear kernels)
  - Random Forest
  - k-Nearest Neighbors

- **Best performing model** (as of training): usually **SVM (RBF)** or **Random Forest** with test set accuracy in the range **96–99%**

- **Evaluation**:
  - Stratified train/test split (80/20)
  - 5-fold RandomizedSearchCV for hyperparameter tuning
  - 10-fold cross-validation on final model
  - Learning curve analysis (training vs validation accuracy vs dataset size)

## Dataset

- **Name**: COIL-100 (Columbia Object Image Library)
- **Source**: [Official Columbia University page](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php)
- **Size**: 7,200 images
- **Classes**: 100 distinct everyday objects
- **Images per class**: 72 (one every 5° rotation: 0°–355°)
- **Resolution**: 128 × 128 pixels
- **Properties**: Single centered object, black homogeneous background, controlled lighting → ideal for classical feature-based methods
- **License**: Commonly used for academic/research purposes (non-commercial)

## Performance

- **Final test accuracy** (best model): **~98.0–99.2%** (exact value depends on random seed and final hyperparameter draw)
- **Cross-validation mean accuracy** (10-fold on training set): **~97.5–98.8%**
- **Inference speed**: very fast — typically <10 ms per image on CPU (feature extraction dominates)

### Learning Curve

The learning curve shows excellent generalization behavior:

- Training accuracy reaches ~100% very quickly (SVM perfectly fits even small subsets due to high separability)
- Validation accuracy rapidly improves with more data and converges close to training accuracy
- Very small gap between train and validation → **minimal overfitting**

![Learning Curve](visualizations/learning_curve.png)
<!-- After uploading the image to the repo, update the path above -->

## Feature Extraction Visualization

This pipeline includes detailed visualization of every extracted feature type to help understand what information each descriptor captures.

Example visualization outputs:
![Complete](visualizations/complete.png)


## How to Use the Model

### Prerequisites
```bash
pip install opencv-python scikit-learn scikit-image joblib huggingface_hub numpy
```
## Usage
```python
from huggingface_hub import hf_hub_download
import joblib
import cv2
import numpy as np
# ... import your extract_features function ...

repo_id = "your-username/coil100-vision-sorting-classical-ml"

# Load pipeline components
model     = joblib.load(hf_hub_download(repo_id, "model.joblib"))
scaler    = joblib.load(hf_hub_download(repo_id, "scaler.joblib"))
selector  = joblib.load(hf_hub_download(repo_id, "selector.joblib"))
pca       = joblib.load(hf_hub_download(repo_id, "pca.joblib"))
le        = joblib.load(hf_hub_download(repo_id, "label_encoder.joblib"))

# Example inference on a new image
img = cv2.imread("path/to/new_object.png")
features = extract_features(img)                      # your feature function
features_scaled = scaler.transform([features])
features_selected = selector.transform(features_scaled)
features_reduced = pca.transform(features_selected)

pred_idx = model.predict(features_reduced)[0]
predicted_class = le.inverse_transform([pred_idx])[0]

print(f"Predicted object class: {predicted_class}")
```
## Citation
```bib
@article{nene1996columbia,
  title={Columbia object image library (coil-100)},
  author={Nene, Sameer A and Nayar, Shree K and Murase, Hiroshi},
  journal={Technical Report CUCS-005-96},
  year={1996},
  institution={Columbia University}
}```

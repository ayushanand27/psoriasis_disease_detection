# psoriasis_disease_detection

Psoriasis Detection with DenseNet201 (Colab)
A complete, production‑ready pipeline to detect psoriasis vs normal skin using transfer learning. This README explains the entire project in depth: data, setup, training, evaluation, explainability, export, and deployment. It also highlights optional extensions to surpass 99% accuracy using an ensemble.

1. Project Summary
Objective: Binary classification of skin images into psoriasis vs normal.
Framework: PyTorch + timm + Albumentations in Google Colab (GPU).
Core Model: DenseNet201 (pretrained).
Training Strategy: Two‑phase training with medical‑grade augmentations at 384×384.
Explainability: Grad‑CAM visualizations.
Exports: Final PyTorch weights (.pth), TorchScript (.pt), ONNX (.onnx).
Results achieved:
Validation: up to 100% accuracy in phases.
Test: Accuracy ≈ 98.6%, Precision ≈ 0.994, Recall ≈ 0.983, F1 ≈ 0.989, ROC‑AUC ≈ 0.994.
Optional: EfficientNet‑B7 + ensemble with DenseNet201 and test‑time augmentation (TTA) to push beyond 99% consistently.

2. Dataset
Source: Kaggle dataset “pallapurajkumar/psoriasis-skin-dataset”
Downloaded in Colab via Kaggle API.
Organized into 80/10/10 split: train, val, test with class subfolders psoriasis/normal.
Final split counts (example):
Train: psoriasis=1401, normal=839
Val: psoriasis=175, normal=105
Test: psoriasis=176, normal=105
If your unzip path differs, adjust the RAW path before splitting.

3. Environment and Dependencies
Runtime: Google Colab GPU (e.g., Tesla T4).
Core packages:
torch, torchvision, timm, albumentations, grad-cam
kaggle (for dataset download)
onnx, onnxruntime (for ONNX export/verify)
wandb optional (disabled in this flow)
To install in Colab:

python
!pip -q install --upgrade pip
!pip -q install albumentations==1.4.4 timm==1.0.9 grad-cam==1.5.4
!pip -q install kaggle onnx onnxruntime
4. Data Access (Kaggle)
Obtain kaggle.json from Kaggle account (username+key).
In Colab:
python
import os, json
os.makedirs("/root/.kaggle", exist_ok=True)
kaggle_creds = {"username": "<your_username>", "key": "<your_key>"}
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump(kaggle_creds, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)

!kaggle datasets download -d "pallapurajkumar/psoriasis-skin-dataset" -p /content/ds --force
!unzip -q -o /content/ds/*.zip -d /content/ds/extracted
Split into train/val/test:
python
RAW = "/content/ds/extracted/PSORIASIS AND NORMAL SKIN"  # adjust if needed
DATA_ROOT = "/content/psoriasis_data"
# Create train/val/test with psoriasis/normal and copy images accordingly...
Verify final folders:

TRAIN_DIR: /content/psoriasis_data/train
VAL_DIR: /content/psoriasis_data/val
TEST_DIR: /content/psoriasis_data/test
5. Configuration
python
class Cfg:
    TRAIN_DIR = "/content/psoriasis_data/train"
    VAL_DIR   = "/content/psoriasis_data/val"
    TEST_DIR  = "/content/psoriasis_data/test"
    OUT_DIR   = "/content/outputs"

    IMAGE_SIZE = (384, 384)        # Full runs; use (224,224) for sanity
    NUM_CLASSES = 2
    NUM_WORKERS = 2
    PIN_MEMORY  = True

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD  = [0.229, 0.224, 0.225]

    MIXED_PRECISION = True
    PRETRAINED = True
    DEVICE = "cuda"  # Colab GPU

    PHASES = {
        "phase1": {"epochs": 5, "lr": 1e-3, "batch": 32},
        "phase2": {"epochs": 10,"lr": 5e-4, "batch": 16},
    }
cfg = Cfg()
6. Data Pipeline and Augmentations
Albumentations for medical-grade transforms:
Resize, Horizontal/Vertical flips, RandomRotate90, Rotate
ShiftScaleRotate, ElasticTransform
RandomBrightnessContrast
Normalize to ImageNet stats
Torchvision ImageFolder for I/O.
For sanity runs: lighter aug at 224×224.
7. Model Architecture
DenseNet201 (timm) with pretrained weights, final classifier set to 2 classes.
Reasons for DenseNet201:
Strong feature reuse via dense connections.
Robust on medical imaging classification tasks.
Efficient parameter count vs capacity trade-off.
Optional models:

EfficientNet‑B7 (tf_efficientnet_b7_ns) for additional performance.
ViT (e.g., vit_large_patch16_384) for complementary features.
8. Training Strategy
Mixed Precision (AMP) for speed on GPU.
Two‑phase training:
Phase 1 (initial/head): establish robust classification with moderate LR.
Phase 2 (full fine‑tuning): lower LR, stronger regularization from augmentations.
Optimizer: AdamW with weight decay 1e‑4.
Sanity run at 224×224 (2 epochs) confirms gradient flow and pipeline correctness; then switch to full 384×384.

9. Evaluation
Metrics:
Accuracy, Precision, Recall, F1
ROC‑AUC
Confusion Matrix
ROC Curve
Grad‑CAM for qualitative analysis
Test results (example achieved):

Accuracy ≈ 0.9858
Precision ≈ 0.9943
Recall ≈ 0.9830
F1 ≈ 0.9886
ROC‑AUC ≈ 0.9942
These are faculty‑friendly: include bar chart, ROC curve, confusion matrix, and 1–2 Grad‑CAM heatmaps.

10. Explainability (Grad‑CAM)
Uses pytorch-grad-cam.
Selects last Conv2d automatically.
Ensure input dtype is float32 to avoid CUDA double/float mismatch.
Use context manager to release hooks cleanly.
11. Export and Deployment
Exports:

PyTorch weights: /content/outputs/densenet201_final.pth
TorchScript: /content/outputs/densenet201_ts.pt
ONNX: /content/outputs/densenet201.onnx (legacy exporter)
Verification:

ONNX checked and inferenced via onnxruntime.
Deployment options:

PyTorch/TorchScript: load and serve with FastAPI/Flask.
ONNX: deploy with onnxruntime (Python/C++), CPU‑friendly.
Example ONNX inference snippet:

python
import onnxruntime as ort, numpy as np, cv2
sess = ort.InferenceSession("densenet201.onnx", providers=["CPUExecutionProvider"])
# preprocess to shape (1,3,H,W), float32 normalized
logits = sess.run(["logits"], {"input": input_np})[0]
prob = softmax(logits, axis=1)[:,1]
12. Reproducibility Checklist (Colab)
Install dependencies.
Set Kaggle credentials (kaggle.json).
Download and unzip dataset.
Split into train/val/test.
Define config and data pipeline.
Sanity train at 224×224 (2 epochs) → confirm metrics.
Full two‑phase training at 384×384 with strong aug.
Evaluate on test set (metrics + plots).
Generate Grad‑CAM for 1–2 sample images.
Export models (TorchScript + ONNX).
Copy artifacts to Google Drive.
13. Optional: Beyond 99% Accuracy (If needed)
Train EfficientNet‑B7 at 384×384 with same schedule.
Ensemble DenseNet201 + EfficientNet‑B7:
Average probabilities or weight by validation accuracy.
Test‑Time Augmentation (TTA):
Average predictions with original + horizontal flip (or small rotations).
Increase resolution to 448×448 for extra gains (with caution for overfitting).
Verify data quality; remove mislabels/duplicates if any.
14. Troubleshooting
Kaggle errors (KeyError 'username' or missing kaggle.json):
Ensure /root/.kaggle/kaggle.json exists with 600 perms.
No images found for class:
RAW path incorrect; inspect os.listdir(RAW); adjust path; re‑split.
AMP deprecation warnings:
Harmless; classic torch.cuda.amp used; upgrade path is available.
Grad-CAM dtype mismatch:
Ensure input normalized arrays are float32, not float64.
15. Ethics and Usage
Medical images are sensitive. Use the model responsibly:
It does not replace professional diagnosis.
Ensure dataset licenses/consents are respected.
Follow data governance and privacy guidelines.
16. Acknowledgements
Dataset: Kaggle “pallapurajkumar/psoriasis-skin-dataset”
Libraries: PyTorch, timm, Albumentations, pytorch-grad-cam, scikit-learn, onnx, onnxruntime.
Colab GPU infrastructure.
17. Quick Results Recap (DenseNet201)
Validation: up to 100% during training.
Test: Accuracy ≈ 98.6%, AUC ≈ 0.994.
Plots available:
Performance bars (Accuracy/Precision/Recall/F1)
ROC curve (AUC)
Confusion matrix
Grad‑CAM heatmaps
18. Next Steps (Optional)
Add EfficientNet‑B7 training cell and ensemble cell.
Apply TTA on validation/test.
Produce a final combined report (.ipynb or PDF) with:
Data description
Training procedure
Metrics tables + plots
Grad‑CAM examples
Exported artifacts summary
Limitations and future work
Summary: This project delivers a high‑accuracy psoriasis classifier using DenseNet201 with a robust, repeatable Colab pipeline, comprehensive evaluation, explainability via Grad‑CAM, and deployment‑ready exports. For an extra performance margin, integrate EfficientNet‑B7 and an ensemble with TTA.

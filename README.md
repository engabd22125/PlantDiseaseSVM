# PlantDiseaseSVM — Project overview and 

This repository implements a classical image-processing pipeline for plant disease classification using handcrafted texture features (Local Binary Patterns and Wavelet sub-band statistics) and Support Vector Machines (SVM). The README starts by describing the preprocessing stage in detail (as requested) and then covers feature extraction, training, inference, repository layout, and operational notes.

Table of contents
- Preprocessing (detailed)
- Feature extraction (LBP and Wavelet)
- Training and evaluation
- Inference / GUI
- File & folder layout
- How to run (quick commands)
- Small implementation contract and edge cases
- Notes: Git LFS, dataset size, next steps

Preprocessing (detailed)
------------------------

Purpose: prepare raw leaf images so the feature extractors (LBP, wavelet) produce consistent and robust vectors. Preprocessing reduces noise, compensates for lighting differences, and ensures fixed-size inputs.

Common preprocessing functions provided (or expected) in `preprocessor.py`:

- load_image(path: str) -> np.ndarray
  - Load an image from disk (OpenCV/PIL). Handles common formats (jpg, png, bmp). Returns an RGB or grayscale array.

- to_grayscale(img: np.ndarray) -> np.ndarray
  - Convert RGB to grayscale when needed by downstream extractors.

- resize_and_crop(img: np.ndarray, size: Tuple[int,int] = (256,256)) -> np.ndarray
  - Resize the image while preserving aspect ratio and center-crop or pad to a fixed target size (recommended 256×256). Consistent shapes make feature vectors fixed-length.

- denoise(img: np.ndarray, method: str = 'bilateral') -> np.ndarray
  - Apply denoising (bilateral, non-local means) to reduce sensor noise but preserve edges.

- normalize(img: np.ndarray) -> np.ndarray
  - Normalize intensities to a common scale (0–1 or zero mean/unit variance) appropriate for scalers used before model training.

- histogram_equalize(img: np.ndarray) -> np.ndarray
  - Optional contrast-limited adaptive histogram equalization (CLAHE) to reduce lighting variations.

- augment_image(img: np.ndarray, ops: List[str]) -> List[np.ndarray]
  - Generate augmented variants (rotations, flips, brightness/contrast adjustments) for training. Not required for model inference.

Batch utilities
- process_folder(src_dir: str, dst_dir: Optional[str], size=(256,256), save_preprocessed=False)
  - Walks the dataset folder structure, applies the preprocessing pipeline to each image, and optionally saves preprocessed images for debugging or reuse.

Why these steps?
- Consistent image size and dynamic range reduce variance in extracted features.
- Denoising and contrast normalization help both local texture descriptors (LBP) and frequency/sub-band descriptors (wavelet) to be more discriminative.

Feature extraction
------------------

1) Local Binary Pattern (LBP)
- Function (example): compute_lbp(img_gray, P=8, R=1, method='uniform') -> np.ndarray
- Returns a normalized histogram of LBP codes (for uniform LBP with P=8, there are usually 59 bins). The normalized histogram is the feature vector for the image.
- Output CSV: `lbp_features.csv` with rows = samples and columns = histogram bins + label.

2) Wavelet features (DWT)
- Function (example): extract_wavelet_features(img_gray, wavelet='bior1.3', level=1) -> np.ndarray
- Decompose image into sub-bands (LL, LH, HL, HH). For each sub-band compute statistics: mean, std, energy (sum squares), entropy (or log-energy). Concatenate to fixed-length vector.
- Output CSV: `wavelet_features.csv`.

Training and evaluation
-----------------------

Main scripts
- `without_wavlent.py` — LBP pipeline:
  - Walks `org_dataset/`, computes LBP histograms per image, builds a feature table (CSV), splits data, scales features with sklearn.preprocessing.StandardScaler, trains an SVM (sklearn.svm.SVC), evaluates metrics, and saves model & scaler (e.g., `models/svm_lbp.pkl`, `models/scaler_ww.pkl`).

- `train_W_svm.py` — Wavelet pipeline:
  - Computes wavelet features, saves to CSV, applies scaling and trains an SVM (RBF kernel by default). Saves outputs to `models/svm_wavelet_bior13.pkl` and `models/scaler.pkl`.

Implementation notes (assumptions)
- Scaling: StandardScaler is applied before training and the same scaler is required at inference.
- Model persistence: joblib.dump/load is used for model and scaler files.
- Classifier: SVC (RBF) with probability=True recommended if you want probability outputs; otherwise decision_function can be used for confidence.

Evaluation
- Scripts perform a train/test split (example 80/20), report accuracy, classification report, and confusion matrix. Optionally cross-validation is used for hyperparameter tuning.

Inference / GUI
-----------------

- `app.py` is a small desktop GUI that loads a saved model and scaler, runs preprocessing + feature extraction on a selected image, and outputs predicted label and confidence. Make sure the model and scaler paths in `app.py` point to the correct files in `models/`.

Files and folder layout
----------------------

- `app.py` — inference GUI
- `preprocessor.py` — preprocessing helpers
- `without_wavlent.py` — LBP feature extraction + training
- `train_W_svm.py` — Wavelet feature extraction + training
- `lbp_features.csv`, `wavelet_features.csv` — feature datasets (generated)
- `models/` — saved models and scalers (tracked with Git LFS)
- `org_dataset/` — dataset folder: `healthy/` and `diseased/` (tracked with Git LFS)

How to run (quick commands)
---------------------------

Set up environment (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pandas opencv-python scikit-image scikit-learn joblib tqdm pillow pywt customtkinter
```

Preprocess the dataset (example):

```powershell
python -c "from preprocessor import process_folder; process_folder('org_dataset','org_dataset_preprocessed',size=(256,256),save_preprocessed=True)"
```

Train LBP model:

```powershell
python without_wavlent.py
```

Train Wavelet model:

```powershell
python train_W_svm.py
```

Run the GUI (after models available):

```powershell
python app.py
```

Implementation contract and edge cases
-------------------------------------

Contract (short):
- Inputs: image file(s) in common formats, or a dataset folder with subfolders per class.
- Outputs: CSV feature files, trained model pickle(s) and scaler(s), predictions with label + confidence.
- Failure modes: missing or corrupt files (skip + log), mismatched scaler/model versions (raise error). Use the scaler saved with the model.

Edge cases:
- Empty/corrupt images: skip and report a warning in logs.
- Very small images: pad or resize to avoid degenerate feature vectors.
- Class imbalance: consider stratified splits or class_weight in SVM.


import os
import cv2
import pywt
import numpy as np
import pandas as pd
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from tqdm import tqdm
import time  

# ---------------- Settings ----------------
DATASET = r"A:\MY project\v3\org_dataset"
MODEL_DIR = r"A:\MY project\v3\models"
CSV_PATH = r"A:\MY project\v3\wavelet_features.csv"

IMG_SIZE = 128
WAVELET = 'bior1.3'
LEVEL = 1
CATEGORIES = ["healthy", "diseased"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Feature Extraction ----------------
def extract_wavelet_features(img_path, size=IMG_SIZE):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0  # normalization

    # 2D DWT
    coeffs = pywt.wavedec2(img, WAVELET, level=LEVEL)
    LL, (LH, HL, HH) = coeffs

    bands = [LL, LH, HL, HH]
    features = []

    for b in bands:
        b = np.nan_to_num(b)
        mean = np.mean(b)
        std = np.std(b)
        energy = np.sum(b**2) / b.size
        entropy = shannon_entropy(np.abs(b))
        features.extend([mean, std, energy, entropy])

    return features

# ---------------- Load Images ----------------
data = []
labels = []

for label, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET, cat)
    if not os.path.exists(folder):
        continue

    for img_name in tqdm(os.listdir(folder), desc=f"Processing {cat}"):
        img_path = os.path.join(folder, img_name)
        feat = extract_wavelet_features(img_path)
        if feat is not None:
            data.append(feat)
            labels.append(label)

# ---------------- Save CSV ----------------
bands = ['LL','LH','HL','HH']
metrics = ['mean','std','energy','entropy']
columns = [f"{b}_{m}" for b in bands for m in metrics]

df = pd.DataFrame(data, columns=columns)
df['label'] = labels
df.to_csv(CSV_PATH, index=False)
print(f"\nFeatures saved => {CSV_PATH}")

# ---------------- Train/Test Split ----------------
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------- Scaling ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Train SVM ----------------
print("\nTraining SVM...")
model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
start_time = time.time()  # <-- بداية القياس
y_pred = model.predict(X_test)
end_time = time.time()    # <-- نهاية القياس
time_per_image = (end_time - start_time) / len(X_test)  # حساب الوقت لكل صورة

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n====== RESULTS ======")
print(f"Accuracy      : {acc:.4f}")
print(f"Precision     : {prec:.4f}")
print(f"Recall        : {rec:.4f}")
print(f"F1-Score      : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print(f"\nTime per image (s): {time_per_image:.6f}")
# ---------------- Save Results ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "svm_wavelet_bior13.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

print(f"\nModel saved → {MODEL_DIR}")
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
CSV_PATH = r"A:\MY project\v3\wavelet_level2_features.csv"

IMG_SIZE = 128
WAVELET = 'bior1.3'
LEVEL = 2
CATEGORIES = ["healthy", "diseased"]

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Feature Extraction ----------------
def extract_wavelet_features(img_path, size=IMG_SIZE):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (size, size))
    img = img.astype(np.float32) / 255.0

    coeffs = pywt.wavedec2(img, WAVELET, level=LEVEL)

    # coeffs structure:
    # [LL2, (LH2,HL2,HH2), (LH1,HL1,HH1)]

    features = []

    for level in coeffs:

        if isinstance(level, tuple):

            for band in level:
                band = np.nan_to_num(band)

                mean = np.mean(band)
                std = np.std(band)
                energy = np.sum(band ** 2) / band.size
                entropy = shannon_entropy(np.abs(band))

                features.extend([mean, std, energy, entropy])

        else:

            band = np.nan_to_num(level)

            mean = np.mean(band)
            std = np.std(band)
            energy = np.sum(band ** 2) / band.size
            entropy = shannon_entropy(np.abs(band))

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
num_features = len(data[0])
columns = [f'feature_{i}' for i in range(num_features)]

df = pd.DataFrame(data, columns=columns)
df['label'] = labels

df.to_csv(CSV_PATH, index=False)

print(f"\nFeatures saved => {CSV_PATH}")

# ---------------- Train/Test Split ----------------
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------------- Scaling ----------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- Train SVM ----------------
print("\nTraining SVM...")

model = SVC(
    kernel='rbf',
    C=10,
    gamma='scale',
    probability=True
)

model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
start_time = time.time()

y_pred = model.predict(X_test)

end_time = time.time()

time_per_image = (end_time - start_time) / len(X_test)

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

# ---------------- Save Model ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "svm_wavelet_level2.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_wavelet.pkl"))

print(f"\nModel saved -> {MODEL_DIR}")
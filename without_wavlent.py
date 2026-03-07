import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from tqdm import tqdm
import time  # لحساب الوقت لكل صورة

# ---------------- Settings ----------------
DATASET = r"A:\MY project\v3\org_dataset"
MODEL_DIR = r"A:\MY project\v3\models"
CSV_PATH = r"A:\MY project\v3\lbp_features.csv"

IMG_SIZE = 128
CATEGORIES = ["healthy", "diseased"]

# LBP parameters
RADIUS = 1
N_POINTS = 8 * RADIUS

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Feature Extraction ----------------
def extract_lbp_features(img_path, size=IMG_SIZE):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (size, size))
    lbp = local_binary_pattern(img, N_POINTS, RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3), range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # normalize
    return hist.tolist()

# ---------------- Load Images ----------------
data = []
labels = []

for label, cat in enumerate(CATEGORIES):
    folder = os.path.join(DATASET, cat)
    if not os.path.exists(folder):
        continue
    for img_name in tqdm(os.listdir(folder), desc=f"Processing {cat}"):
        img_path = os.path.join(folder, img_name)
        feat = extract_lbp_features(img_path)
        if feat is not None:
            data.append(feat)
            labels.append(label)

# ---------------- Save CSV ----------------
columns = [f'LBP_{i}' for i in range(len(data[0]))]
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
start_time = time.time()  # بداية القياس
y_pred = model.predict(X_test)
end_time = time.time()    # نهاية القياس

time_per_image = (end_time - start_time) / len(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n====== RESULTS ======")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1-Score       : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print(f"\nTime per image (s): {time_per_image:.6f}")

# ---------------- Save Model ----------------
joblib.dump(model, os.path.join(MODEL_DIR, "svm_lbp.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_ww.pkl"))
print(f"\nModel saved -> {MODEL_DIR}")
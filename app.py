import os
import cv2
import pywt
import numpy as np
import joblib
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
from skimage.measure import shannon_entropy

# -------- PATHS --------
MODEL_PATH = r"A:\MY project\v3\models\svm_wavelet_bior13.pkl"
SCALER_PATH = r"A:\MY project\v3\models\scaler.pkl"

IMG_SIZE = 128
WAVELET = 'bior1.3'

# -------- Feature Extraction --------
def extract_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    LL, (LH, HL, HH) = pywt.dwt2(img, WAVELET)
    bands = [LL, LH, HL, HH]

    features = []
    for b in bands:
        b = np.nan_to_num(b)
        features.append(np.mean(b))
        features.append(np.std(b))
        features.append(np.sum(b**2) / b.size)
        features.append(shannon_entropy(np.abs(b)))

    return np.array(features).reshape(1, -1)

# -------- GUI --------
class PlantApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Plant Health AI System")
        self.geometry("600x750")
        ctk.set_appearance_mode("dark")

        # تحميل النموذج
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", "Model not found")
            self.destroy()
            return

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        self.setup_gui()

    def setup_gui(self):

        ctk.CTkLabel(
            self,
            text="Plant Health Diagnostic",
            font=("Arial", 28, "bold"),
            text_color="#81C784"
        ).pack(pady=30)

        self.btn = ctk.CTkButton(
            self,
            text="Select Plant Image",
            command=self.predict,
            height=45,
            width=200,
            font=("Arial", 16, "bold")
        )
        self.btn.pack(pady=10)

        self.canvas = ctk.CTkLabel(
            self,
            text="No image selected",
            width=420,
            height=320,
            fg_color="#1E1E1E",
            corner_radius=15
        )
        self.canvas.pack(pady=20)

        self.card = ctk.CTkFrame(self, fg_color="#2B2B2B", corner_radius=15)
        self.card.pack(fill="x", padx=60, pady=20)

        self.res_txt = ctk.CTkLabel(
            self.card,
            text="Status: Ready",
            font=("Arial", 22, "bold")
        )
        self.res_txt.pack(pady=15)

        self.conf_txt = ctk.CTkLabel(
            self.card,
            text="Confidence: 0%",
            font=("Arial", 14)
        )
        self.conf_txt.pack(pady=(0, 15))

    # -------- Prediction --------
    def predict(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )
        if not path:
            return

        try:
            # عرض الصورة
            img = Image.open(path)
            self.tk_img = ctk.CTkImage(img, size=(400, 300))
            self.canvas.configure(image=self.tk_img, text="")

            features = extract_features(path)
            if features is None:
                messagebox.showwarning("Warning", "Cannot process image")
                return

            features = self.scaler.transform(features)

            pred = self.model.predict(features)[0]
            prob = self.model.predict_proba(features).max()

            if pred == 0:
                result = "HEALTHY"
                color = "#66BB6A"
            else:
                result = "DISEASED"
                color = "#EF5350"

            self.res_txt.configure(
                text=f"Result: {result}",
                text_color=color
            )

            self.conf_txt.configure(
                text=f"Confidence Score: {prob*100:.2f}%"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    app = PlantApp()
    app.mainloop()
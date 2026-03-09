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
MODEL_PATH = r"A:\MY project\v3\models\svm_wavelet_level2.pkl"
SCALER_PATH = r"A:\MY project\v3\models\scaler_wavelet.pkl"

IMG_SIZE = 128
WAVELET = "bior1.3"
LEVEL = 2


# -------- Feature Extraction (Wavelet Level-2) --------
def extract_features(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    coeffs = pywt.wavedec2(img, WAVELET, level=LEVEL)

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

    return np.array(features).reshape(1, -1)


# -------- GUI --------
class PlantAI(ctk.CTk):

    def __init__(self):
        super().__init__()

        # DARK MODE
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.title("Plant Health AI")
        self.geometry("620x720")
        self.configure(fg_color="#0D1117")

        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("Error", "Model not found")
            self.destroy()
            return

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        self.setup_ui()

    def setup_ui(self):

        ctk.CTkLabel(
            self,
            text="Plant Health Diagnostic",
            font=("Segoe UI", 28, "bold"),
            text_color="#58A6FF"
        ).pack(pady=25)

        self.btn = ctk.CTkButton(
            self,
            text="Select Leaf Image",
            command=self.predict,
            height=45,
            width=220,
            corner_radius=12,
            fg_color="#238636",
            hover_color="#2EA043",
            font=("Segoe UI", 16, "bold")
        )
        self.btn.pack(pady=10)

        self.canvas = ctk.CTkLabel(
            self,
            text="No image selected",
            width=420,
            height=320,
            fg_color="#161B22",
            corner_radius=15,
            text_color="#8B949E"
        )
        self.canvas.pack(pady=20)

        self.card = ctk.CTkFrame(
            self,
            fg_color="#161B22",
            corner_radius=15,
            border_width=1,
            border_color="#30363D"
        )
        self.card.pack(fill="x", padx=60, pady=15)

        self.result_label = ctk.CTkLabel(
            self.card,
            text="Status: Ready",
            font=("Segoe UI", 22, "bold"),
            text_color="#C9D1D9"
        )
        self.result_label.pack(pady=15)

        self.conf_label = ctk.CTkLabel(
            self.card,
            text="Confidence: 0%",
            font=("Segoe UI", 14),
            text_color="#8B949E"
        )
        self.conf_label.pack(pady=(0, 15))

    # -------- Prediction --------
    def predict(self):

        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.png *.jpeg")]
        )

        if not path:
            return

        try:

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
                color = "#2EA043"
            else:
                result = "DISEASED"
                color = "#F85149"

            self.result_label.configure(
                text=f"Result: {result}",
                text_color=color
            )

            self.conf_label.configure(
                text=f"Confidence: {prob*100:.2f}%"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))


# -------- Run App --------
if __name__ == "__main__":
    app = PlantAI()
    app.mainloop()
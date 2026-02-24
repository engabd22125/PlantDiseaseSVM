# PlantDiseaseSVM (v3)

This repository contains code and data used for plant disease classification using SVMs and feature extraction.

Contents (high level):
- `app.py` — main application / experiment runner
- `preprocessor.py` — preprocessing utilities
- `train_W_svm.py` — training script
- `without_wavlent.py` — LBP feature extraction, SVM training and model saving (no wavelet)
- `dataset/` — image dataset (not included in git)
- `models/` — trained model files (not included in git)
- CSV feature files (excluded from commit by default)

Notes:
- Large data and model files are excluded from the repository via `.gitignore`.
- To push dataset or models, add them to a separate release or storage (Git LFS, cloud storage) and link from here.

How to use
1. Create and activate a Python environment.
2. Install dependencies (see scripts or requirements if provided).
3. Run preprocessing and training scripts as needed.

If you want me to include a `requirements.txt` or CI workflow, tell me which packages you need.

## without_wavlent.py — وصف ووظيفة الملف

هذا الملف يقوم بما يلي:

- يقرأ صورًا من المجلد المحدد في المتغير `DATASET` (الافتراضي: `Organized_Data/`، ويجب أن يحتوي على مجلدات `healthy` و`diseased`).
- يستخرج ميزات LBP (Local Binary Pattern) من كل صورة باستخدام `skimage.feature.local_binary_pattern` ويحفظ التوزيع (histogram) في `lbp_features.csv`.
- يقسم البيانات لقطع train/test، يطبق StandardScaler، يدرب نموذج SVM (kernel='rbf')، ويقيّم الأداء (accuracy, precision, recall, f1, confusion matrix).
- يحسب زمن المعالجة لكل صورة (بالتقريب) ويطبع النتائج.
- يخزن النموذج والمُقَيِّم (scaler) في مجلد `models/` كملفات `svm_lbp.pkl` و`scaler_ww.pkl` على التوالي.

تخصيص وتشغيل

- لتعديل المسارات أو الإعدادات، افتح `without_wavlent.py` وعدّل المتغيرات أعلى الملف: `DATASET`, `MODEL_DIR`, `CSV_PATH`, `IMG_SIZE`, `CATEGORIES`، وما إلى ذلك.
- لتشغيل الملف من سطر الأوامر:

```powershell
python without_wavlent.py
```

المخرجات المتوقعة

- ملف CSV يحتوي ميزات LBP: `lbp_features.csv`.
- نموذج SVM محفوظ: `models/svm_lbp.pkl`.
- StandardScaler محفوظ: `models/scaler_ww.pkl`.

المتطلبات (اقتراحية)

- Python 3.8+
- numpy
- pandas
- opencv-python
- scikit-image
- scikit-learn
- joblib
- tqdm

يمكنني إنشاء `requirements.txt` تلقائيًا بقائمة الحزم أعلاه إذا رغبت بذلك.

---

Files already on GitHub

The file `without_wavlent.py` is already tracked in this repo and was included in the initial push. The README has now been updated and pushed as well.
# PlantDiseaseSVM (v3)

This repository contains code and data used for plant disease classification using SVMs and feature extraction.

Contents (high level):
- `app.py` — main application / experiment runner
- `preprocessor.py` — preprocessing utilities
- `train_W_svm.py` — training script
- `dataset/` — image dataset (not included in git)
- `models/` — trained model files (not included in git)
- CSV feature files (excluded from commit by default)

Notes:
- Large data and model files are excluded from the repository via `.gitignore`.
- To push dataset or models, add them to a separate release or storage (Git LFS, cloud storage) and link from here.

How to use
1. Create and activate a Python environment.
2. Install dependencies (see scripts or requirements if provided).
3. Run preprocessing and training scripts as needed.

If you want me to include a `requirements.txt` or CI workflow, tell me which packages you need.

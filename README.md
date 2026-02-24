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

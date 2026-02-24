import sys
sys.stdout.reconfigure(encoding='utf-8')  # لتجنب مشاكل الترميز في ويندوز

import os
import shutil
import threading
import random
from tqdm import tqdm

# --- Paths ---
DATASET_SOURCE = r'A:\MY project\v3\dataset'
OUTPUT_FOLDER = r'A:\MY project\v3\Organized_Data'

MAX_PER_CLASS = 15000

lock = threading.Lock()

collected = {
    "healthy": [],
    "diseased": [],
    "unknown": []
}

def classify_folder(folder_name):
    name = folder_name.lower()
    if "healthy" in name:
        return "healthy"
    elif any(d in name for d in ["spot","blight","rust","scab","virus","mold","rot","diseased"]):
        return "diseased"
    else:
        return "unknown"

def collect_images_from_folder(path, prefix):
    images = [f for f in os.listdir(path)
              if f.lower().endswith(('.png','.jpg','.jpeg'))]

    for img in images:
        full = os.path.join(path, img)
        category = classify_folder(os.path.basename(path))

        with lock:
            collected[category].append((full, f"{prefix}_{img}"))

def process_standard_modalities():
    modes = ['color','segmented','grayscale']
    print("Thread 1 → scanning standard folders...")

    for mode in modes:
        mode_path = os.path.join(DATASET_SOURCE, mode)
        if not os.path.exists(mode_path):
            continue

        for sub in os.listdir(mode_path):
            sub_path = os.path.join(mode_path, sub)
            if os.path.isdir(sub_path):
                collect_images_from_folder(sub_path, f"STD_{mode}_{sub}")

    print("Thread 1 finished")

def process_other_folders():
    modes = ['color','segmented','grayscale']
    print("Thread 2 => scanning other folders...")

    for item in os.listdir(DATASET_SOURCE):
        item_path = os.path.join(DATASET_SOURCE, item)

        if item in modes or not os.path.isdir(item_path):
            continue

        collect_images_from_folder(item_path, f"OTHER_{item}")

    print("Thread 2 finished")

def balance_and_copy():
    print("\nBalancing dataset...")

    healthy_count = len(collected["healthy"])
    diseased_count = len(collected["diseased"])

    print(f"Healthy found  : {healthy_count}")
    print(f"Diseased found : {diseased_count}")

    if healthy_count == 0 or diseased_count == 0:
        print("ERROR: one class is empty!")
        return

    target_count = min(healthy_count, diseased_count, MAX_PER_CLASS)
    print(f"Balanced count per class: {target_count}")

    for category in ["healthy","diseased"]:
        dest = os.path.join(OUTPUT_FOLDER, category)
        os.makedirs(dest, exist_ok=True)

        random.shuffle(collected[category])
        selected = collected[category][:target_count]

        for src, name in tqdm(selected, desc=f"Copying {category}"):
            try:
                # منع تكرار الأسماء
                dst = os.path.join(dest, name)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(name)
                    dst = os.path.join(dest, base + "_dup" + ext)

                shutil.copy2(src, dst)

            except Exception as e:
                print("Copy error:", e)

def run_system():
    for cat in ['healthy','diseased']:
        os.makedirs(os.path.join(OUTPUT_FOLDER, cat), exist_ok=True)

    t1 = threading.Thread(target=process_standard_modalities)
    t2 = threading.Thread(target=process_other_folders)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    balance_and_copy()

    print("\n" + "="*50)
    print("DONE  Balanced dataset ready")
    print("Saved in:", OUTPUT_FOLDER)
    print("="*50)

if __name__ == "__main__":
    run_system()
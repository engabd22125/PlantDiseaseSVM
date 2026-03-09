import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import shutil
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIG =================
DATASET_SOURCE = r'A:\F_PROJECT\Dataset'
OUTPUT_FOLDER = r'A:\MY project\v3\org_dataset'
# ==========================================

# Data structure: { "apple": {"healthy": [paths], "diseased": [paths]}, "corn": {...} }
plant_data = defaultdict(lambda: {"healthy": [], "diseased": []})

def classify_by_name(path_string):
    name = path_string.lower()
    
    if "healthy" in name:
        return "healthy"
    
    disease_keywords = [
        "spot", "blight", "rust", "scab", 
        "virus", "mold", "rot", "diseased", 
        "septoria", "mite"
    ]
    
    if any(d in name for d in disease_keywords):
        return "diseased"
        
    return None


def extract_plant_name(folder_path):
    name = os.path.basename(folder_path).lower()

    # Clean the folder name to get only the plant type
    if "___" in name:
        name = name.split("___")[0]
    elif "_" in name:
        name = name.split("_")[0]

    return name


def scan_and_group():
    print(f"🔍 Scanning folders in: {DATASET_SOURCE}")
    
    for root, _, files in os.walk(DATASET_SOURCE):
        category = classify_by_name(root)
        
        if category:
            plant_name = extract_plant_name(root)

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    plant_data[plant_name][category].append(full_path)

    print(f"✅ Found {len(plant_data)} plant types.\n")


def process_and_balance():
    
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    os.makedirs(os.path.join(OUTPUT_FOLDER, "healthy"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "diseased"), exist_ok=True)

    total_copied = {"healthy": 0, "diseased": 0}

    for plant, categories in plant_data.items():

        h_list = categories["healthy"]
        d_list = categories["diseased"]

        h_count = len(h_list)
        d_count = len(d_list)

        if h_count == 0 or d_count == 0:
            print(f"⚠️ Skipping {plant}: missing one class (healthy:{h_count}, diseased:{d_count})")
            continue

        # Balance per plant: choose the minimum between healthy and diseased
        target_for_this_plant = min(h_count, d_count)

        print(f"🌿 Processing {plant.upper()}: selecting {target_for_this_plant} images per class.")

        for cat in ["healthy", "diseased"]:
            selected_images = categories[cat][:target_for_this_plant]
            dest_dir = os.path.join(OUTPUT_FOLDER, cat)

            for i, src in enumerate(selected_images):

                # Unique file naming: plant_category_index.jpg
                new_name = f"{plant}_{cat}_{i+1}.jpg"

                shutil.copy2(src, os.path.join(dest_dir, new_name))
                total_copied[cat] += 1


    print("\n" + "="*50)
    print("🎉 Process completed successfully!")
    print(f"📁 Total healthy images: {total_copied['healthy']}")
    print(f"📁 Total diseased images: {total_copied['diseased']}")
    print(f"📍 Output location: {OUTPUT_FOLDER}")
    print("="*50)


if __name__ == "__main__":
    scan_and_group()
    process_and_balance()
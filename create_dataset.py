# create_dataset.py

from icrawler.builtin import BingImageCrawler
from PIL import Image
import os

DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

classes = {
    "Melanoma": "melanoma dermoscopy skin",
    "Benign": "benign nevus dermoscopy skin",
    "Basal_Cell": "basal cell carcinoma dermoscopy skin"
}

for label, keyword in classes.items():

    print(f"\n⬇️ Downloading {label} images...")

    class_path = os.path.join(DATASET_DIR, label)
    os.makedirs(class_path, exist_ok=True)

    crawler = BingImageCrawler(storage={"root_dir": class_path})

    crawler.crawl(
        keyword=keyword,
        max_num=30   # 👈 you asked 20–30 images
    )

    # ✅ Convert ALL images to JPG (important)
    files = os.listdir(class_path)
    count = 0

    for i, file in enumerate(files):
        try:
            path = os.path.join(class_path, file)

            img = Image.open(path).convert("RGB")
            new_path = os.path.join(class_path, f"{label}_{i}.jpg")

            img.save(new_path, "JPEG")

            if not file.endswith(".jpg"):
                os.remove(path)

            count += 1

        except:
            pass

    print(f"✅ Saved {count} JPG images in {label}")

print("\n📁 Dataset ready at:", os.path.abspath(DATASET_DIR))

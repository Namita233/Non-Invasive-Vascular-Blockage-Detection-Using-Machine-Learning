import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ✅ Dataset path
DATA_DIR = r"D:\vascular_blockage\data"

# ✅ Image size (small to reduce memory)
IMG_SIZE = 128
MAX_IMAGES_PER_CLASS = 500  # safe limit for now

def load_data():
    images, labels = [], []
    categories = ["blockage", "no_blockage"]

    for label, category in enumerate(categories):
        path = os.path.join(DATA_DIR, category)
        if not os.path.exists(path):
            print(f"⚠ Folder missing: {path}")
            continue

        image_files = os.listdir(path)[:MAX_IMAGES_PER_CLASS]
        print(f"📂 Loading {len(image_files)} images from {category}")

        for img_name in image_files:
            img_path = os.path.join(path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(label)
            except Exception as e:
                print(f"❌ Error with {img_name}: {e}")

    X = np.array(images, dtype=np.float32) / 255.0
    y = np.array(labels)
    print(f"✅ Total loaded images: {len(X)}")
    return X, y


def get_train_test_data(test_size=0.2, random_state=42):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    get_train_test_data()
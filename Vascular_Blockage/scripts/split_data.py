import os, random, shutil

# Set random seed for reproducibility
random.seed(42)

# Define your main paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset_split')

classes = ['blockage', 'no_blockage']

# Ratios for split
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

def make_dirs():
    for split in ['train', 'val', 'test']:
        for cls in classes:
            path = os.path.join(OUT_DIR, split, cls)
            os.makedirs(path, exist_ok=True)

def split_data():
    make_dirs()
    for cls in classes:
        cls_path = os.path.join(BASE_DIR, cls)
        all_imgs = os.listdir(cls_path)
        random.shuffle(all_imgs)

        n_total = len(all_imgs)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_imgs = all_imgs[:n_train]
        val_imgs = all_imgs[n_train:n_train + n_val]
        test_imgs = all_imgs[n_train + n_val:]

        for split_name, img_list in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            for img in img_list:
                src = os.path.join(cls_path, img)
                dst = os.path.join(OUT_DIR, split_name, cls, img)
                shutil.copy(src, dst)

        print(f"{cls}: {n_total} images → Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

if __name__ == "__main__":
    split_data()
    print("\n✅ Data split successfully! Check 'dataset_split' folder.")
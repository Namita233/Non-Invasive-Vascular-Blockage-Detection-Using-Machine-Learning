import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from PIL import Image

# ========== STEP 1: VERIFY IMAGES (REMOVE CORRUPTED ONES) ==========
def verify_images(directory):
    print("\n🔍 Checking for corrupted images...")
    removed = 0
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):
            continue
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path)
                img.verify()  # verifies if image is corrupted
            except Exception:
                print(f"❌ Removed corrupted image: {file_path}")
                os.remove(file_path)
                removed += 1
    print(f"✅ Image check complete. Removed {removed} bad images.\n")

# Path to your dataset
data_dir = r"D:\Vascular_Blockage\data"

# Verify all images first
verify_images(data_dir)

# ========== STEP 2: IMAGE DATA GENERATORS ==========
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ========== STEP 3: CNN MODEL ==========
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# ========== STEP 4: COMPILE MODEL ==========
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# ========== STEP 5: TRAIN MODEL ==========
print("\n🚀 Starting training...\n")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# ========== STEP 6: SAVE MODEL ==========
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/vascular_blockage_model.h5")
print("\n✅ Model training complete and saved at: saved_model/vascular_blockage_model.h5")
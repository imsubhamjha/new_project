import numpy as np
from segmentation_models import Unet
from sklearn.model_selection import train_test_split
import cv2
import os

# Load data (assuming preprocessed and CLAHE applied)
def load_data(data_dir):
    images, masks = [], []
    for img_file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_file)
        mask_path = img_path.replace('.jpg', '_mask.jpg')  # Example mask format
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img = img / 255.0
        mask = mask / 255.0

        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load dataset
train_images, train_masks = load_data('data/train/')
train_images = train_images.reshape(-1, 128, 128, 1)  # Adjust the shape as per your data
train_masks = train_masks.reshape(-1, 128, 128, 1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.2, random_state=42)

# Define U-Net++ model
model_unet_plus = Unet('resnet34', encoder_weights='imagenet', classes=1, activation='sigmoid')
model_unet_plus.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model_unet_plus.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8)

# Save the model
model_unet_plus.save('models/unet_plus_plus.h5')
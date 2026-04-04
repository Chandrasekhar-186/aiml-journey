# Day 20 — OpenCV Advanced
# Date: April 1, 2026
# Image preprocessing pipeline for YOLOv8!

import cv2
import numpy as np
import mlflow

def preprocess_for_yolo(image_path,
                         target_size=(640, 640)):
    """Complete preprocessing pipeline for YOLOv8"""

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        # Create synthetic image for testing
        img = np.random.randint(
            0, 255, (480, 640, 3),
            dtype=np.uint8
        )

    original_shape = img.shape
    print(f"Original: {original_shape}")

    # 2. Color space conversion
    img_rgb = cv2.cvtColor(img,
                            cv2.COLOR_BGR2RGB)

    # 3. Resize with aspect ratio preserved
    h, w = img.shape[:2]
    scale = min(target_size[0]/h,
                target_size[1]/w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    img_resized = cv2.resize(
        img_rgb, (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    # 4. Letterbox padding (YOLOv8 style!)
    pad_h = target_size[0] - new_h
    pad_w = target_size[1] - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 5. Normalize to [0, 1]
    img_normalized = img_padded.astype(
        np.float32) / 255.0

    # 6. Image augmentations
    # Horizontal flip
    img_flipped = cv2.flip(img_padded, 1)

    # Brightness adjustment
    img_bright = cv2.convertScaleAbs(
        img_padded, alpha=1.2, beta=10
    )

    # Gaussian blur (reduce noise)
    img_blurred = cv2.GaussianBlur(
        img_padded, (5, 5), 0
    )

    print(f"After preprocessing: {img_padded.shape}")
    print(f"Normalized range: "
          f"[{img_normalized.min():.2f}, "
          f"{img_normalized.max():.2f}]")

    # 7. Save preprocessed images
    cv2.imwrite('preprocessed.jpg', img_padded)
    cv2.imwrite('augmented_flip.jpg', img_flipped)
    cv2.imwrite('augmented_bright.jpg', img_bright)

    return img_normalized

# 8. Log preprocessing config to MLflow
mlflow.set_experiment("cv_preprocessing")
with mlflow.start_run(run_name="YOLOv8_preprocess"):
    mlflow.log_param("target_size", "640x640")
    mlflow.log_param("letterbox", True)
    mlflow.log_param("normalize", "0-1")
    mlflow.log_param("augmentations",
                     "flip,brightness,blur")

    result = preprocess_for_yolo("test_image.jpg")
    mlflow.log_metric("output_h", result.shape[0])
    mlflow.log_metric("output_w", result.shape[1])
    mlflow.log_artifact('preprocessed.jpg')
    print("Preprocessing logged to MLflow!")

print("\nComplete CV Preprocessing Pipeline:")
print("Load → RGB → Resize → Letterbox →"
      " Normalize → Augment → MLflow log")

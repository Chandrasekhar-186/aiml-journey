pip install ultralytics

# Day 18 — YOLOv8 Object Detection
# Date: March 30, 2026
# State-of-the-art CV — your Phase 5 project uses this!

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import cv2
import numpy as np

print("="*50)
print("YOLOv8 — You Only Look Once v8")
print("Real-time object detection!")
print("="*50)

# 1. Load pretrained YOLOv8 model
print("\nLoading YOLOv8n (nano — fastest)...")
model = YOLO('yolov8n.pt')  # downloads automatically
print(f"Model loaded!")
print(f"Task: {model.task}")

# 2. Model info
info = model.info()
print(f"\nYOLOv8 Architecture:")
print(f"  Parameters: ~3.2M (nano version)")
print(f"  Classes:    80 (COCO dataset)")
print(f"  Input size: 640×640")

# 3. Inference on a test image
# Create synthetic test image
test_img = np.random.randint(
    0, 255, (640, 640, 3),
    dtype=np.uint8
)
cv2.imwrite('test_yolo.jpg', test_img)

# Run inference
results = model('test_yolo.jpg',
                verbose=False)
print(f"\nInference complete!")
print(f"  Detections: {len(results[0].boxes)}")

# 4. Key YOLO concepts
print("\n" + "="*50)
print("YOLO Architecture Explained:")
print("="*50)
print("""
Input Image (640×640×3)
        ↓
Backbone (CSPDarknet)
→ Extracts features at multiple scales
        ↓
Neck (FPN + PAN)
→ Feature Pyramid Network
→ Combines features from different scales
        ↓
Head (Detect)
→ Predicts bounding boxes + class probabilities
→ Output: [x, y, w, h, confidence, class_probs]
        ↓
NMS (Non-Maximum Suppression)
→ Removes duplicate detections
→ Keeps box with highest confidence
""")

# 5. Log to MLflow
mlflow.set_experiment("yolov8_experiments")
with mlflow.start_run(run_name="YOLOv8n_baseline"):
    mlflow.log_param("model", "yolov8n")
    mlflow.log_param("pretrained", True)
    mlflow.log_param("dataset", "COCO")
    mlflow.log_param("num_classes", 80)
    mlflow.log_param("input_size", 640)
    mlflow.log_metric("num_params", 3200000)
    mlflow.log_artifact('test_yolo.jpg')
    print("\nYOLOv8 experiment logged to MLflow!")

# 6. Phase 5 project preview
print("\n" + "="*50)
print("Phase 5 Project: CV Pipeline on Databricks")
print("="*50)
print("""
Architecture:
Video frames → OpenCV preprocessing
            → YOLOv8 detection
            → Results → Delta Lake storage
            → MLflow experiment tracking
            → PySpark aggregations
            → Databricks dashboard

This project shows:
✅ Computer Vision (YOLOv8)
✅ Data Engineering (Delta Lake + Spark)
✅ MLOps (MLflow tracking)
✅ Databricks platform knowledge
→ 4 Databricks competencies in 1 project!
""")

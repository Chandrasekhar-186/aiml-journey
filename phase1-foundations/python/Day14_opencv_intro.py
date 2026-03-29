
pip install opencv-python

# Day 14 — OpenCV Introduction
# Date: March 26, 2026
# Computer Vision toolkit!

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Create a test image (since no webcam needed)
img = np.zeros((400, 600, 3), dtype=np.uint8)

# Draw shapes — basic CV operations
cv2.rectangle(img, (50, 50), (200, 200),
              (0, 255, 0), 3)      # green rectangle
cv2.circle(img, (350, 150), 80,
           (255, 0, 0), -1)        # blue filled circle
cv2.line(img, (0, 300), (600, 300),
         (0, 0, 255), 2)           # red line
cv2.putText(img, "OpenCV + Databricks",
            (50, 350),
            cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 255, 255), 2)

# Save image
cv2.imwrite('test_image.png', img)
print("Image created: test_image.png")

# 2. Image transformations
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
edges = cv2.Canny(blurred, 50, 150)

# 3. Contour detection
contours, _ = cv2.findContours(
    edges, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
print(f"Contours found: {len(contours)}")

# 4. Save results
cv2.imwrite('gray_image.png', gray)
cv2.imwrite('edges_image.png', edges)
print("Transformations saved!")

# 5. Key CV concepts
print("\nCV Pipeline for Databricks:")
print("Image → OpenCV preprocess →"
      " PyTorch CNN → MLflow log →"
      " Delta Lake store → Spark serve")

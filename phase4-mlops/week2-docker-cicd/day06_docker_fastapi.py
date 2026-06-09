# Phase 4 Day 6 — Docker + FastAPI for ML
# Date: June 17, 2026
# Containerized model deployment!

print("="*60)
print("Docker + FastAPI — ML Serving")
print("="*60)

"""
WHY DOCKER FOR ML?

Problem without Docker:
"It works on my machine!"
→ Different Python versions
→ Different library versions
→ Different OS dependencies
→ Works in dev, breaks in prod!

Docker solution:
→ Package code + dependencies + runtime
→ Same container runs EVERYWHERE
→ Dev = Staging = Production
→ Reproducible builds always!

Docker concepts:
Image:     blueprint (snapshot of filesystem)
Container: running instance of image
Layer:     each Dockerfile instruction = layer
           layers are CACHED → fast rebuilds!
Registry:  store images (Docker Hub, ECR, ACR)

For ML specifically:
→ Lock Python + library versions
→ Include model artifacts
→ No "works on my machine" ever again!
"""

# 1. Dockerfile for ML serving
DOCKERFILE = '''
# ML Model Serving Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer caching!)
# Copy requirements before code
# → only rebuilds if requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.main:app",
     "--host", "0.0.0.0",
     "--port", "8000",
     "--workers", "4"]
'''

# 2. requirements.txt
REQUIREMENTS = '''
fastapi==0.104.1
uvicorn[standard]==0.24.0
mlflow==2.9.2
scikit-learn==1.3.2
numpy==1.26.2
pandas==2.1.3
pydantic==2.5.0
prometheus-client==0.19.0
'''

# 3. FastAPI application
FASTAPI_APP = '''
# src/main.py — Production ML API
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import mlflow
import numpy as np
import pandas as pd
import logging
import time
from prometheus_client import (
    Counter, Histogram, generate_latest
)
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "status"]
)
INFERENCE_TIME = Histogram(
    "inference_duration_seconds",
    "Model inference duration"
)
PREDICTION_DIST = Counter(
    "predictions_total",
    "Prediction distribution",
    ["class_label"]
)

app = FastAPI(
    title="ML Model API",
    description="Production ML serving",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model at startup (not per request!)
MODEL = None
MODEL_VERSION = None

@app.on_event("startup")
async def load_model():
    global MODEL, MODEL_VERSION
    try:
        MODEL = mlflow.sklearn.load_model(
            "models:/ProductionModel/Production"
        )
        MODEL_VERSION = "1.0"
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        # Fall back to local model
        import joblib
        MODEL = joblib.load("models/model.pkl")

# Request/Response schemas
class PredictionRequest(BaseModel):
    features: List[List[float]]
    request_id: Optional[str] = None

    @validator("features")
    def validate_features(cls, v):
        if not v:
            raise ValueError("Empty features!")
        if len(v[0]) != 9:  # expected features
            raise ValueError(
                f"Expected 9 features, "
                f"got {len(v[0])}"
            )
        return v

class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    model_version: str
    latency_ms: float
    request_id: Optional[str]

# Endpoints
@app.get("/health")
async def health_check():
    """Kubernetes/Docker health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_version": MODEL_VERSION
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/predict",
           response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    if MODEL is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded!"
        )

    start_time = time.time()
    try:
        X = np.array(request.features)
        predictions = MODEL.predict(X).tolist()
        probabilities = MODEL.predict_proba(
            X
        ).tolist()

        latency = (time.time() - start_time)*1000

        # Update metrics
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="success"
        ).inc()
        INFERENCE_TIME.observe(latency / 1000)
        for pred in predictions:
            PREDICTION_DIST.labels(
                class_label=str(pred)
            ).inc()

        logger.info(
            f"Predicted {len(predictions)} samples"
            f" in {latency:.2f}ms"
        )

        return PredictionResponse(
            predictions=predictions,
            probabilities=probabilities,
            model_version=MODEL_VERSION,
            latency_ms=latency,
            request_id=request.request_id
        )

    except Exception as e:
        REQUEST_COUNT.labels(
            endpoint="/predict",
            status="error"
        ).inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.post("/predict/batch")
async def predict_batch(
    request: PredictionRequest
):
    """Batch prediction endpoint"""
    # Same as predict but optimized for large batches
    return await predict(request)
'''

# 4. Docker compose for full stack
DOCKER_COMPOSE = '''
# docker-compose.yml
version: "3.8"

services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - MODEL_NAME=ProductionModel
    depends_on:
      - mlflow
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "2"

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlflow/mlruns
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:///mlflow.db
      --default-artifact-root /mlflow/mlruns

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
'''

# 5. Print all files
print("\n=== DOCKERFILE ===")
print(DOCKERFILE)

print("\n=== FASTAPI APP (key parts) ===")
# Print key sections
for section in [
    "# Load model at startup",
    "class PredictionRequest",
    "@app.post(\"/predict\""
]:
    start = FASTAPI_APP.find(section)
    print(FASTAPI_APP[start:start+200] + "...")

print("\n=== DOCKER COMMANDS ===")
print("""
# Build image
docker build -t ml-api:latest .

# Run container
docker run -p 8000:8000 ml-api:latest

# Run full stack
docker-compose up -d

# Test the API
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"features": [[1.2,0.5,3.1,2.0,
                      1.44,0.41,0.31,2.4,1]]}'

# Check health
curl http://localhost:8000/health

# View logs
docker logs ml-api --tail 100

# Scale up
docker-compose up -d --scale ml-api=3
""")

# 6. Write files to disk
import os
os.makedirs(
    "phase4-mlops/week2-docker-cicd",
    exist_ok=True
)

files = {
    "phase4-mlops/week2-docker-cicd/Dockerfile":
        DOCKERFILE,
    "phase4-mlops/week2-docker-cicd/requirements.txt":
        REQUIREMENTS,
    "phase4-mlops/week2-docker-cicd/docker-compose.yml":
        DOCKER_COMPOSE,
}

for path, content in files.items():
    with open(path, 'w') as f:
        f.write(content)
    print(f"  ✅ Written: {path}")

# Also write FastAPI app
os.makedirs(
    "phase4-mlops/week2-docker-cicd/src",
    exist_ok=True
)
with open(
    "phase4-mlops/week2-docker-cicd/src/main.py",
    'w'
) as f:
    f.write(FASTAPI_APP)
print("  ✅ Written: src/main.py")

import mlflow
mlflow.set_experiment("phase4_docker_fastapi")
with mlflow.start_run(
        run_name="docker_api_design"):
    mlflow.log_params({
        "serving": "FastAPI",
        "container": "Docker",
        "workers": 4,
        "monitoring": "Prometheus+Grafana",
        "validation": "Pydantic"
    })
    print("\nDocker+FastAPI design logged!")

print("\n" + "="*60)
print("Docker + FastAPI — MASTERED! 🐳")
print("Phase 4 Week 2 Day 1 COMPLETE!")
print("="*60)

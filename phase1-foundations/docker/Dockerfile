 # Day 06 — First Dockerfile
# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching!)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Environment variable
ENV MODEL_NAME=RandomForest
ENV ACCURACY_THRESHOLD=85

# Run script
CMD ["python", "train.py"]

# Phase 1 Capstone — Day 4
# RAG System + CV Integration
# Date: April 5, 2026

import mlflow
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import mlflow.pytorch
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)
from langchain.schema import Document

print("="*55)
print("Capstone Day 4: RAG + CV Integration")
print("="*55)

# ── PART 1: RAG System ───────────────────────────

# 1. Load experiment data
df = pd.read_csv('experiments_dataset.csv')

# 2. Convert experiments to documents
print("\n📚 Building RAG knowledge base...")
documents = []
for _, row in df.iterrows():
    content = f"""
Experiment {row['exp_id']}:
Model: {row['model_type']}
Dataset: {row['dataset']}
Accuracy: {row['accuracy']:.4f}
Train time: {row['train_time']:.2f}s
Status: {'PASSED' if row['passed'] else 'FAILED'}
Date: {row['date']}
"""
    documents.append(Document(
        page_content=content,
        metadata={
            "exp_id": int(row['exp_id']),
            "model_type": row['model_type'],
            "passed": bool(row['passed'])
        }
    ))

# 3. Build FAISS vector store
embeddings = HuggingFaceEmbeddings(
    model_name=(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150, chunk_overlap=20
)
chunks = splitter.split_documents(documents[:50])
vectorstore = FAISS.from_documents(
    chunks, embeddings
)
print(f"RAG index built: {len(chunks)} chunks")

# 4. RAG query function
def query_experiments(question, k=3):
    docs = vectorstore.similarity_search(
        question, k=k
    )
    context = "\n".join(
        [d.page_content for d in docs]
    )
    prompt = f"""Based on experiment data:

{context}

Question: {question}
Answer:"""
    return prompt, docs

# 5. Test RAG queries
queries = [
    "Which XGBoost experiments passed?",
    "What was the best accuracy on wine dataset?",
    "Which models trained in under 2 seconds?"
]

print("\n🔍 RAG Query Results:")
for q in queries:
    prompt, docs = query_experiments(q)
    print(f"\nQ: {q}")
    print(f"Retrieved {len(docs)} relevant experiments")
    print(f"Top result: "
          f"{docs[0].page_content[:80].strip()}...")

# ── PART 2: CV Chart Classifier ─────────────────

print("\n\n👁️ Building CV Chart Classifier...")

# 6. Generate synthetic performance charts
def create_performance_chart(
        accuracy, model_name, chart_type="bar"):
    """Create a synthetic performance chart image"""
    img = np.ones((224, 224, 3),
                   dtype=np.uint8) * 240

    if chart_type == "bar":
        # Draw bar chart
        bar_height = int(accuracy * 180)
        cv2.rectangle(
            img,
            (60, 220 - bar_height),
            (160, 220),
            (50, 150, 50), -1
        )
        # Add accuracy text
        cv2.putText(
            img, f"{accuracy:.2f}",
            (75, 220 - bar_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 0), 2
        )
        cv2.putText(
            img, model_name[:8],
            (55, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 0), 1
        )

    # Add border
    cv2.rectangle(img, (0, 0),
                   (223, 223), (0, 0, 0), 2)
    return img

# 7. Create chart dataset
print("Generating chart images...")
chart_data = []
labels = []
for _, row in df.head(100).iterrows():
    chart = create_performance_chart(
        row['accuracy'],
        row['model_type']
    )
    chart_data.append(chart)
    labels.append(1 if row['passed'] else 0)

print(f"Generated {len(chart_data)} charts")

# 8. Simple CNN chart classifier
class ChartClassifierCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 9. Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

tensors = torch.stack([
    transform(cv2.cvtColor(img,
              cv2.COLOR_BGR2RGB))
    for img in chart_data
])
label_tensor = torch.LongTensor(labels)

# Split
split = int(0.8 * len(tensors))
X_train = tensors[:split]
y_train = label_tensor[:split]
X_test = tensors[split:]
y_test = label_tensor[split:]

# 10. Train CNN — log to MLflow!
cnn_model = ChartClassifierCNN()
optimizer = torch.optim.Adam(
    cnn_model.parameters(), lr=0.001
)
criterion = nn.CrossEntropyLoss()

mlflow.set_experiment("capstone_cv_classifier")
with mlflow.start_run(
        run_name="ChartCNN_v1") as run:
    mlflow.log_param("model",
                     "ChartClassifierCNN")
    mlflow.log_param("input_size", "224x224")
    mlflow.log_param("epochs", 5)

    for epoch in range(5):
        cnn_model.train()
        optimizer.zero_grad()
        out = cnn_model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        cnn_model.eval()
        with torch.no_grad():
            preds = cnn_model(X_test).argmax(1)
            acc = (preds == y_test).float().mean()

        mlflow.log_metric(
            "loss", loss.item(), step=epoch
        )
        mlflow.log_metric(
            "accuracy", acc.item(), step=epoch
        )
        print(f"Epoch {epoch+1}: "
              f"Loss={loss.item():.4f} "
              f"Acc={acc.item():.4f}")

    mlflow.pytorch.log_model(
        cnn_model, "chart_classifier_cnn"
    )
    print("\nCV chart classifier logged!")

# 11. Save sample chart
sample_chart = create_performance_chart(
    0.95, "XGBoost"
)
cv2.imwrite('sample_chart.png', sample_chart)
mlflow.log_artifact('sample_chart.png')

print("\n" + "="*55)
print("Capstone Day 4 Complete!")
print("Components built:")
print("✅ RAG system for experiment queries")
print("✅ CNN chart classifier")
print("✅ All logged to MLflow")
print("\nTomorrow: Final integration + README!")
print("="*55)

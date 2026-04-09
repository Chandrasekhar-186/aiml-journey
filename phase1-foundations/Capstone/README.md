# 🧠 Intelligent ML Experiment Analyzer

> End-to-end ML system combining PySpark,
> Delta Lake, MLflow, RAG, and Computer Vision
> — built on Databricks architecture

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Spark](https://img.shields.io/badge/Apache_Spark-3.x-orange)
![MLflow](https://img.shields.io/badge/MLflow-2.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)

## 🎯 Project Overview

An intelligent system that analyzes ML
experiments at scale, predicts experiment
success, and enables natural language
queries over experiment history.

## 🏗️ Architecture
```
Raw Data → PySpark Ingestion
        → Delta Lake (Bronze/Silver/Gold)
        → Meta-Model Training (XGBoost+NN)
        → MLflow Model Registry
        → RAG Query System (LangChain+FAISS)
        → CNN Chart Classifier (PyTorch)
        → Unified API
```

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Engineering | PySpark, Delta Lake |
| ML Training | XGBoost, PyTorch, Scikit-learn |
| Experiment Tracking | MLflow |
| GenAI/RAG | LangChain, FAISS, HuggingFace |
| Computer Vision | OpenCV, CNN, YOLOv8 |
| Infrastructure | Docker, Git |

## 📊 Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| XGBoost Meta | 0.91 | 0.91 |
| GBM Meta | 0.89 | 0.89 |
| PyTorch NN | 0.88 | 0.88 |
| CNN Chart | 0.85 | 0.84 |

## 🚀 Quick Start
```bash
git clone https://github.com/Chandrasekhar-186/aiml-journey
cd phase1-foundations/capstone
pip install -r requirements.txt
python capstone_final.py
mlflow ui  # view experiments at localhost:5000
```

## 📁 Project Structure
```
capstone/
├── capstone_day1_data.py      # Data generation
├── capstone_day2_spark.py     # PySpark + Delta Lake
├── capstone_day3_metamodel.py # Meta-model training
├── capstone_day4_rag_cv.py    # RAG + CNN
├── capstone_final.py          # Integration demo
└── README.md                  # This file
```

## 💡 Key Learnings

1. **Delta Lake MERGE** for upsert operations
2. **MLflow Registry** staging → production
3. **RAG** enables natural language data queries
4. **CNN** can classify chart images effectively
5. **PySpark window functions** for ranking

## 👨‍💻 Author

Nuthalapati Chandrasekhar
B.Tech AIML 2027 | Building toward
Databricks MLE role

---
*Built as Phase 1 capstone — Day 25 of
6-month Databricks preparation journey*

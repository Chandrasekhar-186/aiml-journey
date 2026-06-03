## Phase 3 Master Summary

Week 1 — ML Algorithms:
Linear Reg: GD + Normal Eq + Ridge/Lasso
Logistic:   sigmoid + log loss + threshold
DT + RF:    Gini + IG + OOB + bagging
SVM:        margin max + kernel trick
Clustering: K-Means++ + DBSCAN + ward
PCA:        eigendecomposition + t-SNE
XGBoost:    GBM + L1+L2 + second-order

Week 2 — Deep Learning:
NN:         forward + backprop + He init
Optimizers: Adam = momentum + RMSProp + BC
ResNet:     y=F(x)+x → gradient highway
LSTM:       4 gates, cell state = memory
Attention:  QKᵀ/√dk → softmax → V

Week 3 — Computer Vision:
Transfer:   freeze backbone + diffLR
YOLOv8:    C2f + PAN-FPN + anchor-free
U-Net:     encoder + decoder + skip
ViT:        16×16 patches + [CLS]
CLIP:       InfoNCE + zero-shot

Week 4 — LLMs + GenAI:
BERT:       encoder-only + MLM
GPT:        decoder-only + CLM + causal mask
LoRA:       ΔW=BA, B=0, save 96% params
RAG:        hybrid BM25+dense + RRF + rerank
Agents:     ReAct: thought→action→observe

## N-Sum Pattern
Sort array
Fix N-2 outer pointers (skip duplicates!)
Two pointers for innermost
Works for 2Sum, 3Sum, 4Sum, kSum!

## Phase 4 Focus
Models → Production
Training → Serving
Experiments → Workflows
Notebooks → CI/CD pipelines

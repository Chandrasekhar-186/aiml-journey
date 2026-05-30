# Phase 3 LLM Day 2 — LoRA + QLoRA
# Date: May 30, 2026
# Fine-tuning LLMs efficiently!

import torch
import torch.nn as nn
import math
import mlflow
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np

print("="*60)
print("LoRA + QLoRA — Efficient LLM Fine-tuning")
print("="*60)

"""
THE FINE-TUNING PROBLEM:

GPT-3:   175B parameters
LLaMA-2: 70B parameters
Mistral: 7B parameters

Full fine-tuning 7B model:
→ 7B × 4 bytes = 28GB just for weights!
→ + gradients: 28GB more
→ + optimizer states (Adam): 56GB more
→ Total: ~112GB GPU memory
→ Need: 2-3 × A100 80GB cards = $$$!

Solution: Parameter-Efficient Fine-Tuning (PEFT)
→ Freeze most parameters
→ Train only a small adapter!
→ LoRA: train ~0.1-1% of parameters!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LORA — LOW-RANK ADAPTATION:

Key insight: weight updates during fine-tuning
             have LOW INTRINSIC RANK!

Full fine-tuning: update W ∈ ℝ^(d×k)
→ d×k parameters per layer

LoRA: decompose update ΔW = B × A
  A ∈ ℝ^(r×k)  (down-projection)
  B ∈ ℝ^(d×r)  (up-projection)
  r << min(d,k) (rank, typically 4-64)

Forward pass:
h = W₀x + ΔWx = W₀x + BAx
             ↑              ↑
      frozen weights    trainable adapter

Parameters comparison:
Full:  d × k = 768 × 768 = 589,824
LoRA:  r×k + d×r = 16×768 + 768×16 = 24,576
Saving: 96% fewer parameters!

Initialization:
A: random Gaussian (N(0, σ²))
B: ZERO (so ΔW=0 at start — no change!)
Scale: α/r (α is hyperparameter, usually=r)

Why B=0 at start?
→ Fine-tuning starts from pretrained model
→ No disruption at beginning of training
→ Model gradually adapts!

WHERE to apply LoRA:
In transformer: Q, K, V, O projections
Sometimes: FFN layers too
Typical: just Q + V (most impactful!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QLORA — QUANTIZED LORA:

LoRA still loads full model in fp16/fp32
QLoRA: quantize base model to 4-bit NF4!

Memory breakdown:
LoRA on 7B:   7B × 2 bytes = 14GB (fp16)
QLoRA on 7B:  7B × 0.5 bytes = 3.5GB (4-bit!)

4-bit NF4 quantization:
→ Normal Float 4 (NF4) format
→ Optimized for normally distributed weights
→ Minimal accuracy loss vs fp16!

Double quantization:
→ Quantize the quantization constants too!
→ Saves additional 0.37 bits/parameter

Result: fine-tune 7B model on single 24GB GPU!
(Previously needed 4× A100 = $20,000+)

QLoRA enabled: anyone can fine-tune LLMs!
"""

# 1. LoRA implementation from scratch
class LoRALayer(nn.Module):
    """
    LoRA adapter for a linear layer.
    Adds low-rank decomposition ΔW = BA
    """
    def __init__(self, in_features,
                  out_features, rank=16,
                  alpha=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Frozen original weights
        self.weight = nn.Parameter(
            torch.randn(out_features,
                         in_features),
            requires_grad=False  # FROZEN!
        )

        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features)
            * (1 / math.sqrt(in_features))
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
            # B=0 → no change at start!
        )

        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

    def forward(self, x):
        # Original: W₀x
        base_out = F.linear(
            x, self.weight, self.bias
        )
        # LoRA: scale * B @ A @ x
        lora_out = (
            self.scale *
            F.linear(
                F.linear(
                    self.dropout(x),
                    self.lora_A
                ),
                self.lora_B
            )
        )
        return base_out + lora_out

    def trainable_params(self):
        return (self.lora_A.numel() +
                self.lora_B.numel())

    def total_params(self):
        return (self.weight.numel() +
                self.lora_A.numel() +
                self.lora_B.numel() +
                self.bias.numel())

import torch.nn.functional as F

# Test LoRA layer
lora = LoRALayer(
    in_features=768,
    out_features=768,
    rank=16, alpha=16
)
x = torch.randn(4, 128, 768)
out = lora(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Trainable: {lora.trainable_params():,}")
print(f"Total:     {lora.total_params():,}")
print(f"Ratio: {lora.trainable_params()/"
      f"{lora.total_params():.1%}")

# 2. Apply LoRA to BERT
class LoRABERT(nn.Module):
    """BERT with LoRA adapters on Q + V"""
    def __init__(self, num_classes=2,
                  rank=8, alpha=16):
        super().__init__()
        from transformers import BertModel
        self.bert = BertModel.from_pretrained(
            'bert-base-uncased'
        )

        # Freeze ALL BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        hidden = 768
        # Add LoRA to Q and V in each layer
        self.lora_layers = nn.ModuleDict()
        for i in range(12):  # 12 BERT layers
            # Query projection
            self.lora_layers[f'q_{i}'] = \
                LoRALayer(hidden, hidden,
                           rank, alpha)
            # Value projection
            self.lora_layers[f'v_{i}'] = \
                LoRALayer(hidden, hidden,
                           rank, alpha)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, input_ids,
                 attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]
        return self.classifier(cls_output)

    def count_params(self):
        trainable = sum(
            p.numel() for p in
            self.parameters()
            if p.requires_grad
        )
        total = sum(
            p.numel() for p in
            self.parameters()
        )
        return trainable, total

# Analyze LoRA BERT
lora_bert = LoRABERT(rank=8)
trainable, total = lora_bert.count_params()
print(f"\nLoRA BERT analysis:")
print(f"Total params:     {total:,}")
print(f"Trainable params: {trainable:,}")
print(f"Training ratio:   {trainable/total:.2%}")

# 3. Real LoRA with PEFT library
print("\n=== PEFT LIBRARY (PRODUCTION) ===")
print("""
# In production: use Hugging Face PEFT

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

# Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,              # rank
    lora_alpha=32,     # scaling factor
    target_modules=[   # which layers
        "query",
        "value"
    ],
    lora_dropout=0.1,
    bias="none"        # don't train biases
)

# Apply to model
model = AutoModelForSequenceClassification\\
    .from_pretrained('bert-base-uncased')
peft_model = get_peft_model(
    model, lora_config
)

peft_model.print_trainable_parameters()
# trainable params: 294,912 || all params: 109,776,138
# → 0.27% trainable!
""")

# 4. QLoRA setup
print("\n=== QLORA SETUP ===")
print("""
# QLoRA: 4-bit quantization + LoRA

from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",    # NF4 format!
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load 7B model in 4-bit (fits on 1 GPU!)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# Add LoRA on top of 4-bit model
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=[
        "q_proj", "v_proj",
        "k_proj", "o_proj"  # all attention!
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Memory comparison:
# Full 7B fp16:   14GB
# LoRA 7B fp16:   14GB (same base!)
# QLoRA 7B 4bit:  ~5GB  ← game changer!
""")

# 5. Fine-tuning with synthetic data
print("\n=== FINE-TUNING DEMO ===")
# Synthetic sentiment dataset
texts = [
    "Databricks Delta Lake is amazing",
    "The pipeline failed with OOM error",
    "MLflow tracking works perfectly",
    "Spark job ran for 10 hours and failed",
    "The model accuracy improved significantly",
    "Performance degraded after the update",
] * 20

labels = [1, 0, 1, 0, 1, 0] * 20

tokenizer = AutoTokenizer.from_pretrained(
    'bert-base-uncased'
)

# Tokenize
encodings = tokenizer(
    texts, truncation=True,
    padding=True, max_length=64,
    return_tensors='pt'
)

dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': torch.tensor(labels)
})

# Train with Trainer
model = AutoModelForSequenceClassification\
    .from_pretrained(
        'bert-base-uncased',
        num_labels=2
    )

training_args = TrainingArguments(
    output_dir='/tmp/lora_demo',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    learning_rate=2e-4,  # Higher for LoRA!
    logging_steps=10,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

print("Training LoRA BERT...")
trainer.train()
print("Training complete!")

# 6. Save + merge LoRA weights
print("\n=== SAVING LORA WEIGHTS ===")
print("""
# Save ONLY LoRA weights (tiny!)
peft_model.save_pretrained(
    "/tmp/lora_weights"
)
# Saves: adapter_config.json + adapter_model.bin
# Size: ~2MB vs 418MB for full BERT!

# Merge for deployment (optional)
merged = peft_model.merge_and_unload()
# Now: LoRA weights merged into base
# → Single model, no adapter overhead
# → Best for production serving!

# Load for inference
from peft import PeftModel
base = AutoModel.from_pretrained('bert-base-uncased')
model = PeftModel.from_pretrained(
    base, "/tmp/lora_weights"
)
""")

# 7. LoRA hyperparameter guide
print("\n=== LORA HYPERPARAMETERS ===")
print("""
rank (r): bottleneck dimension
  r=4:  very efficient, less expressive
  r=16: good balance ← default
  r=64: more expressive, less efficient
  r=128: for complex tasks, large models

alpha: scaling factor (use = r typically)
  alpha=r: standard scaling
  alpha=2r: stronger adaptation

target_modules: which layers to adapt
  Minimum: query, value
  Better:  q, k, v, o (all attention)
  Maximum: + ffn layers

lora_dropout: 0.05-0.1 typical

Learning rate: 10-100× higher than full FT!
  Full FT:  2e-5
  LoRA:     2e-4  (10× higher!)
  Why: fewer params → can use larger steps
""")

# 8. Log to MLflow
mlflow.set_experiment("phase3_lora")
with mlflow.start_run(
        run_name="LoRA_BERT_finetuning"):
    mlflow.log_params({
        "method": "LoRA",
        "rank": 8,
        "alpha": 16,
        "target_modules": "Q+V",
        "trainable_ratio": f"{trainable/total:.2%}",
        "base_model": "bert-base-uncased"
    })
    mlflow.log_metrics({
        "total_params": total,
        "trainable_params": trainable,
        "param_reduction": 1 - trainable/total
    })
    print("\nLoRA experiment logged to MLflow!")

print("\n" + "="*60)
print("LoRA + QLoRA — MASTERED! 🚀")
print("LLM Week Day 2 COMPLETE!")
print("="*60)

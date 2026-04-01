# Day 17 — LoRA Fine-tuning Introduction
# Date: March 29, 2026
# How Databricks fine-tunes LLMs efficiently!

import torch
import torch.nn as nn

print("="*50)
print("LoRA — Low-Rank Adaptation")
print("Fine-tune LLMs with 99% fewer parameters!")
print("="*50)

# 1. LoRA concept demonstration
class LoRALayer(nn.Module):
    """
    LoRA adds two small matrices A and B
    instead of updating the full weight matrix W.

    W_new = W_original + B @ A
    Where:
    - W_original is FROZEN (no gradient)
    - A shape: (rank, input_dim)
    - B shape: (output_dim, rank)
    - rank << min(input_dim, output_dim)
    """
    def __init__(self, in_features, out_features,
                 rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Original weight — FROZEN!
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features),
            requires_grad=False  # ← FROZEN!
        )

        # LoRA matrices — TRAINABLE!
        self.lora_A = nn.Parameter(
            torch.randn(rank, in_features) * 0.01
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank)
        )

    def forward(self, x):
        # Original + LoRA adaptation
        original = x @ self.weight.T
        lora = x @ self.lora_A.T @ self.lora_B.T
        return original + self.scaling * lora

# 2. Compare parameter counts
in_dim, out_dim = 768, 768  # BERT-like dimensions
rank = 8

full_params = in_dim * out_dim
lora_params = rank * in_dim + out_dim * rank

print(f"\nFull fine-tuning parameters: "
      f"{full_params:,}")
print(f"LoRA parameters (rank={rank}):  "
      f"{lora_params:,}")
print(f"Parameter reduction: "
      f"{(1 - lora_params/full_params)*100:.1f}%!")
print(f"LoRA uses {full_params//lora_params}× "
      f"fewer parameters!")

# 3. Test LoRA layer
lora_layer = LoRALayer(
    in_features=64,
    out_features=64,
    rank=4
)
x = torch.randn(2, 10, 64)  # batch, seq, features
output = lora_layer(x)
print(f"\nInput shape:  {x.shape}")
print(f"Output shape: {output.shape}")

trainable = sum(p.numel() for p in
                lora_layer.parameters()
                if p.requires_grad)
frozen = sum(p.numel() for p in
             lora_layer.parameters()
             if not p.requires_grad)
print(f"Trainable params: {trainable:,}")
print(f"Frozen params:    {frozen:,}")

# 4. LoRA in HuggingFace (PEFT library)
print("\n# In production with PEFT library:")
print("""
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,              # rank
    lora_alpha=32,    # scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA — only 0.1% params trainable!
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || 
# all params: 6,738,415,616 || 
# trainable%: 0.0623%
""")

print("\nThis is exactly how Databricks fine-tunes")
print("LLaMA and Mistral models in Mosaic AI!")

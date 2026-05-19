# Phase 3 Day 11 — RNN + LSTM + GRU
# Date: May 19, 2026
# Sequence modeling from scratch!

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow

print("="*60)
print("RNN + LSTM + GRU — Sequence Modeling")
print("="*60)

"""
RNN — RECURRENT NEURAL NETWORKS

Problem: standard NNs ignore SEQUENCE ORDER
→ "I love dogs" ≠ "dogs love I" to a bag-of-words
→ But sequence matters for language, time series!

RNN solution: maintain HIDDEN STATE
→ Process one token at a time
→ Hidden state carries "memory" forward

RNN equations:
h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y

Where:
  x_t: input at time t
  h_t: hidden state at time t (memory!)
  y_t: output at time t
  W_hh: hidden-to-hidden weights
  W_xh: input-to-hidden weights

KEY: W_hh and W_xh SHARED across time steps!
     → Same weights for all positions
     → Enables processing variable-length sequences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THE VANISHING GRADIENT PROBLEM (again!):

BPTT (Backpropagation Through Time):
∂L/∂h_0 = ∂L/∂h_T * Π_{t=1}^{T} ∂h_t/∂h_{t-1}

Each ∂h_t/∂h_{t-1} ≈ tanh'(.) * W_hh
tanh' ≤ 1 and |W_hh| < 1 typically

For T=100: product of 100 terms < 1
→ Gradient → 0 exponentially fast!
→ RNN forgets long-range dependencies!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LSTM — LONG SHORT-TERM MEMORY:

Solves vanishing gradient with GATES!

4 gates (all use sigmoid → [0,1]):

Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
              "How much of cell state to keep?"

Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
              "How much new info to store?"

Cell gate:    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)
              "What new info to potentially store?"

Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
              "What to output from cell state?"

Cell state update (the "memory highway"!):
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
  f_t ⊙ c_{t-1}: forget part of old memory
  i_t ⊙ g_t:     add new information

Hidden state:
h_t = o_t ⊙ tanh(c_t)

WHY LSTM WORKS:
→ Cell state c_t = "conveyor belt" of memory
→ Gradient flows through addition (not multiplication!)
→ Forget gate near 1 → gradient passes through
→ Can learn dependencies 1000+ steps back!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRU — GATED RECURRENT UNIT:

Simplified LSTM with 2 gates (not 4!):

Reset gate: r_t = σ(W_r · [h_{t-1}, x_t])
            "How much past to forget?"

Update gate: z_t = σ(W_z · [h_{t-1}, x_t])
             "How much to update hidden state?"

Candidate: h̃_t = tanh(W · [r_t ⊙ h_{t-1}, x_t])

Update:    h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

GRU vs LSTM:
→ Fewer parameters (faster training)
→ Similar performance on most tasks
→ No separate cell state (simpler!)
→ GRU merges forget + input gates
"""

# 1. RNN from scratch
class RNNScratch:
    def __init__(self, input_size,
                  hidden_size, output_size):
        self.hidden_size = hidden_size
        # Initialize weights
        self.W_xh = np.random.randn(
            hidden_size, input_size
        ) * 0.01
        self.W_hh = np.random.randn(
            hidden_size, hidden_size
        ) * 0.01
        self.W_hy = np.random.randn(
            output_size, hidden_size
        ) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, inputs):
        """inputs: list of column vectors"""
        h = np.zeros((self.hidden_size, 1))
        outputs = []
        self.cache = {'h': [h], 'x': inputs}

        for x in inputs:
            # h_t = tanh(W_xh*x + W_hh*h + b)
            h = np.tanh(
                self.W_xh @ x +
                self.W_hh @ h +
                self.b_h
            )
            # y_t = W_hy * h + b_y
            y = self.W_hy @ h + self.b_y
            outputs.append(y)
            self.cache['h'].append(h)

        return outputs, h

# Test RNN
rnn = RNNScratch(
    input_size=10,
    hidden_size=32,
    output_size=5
)
# Sequence of 5 timesteps
inputs = [np.random.randn(10, 1)
           for _ in range(5)]
outputs, final_h = rnn.forward(inputs)
print(f"RNN output shape:  {outputs[-1].shape}")
print(f"Hidden state shape: {final_h.shape}")

# 2. LSTM in PyTorch
print("\n=== LSTM IN PYTORCH ===")

class LSTMClassifier(nn.Module):
    def __init__(self, input_size,
                  hidden_size, num_layers,
                  num_classes, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,    # (B, T, F)!
            dropout=dropout if num_layers > 1
                    else 0,
            bidirectional=True   # BiLSTM!
        )
        # *2 for bidirectional
        self.fc = nn.Linear(
            hidden_size * 2, num_classes
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use last timestep output
        # For bidirectional: concat forward+backward
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out)

# Create synthetic sequence data
print("Creating sequence classification task...")
B, T, F = 256, 20, 16  # batch, time, features
n_classes = 3
X_seq = torch.randn(B, T, F)
y_seq = torch.randint(0, n_classes, (B,))

X_tr = X_seq[:200]
y_tr = y_seq[:200]
X_te = X_seq[200:]
y_te = y_seq[200:]

# Train LSTM
lstm_model = LSTMClassifier(
    input_size=F,
    hidden_size=64,
    num_layers=2,
    num_classes=n_classes,
    dropout=0.3
)
optimizer = optim.Adam(
    lstm_model.parameters(), lr=0.001
)
criterion = nn.CrossEntropyLoss()

mlflow.set_experiment("phase3_lstm")
with mlflow.start_run(run_name="BiLSTM_classifier"):
    mlflow.log_param("hidden_size", 64)
    mlflow.log_param("num_layers", 2)
    mlflow.log_param("bidirectional", True)
    mlflow.log_param("seq_len", T)

    for epoch in range(50):
        lstm_model.train()
        optimizer.zero_grad()
        out = lstm_model(X_tr)
        loss = criterion(out, y_tr)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            lstm_model.eval()
            with torch.no_grad():
                pred = lstm_model(X_te).argmax(1)
                acc = (pred == y_te).float().mean()
            print(f"Epoch {epoch:3d}: "
                  f"loss={loss:.4f} "
                  f"val_acc={acc:.4f}")
            mlflow.log_metric("loss", loss.item(),
                               step=epoch)
            mlflow.log_metric("val_acc", acc.item(),
                               step=epoch)

# 3. GRU comparison
print("\n=== GRU vs LSTM ===")

class GRUClassifier(nn.Module):
    def __init__(self, input_size,
                  hidden_size, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size,
            num_layers=2, batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(
            hidden_size * 2, num_classes
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

gru_model = GRUClassifier(F, 64, n_classes)
optimizer_gru = optim.Adam(
    gru_model.parameters(), lr=0.001
)

for epoch in range(50):
    gru_model.train()
    optimizer_gru.zero_grad()
    out = gru_model(X_tr)
    loss = criterion(out, y_tr)
    loss.backward()
    optimizer_gru.step()

gru_model.eval()
with torch.no_grad():
    pred = gru_model(X_te).argmax(1)
    gru_acc = (pred == y_te).float().mean()

lstm_model.eval()
with torch.no_grad():
    pred = lstm_model(X_te).argmax(1)
    lstm_acc = (pred == y_te).float().mean()

print(f"LSTM accuracy: {lstm_acc:.4f}")
print(f"GRU accuracy:  {gru_acc:.4f}")
lstm_params = sum(
    p.numel() for p in lstm_model.parameters()
)
gru_params = sum(
    p.numel() for p in gru_model.parameters()
)
print(f"LSTM params:   {lstm_params:,}")
print(f"GRU params:    {gru_params:,}")

# 4. Practical guide
print("\n=== WHEN TO USE WHAT ===")
print("""
Vanilla RNN:
→ Short sequences only (<10 steps)
→ Educational purposes
→ Never in production!

LSTM:
→ Long sequences (100+ steps)
→ When you need to remember far back
→ NLP, time series forecasting
→ More parameters → slower training

GRU:
→ When LSTM is too slow/complex
→ Similar performance with less compute
→ Good default for sequences

Bidirectional:
→ When you have access to full sequence
→ NOT for real-time prediction
→ NLP classification, translation

Stacked (multi-layer):
→ More capacity for complex patterns
→ Use dropout between layers!
→ Usually 2-3 layers is enough

Modern replacement: Transformers!
→ Parallelizable (not sequential)
→ Better long-range dependencies
→ Now the default for most NLP tasks
""")

print("\n" + "="*60)
print("RNN + LSTM + GRU — MASTERED! 🔄")
print("="*60)

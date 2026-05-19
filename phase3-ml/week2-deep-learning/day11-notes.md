## RNN Core Equations
h_t = tanh(W_xh*x_t + W_hh*h_{t-1} + b_h)
y_t = W_hy*h_t + b_y
Shared weights across time → variable length!
Problem: vanishing gradient for long sequences

## LSTM Gates (memorize for interviews!)
f_t = σ(W_f·[h_{t-1},x_t]) ← forget
i_t = σ(W_i·[h_{t-1},x_t]) ← input
g_t = tanh(W_g·[h_{t-1},x_t]) ← candidate
o_t = σ(W_o·[h_{t-1},x_t]) ← output

Cell: c_t = f_t⊙c_{t-1} + i_t⊙g_t
Hide: h_t = o_t⊙tanh(c_t)

The + in cell update = gradient highway!

## GRU Gates (simpler!)
r_t = σ(W_r·[h_{t-1},x_t]) ← reset
z_t = σ(W_z·[h_{t-1},x_t]) ← update
h̃_t = tanh(W·[r_t⊙h_{t-1},x_t])
h_t = (1-z_t)⊙h_{t-1} + z_t⊙h̃_t

## PyTorch LSTM Setup
nn.LSTM(input_size, hidden_size,
        num_layers, batch_first=True,
        bidirectional=True)
Output: (output, (h_n, c_n))
output shape: (B, T, H*2) if bidirectional

## Linked List Patterns
Reverse:      prev=None, curr=head
              track next before breaking link
Find middle:  slow/fast pointers
Merge sorted: two pointer merge
Reorder:      all three combined!

## 3 Days to CV Week!
LSTM → Attention (tomorrow)
Attention → Transformers
Transformers → Vision Transformers (ViT)
ViT → YOLOv8 + CLIP in CV week!
Everything connects! 🎯

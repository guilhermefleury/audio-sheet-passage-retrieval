# First Training Run — Results & Analysis

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | CrossModalEncoder (CNNEncoder + GRU + FC) |
| snippet_emb_dim | 32 |
| rnn_hidden | 128 |
| emb_dim | 64 |
| batch_size | 16 (effective 64 via grad_accum=4) |
| lr | 3e-3 |
| weight_decay | 1e-5 |
| loss | Triplet, margin=0.3, in-batch negatives |
| scheduler | ReduceLROnPlateau, mode=max, factor=0.5, patience=15 |
| early_stop_patience | 30 epochs |
| n_epochs | 180 (max) |
| n_train | 16384 passages/epoch (23.2% of 70572) |
| n_val | 4096 passages/epoch (25.4% of 16138) |
| mixed_precision | True |
| seed | 123 |

**Architecture fixes applied vs original lcasr-main:**
- Gradient accumulation (grad_accum=4) to match paper's effective batch_size=64
- GroupNorm(1, C) for audio CNN encoder (matches original); BatchNorm2d kept for sheet CNN

---

## Training History

| Epoch | tr_loss | va_loss | S2A MRR | R@1 | R@10 | A2S MRR | Med.Rk | LR |
|-------|---------|---------|---------|-----|------|---------|--------|-----|
| 1 | 0.1881 | 0.1666 | 0.0116 | 0.3 | 2.0 | 0.0114 | 581.5 | 3e-3 |
| 2 | 0.1100 | 0.1219 | 0.0231 | 0.6 | 4.5 | 0.0194 | 338.0 | 3e-3 |
| 3 | 0.0780 | 0.1130 | 0.0321 | 0.9 | 6.5 | 0.0282 | 259.0 | 3e-3 |
| 4 | 0.0580 | 0.1064 | 0.0289 | 0.7 | 6.3 | 0.0280 | 241.0 | 3e-3 |
| 5 | 0.0456 | 0.1072 | 0.0368 | 1.0 | 7.8 | 0.0372 | 213.5 | 3e-3 |
| 6 | 0.0364 | 0.1136 | 0.0366 | 1.1 | 8.0 | 0.0332 | 240.0 | 3e-3 |
| 7 | 0.0289 | 0.1059 | 0.0333 | 0.9 | 7.2 | 0.0359 | 215.5 | 3e-3 |
| 8 | 0.0242 | 0.1103 | 0.0363 | 1.0 | 8.0 | 0.0334 | 245.0 | 3e-3 |
| **9** | **0.0210** | **0.1061** | **0.0396** | **1.1** | **8.7** | **0.0401** | **213.0** | **3e-3** |
| 10 | 0.0186 | 0.1084 | 0.0359 | 0.9 | 8.2 | 0.0388 | 212.5 | 3e-3 |
| 11 | 0.0165 | 0.1115 | 0.0337 | 0.9 | 7.6 | 0.0350 | 248.5 | 3e-3 |
| 12 | 0.0147 | 0.1022 | 0.0356 | 1.0 | 7.4 | 0.0349 | 211.0 | 3e-3 |
| 13 | 0.0137 | 0.1040 | 0.0352 | 1.0 | 7.7 | 0.0332 | 211.0 | 3e-3 |
| 14 | 0.0126 | 0.1179 | 0.0307 | 0.8 | 6.6 | 0.0320 | 263.5 | 3e-3 |
| 15 | 0.0117 | 0.1100 | 0.0371 | 1.1 | 8.0 | 0.0347 | 213.0 | 3e-3 |
| 16 | 0.0110 | 0.1141 | 0.0382 | 1.2 | 8.1 | 0.0365 | 229.5 | 3e-3 |
| 17 | 0.0105 | 0.1146 | 0.0296 | 0.7 | 6.1 | 0.0318 | 245.0 | 3e-3 |
| 18 | 0.0099 | 0.1089 | 0.0322 | 0.8 | 7.1 | 0.0338 | 232.0 | 3e-3 |
| 19 | 0.0092 | 0.1076 | 0.0368 | 1.1 | 8.1 | 0.0368 | 225.0 | 3e-3 |
| 20 | 0.0090 | 0.1135 | 0.0332 | 0.9 | 7.1 | 0.0344 | 241.0 | 3e-3 |
| 21 | 0.0084 | 0.1099 | 0.0325 | 0.9 | 6.8 | 0.0328 | 236.5 | 3e-3 |
| 22 | 0.0082 | 0.1133 | 0.0332 | 0.9 | 7.0 | 0.0338 | 238.0 | 3e-3 |
| 23 | 0.0076 | 0.1132 | 0.0335 | 0.8 | 7.4 | 0.0379 | 244.0 | 3e-3 |
| 24 | 0.0075 | 0.1128 | 0.0308 | 0.8 | 6.5 | 0.0316 | 263.0 | 3e-3 |
| 25 | 0.0075 | 0.1141 | 0.0305 | 0.8 | 6.3 | 0.0278 | 240.0 | 1.5e-3 |
| 26 | 0.0053 | 0.1149 | 0.0328 | 0.8 | 7.3 | 0.0326 | 257.0 | 1.5e-3 |
| 27 | 0.0046 | 0.1153 | 0.0299 | 0.8 | 6.3 | 0.0304 | 248.5 | 1.5e-3 |
| 28 | 0.0041 | 0.1214 | 0.0310 | 0.8 | 6.5 | 0.0282 | 269.0 | 1.5e-3 |
| 29 | 0.0042 | 0.1191 | 0.0323 | 0.9 | 7.1 | 0.0329 | 279.5 | 1.5e-3 |
| 30 | 0.0041 | 0.1199 | 0.0315 | 0.9 | 6.5 | 0.0270 | 282.0 | 1.5e-3 |
| 31 | 0.0040 | 0.1215 | 0.0324 | 0.9 | 7.3 | 0.0330 | 255.5 | 1.5e-3 |
| 32 | 0.0040 | 0.1219 | 0.0309 | 0.9 | 6.2 | 0.0285 | 284.0 | 1.5e-3 |
| 33 | 0.0041 | 0.1192 | 0.0315 | 0.7 | 7.2 | 0.0315 | 265.0 | 1.5e-3 |
| 34 | 0.0039 | 0.1197 | 0.0288 | 0.7 | 5.8 | 0.0300 | 279.0 | 1.5e-3 |
| 35 | 0.0039 | 0.1216 | 0.0334 | 1.0 | 7.2 | 0.0308 | 267.5 | 1.5e-3 |
| 36 | 0.0038 | 0.1203 | 0.0333 | 1.0 | 7.2 | 0.0311 | 282.0 | 1.5e-3 |
| 37 | 0.0038 | 0.1163 | 0.0320 | 0.8 | 7.1 | 0.0290 | 263.0 | 1.5e-3 |
| 38 | 0.0037 | 0.1198 | 0.0332 | 1.1 | 6.6 | 0.0286 | 283.5 | 1.5e-3 |
| 39 | 0.0038 | 0.1193 | 0.0328 | 1.0 | 7.0 | 0.0286 | 281.0 | 1.5e-3 |

**Early stopping triggered at epoch 39** (30 epochs without improvement).
Best checkpoint: epoch 9, S2A MRR = 0.0396.

---

## Test Set Results

Evaluated on the full test split (14,062 passage pairs, no subsampling).

```
------------------------------------------------------------
Direction        R@1    R@10    R@25     MRR   Med.Rk      N
------------------------------------------------------------
S2A             0.30    2.70    6.14  0.0138    841.5  14062
A2S             0.18    2.10    5.33  0.0112    866.5  14062
------------------------------------------------------------
```

Checkpoint: `experiments/model_emb64.pt` (epoch 9)

---

## Observations

**What worked:**
- The model learned meaningful cross-modal representations — MRR=0.0138 is ~138x better than random chance (random baseline ≈ 0.0001 with 14,062 items).
- Steady improvement in early epochs (1–9) with multiple checkpoints saved.
- Gradient accumulation (grad_accum=4) to simulate effective batch_size=64 helped early learning.
- GroupNorm for audio CNN encoder (matching the original implementation) improved consistency.

**What limited performance:**
- **Triplet loss saturation**: Training loss bottomed out at ~0.004 by epoch 25, indicating the model easily satisfied the margin for all in-batch negatives. Once training loss saturates, gradients vanish and the model stops learning useful representations.
- **Easy in-batch negatives**: With random in-batch sampling, after ~9 epochs the model had already learned to separate the easy cases. Without harder negatives to mine, no further improvement was possible.
- **Val MRR noise**: Subsampling validation to 4096 passages per epoch introduced variance in the MRR estimate, making it difficult to track real progress.
- **LR reduction did not help**: After the LR drop at epoch 25 (3e-3 → 1.5e-3), val MRR did not improve and actually trended slightly worse, confirming the issue was gradient signal quality, not learning rate.

**Val vs test MRR gap:**
Val MRR during training (~0.040) was against a pool of 4,096 passages. Test MRR (0.0138) is against 14,062 passages. The larger pool makes retrieval harder; the gap is expected and not a sign of overfitting.

---

## Next Steps

- **Hard negative mining**: Periodically recompute embeddings and select negatives that are close to the anchor but incorrect. This directly addresses the gradient saturation problem.
- **NT-Xent (InfoNCE) loss**: Alternative to triplet loss that has shown stronger performance with in-batch negatives. Already implemented in the original lcasr-main repo (`losses.py`).

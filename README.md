# Quantization-Aware Healing — Small-Scale Reproduction

Small-scale reproduction of **Quantization-Aware Healing (QAH)**, based on the 2026 paper by **Multiverse Computing**:

> **Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs**
> Multiverse Computing, 2026
> [arXiv:2608.20953](https://arxiv.org/abs/2608.20953)

The goal of this project is to study whether distilling a **compressed and quantized student directly from the original uncompressed model** can recover more performance than conventional Quantization-Aware Training (QAT) or Quantization-Aware Distillation (QAD).

The experiments are designed to run on a CPU-only machine.

---

## Model

The project uses:

**[roneneldan/TinyStories-8M](https://huggingface.co/roneneldan/TinyStories-8M)**

TinyStories-8M is a small GPT-Neo causal language model with 8 Transformer blocks, making it suitable for a lightweight reproduction of structural compression and quantization experiments.

The model is structurally compressed from:

```text
8 Transformer layers
        ↓
4 Transformer layers
```

by keeping layers:

```text
[0, 2, 4, 6]
```

---

## Experimental Pipeline

```text
Original TinyStories-8M
        │
        │ Structural pruning
        ▼
Pruned 4-layer model
        │
        │ KL distillation
        ▼
Recovered FP32 model
        │
        ├── PTQ
        ├── QAT
        ├── QAD
        └── QAH
```

### Recovery

After pruning, the student is recovered using knowledge distillation:

$$
KL(p_{\text{original 8L}} \parallel p_{\text{student 4L}})
$$

The original model remains frozen while the compressed student is trained to reproduce its next-token probability distributions.

---

## Quantization

The current implementation uses **fake symmetric INT4 quantization**.

For each weight tensor:

$$
s = \frac{\max |W|}{7}
$$

$$
q = \text{clip}(\text{round}(W/s), -8, 7)
$$

$$
\hat W = qs
$$

The stored parameters remain FP32, but the forward pass uses the quantized/dequantized weights.

A **Straight-Through Estimator (STE)** allows gradients to update the underlying FP32 parameters.

This simulates the numerical effects of INT4 quantization without requiring native INT4 CPU kernels.

> The original Multiverse Computing paper uses **MXFP4**, so the INT4 implementation in this repository is a simplification.

---

## Training Methods

### PTQ — Post-Training Quantization

```text
Recovered FP32 model
        ↓
Fake INT4
```

No additional training is performed.

### QAT — Quantization-Aware Training

The fake-INT4 student is trained using normal next-token Cross-Entropy:

$$
L = CE(y, p_S)
$$

### QAD — Quantization-Aware Distillation

Teacher:

```text
Recovered 4-layer FP32 model
```

Student:

```text
Recovered 4-layer fake-INT4 model
```

Objective:

$$
KL(p_{\text{recovered}} \parallel p_S)
$$

### QAH — Quantization-Aware Healing

QAH instead returns to the **original 8-layer model** as teacher:

$$
KL(p_{\text{original}} \parallel p_S)
$$

The key comparison is therefore:

```text
QAD → compressed teacher → quantized student

QAH → original teacher → quantized compressed student
```

The hypothesis is that the original model retains information that was already lost during structural compression.

---

## Evaluation

All models are evaluated on the same pre-tokenized TinyStories validation set.

Main metrics:

* Cross-Entropy (CE) ↓
* Perplexity (PPL) ↓
* Quantization damage recovery %

Perplexity is calculated as:

$$
PPL = e^{CE}
$$

The main models compared are:

| Model     | Precision | Training                  |
| --------- | --------- | ------------------------- |
| Original  | FP32      | —                         |
| Pruned    | FP32      | —                         |
| Recovered | FP32      | KL from original          |
| PTQ       | Fake INT4 | None                      |
| QAT       | Fake INT4 | Cross-Entropy             |
| QAD       | Fake INT4 | KL from recovered teacher |
| QAH       | Fake INT4 | KL from original teacher  |

---

## Running the Project

```bash
python src/prepare_data.py
python src/tokenize_data.py
python src/prune_model.py
python src/recover.py
python src/evaluate_quantized.py
python src/train_qat.py
python src/train_qad.py
python src/train_qah.py
python src/compare_results.py
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
qah_reproduction/
├── src/
├── data/
├── checkpoints/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

`data/` and `checkpoints/` are generated locally and should normally not be committed to Git.

---

## Limitations

This is a **small-scale mechanistic reproduction**, not an exact reproduction of the Multiverse Computing experiments.

Major differences include:

* TinyStories-8M instead of a large LLM
* 8 → 4 layer structural compression
* fake INT4 instead of MXFP4
* small training dataset
* CPU-only training
* TinyStories perplexity rather than a large benchmark suite

The objective is to reproduce and study the **QAH training principle**, rather than match the paper's reported benchmark numbers.

---

## References

**Multiverse Computing — Quantization-Aware Healing**
[Quantization-Aware Healing: A Practical Recipe for Recovering Compressed, 4-Bit LLMs](https://arxiv.org/abs/2608.20953)

**Model**
[roneneldan/TinyStories-8M — Hugging Face](https://huggingface.co/roneneldan/TinyStories-8M)

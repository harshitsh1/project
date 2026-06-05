# SDPO v5 — Self-Distillation Policy Optimization

> Fine-tuning a small language model using reinforcement learning with self-teacher KL distillation and GRPO-based optimization.

**GitHub:** [https://github.com/harshitsh1/project.git](https://github.com/harshitsh1/project.git)

---

## Problem Statement

Implement and evaluate True Self-Distillation Policy Optimization (SDPO) for improving response quality of a language model using reinforcement learning with self-teacher KL distillation and GRPO-based optimization.

---

## Overview

SDPO trains a model to improve its own responses by treating the same model in two roles simultaneously:

- **Student** — the model as it normally answers a question
- **Self-Teacher** — the same model, but re-prompted with feedback about what went wrong

The student is then trained to match the teacher's richer, feedback-informed output distribution. This removes the need for any external stronger model — the model bootstraps its own improvement.

---

## Architecture

```
Question
   │
   ├──► Student (bare prompt)          ──► KL divergence ──► SDPO loss
   │                                              ▲
   └──► Self-Teacher (prompt + feedback) ─────────┘ (no grad)

Combined loss = λ × GRPO_loss + (1−λ) × scale × SDPO_KL_loss
```

### Key Components

| Component | Description |
|---|---|
| `SDPOTrainer` | Extends GRPOTrainer; injects SDPO KL loss alongside GRPO |
| `EMATeacher` | Exponential moving average copy of student weights to stabilize training |
| `NLIEnvironment` | NLI-based scorer using cross-encoder + token F1 for reward computation |
| `compute_sdpo_kl` | Top-K KL/JSD divergence between student and teacher distributions |
| `build_self_teacher_context` | Constructs the enriched teacher prompt (Table 2 of SDPO paper) |

---

## Dataset

**SQuAD v2** — Stanford Question Answering Dataset v2

- Contains both answerable and unanswerable question-answer pairs
- Unanswerable questions (15% of training data) teach the model to say "unanswerable" instead of hallucinating
- Prompts include the full passage context so the model learns grounded extraction

```
Context: {passage}

Question: {question}
Answer briefly based on the context.
If the answer is not in the context, say 'unanswerable'.
```

---

## Model

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`

**Fine-tuning method:** LoRA (Low-Rank Adaptation)

| LoRA parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Dropout | 0.05 |
| Trainable params | ~13M (out of 1.5B) |

---

## Training Configuration

| Parameter | Value | Reason |
|---|---|---|
| `grpo_lambda` | 0.9 | 90% GRPO + 10% SDPO — hybrid needed for small 1.5B model |
| `divergence` | JSD | Jensen-Shannon is smoother than KL for small models |
| `ema_alpha` | 0.05 | Slow teacher tracking prevents KL collapse |
| `learning_rate` | 2e-6 | Conservative to avoid reward hacking |
| `num_generations` | 2 | Two rollouts per prompt for GRPO advantage estimation |
| `beta` | 0.1 | KL penalty to reference model |
| `top_k_distill` | 20 | Only top-20 logits for memory-efficient KL approximation |
| `max_new_tokens` | 128 | Enough headroom for SQuAD-style short answers |

---

## Loss Function

```
L_total = λ · L_GRPO + (1 − λ) · scale · L_SDPO

L_SDPO = Σ_t JSD( πθ(·|x, y<t) || stopgrad(πθ_EMA(·|x, f, y<t)) )
```

Where:
- `x` = question, `y` = student response, `f` = environment feedback
- `πθ_EMA` = EMA-smoothed teacher (same architecture, slowly-updated weights)
- `stopgrad` prevents gradients from flowing through the teacher

---

## Reward Function

Rewards are computed by `NLIEnvironment` using three signals:

1. **Exact match** — if the ground truth string appears verbatim in the response (reward = 1.0)
2. **Token F1** — SQuAD-style overlap with stemming for morphological variants
3. **NLI entailment** — cross-encoder (`nli-deberta-v3-small`) checks semantic consistency

For **unanswerable questions**, reward = 1.0 if the model says "unanswerable" or refuses, 0.0 if it hallucinate an answer.

---

## Results (V5)

| Metric | Baseline | Final | Δ |
|---|---|---|---|
| Avg Reward | 0.821 | 0.813 | −0.008 |
| Exact Match | 0.800 | 0.770 | −0.030 |
| Refusal Rate | ~0.00 | ~0.00 | stable |
| Hallucination Rate | ~0.50 | ~0.52 | slight increase |
| Avg Response Length | 12.9w | 12.7w | −0.200 |

**Observation:** Reward is nearly flat (Δ−0.008) rather than clearly improving. The paper confirms this is expected for Qwen2.5-1.5B — pure SDPO underperforms GRPO at this scale; the hybrid lambda=0.9 mode used here partially mitigates this. Larger models (≥7B) show clear gains.

**Training batch reward** trended upward (0.65 → 0.85+), indicating the model is learning signal within batches, but this does not fully transfer to held-out evaluation.

---

## File Structure

```
sdpo_v5.py                   # Main training script
outputs/
  v5_sdpo_grpo/
    best_model/              # Checkpoint with highest eval reward
    eval_log.json            # Per-step evaluation metrics
    train_log.json           # Per-step training metrics (loss, grad norm, entropy, KL)
    training_plot.png        # 9-panel diagnostic plot
```

---

## How to Run

```bash
# Smoke test (5 prompts, verifies the full loop works)
python sdpo_v5.py --mode smoke

# Small training run (120 SQuAD samples)
python sdpo_v5.py --mode small

# Large training run (1000 SQuAD samples)
python sdpo_v5.py --mode large

# Regenerate plots from saved logs
python sdpo_v5.py --mode plot

# Override key hyperparameters
python sdpo_v5.py --mode small --lambda 1.0        # pure GRPO baseline
python sdpo_v5.py --mode small --lambda 0.0        # pure SDPO
python sdpo_v5.py --mode small --divergence kl     # use KL instead of JSD
python sdpo_v5.py --mode small --no-ema            # disable EMA teacher
```

---

## Dependencies

```bash
pip install torch transformers trl peft datasets bitsandbytes
pip install sentence-transformers   # for NLI scorer
pip install matplotlib psutil
```

---

## Design Decisions & Why

**Why JSD instead of KL?**
Forward KL(student||teacher) can blow up when the teacher assigns near-zero probability to a token the student likes. JSD is bounded [0, log2] and numerically stable.

**Why EMA teacher?**
Without EMA, after a few gradient steps the teacher and student are nearly identical, the KL collapses to ~0, and you get random gradient spikes. EMA keeps the teacher slightly behind the student so there's always a meaningful learning signal.

**Why lambda=0.9 for 1.5B?**
The original paper (Figure 17) shows Qwen2.5-1.5B underperforms GRPO with pure SDPO. The hybrid (mostly GRPO + small SDPO nudge) is the recommended setting for models below 7B.

---

## References

- Hübotter et al. (2026). *Reinforcement Learning via Self-Distillation.* arXiv:2601.20802
- Shao et al. (2024). *DeepSeekMath: GRPO.* arXiv:2402.03300
- Rajpurkar et al. (2018). *SQuAD 2.0.* arXiv:1806.03822

# SDPO v2 — Manual Self-Distillation Policy Optimization

> True SDPO implemented as a raw PyTorch training loop — no GRPOTrainer, no external teacher. The same model plays both student and self-teacher roles simultaneously using two forward passes per step.

**GitHub:** [https://github.com/harshitsh1/project.git](https://github.com/harshitsh1/project.git)

---

## Problem Statement

Implement the core SDPO algorithm from scratch without any RL framework wrapper. The model generates a response (student), receives heuristic feedback, and is trained to match the distribution of the same model re-prompted with that feedback (self-teacher). This version establishes the fundamental SDPO training loop that all later versions build on.

---

## What Changed from v1 → v2

| Feature | v1 | v2 |
|---|---|---|
| Algorithm | DPO (preference pairs) | **True SDPO (KL distillation)** |
| Teacher | External 7B model | **Same model (self-teacher)** |
| Training loop | HuggingFace DPOTrainer | **Manual PyTorch loop** |
| Loss function | DPO contrastive loss | **Top-K KL divergence** |
| Feedback | Heuristic score only | **Rich text feedback to self-teacher** |
| External model needed | Yes (7B teacher) | **No — single 1.5B model** |
| Dataset | UltraFeedback pairs | **UltraFeedback prompts** |

v2 is the first version that implements true SDPO as described in the paper — one model, two contexts, KL loss.

---

## How SDPO Works (v2)

Each training step does the following:

```
Step 1: Generate response
  model.eval() → response = generate(question)

Step 2: Get feedback
  reward, feedback = SimpleEnvironment.evaluate(question, response)

Step 3: Two forward passes through the same model
  ┌─────────────────────────────────────────────────────────────┐
  │  Student context:   [question] + response                   │
  │  Teacher context:   [question + feedback + hint] + response │
  │                                                             │
  │  s_logits = model(student_ids)      ← grad ON              │
  │  t_logits = model(teacher_ids)      ← grad OFF             │
  └─────────────────────────────────────────────────────────────┘

Step 4: Compute KL divergence (top-K approximation)
  loss = KL(student_probs || teacher_probs)  over top-50 tokens

Step 5: Backprop and update
  loss.backward() → optimizer.step()
```

The key insight is that the teacher sees richer context (what went wrong + how to fix it), so its token distribution is better. The student is trained to match it — learning from its own mistakes without any external model.

---

## SimpleEnvironment — Feedback Generation

`SimpleEnvironment` evaluates each response and produces rich text feedback for the self-teacher:

| Condition | Reward | Feedback given to self-teacher |
|---|---|---|
| Empty / < 5 words | 0.0 | "Response is too short. Provide a full answer." |
| Refusal detected | 0.0 | "You refused. Provide a direct response." |
| Hallucinated URL/image | 0.0 | "Contains hallucinated links. Plain text only." |
| Trigram repetition > 40% | 0.0 | "Highly repetitive. Use diverse vocabulary." |
| < 15 words | 0.0 | "Too brief. Provide a comprehensive explanation." |
| Passes all checks, unstructured | 0.8 | "Correct! Consider using bullet points or headers." |
| Passes all checks, structured | 1.0 | "Correct! Detailed, non-repetitive, valid answer." |

The feedback text is inserted into the self-teacher's prompt (Table 2 of the SDPO paper), allowing the teacher to see what went wrong and produce a better-informed distribution.

---

## Dataset

**UltraFeedback** (`trl-lib/ultrafeedback_binarized`) loaded via streaming.

- Prompts are extracted from the `chosen` messages (user turns only)
- Truncated to 500 characters to avoid very long prompts
- Shuffled each epoch for diversity
- Falls back to 20 hand-crafted CS/ML questions if download fails

No ground-truth answers are used — SimpleEnvironment scores based on response quality heuristics alone, making it dataset-agnostic.

---

## Model

**Base:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Fine-tuning:** LoRA

| Parameter | Value |
|---|---|
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Trainable params | ~7M of 1.5B |
| Optimizer | AdamW 8-bit (bitsandbytes) |
| Scheduler | Cosine with 10 warmup steps |

The same LoRA weights are used for both student and self-teacher passes — there is no separate teacher model copy. The teacher's output is computed `with torch.no_grad()` to prevent gradients from flowing through it.

---

## Training Configuration

| Parameter | Value | Note |
|---|---|---|
| `max_new_tokens` | 128 | Full-length responses |
| `top_k_distill` | 50 | Top-50 logits for KL (6GB GPU safe) |
| `learning_rate` | 1e-5 | Standard SDPO LR |
| `grad_accum` | 4 | Effective batch = 4 |
| `num_epochs` | 3 | Full passes through all prompts |
| `max_grad_norm` | 1.0 | Gradient clipping |
| `temperature` | 0.8 | Student generation temperature |

---

## Loss Function

```
L_SDPO = (1/T) * Σ_t  KL( π_θ(·|x, y<t)  ||  stopgrad(π_θ(·|x, f, y<t)) )

Top-K approximation (K=50):
  L ≈ Σ_{i∈topK} p_s(i) * log(p_s(i)/p_t(i))
     + p_s(tail) * log(p_s(tail)/p_t(tail))
```

Where `x` = question, `y` = student response, `f` = SimpleEnvironment feedback, `T` = response length. The tail term accounts for all probability mass outside the top-K tokens.

---

## Training Results (v2)

The plot shows two panels over ~900 steps (300 prompts × 3 epochs):

### Training Loss

Loss oscillates in the range **0.75 – 1.90** throughout training with no clear downward trend. This high and noisy loss is expected for v2 for three reasons:

1. **No ground truth** — SimpleEnvironment gives reward=0.8 for almost any non-empty, non-repetitive response, so the feedback text is often `"Correct! Consider using bullet points..."` — which is only weakly informative to the self-teacher. The KL between student and teacher is therefore large and noisy.

2. **No EMA teacher** — the teacher weights are identical to the student at each step. As the student updates, the teacher moves with it, creating a moving target. This instability is fixed in v5 with EMA teacher stabilization.

3. **No GRPO baseline** — without a reward-weighted advantage signal, the model has no objective to minimize loss on high-reward responses more aggressively than low-reward ones.

### Batch Reward

Reward oscillates between **0.0 and 1.0** with most values at **0.8 or 1.0** from step ~150 onward. The early 0.0 values correspond to refusals or very short responses from the untrained model; these disappear quickly as SDPO feedback conditioning suppresses refusal behavior. The remaining 0.0 spikes later in training correspond to UltraFeedback prompts where the model occasionally generates repetitive or very short responses. The dominance of 0.8–1.0 rewards confirms the model is producing valid, detailed responses throughout training.

---

## File Structure

```
sdpo_v2.py
outputs/
  v2_sdpo_manual/
    checkpoint-50/       # saved every 50 steps
    checkpoint-100/
    ...
    (final model)        # saved at end of training
    training_log.json    # step, loss, reward, lr per logging interval
    training_plot.png    # 2-panel: loss + batch reward
```

---

## How to Run

```bash
# Smoke test — 5 steps, verifies full SDPO loop
python sdpo_v2.py --mode smoke

# Small training run (20 prompts, 3 epochs = 60 steps)
python sdpo_v2.py --mode small

# Large training run (300 prompts, 3 epochs = 900 steps)
python sdpo_v2.py --mode large

# Regenerate plot from saved log
python sdpo_v2.py --mode plot
```

---

## Limitations of v2 (Fixed in Later Versions)

| Problem | Impact | Fixed in |
|---|---|---|
| No train/test split | Cannot measure generalization | v3 |
| No verifiable eval | exact_match metric unavailable | v3 |
| SimpleEnvironment only | Coarse feedback, high loss variance | v3 (SemanticEnv), v4 (NLI) |
| No EMA teacher | Moving target, training instability | v5 |
| No GRPO baseline | No reward-weighted advantage signal | v5 |
| No hard questions | Model not challenged on CS/ML topics | v3 |

---

## Dependencies

```bash
pip install torch transformers datasets peft accelerate
pip install bitsandbytes       # 8-bit AdamW optimizer
pip install matplotlib psutil
```

Or use the shared `requirements.txt` at the repo root.

---

## Comparison Across Versions

| Version | Key addition | Teacher | Loss | Dataset |
|---|---|---|---|---|
| v1 | DPO with external teacher | External 7B | DPO contrastive | UltraFeedback |
| **v2** | **True SDPO manual loop** | **Self (no EMA)** | **Top-K KL** | **UltraFeedback** |
| v3 | Train/test split + hard Qs | Self (no EMA) | Top-K KL | Fallback + UF |
| v4 | NLI entailment scorer | Self (no EMA) | Top-K KL | Fallback + UF |
| v5 | GRPO + EMA teacher | Self + EMA | JSD hybrid | SQuAD v2 |

---

## References

- Hübotter et al. (2026). *Reinforcement Learning via Self-Distillation.* arXiv:2601.20802
- Shao et al. (2024). *DeepSeekMath: GRPO.* arXiv:2402.03300

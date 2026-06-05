import os, sys, time, re

# -- ALL DOWNLOADS GO HERE -- must be before any other import --
MODELS_DIR = r"D:\deep_learning\models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"]                       = MODELS_DIR
os.environ["HF_HUB_CACHE"]                  = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"]         = MODELS_DIR
os.environ["TORCH_HOME"]                    = MODELS_DIR
os.environ["BNB_CACHE_DIR"]                 = MODELS_DIR
os.environ["TRITON_CACHE_DIR"]              = MODELS_DIR
os.environ["XDG_CACHE_HOME"]               = MODELS_DIR
os.environ["HF_DATASETS_CACHE"]             = MODELS_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"
print(f"[CACHE] All caches -> {MODELS_DIR}")

# -- NOW import everything else --
import torch
import json
import gc
import random
import psutil
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig, get_peft_model, TaskType



# --- Configuration ---

@dataclass
class SDPOConfig:
    teacher_model  : str   = "Qwen/Qwen2.5-7B-Instruct"
    student_model  : str   = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir     : str   = r"D:\deep_learning\outputs\v1_sdpo_teacher"

    use_small_data : bool  = False   # True = small_n samples, False = large_n
    small_n        : int   = 20
    large_n        : int   = 300

    # Teacher generation
    num_candidates : int   = 3       # 3 candidates: wider diversity, still fast
    max_new_tokens : int   = 80      # enough for meaningful responses
    temperature    : float = 0.8
    top_p          : float = 0.9

    # Student DPO
    beta           : float = 0.1
    learning_rate  : float = 2e-5    # slightly lower for stability
    num_epochs     : int   = 3
    batch_size     : int   = 1
    grad_accum     : int   = 8       # larger effective batch = smoother gradients
    max_length     : int   = 512
    max_prompt_len : int   = 256
    warmup_ratio   : float = 0.1     # warmup first 10% of steps

    # LoRA
    lora_r         : int   = 16      # doubled: more capacity
    lora_alpha     : int   = 32      # keep alpha = 2*r
    lora_dropout   : float = 0.05

    # Evaluation
    eval_split     : float = 0.1     # 10% held out for eval
    eval_prompts_n : int   = 10      # number of prompts for before/after comparison

    device: str = field(default_factory=lambda:
                        "cuda" if torch.cuda.is_available() else "cpu")


CFG = SDPOConfig()


def safe_print(text: str, max_len: int = 80) -> str:
    """Make text safe for Windows cp1252 terminal output."""
    return text[:max_len].encode('ascii', 'backslashreplace').decode('ascii')



# --- Helper Functions ---

def ram_status(label=""):
    ram = psutil.virtual_memory()
    vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[MEM] {label} | RAM free: {ram.available/1e9:.1f}GB "
          f"/ {ram.total/1e9:.1f}GB | VRAM: {vram:.2f}GB")


def free_memory(*objects):
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()



# --- Dataset ---

def load_prompts(use_small: bool, small_n: int, large_n: int) -> List[str]:
    """
    Loads prompts from UltraFeedback (64k high-quality instruction prompts).
    Streaming mode: only downloads what you need.
    """
    n = small_n if use_small else large_n
    print(f"[Dataset] Loading UltraFeedback prompts (n={n}, streaming=True)...")

    try:
        ds = load_dataset(
            "trl-lib/ultrafeedback_binarized",
            split="train",
            streaming=True,
        )
        prompts = []
        for row in ds:
            for msg in row["chosen"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"])
                    break
            if len(prompts) >= n:
                break
        print(f"[Dataset] Loaded {len(prompts)} prompts from UltraFeedback.")
        return prompts

    except Exception as e:
        print(f"[Dataset] UltraFeedback failed: {e}")
        print(f"[Dataset] Falling back to built-in prompts.")
        return _fallback_prompts(n)


def _fallback_prompts(n: int) -> List[str]:
    base = [
        "Explain what a neural network is.",
        "What is gradient descent?",
        "Write a Python hello world program.",
        "What is the capital of Japan?",
        "Explain reinforcement learning in simple terms.",
        "What is the difference between supervised and unsupervised learning?",
        "What is 15 multiplied by 17?",
        "How does backpropagation work?",
        "What is an attention mechanism in transformers?",
        "Write a Python function that returns the factorial of n.",
        "Explain what overfitting is.",
        "What is the purpose of a loss function?",
        "What is transfer learning?",
        "Explain the concept of embeddings.",
        "What is the difference between BERT and GPT?",
        "Explain regularization techniques in ML.",
        "What is batch normalization?",
        "Describe the transformer architecture.",
        "How does dropout prevent overfitting?",
        "What is a convolutional neural network?",
    ]
    pool = base * (n // len(base) + 1)
    random.shuffle(pool)
    return pool[:n]



# --- Scoring Heuristic ---

def score_response(prompt: str, response: str) -> float:
    """
    Multi-feature scoring heuristic. Designed to maximize spread between
    candidates so DPO gets a strong training signal.
    
    Features:
      1. Length adequacy (0-2 pts)
      2. Refusal penalty (-3 pts)
      3. Code detection for code prompts (+2 pts)
      4. Definitional keywords (+0.3 each, max 1.5)
      5. Structure bonus: bullets/numbered lists (+0.5)
      6. Vocabulary richness (+0-1 pt)
      7. Hallucination penalty: fake URLs, image markdown (-2 pts)
      8. Repetition penalty (-0-2 pts)
    """
    if not response or len(response.strip()) < 3:
        return -1.0

    resp_lower = response.lower()
    words = response.split()
    n_words = len(words)
    score = 0.0

    # 1. Length adequacy (reward 20-80 words, penalize very short)
    if n_words < 5:
        score -= 1.0
    else:
        score += min(n_words / 40.0, 1.0) * 2.0

    # 2. Refusal penalty
    refusals = ["i don't know", "i cannot", "i'm not sure", "i am not sure",
                "i do not know", "i can't", "as an ai", "i'm unable"]
    if any(p in resp_lower for p in refusals):
        score -= 3.0

    # 3. Code detection for code-related prompts
    code_keywords = ["python", "code", "function", "program", "write",
                     "script", "implement", "c#", "java", "javascript"]
    if any(k in prompt.lower() for k in code_keywords):
        code_markers = ["```", "def ", "return ", "print(", "class ",
                        "import ", "function ", "var ", "const ", "let "]
        if any(k in response for k in code_markers):
            score += 2.0

    # 4. Definitional / explanatory keywords
    explain_words = ["is", "are", "means", "defined as", "refers to",
                     "because", "therefore", "however", "for example",
                     "such as", "in other words", "specifically"]
    matches = sum(1 for w in explain_words if w in resp_lower)
    score += min(matches * 0.3, 1.5)

    # 5. Structure bonus (bullets, numbered lists, headers)
    if any(marker in response for marker in ["- ", "* ", "1.", "2.", "###", "**"]):
        score += 0.5

    # 6. Vocabulary richness (unique words / total words)
    if n_words > 10:
        unique_ratio = len(set(w.lower() for w in words)) / n_words
        score += unique_ratio  # 0.0 to ~1.0

    # 7. Hallucination penalty (fake URLs, image markdown)
    if "![" in response or "https://qwen" in resp_lower or "oss-cn" in resp_lower:
        score -= 2.0

    # 8. Repetition penalty (repeated 3-grams)
    if n_words > 15:
        trigrams = [tuple(words[i:i+3]) for i in range(n_words - 2)]
        unique_trigrams = len(set(trigrams))
        repeat_ratio = 1.0 - (unique_trigrams / len(trigrams))
        score -= repeat_ratio * 2.0  # max -2.0 if all trigrams repeat

    return round(score, 3)



# --- Teacher ---

class TeacherGenerator:
    def __init__(self, cfg: SDPOConfig):
        print(f"[Teacher] Loading {cfg.teacher_model} in 4-bit ...")
        ram_status("before teacher load")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.teacher_model, trust_remote_code=True, cache_dir=MODELS_DIR
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.teacher_model,
            trust_remote_code=True,
            quantization_config=bnb_config,
            device_map={"": 0},
            cache_dir=MODELS_DIR,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # Reset generation_config to suppress Qwen's baked-in warnings
        self.model.generation_config = GenerationConfig()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.cfg = cfg
        ram_status("after teacher load")
        print("[Teacher] Loaded.")

    def generate_candidates(self, prompt: str) -> List[str]:
        """Generate diverse candidates via greedy decoding with varying repetition penalties.
        
        Sampling crashes on 4-bit quantized models (NaN logits -> torch.multinomial crash).
        Wider penalty spread (1.0 vs 1.25 vs 1.5) forces very different generation paths.
        """
        try:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        # Wide penalty spread = very different outputs
        penalties = [1.0, 1.25, 1.5, 1.75]
        candidates = []

        for penalty in penalties[:self.cfg.num_candidates]:
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=self.cfg.max_new_tokens,
                        do_sample=False,
                        repetition_penalty=penalty,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                raw = self.tokenizer.decode(
                    out[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                # Clean up artifacts from high repetition penalties
                cleaned = re.sub(r'!\s+', ' ', raw)
                cleaned = re.sub(r'([!?.,;:]){3,}', r'\1', cleaned)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if len(cleaned) > 5:
                    candidates.append(cleaned)
            except RuntimeError as e:
                print(f"    [WARN] Generation failed (penalty={penalty}): {e}")

        if not candidates:
            candidates = [""]

        print(f"    Sample: {safe_print(candidates[0])!r}")
        return candidates

    def unload(self):
        print("[Teacher] Unloading...")
        free_memory(self.model, self.tokenizer)
        ram_status("after teacher unload")



# --- Build SDPO Dataset ---

def build_sdpo_dataset(
    teacher: TeacherGenerator,
    prompts: List[str],
    eval_split: float = 0.1,
) -> Tuple[Dataset, Dataset, List[Dict]]:
    """Returns (train_dataset, eval_dataset, raw_log)."""
    records, raw_log = [], []
    skipped = {"identical": 0, "too_few": 0, "weak": 0}
    t0 = time.time()

    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] '{safe_print(prompt, 60)}'")
        candidates = teacher.generate_candidates(prompt)

        if len(candidates) < 2:
            print(f"    [SKIP] only {len(candidates)} candidate(s)")
            skipped["too_few"] += 1
            continue

        scores = [score_response(prompt, c) for c in candidates]
        print(f"    Scores: {[round(s, 2) for s in scores]}")

        best_idx  = scores.index(max(scores))
        worst_idx = scores.index(min(scores))
        chosen    = candidates[best_idx]
        rejected  = candidates[worst_idx]

        if chosen == rejected:
            print(f"    [SKIP] chosen == rejected")
            skipped["identical"] += 1
            continue

        spread = max(scores) - min(scores)
        label = "[WEAK]" if spread < 0.3 else "[OK]" if spread < 1.0 else "[STRONG]"
        print(f"    Spread: {spread:.2f} {label}")

        records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        raw_log.append({
            "prompt": prompt, "candidates": candidates,
            "scores": scores, "chosen": chosen, "rejected": rejected,
            "spread": spread,
        })

        # Progress estimate
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            print(f"    [PROGRESS] {i+1}/{len(prompts)} | "
                  f"{len(records)} valid pairs | "
                  f"ETA: {eta/60:.1f} min")

    # Summary
    print(f"\n[Dataset] {len(records)} valid pairs / {len(prompts)} prompts")
    print(f"[Dataset] Skipped: {skipped}")
    if records:
        spreads = [r["spread"] for r in raw_log]
        print(f"[Dataset] Spread stats: min={min(spreads):.2f} "
              f"avg={sum(spreads)/len(spreads):.2f} max={max(spreads):.2f}")

    # Train/test split
    random.shuffle(records)
    split_idx = max(1, int(len(records) * (1 - eval_split)))
    train_records = records[:split_idx]
    eval_records  = records[split_idx:]

    print(f"[Dataset] Train: {len(train_records)} | Eval: {len(eval_records)}")

    train_ds = Dataset.from_list(train_records) if train_records else Dataset.from_list([])
    eval_ds  = Dataset.from_list(eval_records)  if eval_records  else Dataset.from_list([])

    return train_ds, eval_ds, raw_log



# --- Student ---

def load_student(cfg: SDPOConfig):
    print(f"\n[Student] Loading {cfg.student_model} ...")
    ram_status("before student load")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student_model, trust_remote_code=True, cache_dir=MODELS_DIR
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.student_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        cache_dir=MODELS_DIR,
        low_cpu_mem_usage=True,
    )
    print(f"[Student] Base loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # LoRA: target all attention projections for maximum learning capacity
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    print(f"[Student] LoRA applied. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return tokenizer, model


def train_student(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    model,
    cfg: SDPOConfig,
):
    os.makedirs(cfg.output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        beta=cfg.beta,
        max_length=cfg.max_length,
        max_prompt_length=cfg.max_prompt_len,
        logging_steps=5,
        save_steps=50,
        remove_unused_columns=False,
        report_to="none",
        precompute_ref_log_probs=True,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type="cosine",
        # Evaluation
        eval_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=25 if len(eval_dataset) > 0 else None,
        per_device_eval_batch_size=1,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        processing_class=tokenizer,
    )

    print(f"[Student] Training on {len(train_dataset)} pairs "
          f"(eval: {len(eval_dataset)})...")
    print(f"[Student] VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    trainer.train()
    trainer.save_model(cfg.output_dir)
    print(f"[Student] Saved to {cfg.output_dir}")
    return trainer



# --- Evaluation ---

def evaluate_student(tokenizer, model, eval_prompts: List[str], label: str = ""):
    """Generate responses for eval prompts and score them."""
    print(f"\n[Eval] {label} -- Generating on {len(eval_prompts)} prompts...")
    results = []

    model.eval()
    for i, prompt in enumerate(eval_prompts):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        sc = score_response(prompt, response)
        results.append({
            "prompt": prompt,
            "response": response,
            "score": sc,
        })
        print(f"  [{i+1}] score={sc:.2f} | {safe_print(response, 60)}")

    avg_score = sum(r["score"] for r in results) / max(len(results), 1)
    print(f"[Eval] {label} -- Avg score: {avg_score:.3f}")
    return results, avg_score



# --- Main ---

def main():
    print("=" * 60)
    print("SDPO: Self-Distillation Policy Optimization (v2)")
    print(f"Teacher : {CFG.teacher_model}")
    print(f"Student : {CFG.student_model}")
    mode_str = f"SMALL ({CFG.small_n})" if CFG.use_small_data else f"LARGE ({CFG.large_n})"
    print(f"Mode    : {mode_str}")
    print(f"Device  : {CFG.device}")
    print(f"LoRA    : r={CFG.lora_r}, alpha={CFG.lora_alpha}, "
          f"targets=[q,k,v,o]_proj")
    print(f"Eval    : {CFG.eval_split*100:.0f}% held out, "
          f"{CFG.eval_prompts_n} before/after prompts")
    print("=" * 60)
    ram_status("startup")

    # Step 1: load prompts
    prompts = load_prompts(CFG.use_small_data, CFG.small_n, CFG.large_n)

    # Reserve some prompts for before/after evaluation
    eval_prompts = prompts[:CFG.eval_prompts_n]

    # Step 2: teacher generates SDPO pairs
    print("\n[Step 2] Teacher generating SDPO pairs...")
    teacher = TeacherGenerator(CFG)
    train_ds, eval_ds, raw_log = build_sdpo_dataset(
        teacher, prompts, CFG.eval_split
    )

    log_path = os.path.join(CFG.output_dir, "sdpo_pairs_log.json")
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(raw_log, f, indent=2, ensure_ascii=False)
    print(f"[Step 2] Log saved -> {log_path}")

    if len(train_ds) == 0:
        print("[ERROR] No valid training pairs. Check teacher output and scorer.")
        return

    # Unload teacher
    teacher.unload()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    ram_status("after teacher unload")

    # Step 3: load student and evaluate BEFORE training
    print("\n[Step 3] Loading student for training...")
    tokenizer, model = load_student(CFG)

    before_results, before_avg = evaluate_student(
        tokenizer, model, eval_prompts, label="BEFORE training"
    )

    # Step 4: train student with DPO
    print("\n[Step 4] DPO Training...")
    trainer = train_student(train_ds, eval_ds, tokenizer, model, CFG)

    # Step 5: evaluate AFTER training
    after_results, after_avg = evaluate_student(
        tokenizer, model, eval_prompts, label="AFTER training"
    )

    # Step 6: compute and display improvement metrics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Avg score BEFORE : {before_avg:.3f}")
    print(f"  Avg score AFTER  : {after_avg:.3f}")
    improvement = after_avg - before_avg
    pct = (improvement / max(abs(before_avg), 0.01)) * 100
    print(f"  Improvement      : {improvement:+.3f} ({pct:+.1f}%)")

    # Win rate: how often did the trained model score higher?
    wins, ties, losses = 0, 0, 0
    for b, a in zip(before_results, after_results):
        if a["score"] > b["score"]:
            wins += 1
        elif a["score"] == b["score"]:
            ties += 1
        else:
            losses += 1
    total = max(wins + ties + losses, 1)
    print(f"  Win/Tie/Loss     : {wins}/{ties}/{losses} "
          f"(win rate: {wins/total*100:.0f}%)")

    # Training stats from trainer
    if hasattr(trainer, 'state') and trainer.state.log_history:
        train_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        eval_losses  = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
        if train_losses:
            print(f"  Train loss       : {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
        if eval_losses:
            print(f"  Eval loss        : {eval_losses[0]:.4f} -> {eval_losses[-1]:.4f}")

        # Reward accuracy (if logged by DPOTrainer)
        reward_accs = [h.get("rewards/accuracies", h.get("train_rewards/accuracies"))
                       for h in trainer.state.log_history
                       if "rewards/accuracies" in h or "train_rewards/accuracies" in h]
        if reward_accs:
            print(f"  Reward accuracy  : {reward_accs[0]:.3f} -> {reward_accs[-1]:.3f}")

    print("=" * 60)

    # Save evaluation report
    report = {
        "before_avg_score": before_avg,
        "after_avg_score": after_avg,
        "improvement": improvement,
        "improvement_pct": pct,
        "win_rate": wins / total * 100,
        "wins": wins, "ties": ties, "losses": losses,
        "before_results": before_results,
        "after_results": after_results,
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "config": {
            "teacher": CFG.teacher_model,
            "student": CFG.student_model,
            "lora_r": CFG.lora_r,
            "lora_alpha": CFG.lora_alpha,
            "learning_rate": CFG.learning_rate,
            "num_epochs": CFG.num_epochs,
            "beta": CFG.beta,
        }
    }
    report_path = os.path.join(CFG.output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[DONE] Eval report -> {report_path}")

    print(f"\n[DONE] SDPO training complete.")
    print(f"[DONE] Model -> {CFG.output_dir}")
    
    plot_results(CFG.output_dir)

# --- Plotting ---
def plot_results(output_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    except ImportError:
        print("[Plot] matplotlib not installed, skipping plot.")
        return

    eval_path  = os.path.join(output_dir, "eval_report.json")
    pairs_path = os.path.join(output_dir, "sdpo_pairs_log.json")
    
    if not os.path.exists(eval_path):
        print("[Plot] No eval_report.json found, skipping.")
        return

    with open(eval_path, "r") as f:
        report = json.load(f)
    
    pairs = []
    if os.path.exists(pairs_path):
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('SDPO v1 (Teacher-Student) Results', fontsize=14, fontweight='bold')

    # Plot 1: Before/After Eval Scores
    ax1 = axes[0]
    b_avg = report.get("before_avg_score", 0)
    a_avg = report.get("after_avg_score", 0)
    
    bars = ax1.bar(['Before', 'After'], [b_avg, a_avg], color=['#FF9800', '#4CAF50'])
    ax1.set_title('Evaluation Score (0 to 10)', fontweight='bold')
    ax1.set_ylim(0, max(b_avg, a_avg) * 1.2 + 1)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.4)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
        
    imp_pct = report.get("improvement_pct", 0)
    ax1.annotate(f'Improvement: {imp_pct:+.1f}%', xy=(0.5, 0.9), xycoords='axes fraction',
                 ha='center', fontsize=10, color='green' if imp_pct > 0 else 'red',
                 fontweight='bold')

    # Plot 2: Pair Spread Distribution
    ax2 = axes[1]
    if pairs:
        spreads = [p.get("spread", 0) for p in pairs]
        ax2.hist(spreads, bins=15, color='#2196F3', alpha=0.8, edgecolor='black')
        ax2.set_title('DPO Pair Score Spread', fontweight='bold')
        ax2.set_xlabel('Score Difference (Chosen - Rejected)')
        ax2.set_ylabel('Count')
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        avg_spread = sum(spreads)/len(spreads) if spreads else 0
        ax2.axvline(avg_spread, color='red', linestyle='dashed', linewidth=2, label=f'Avg: {avg_spread:.2f}')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No pairs log found", ha='center', va='center')
        ax2.set_title('DPO Pair Score Spread', fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, "training_plot.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n[Plot] Saved -> {out}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "small", "large", "plot"],
                        default="smoke")
    args = parser.parse_args()

    if args.mode == "smoke":
        CFG.use_small_data = True
        CFG.small_n = 5
        CFG.num_epochs = 1
        main()
    elif args.mode == "small":
        CFG.use_small_data = True
        main()
    elif args.mode == "large":
        CFG.use_small_data = False
        main()
    elif args.mode == "plot":
        plot_results(CFG.output_dir)
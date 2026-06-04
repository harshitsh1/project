"""
Manual SDPO implementation using a single model for both student and teacher passes.
"""

import os, sys

#  ALL DOWNLOADS -> D drive 
MODELS_DIR = r"D:\deep_learning\models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"]                       = MODELS_DIR
os.environ["HF_HUB_CACHE"]                  = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"]         = MODELS_DIR
os.environ["TORCH_HOME"]                    = MODELS_DIR
os.environ["BNB_CACHE_DIR"]                 = MODELS_DIR
os.environ["HF_DATASETS_CACHE"]             = MODELS_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]       = "expandable_segments:True"
print(f"[CACHE] All caches -> {MODELS_DIR}")

import torch
import torch.nn.functional as F
import json
import gc
import random
import psutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType



# --- Configuration ---

@dataclass
class TrueSDPOConfig:
    # Model -- single model plays both student and self-teacher roles
    model_name     : str   = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir     : str   = r"D:\deep_learning\outputs\true_sdpo"

    # Dataset
    use_small_data : bool  = True   # True=20 samples (smoke test), False=full
    small_n        : int   = 20
    large_n        : int   = 500

    # Generation (student rollout)
    max_new_tokens : int   = 128
    temperature    : float = 0.8
    top_p          : float = 0.9

    # SDPO loss
    top_k_distill  : int   = 50     #  K=100; use 50 for 6GB GPU
    beta           : float = 0.1    # KL regularization weight

    # Training
    learning_rate  : float = 1e-5   # 1e-5 for SDPO
    num_epochs     : int   = 3
    batch_size     : int   = 1      # one question at a time (no parallel)
    grad_accum     : int   = 4
    warmup_steps   : int   = 10
    max_grad_norm  : float = 1.0
    logging_steps  : int   = 5
    save_steps     : int   = 50

    # LoRA (student is fine-tuned, not teacher -- same model here)
    lora_r         : int   = 8
    lora_alpha     : int   = 16
    lora_dropout   : float = 0.05

    # Speed optimizations
    use_bf16       : bool  = True
    use_adamw_8bit : bool  = True   # requires bitsandbytes

    device: str = field(default_factory=lambda:
                        "cuda" if torch.cuda.is_available() else "cpu")


CFG = TrueSDPOConfig()



# --- Helper Functions ---

def ram_status(label=""):
    ram = psutil.virtual_memory()
    vram = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[MEM] {label} | RAM free: {ram.available/1e9:.1f}GB "
          f"/ {ram.total/1e9:.1f}GB | VRAM: {vram:.2f}GB")


def free_memory(*objects):
    for obj in objects:
        try: del obj
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


 
# --- Environment ---

class SimpleEnvironment:
    """
    Evaluates responses using a heuristic rule-based approach to generate
    rich textual feedback. This allows us to train on diverse datasets like
    UltraFeedback without needing an LLM-as-a-judge.
    """

    def evaluate(self, question: str, response: str) -> Tuple[float, str]:
        """
        Returns (reward, feedback_text).
        Checks for refusals, length, repetition, and hallucinations.
        """
        if not response or len(response.strip()) < 5:
            return 0.0, (
                "Incorrect. Your response is completely missing or far too short. "
                "You must provide a full, detailed answer."
            )
            
        resp_lower = response.lower()
        words = response.split()
        n_words = len(words)
        
        # 1. Check for Refusals
        refusals = ["i don't know", "i cannot", "i'm not sure", "i am not sure",
                    "i do not know", "i can't", "as an ai", "i'm unable"]
        if any(p in resp_lower for p in refusals):
            return 0.0, (
                "Incorrect. You refused to answer the prompt. "
                "You must attempt to provide a helpful and direct response instead of apologizing or refusing."
            )
            
        # 2. Check for Hallucinated markdown/links
        if "![" in response or "https://qwen" in resp_lower or "oss-cn" in resp_lower:
            return 0.0, (
                "Incorrect. Your response contains hallucinated image links or fake URLs. "
                "Please provide plain text explanations only."
            )
            
        # 3. Check for Repetition (a common failure mode for small/quantized models)
        if n_words > 15:
            trigrams = [tuple(words[i:i+3]) for i in range(n_words - 2)]
            unique_trigrams = len(set(trigrams))
            if len(trigrams) > 0 and (unique_trigrams / len(trigrams)) < 0.6:
                return 0.0, (
                    "Incorrect. Your response is highly repetitive. "
                    "You must use diverse vocabulary and avoid getting stuck in a loop repeating the same phrases."
                )
                
        # 4. Check for Length adequacy
        if n_words < 15:
            return 0.0, (
                "Incorrect. Your response is too brief. "
                "Please provide a more detailed and comprehensive explanation that fully addresses the user's prompt."
            )

        # 5. Passed basic checks - Positive feedback
        feedback = "Correct! The response is sufficiently detailed, avoids repetition, and provides a valid answer."
        reward = 1.0
        
        # Give constructive feedback to improve formatting even if it's correct
        if not any(marker in response for marker in ["- ", "* ", "1.", "2.", "###", "**"]):
            feedback += " For future reference, using bullet points, bold text, or numbered lists can make your answers even clearer and easier to read."
            reward = 0.8  # slightly lower reward if unstructured
            
        return reward, feedback


 
# --- Self-Teacher Template ---
 
def build_self_teacher_prompt(
    question: str,
    original_response: str,
    feedback: str,
    successful_rollout: Optional[str] = None,
) -> str:
    """
    Constructs the self-teacher 
    
    Template:
      User: {question}
            [Correct solution: {successful_rollout}]   ← if available
            The following is feedback from my unsuccessful earlier attempt:
            {feedback}
            Correctly solve the original question.
      Assistant: {original_response}   ← re-evaluate log-probs of THIS
    """
    user_content = question + "\n"

    if successful_rollout:
        user_content += f"\nCorrect solution:\n{successful_rollout}\n"

    if feedback and "Correct!" not in feedback:
        user_content += (
            f"\nThe following is feedback from my unsuccessful earlier attempt:\n"
            f"{feedback}\n"
            f"\nCorrectly solve the original question."
        )

    return user_content


 
# --- SDPO Loss ---
 
def compute_sdpo_loss(
    model,
    tokenizer,
    question: str,
    response: str,
    feedback: str,
    successful_rollout: Optional[str],
    cfg: TrueSDPOConfig,
) -> Optional[torch.Tensor]:
    """
    Computes the SDPO loss for one (question, response, feedback) triple.
    
    Steps:
      1. Forward pass WITHOUT feedback -> student log-probs
      2. Forward pass WITH feedback    -> self-teacher log-probs (stop_grad)
      3. Loss = sum_t KL(student_t || self_teacher_t)  [top-K approximation]
    
    This is the core of SDPO -- same model, two contexts.
    """

    #  Build student context (no feedback) 
    try:
        student_messages = [{"role": "user", "content": question}]
        student_prefix = tokenizer.apply_chat_template(
            student_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        student_prefix = f"User: {question}\nAssistant:"

    #  Build self-teacher context (with feedback) 
    teacher_question = build_self_teacher_prompt(
        question, response, feedback, successful_rollout
    )
    try:
        teacher_messages = [{"role": "user", "content": teacher_question}]
        teacher_prefix = tokenizer.apply_chat_template(
            teacher_messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        teacher_prefix = f"User: {teacher_question}\nAssistant:"

    #  Tokenize both contexts + response 
    full_student = student_prefix + response
    full_teacher = teacher_prefix + response

    student_enc = tokenizer(
        full_student, return_tensors="pt", truncation=True,
        max_length=cfg.max_new_tokens + 256
    )
    teacher_enc = tokenizer(
        full_teacher, return_tensors="pt", truncation=True,
        max_length=cfg.max_new_tokens + 512
    )

    student_ids = student_enc["input_ids"].to(cfg.device)
    teacher_ids = teacher_enc["input_ids"].to(cfg.device)

    # Find where response tokens start in each encoding
    prefix_student = tokenizer(student_prefix, return_tensors="pt")
    prefix_teacher = tokenizer(teacher_prefix, return_tensors="pt")
    resp_start_s   = prefix_student["input_ids"].shape[1]
    resp_start_t   = prefix_teacher["input_ids"].shape[1]

    if student_ids.shape[1] <= resp_start_s or teacher_ids.shape[1] <= resp_start_t:
        print("    [SKIP] Response too short after tokenization")
        return None

    #  Forward pass 1: student (gradients flow through) 
    student_out = model(input_ids=student_ids)
    # logits for response tokens only: shape (response_len, vocab)
    student_logits = student_out.logits[0, resp_start_s-1:-1]

    #  Forward pass 2: self-teacher (no gradients) 
    with torch.no_grad():
        teacher_out = model(input_ids=teacher_ids)
        teacher_logits = teacher_out.logits[0, resp_start_t-1:-1]

    # Align lengths (take the shorter)
    min_len = min(student_logits.shape[0], teacher_logits.shape[0])
    if min_len == 0:
        return None

    student_logits = student_logits[:min_len]
    teacher_logits = teacher_logits[:min_len]

    #  Top-K approximation of KL 
    # Only compute KL over top-K tokens under student to save memory
    # This avoids keeping full vocab logits for both passes
    K = cfg.top_k_distill
    student_probs = F.softmax(student_logits, dim=-1)   # (T, V)
    teacher_probs = F.softmax(teacher_logits, dim=-1)   # (T, V)

    # Get top-K indices under student
    topk_vals, topk_idx = student_probs.topk(K, dim=-1)  # (T, K)

    # Gather teacher probs at those same indices
    teacher_topk = teacher_probs.gather(1, topk_idx)     # (T, K)

    # Tail mass (probability outside top-K)
    student_tail = 1.0 - topk_vals.sum(dim=-1, keepdim=True)   # (T, 1)
    teacher_tail = 1.0 - teacher_topk.sum(dim=-1, keepdim=True) # (T, 1)

    # Clamp to avoid log(0)
    eps = 1e-8
    student_topk_clamped = topk_vals.clamp(min=eps)
    teacher_topk_clamped = teacher_topk.clamp(min=eps)
    student_tail_clamped = student_tail.clamp(min=eps)
    teacher_tail_clamped = teacher_tail.clamp(min=eps)

    # KL over top-K: sum p * log(p/q)
    kl_topk = (
        student_topk_clamped *
        (student_topk_clamped.log() - teacher_topk_clamped.log())
    ).sum(dim=-1)   # (T,)

    # KL tail term
    kl_tail = (
        student_tail_clamped *
        (student_tail_clamped.log() - teacher_tail_clamped.log())
    ).squeeze(-1)   # (T,)

    # Per-token KL, averaged over sequence
    per_token_kl = kl_topk + kl_tail             # (T,)
    loss = per_token_kl.mean()

    return loss



# --- Dataset ---

def load_prompts(cfg: TrueSDPOConfig) -> List[str]:
    n = cfg.small_n if cfg.use_small_data else cfg.large_n
    print(f"[Dataset] Loading prompts (n={n})...")
    try:
        ds = load_dataset(
            "trl-lib/ultrafeedback_binarized",
            split="train", streaming=True,
        )
        prompts = []
        for row in ds:
            for msg in row["chosen"]:
                if msg["role"] == "user":
                    prompts.append(msg["content"][:500])  # truncate long prompts
                    break
            if len(prompts) >= n:
                break
        print(f"[Dataset] Loaded {len(prompts)} prompts from UltraFeedback.")
        return prompts
    except Exception as e:
        print(f"[Dataset] UltraFeedback failed: {e}, using fallback prompts")
        return _fallback_prompts(n)


def _fallback_prompts(n: int) -> List[str]:
    base = [
        "What is 2 + 2?",
        "What is the capital of Germany?",
        "What is gravity?",
        "What is photosynthesis?",
        "What does CPU stand for?",
        "What is a neural network?",
        "What is gradient descent?",
        "What is backpropagation?",
        "What is overfitting?",
        "What is transfer learning?",
        "Explain what a for loop is.",
        "What is the speed of light?",
        "What is a variable in programming?",
        "How does sorting work?",
        "What is machine learning?",
        "Explain the concept of recursion.",
        "What is an API?",
        "What is a database?",
        "Explain binary search.",
        "What is object-oriented programming?",
    ]
    pool = (base * (n // len(base) + 1))[:n]
    random.shuffle(pool)
    return pool



# --- Model Setup ---
 
def load_model(cfg: TrueSDPOConfig):
    print(f"\n[Model] Loading {cfg.model_name} ...")
    ram_status("before model load")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[Model] Tokenizer loaded. Vocab: {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    print(f"[Model] Base loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    # Apply LoRA -- student trains; self-teacher uses same weights (no separate copy)
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # more modules
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"[Model] LoRA applied. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    return tokenizer, model



# --- Student Rollout ---

def generate_response(model, tokenizer, question: str, cfg: TrueSDPOConfig) -> str:
    """Generate one response from the student (current policy)."""
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True
        )
    except Exception:
        text = f"User: {question}\nAssistant:"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(cfg.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()



# --- Training Loop ---

def train_sdpo(cfg: TrueSDPOConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    ram_status("startup")

    # Load single model (student = self-teacher)
    tokenizer, model = load_model(cfg)
    model.train()

    # Optimizer
    if cfg.use_adamw_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=0.01,
            )
            print("[Optimizer] Using AdamW 8-bit")
        except Exception as e:
            print(f"[Optimizer] 8-bit failed ({e}), using standard AdamW")
            optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    # Load prompts
    prompts = load_prompts(cfg)
    env     = SimpleEnvironment()
    total_steps = len(prompts) * cfg.num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=total_steps // cfg.grad_accum,
    )

    print(f"\n[Train] Starting True SDPO training")
    print(f"[Train] Prompts: {len(prompts)} | Epochs: {cfg.num_epochs} | Steps: {total_steps}")
    ram_status("before training")

    global_step  = 0
    accum_loss   = 0.0
    accum_count  = 0
    log_history  = []
    optimizer.zero_grad()

    for epoch in range(cfg.num_epochs):
        random.shuffle(prompts)
        print(f"\n[Epoch {epoch+1}/{cfg.num_epochs}]")

        for q_idx, question in enumerate(prompts):
            global_step += 1

            #  Step 1: Student generates response 
            model.eval()  # eval mode for generation
            with torch.no_grad():
                response = generate_response(model, tokenizer, question, cfg)
            model.train()  # back to train for loss computation

            #  Step 2: Environment gives rich feedback 
            reward, feedback = env.evaluate(question, response)

            #  Step 3: Find successful rollout if reward=0 
            # In a full impl, this would be another rollout from the same batch
            # Here we skip it (no group rollouts) 
            successful_rollout = response if reward == 1.0 else None

            #  Step 4: Compute SDPO loss 
            # student(no feedback) vs self-teacher(with feedback)
            loss = compute_sdpo_loss(
                model, tokenizer, question, response,
                feedback, successful_rollout, cfg
            )

            if loss is None:
                continue

            # Scale loss for gradient accumulation
            loss = loss / cfg.grad_accum
            loss.backward()

            accum_loss  += loss.item() * cfg.grad_accum
            accum_count += 1

            #  Step 5: Optimizer step every grad_accum steps 
            if global_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % (cfg.logging_steps * cfg.grad_accum) == 0:
                    avg_loss = accum_loss / max(accum_count, 1)
                    lr_now   = scheduler.get_last_lr()[0]
                    vram     = torch.cuda.memory_allocated()/1e9
                    print(
                        f"  Step {global_step:4d}/{total_steps} | "
                        f"loss={avg_loss:.4f} | reward={reward:.1f} | "
                        f"lr={lr_now:.2e} | VRAM={vram:.2f}GB"
                    )
                    # Debug: log KL details
                    print(f"    Q: {question[:50]!r}")
                    print(f"    A: {response[:60]!r}")
                    print(f"    F: {feedback[:80]!r}")

                    log_history.append({
                        "step": global_step,
                        "loss": avg_loss,
                        "reward": reward,
                        "lr": lr_now,
                    })
                    accum_loss  = 0.0
                    accum_count = 0

            #  Save checkpoint 
            if global_step % cfg.save_steps == 0:
                ckpt = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt)
                print(f"  [Saved] {ckpt}")

    # Final save
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    log_path = os.path.join(cfg.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)

    print(f"\n[DONE] True SDPO training complete.")
    print(f"[DONE] Model saved -> {cfg.output_dir}")
    print(f"[DONE] Log saved   -> {log_path}")
    return model, tokenizer



# --- Smoke Test ---

def smoke_test():
    """Run 5 steps to verify the full SDPO loop works."""
    print("\n" + "="*60)
    print("SMOKE TEST: True SDPO (5 steps)")
    print("="*60)

    test_cfg = TrueSDPOConfig()
    test_cfg.use_small_data = True
    test_cfg.small_n        = 5      # just 5 prompts
    test_cfg.num_epochs     = 1
    test_cfg.logging_steps  = 1
    test_cfg.output_dir     = r"D:\deep_learning\outputs\sdpo_smoke"

    tokenizer, model = load_model(test_cfg)
    env  = SimpleEnvironment()
    prompts = _fallback_prompts(5)

    model.eval()
    print("\n[Smoke] Running 5 SDPO steps...")

    for i, question in enumerate(prompts):
        # Generate
        response = generate_response(model, tokenizer, question, test_cfg)
        # Evaluate
        reward, feedback = env.evaluate(question, response)
        # Loss
        model.train()
        loss = compute_sdpo_loss(
            model, tokenizer, question, response,
            feedback, None, test_cfg
        )
        model.eval()

        status = "OK" if loss is not None else "SKIP"
        loss_val = f"{loss.item():.4f}" if loss is not None else "None"
        print(f"  [{i+1}] {status} | Q: {question[:40]!r} | "
              f"reward={reward:.1f} | loss={loss_val}")
        print(f"       A: {response[:60]!r}")

        if loss is not None:
            loss.backward()  # verify gradients flow

        free_memory()

    print("\n[Smoke] PASSED -- SDPO loop works correctly.")
    print("[Smoke] Ready for full training with train_sdpo(CFG)")



# --- Entry Point ---

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "small", "large"],
                        default="smoke",
                        help="smoke=5 steps, small=20 samples, large=500 samples")
    args = parser.parse_args()

    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "small":
        CFG.use_small_data = True
        train_sdpo(CFG)
    elif args.mode == "large":
        CFG.use_small_data = False
        train_sdpo(CFG)
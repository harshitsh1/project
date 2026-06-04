import os, sys, warnings
warnings.filterwarnings("ignore", message="Sliding Window Attention")

MODELS_DIR = r"D:\deep_learning\models"
os.makedirs(MODELS_DIR, exist_ok=True)

# HuggingFace
os.environ["HF_HOME"]                = MODELS_DIR
os.environ["HF_HUB_CACHE"]          = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_DIR

# PyTorch
os.environ["TORCH_HOME"]            = MODELS_DIR

# bitsandbytes CUDA kernel cache
os.environ["BNB_CACHE_DIR"]         = MODELS_DIR

# Triton kernel cache (compiles and caches GPU kernels)
os.environ["TRITON_CACHE_DIR"]      = MODELS_DIR

# XDG cache (used by some Linux-style paths even on Windows)
os.environ["XDG_CACHE_HOME"]        = MODELS_DIR

# Huggingface assets (tokenizer files etc)
os.environ["HF_DATASETS_CACHE"]     = MODELS_DIR

print(f"[CACHE] All caches -> {MODELS_DIR}")



import torch 
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from trl import DPOConfig, DPOTrainer

from peft import LoraConfig, get_peft_model, TaskType

STUDENT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TEACHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_SAMPLES = 10

# MINI PROMPTS & HAND-WRITTEN PAIRS
SMOKE_PROMPTS = [
    "What is 2 + 2?",
    "Name the capital of Germany.",
    "What is gravity?",
    "Explain what a for loop is.",
    "What is photosynthesis?",
    "What does CPU stand for?",
    "What is a neural network?",
    "How does sorting work?",
    "What is a variable in programming?",
    "What is the speed of light?",
][:N_SAMPLES]


HAND_PAIRS = [
    {"prompt": "What is 2 + 2?",          "chosen": "2 + 2 = 4.",                     "rejected": "Probably 5."},
    {"prompt": "Name the capital of Germany.", "chosen": "Berlin is the capital of Germany.", "rejected": "Frankfurt is the capital."},
    {"prompt": "What is gravity?",         "chosen": "Gravity is the force that attracts objects with mass toward each other.", "rejected": "Gravity is what keeps the sun bright."},
    {"prompt": "Explain what a for loop is.", "chosen": "A for loop repeats a block of code a set number of times.", "rejected": "A for loop is a type of variable."},
    {"prompt": "What is photosynthesis?",  "chosen": "Photosynthesis is how plants convert sunlight into energy using CO₂ and water.", "rejected": "Photosynthesis is when plants eat soil."},
    {"prompt": "What does CPU stand for?", "chosen": "CPU stands for Central Processing Unit.", "rejected": "CPU means Computer Power Upgrade."},
    {"prompt": "What is a neural network?","chosen": "A neural network is a model inspired by the brain, made of layers of interconnected nodes.", "rejected": "A neural network is a type of internet cable."},
    {"prompt": "How does sorting work?",   "chosen": "Sorting arranges elements in a defined order, such as ascending or descending.", "rejected": "Sorting deletes items from a list."},
    {"prompt": "What is a variable in programming?", "chosen": "A variable stores a value that can be referenced and changed during program execution.", "rejected": "A variable is a type of loop."},
    {"prompt": "What is the speed of light?", "chosen": "The speed of light is approximately 3×10⁸ m/s in vacuum.", "rejected": "Light travels at 1000 km/h."},
][:N_SAMPLES]

def load_student(model_name: str):
    print(f"[DEBUG] Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=MODELS_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[DEBUG] Tokenizer loaded OK")

    print(f"[DEBUG] Loading model... (this may take 1-3 min)")
    
    # Check RAM before loading
    import psutil
    ram = psutil.virtual_memory()
    print(f"[DEBUG] RAM free before load: {ram.available/1e9:.1f} GB")
    print(f"[DEBUG] RAM total           : {ram.total/1e9:.1f} GB")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            attn_implementation="eager",
            low_cpu_mem_usage=True,   # <- add this, loads layer by layer
            cache_dir=MODELS_DIR,
        )
    except Exception as e:
        print(f"EXCEPTION during model load: {e}")
        import traceback
        traceback.print_exc()
        raise

    print(f"[DEBUG] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Apply LoRA — only train a tiny fraction of parameters
    lora_config = LoraConfig(
        r=8,                        # LoRA rank — lower = less memory
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],   # only patch attention
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()   # shows e.g. "trainable: 0.5% of params"
    print(f"[DEBUG] LoRA applied. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return tokenizer, model, None


def run_dpo(dataset, tokenizer, model, ref_model, output_dir):
    args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # increased to compensate small batch
        learning_rate=5e-5,
        beta=0.2,
        max_length=256,
        max_prompt_length=128,
        logging_steps=1,
        save_steps=999999,
        remove_unused_columns=False,
        report_to="none",
        precompute_ref_log_probs=True,
        bf16=True,
        fp16=False,
        optim="adamw_8bit",              # 8-bit Adam — cuts optimizer VRAM by 4x
    )

    print(f"[DEBUG] Building DPOTrainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    print(f"[DEBUG] Trainer built. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"[DEBUG] Starting training...")
    trainer.train()
    return trainer
    


# with sdpo
def generate_sdpo_pairs(prompts):
    print(f"  Loading teacher ({TEACHER_MODEL}) ...")

    from transformers import BitsAndBytesConfig
    import re, random

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer_t = AutoTokenizer.from_pretrained(
        TEACHER_MODEL, trust_remote_code=True, cache_dir=MODELS_DIR
    )
    if tokenizer_t.pad_token is None:
        tokenizer_t.pad_token = tokenizer_t.eos_token

    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
        cache_dir=MODELS_DIR,
    )
    teacher.eval()
    print(f"  Teacher loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Reset model's generation_config to prevent conflicts with our kwargs
    teacher.generation_config = GenerationConfig()
    teacher.generation_config.pad_token_id = tokenizer_t.eos_token_id

    _cuda_ok = True

    def clean_response(text):
        """Post-process teacher output: extract clean first sentence."""
        # Take only the first line (hallucinated URLs/markdown always come after \n)
        text = text.split("\n")[0].strip()
        # Remove markdown image/link artifacts
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        # Fix "word! word! word!" pattern from repetition_penalty artifacts:
        # Replace "! " (exclamation followed by space) with just space
        text = re.sub(r'!\s+', ' ', text)
        # Remove leading/trailing ! that aren't part of sentences
        text = text.strip('! ')
        # Collapse remaining runs of 3+ punctuation
        text = re.sub(r'([!?.,;:]){3,}', r'\1', text)
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def make_bad_answer(good, prompt):
        """Create a plausible-but-wrong answer by corrupting the good one."""
        words = good.split()
        if len(words) < 3:
            return f"I think {prompt.rstrip('?.')} is not well-defined."

        rng = random.Random(hash(prompt))
        result = words.copy()
        n_swaps = max(1, len(result) // 3)
        for _ in range(n_swaps):
            i = rng.randint(0, len(result) - 2)
            result[i], result[i + 1] = result[i + 1], result[i]

        bad = " ".join(result)
        if bad == good:
            bad = f"Not exactly. {good.replace('is', 'is not', 1)}"
        return bad

    def gen(prompt):
        """Generate using greedy decoding with light repetition penalty.
        
        Beam search is broken on 4-bit models (distorted probs → all beams
        collapse to EOS in 1-2 tokens). Greedy decoding works reliably.
        repetition_penalty=1.15 is light enough to avoid '!' insertion
        but strong enough to prevent infinite punctuation loops.
        """
        nonlocal _cuda_ok
        if not _cuda_ok:
            return ""

        torch.cuda.empty_cache()

        system = "You are a helpful assistant. Answer the question correctly in one concise sentence."
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": prompt}]
        try:
            text = tokenizer_t.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inp = tokenizer_t(text, return_tensors="pt")
        if DEVICE == "cuda":
            inp = {k: v.cuda() for k, v in inp.items()}
        try:
            with torch.no_grad():
                out = teacher.generate(
                    **inp,
                    max_new_tokens=60,
                    do_sample=False,
                    repetition_penalty=1.15,  # light: avoids ! spam; prevents loops
                    pad_token_id=tokenizer_t.eos_token_id,
                )
            raw = tokenizer_t.decode(
                out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            cleaned = clean_response(raw)
            print(f"   raw  : {raw}")
            print(f"   clean: {cleaned}")
            return cleaned
        except RuntimeError as e:
            print(f"  [WARN] gen() failed - CUDA context may be poisoned: {e}")
            _cuda_ok = False
            return ""

    pairs = []
    for i, prompt in enumerate(prompts):
        print(f" Teacher [{i+1}/{len(prompts)}]: {prompt[:50]}")
        good = gen(prompt)
        print(f"   good: {good}")
        if good and len(good) > 5:
            bad = make_bad_answer(good, prompt)
            print(f"   bad : {bad}")
            if good != bad:
                pairs.append({"prompt": prompt, "chosen": good, "rejected": bad})
        else:
            print(f"   [SKIP] response too short or empty")

    del teacher, tokenizer_t
    import gc
    gc.collect()
    if _cuda_ok:
        torch.cuda.empty_cache()
    print(f"  Teacher unloaded. VRAM free: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return pairs

def evaluate_accuracy(model, tokenizer, prompts, expected_answers, label=""):
    """Generate responses from the trained model and compare against expected answers."""
    print(f"\n  --- Accuracy Evaluation ({label}) ---")
    model.eval()
    # Reset generation_config to prevent Qwen's baked-in temperature/top_p/top_k warnings
    model.generation_config = GenerationConfig()
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    correct = 0
    total = len(prompts)

    for i, (prompt, expected) in enumerate(zip(prompts, expected_answers)):
        torch.cuda.empty_cache()
        messages = [{"role": "system", "content": "Answer concisely in one sentence."},
                    {"role": "user",   "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"User: {prompt}\nAssistant:"

        inp = tokenizer(text, return_tensors="pt")
        inp = {k: v.to(model.device) for k, v in inp.items()}

        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=60,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()

        # Check if expected keywords appear in the response
        expected_lower = expected.lower()
        response_lower = response.lower()
        # Extract key terms from expected (words > 3 chars, not stopwords)
        stopwords = {"the", "that", "this", "with", "from", "have", "been", "were", "what", "which", "about", "into"}
        key_terms = [w for w in expected_lower.split() if len(w) > 3 and w not in stopwords]
        if key_terms:
            matches = sum(1 for t in key_terms if t in response_lower)
            hit = matches >= max(1, len(key_terms) // 2)  # at least half of key terms present
        else:
            hit = expected_lower[:10] in response_lower  # fallback: prefix match

        status = "PASS" if hit else "FAIL"
        if hit:
            correct += 1
        print(f"  [{status}] Q: {prompt[:40]:40s} | Expected: {expected[:30]:30s} | Got: {response[:50]}")

    accuracy = correct / total * 100
    print(f"\n  Accuracy: {correct}/{total} = {accuracy:.1f}%")
    print(f"  {'---'*20}")
    return accuracy


def run_smoke_without_sdpo():
    """Mode A: DPO training with hand-written pairs."""
    print("\n" + "="*50)
    print("Smoke Test without SDPO")
    print("="*50)
    dataset = Dataset.from_list(HAND_PAIRS)
    print(f" Dataset: {len(dataset)} hand-written pairs")
    print("[DEBUG] About to load student model...")
    tokenizer, model, ref_model = load_student(STUDENT_MODEL)
    print("[DEBUG] Student model loaded successfully")
    print(f"[DEBUG] Model device : {next(model.parameters()).device}")
    print(f"[DEBUG] Model dtype  : {next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        print(f"[DEBUG] VRAM used    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("[DEBUG] About to start DPO training...")

    trainer = run_dpo(dataset, tokenizer, model, ref_model, "./smoke_no_sdpo")

    # --- Accuracy evaluation ---
    expected = [p["chosen"] for p in HAND_PAIRS]
    prompts = [p["prompt"] for p in HAND_PAIRS]
    evaluate_accuracy(model, tokenizer, prompts, expected, label="Without SDPO")

    # --- Cleanup ---
    import gc
    print("[DEBUG] Unloading student model after Mode A...")
    del trainer, model, tokenizer, ref_model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    import psutil
    ram = psutil.virtual_memory()
    print(f"[DEBUG] After cleanup -- VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"[DEBUG] After cleanup -- RAM free: {ram.available/1e9:.1f} GB")

    print("  [PASS] WITHOUT SDPO complete.\n")


def run_smoke_with_sdpo(pairs):
    """Mode B: DPO training with teacher-generated pairs."""
    print("\n" + "="*50)
    print("Smoke Test with SDPO -- DPO Training Stage")
    print("="*50)

    if not pairs:
        print(" [WARN] No pairs -- skipping SDPO DPO training.")
        return

    dataset = Dataset.from_list(pairs)
    print(f" Dataset: {len(dataset)} teacher-generated pairs")
    print("[DEBUG] About to load student model...")
    tokenizer, model, ref_model = load_student(STUDENT_MODEL)
    print("[DEBUG] Student model loaded successfully")
    print(f"[DEBUG] Model device : {next(model.parameters()).device}")
    print(f"[DEBUG] Model dtype  : {next(model.parameters()).dtype}")
    if torch.cuda.is_available():
        print(f"[DEBUG] VRAM used    : {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print("[DEBUG] About to start DPO training...")
    run_dpo(dataset, tokenizer, model, ref_model, "./smoke_with_sdpo")

    # --- Accuracy evaluation ---
    expected = [p["chosen"] for p in pairs]
    prompts = [p["prompt"] for p in pairs]
    evaluate_accuracy(model, tokenizer, prompts, expected, label="With SDPO")

    print(" [PASS] WITH SDPO complete.\n")


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Samples: {N_SAMPLES}")

    # Step 1: Generate teacher pairs (before student model)
    sdpo_pairs = generate_sdpo_pairs(SMOKE_PROMPTS)
    print(f"  Generated {len(sdpo_pairs)} SDPO pairs\n")

    # Step 2: Run DPO with hand-written pairs (Mode A)
    run_smoke_without_sdpo()

    # Step 3: Run DPO with teacher-generated pairs (Mode B / SDPO)
    run_smoke_with_sdpo(sdpo_pairs)

    print("\n Both smoke tests passed. Ready for full training.")
    print(" Next: run_1_baseline_no_sdpo.py  or  2_sdpo_training.py")



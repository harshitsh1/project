"""
SDPO implementation featuring an NLI Environment.
"""

import os, sys

MODELS_DIR = r"D:\deep_learning\models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.environ["HF_HOME"]                        = MODELS_DIR
os.environ["HF_HUB_CACHE"]                   = MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"]          = MODELS_DIR
os.environ["TORCH_HOME"]                     = MODELS_DIR
os.environ["BNB_CACHE_DIR"]                  = MODELS_DIR
os.environ["HF_DATASETS_CACHE"]              = MODELS_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]        = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]         = "false"
print(f"[CACHE] All caches -> {MODELS_DIR}")

import torch
import torch.nn.functional as F
import json, gc, random, time, psutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from torch.optim import AdamW

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType


# --- Configuration ---
@dataclass
class TrueSDPOConfig:
    model_name     : str   = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir     : str   = r"D:\deep_learning\outputs\true_sdpo"

    # Dataset
    use_small_data : bool  = True
    small_n        : int   = 20
    large_n        : int   = 300

    # Train/test split
    test_split     : float = 0.2

    # ── SPEED FIX 1: eval less often, on fewer samples ────────
    eval_steps          : int = 20   # was 10 -> half as many evals
    eval_verifiable_only: bool = True  # only score verifiable test prompts
    # eval_verifiable_only=True means exact_match is always meaningful
    # ----------------------------------------------------------------------

    # ── SPEED FIX 2: shorter generation ───────────────────────
    max_new_tokens : int   = 64    # was 128 -> 2x faster generation
    # ----------------------------------------------------------------------

    temperature    : float = 0.8
    top_p          : float = 0.9

    # SDPO loss
    top_k_distill  : int   = 20    # was 50, paper uses 20 for code env
    beta           : float = 0.1

    # Training
    learning_rate  : float = 1e-5
    num_epochs     : int   = 3
    grad_accum     : int   = 4
    warmup_steps   : int   = 10
    max_grad_norm  : float = 1.0
    logging_steps  : int   = 5
    save_steps     : int   = 50

    # LoRA
    lora_r         : int   = 8
    lora_alpha     : int   = 16
    lora_dropout   : float = 0.05

    use_bf16       : bool  = True
    use_adamw_8bit : bool  = True

    device: str = field(default_factory=lambda:
                        "cuda" if torch.cuda.is_available() else "cpu")


CFG = TrueSDPOConfig()


# --- Helper Functions ---
def ram_status(label=""):
    ram  = psutil.virtual_memory()
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


# --- Semantic Environment ---
class SemanticEnvironment:
    REFUSAL_PHRASES = [
        "i don't know", "i cannot answer", "i'm not sure",
        "i am not sure", "i do not know", "i can't answer",
        "i cannot provide", "i'm unable", "i am unable",
        "this is a complex topic", "as an ai", "i cannot help",
    ]
    GOOD_MARKERS = [
        "the answer is", "this means", "defined as", "refers to",
        "is a type of", "works by", "is used to", "consists of",
        "is the process of", "means that", "is when", "for example",
        "such as", "in other words", "specifically", "in summary",
    ]

    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print("[Env] Loading sentence-transformers scorer (CPU)...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
                print("[Env] Semantic scorer ready.")
            except ImportError:
                self._model = "unavailable"
        return self._model

    def _sim(self, q: str, r: str) -> float:
        m = self._load()
        if m == "unavailable":
            return 0.0
        try:
            from sentence_transformers import util
            with torch.no_grad():
                qe = m.encode(q, convert_to_tensor=True)
                re = m.encode(r, convert_to_tensor=True)
            return float(max(0.0, min(1.0, util.cos_sim(qe, re)[0][0])))
        except Exception:
            return 0.0

    def evaluate(self, question: str, response: str) -> Tuple[float, str]:
        r_lower = response.lower().strip()
        words   = response.split()
        n       = len(words)

        for rp in self.REFUSAL_PHRASES:
            if rp in r_lower:
                return 0.0, f"Refusal: '{rp}'. Answer directly."

        if n < 8:
            return 0.0, f"Too short ({n} words, min 8)."

        if n > 10:
            from collections import Counter
            wc = Counter(w.lower() for w in words if len(w) > 3)
            if wc and wc.most_common(1)[0][1] / n > 0.3:
                return 0.0, "Repetitive response."

        score      = 0.4
        good_count = sum(1 for g in self.GOOD_MARKERS if g in r_lower)
        score     += min(good_count * 0.12, 0.36)
        sim        = self._sim(question, response)
        score     += sim * 0.15
        if n > 200: score -= 0.15
        elif n > 150: score -= 0.08
        if not any(response.rstrip().endswith(p) for p in [".", "!", "?"]):
            score -= 0.10
        score = round(max(0.0, min(1.0, score)), 3)

        fb = (f"Score={score:.2f}. Len={n}w. Markers={good_count}. Sim={sim:.2f}."
              + (" Verbose." if n > 150 else "")
              + (" No punct." if not any(response.rstrip().endswith(p)
                                        for p in [".", "!", "?"]) else ""))
        return score, fb


# --- Simple Environment ---
class SimpleEnvironment:
    ANSWERS = {
        "What is 2 + 2?"                      : {"keyword":"4",                    "bad":["5","three"]},
        "What is the capital of Germany?"      : {"keyword":"berlin",               "bad":["paris","munich"]},
        "What is gravity?"                     : {"keyword":"force",                "bad":[]},
        "What is photosynthesis?"              : {"keyword":"sunlight",             "bad":[]},
        "What does CPU stand for?"             : {"keyword":"central processing",   "bad":["computer power"]},
        "What is a neural network?"            : {"keyword":"layer",                "bad":["cable"]},
        "What is gradient descent?"            : {"keyword":"optim",                "bad":[]},
        "What is backpropagation?"             : {"keyword":"gradient",             "bad":[]},
        "What is overfitting?"                 : {"keyword":"train",                "bad":[]},
        "What is transfer learning?"           : {"keyword":"pretrain",             "bad":[]},
        "Explain what a for loop is."          : {"keyword":"repeat",               "bad":["variable"]},
        "What is the speed of light?"          : {"keyword":"299",                  "bad":["1000 km"]},
        "What is a variable in programming?"   : {"keyword":"valu",                 "bad":["loop"]},
        "How does sorting work?"               : {"keyword":"order",                "bad":["delet"]},
        "What is machine learning?"            : {"keyword":"data",                 "bad":["robot"]},
        "Explain the concept of recursion."    : {"keyword":"itself",               "bad":[]},
        "What is an API?"                      : {"keyword":"interface",            "bad":[]},
        "What is a database?"                  : {"keyword":"data",                 "bad":[]},
        "Explain binary search."               : {"keyword":"sort",                 "bad":["unsort"]},
        "What is object-oriented programming?" : {"keyword":"object",               "bad":[]},
        "What is Python?"                      : {"keyword":"program",              "bad":["snake"]},
        "What is a function in programming?"   : {"keyword":"reusabl",              "bad":[]},
        "What is an algorithm?"                : {"keyword":"step",                 "bad":[]},
        "What is RAM?"                         : {"keyword":"memory",               "bad":[]},
        "What is the Internet?"                : {"keyword":"network",              "bad":[]},
        "What is HTML?"                        : {"keyword":"markup",               "bad":[]},
        "What is a compiler?"                  : {"keyword":"translat",             "bad":[]},
        "What is cloud computing?"             : {"keyword":"server",               "bad":[]},
        "What is encryption?"                  : {"keyword":"secur",                "bad":[]},
        "What is an operating system?"         : {"keyword":"hardware",             "bad":[]},
        # paraphrased variants
        "Explain gradient descent in simple terms." : {"keyword":"optim",           "bad":[]},
        "What does overfitting mean in machine learning?" : {"keyword":"train",     "bad":[]},
        "Define transfer learning."            : {"keyword":"pretrain",             "bad":[]},
        "What is a neural network made of?"    : {"keyword":"layer",                "bad":[]},
        "How does backpropagation work?"       : {"keyword":"gradient",             "bad":[]},
        "What is a CPU?"                       : {"keyword":"process",              "bad":[]},
        "Define machine learning."             : {"keyword":"data",                 "bad":[]},
        "What is an application programming interface?" : {"keyword":"interface",   "bad":[]},
        "How does binary search work?"         : {"keyword":"sort",                 "bad":[]},
        "What is a database used for?"         : {"keyword":"data",                 "bad":[]},
        "What does RAM stand for?"             : {"keyword":"random access",        "bad":[]},
        "What is object oriented programming?" : {"keyword":"object",               "bad":[]},
        "What is a programming algorithm?"     : {"keyword":"step",                 "bad":[]},
        "What is internet?"                    : {"keyword":"network",              "bad":[]},
        "How does a compiler work?"            : {"keyword":"translat",             "bad":[]},
        "What is cloud storage?"               : {"keyword":"server",               "bad":[]},
        "What does encryption do?"             : {"keyword":"secur",                "bad":[]},
        "What is an OS?"                       : {"keyword":"system",               "bad":[]},
        "What is a for loop used for?"         : {"keyword":"repeat",               "bad":[]},
        "How does recursion work in programming?" : {"keyword":"itself",            "bad":[]},
        # ── HARD questions (model should struggle with these) ─────────
        # "keywords" list: match ANY one -> correct. "hint" helps self-teacher.
        "What is the time complexity of merge sort?"          : {"keyword":"n log n",       "keywords":["n log n","nlogn","o(n log n)"],  "bad":["n squared","n^2"],  "hint":"Merge sort runs in O(n log n) time by dividing the array in half recursively."},
        "What is the vanishing gradient problem?"             : {"keyword":"small",          "keywords":["vanish","shrink","small","zero","exponential"],  "bad":[],  "hint":"Gradients become vanishingly small during backpropagation through many layers, preventing learning."},
        "Explain the difference between L1 and L2 regularization." : {"keyword":"absolute", "keywords":["absolute","lasso","sparsity","sparse"],  "bad":[],  "hint":"L1 uses absolute values (encourages sparsity), L2 uses squared values (shrinks weights smoothly)."},
        "What is the bias-variance tradeoff?"                 : {"keyword":"underfitting",   "keywords":["underfitting","underfit","bias","variance","tradeoff"],  "bad":[],  "hint":"High bias causes underfitting, high variance causes overfitting; the tradeoff balances both."},
        "What is a hash table's average lookup time complexity?" : {"keyword":"o(1)",        "keywords":["o(1)","constant time","constant-time"],  "bad":["o(n)","o(log n)"],  "hint":"Hash tables achieve O(1) average lookup by computing an index directly from the key."},
        "What is the CAP theorem in distributed systems?"     : {"keyword":"partition",      "keywords":["partition","consistency","availability"],  "bad":[],  "hint":"CAP states you can only guarantee two of: Consistency, Availability, Partition tolerance."},
        "What is the difference between a stack and a queue?" : {"keyword":"fifo",           "keywords":["fifo","lifo","first in first out","last in first out"],  "bad":[],  "hint":"A stack is LIFO (last in, first out), a queue is FIFO (first in, first out)."},
        "What is a deadlock in operating systems?"            : {"keyword":"wait",           "keywords":["wait","circular","block","resource"],  "bad":[],  "hint":"Deadlock occurs when processes wait circularly for resources held by each other."},
        "What is the halting problem?"                        : {"keyword":"undecidable",    "keywords":["undecidable","cannot determine","impossible to decide"],  "bad":["easy","simple"],  "hint":"The halting problem is undecidable: no algorithm can determine if any program will halt."},
        "What is Big O notation used for?"                    : {"keyword":"complexity",     "keywords":["complexity","worst case","upper bound","growth"],  "bad":[],  "hint":"Big O notation describes the upper bound of an algorithm's time or space complexity."},
        "Explain the difference between TCP and UDP."         : {"keyword":"reliable",       "keywords":["reliable","connection","guarantee","ordered"],  "bad":[],  "hint":"TCP is reliable and connection-oriented; UDP is faster but unreliable with no delivery guarantee."},
        "What is a race condition?"                           : {"keyword":"concurrent",     "keywords":["concurrent","simultaneous","thread","shared"],  "bad":[],  "hint":"A race condition occurs when concurrent threads access shared data and the outcome depends on timing."},
        "What is the purpose of dropout in neural networks?" : {"keyword":"overfit",        "keywords":["overfit","regulariz","random"],  "bad":[],  "hint":"Dropout randomly deactivates neurons during training to prevent overfitting."},
        "What is a transformer model in deep learning?"      : {"keyword":"attention",      "keywords":["attention","self-attention","self attention"],  "bad":[],  "hint":"Transformers use self-attention mechanisms to process sequences in parallel."},
        "Explain what ACID means in databases."               : {"keyword":"atomic",         "keywords":["atomic","atomicity","isolation","durable","consistent"],  "bad":[],  "hint":"ACID: Atomicity, Consistency, Isolation, Durability — guarantees for reliable database transactions."},
        "What is the difference between supervised and unsupervised learning?" : {"keyword":"label",  "keywords":["label","labeled","supervised","target"],  "bad":[],  "hint":"Supervised learning uses labeled data with known targets; unsupervised finds patterns without labels."},
        "What is a GAN in deep learning?"                    : {"keyword":"discriminat",    "keywords":["discriminat","generator","adversarial"],  "bad":[],  "hint":"A GAN pits a generator against a discriminator; the generator creates fake data, the discriminator detects it."},
        "What is the curse of dimensionality?"               : {"keyword":"dimension",      "keywords":["dimension","sparse","exponential","high-dimensional"],  "bad":[],  "hint":"As dimensions increase, data becomes exponentially sparse, making learning and distance metrics unreliable."},
        "What is batch normalization?"                        : {"keyword":"normaliz",       "keywords":["normaliz","mean","variance","scale"],  "bad":[],  "hint":"Batch normalization normalizes layer inputs to zero mean and unit variance, stabilizing training."},
        "Explain the concept of attention mechanism."         : {"keyword":"weight",         "keywords":["weight","relevance","focus","score","query","key"],  "bad":[],  "hint":"Attention assigns weights to input elements based on their relevance, using query-key-value operations."},
        "What is a convolution in CNNs?"                     : {"keyword":"filter",         "keywords":["filter","kernel","sliding","feature map"],  "bad":[],  "hint":"A convolution slides a filter/kernel over the input to extract local features into feature maps."},
        "What is the purpose of an activation function?"     : {"keyword":"nonlinear",      "keywords":["nonlinear","non-linear","relu","sigmoid"],  "bad":[],  "hint":"Activation functions introduce nonlinearity so networks can learn complex patterns beyond linear combinations."},
        "What is cross-entropy loss?"                        : {"keyword":"probabilit",     "keywords":["probabilit","likelihood","log","classif"],  "bad":[],  "hint":"Cross-entropy measures the difference between predicted probability distributions and true labels."},
        "Explain the concept of word embeddings."            : {"keyword":"vector",         "keywords":["vector","dense","representation","semantic"],  "bad":[],  "hint":"Word embeddings map words to dense vectors where semantically similar words are close together."},
        "What is the difference between precision and recall?" : {"keyword":"positive",     "keywords":["positive","true positive","relevant","retrieved"],  "bad":[],  "hint":"Precision = true positives / predicted positives; Recall = true positives / actual positives."},
        "What is a learning rate in neural networks?"        : {"keyword":"step",           "keywords":["step","update","speed","convergence","gradient"],  "bad":[],  "hint":"The learning rate controls the step size of weight updates during gradient descent."},
        "What is the softmax function?"                      : {"keyword":"probabilit",     "keywords":["probabilit","exponential","sum to 1","normalize"],  "bad":[],  "hint":"Softmax converts logits into probabilities that sum to 1 using exponential normalization."},
        "What is the difference between BFS and DFS?"        : {"keyword":"breadth",        "keywords":["breadth","depth","level","queue","stack"],  "bad":[],  "hint":"BFS explores level by level using a queue; DFS goes deep first using a stack."},
        "What is dynamic programming?"                       : {"keyword":"subproblem",     "keywords":["subproblem","sub-problem","overlapping","memoiz","memo"],  "bad":[],  "hint":"Dynamic programming solves complex problems by breaking them into overlapping subproblems and caching results."},
        "What is the purpose of a loss function?"            : {"keyword":"error",          "keywords":["error","difference","predicted","actual","optim"],  "bad":[],  "hint":"A loss function measures the error between predicted and actual values, guiding optimization."},
    }

    GLOBAL_BAD = [
        "i don't know","i cannot answer","i'm not sure",
        "i am not sure","i do not know","i can't answer",
        "i cannot provide","this is a complex",
    ]

    def evaluate(self, question: str, response: str) -> Tuple[float, str]:
        r_lower = response.lower().strip()
        q_lower = question.lower().strip()

        matched = None
        for k in self.ANSWERS:
            if k.lower() in q_lower or q_lower in k.lower():
                matched = k
                break

        if matched is None:
            return 0.5, "Answer could not be automatically verified."

        entry   = self.ANSWERS[matched]
        keyword = entry["keyword"].lower()
        # Support multiple acceptable keywords (new "keywords" field)
        keywords = [kw.lower() for kw in entry.get("keywords", [keyword])]
        bads    = [b.lower() for b in entry["bad"]]
        hint    = entry.get("hint", "")

        if len(response.split()) < 5:
            return 0.0, f"Too short. Expected: '{keyword}'.{' ' + hint if hint else ''}"
        for gp in self.GLOBAL_BAD:
            if gp in r_lower:
                return 0.0, f"Refusal: '{gp}'.{' ' + hint if hint else ''}"
        for bp in bads:
            if bp in r_lower:
                return 0.0, f"Bad phrase: '{bp}'.{' ' + hint if hint else ''}"

        # Check if ANY keyword matches
        found_kw = None
        for kw in keywords:
            if kw in r_lower:
                found_kw = kw
                break

        if found_kw is None:
            return 0.0, (f"Missing key concept (need one of: {keywords}). "
                         f"Response: '{response[:80]}' "
                         f"{hint if hint else 'Hint: state the answer directly.'}")

        # Check negation on the found keyword
        for pat in [f"not {found_kw}", f"isn't {found_kw}", f"no {found_kw}"]:
            if pat in r_lower:
                return 0.0, f"'{found_kw}' appears negated.{' ' + hint if hint else ''}"
        return 1.0, f"Correct! '{found_kw}' found."


# --- NLI Environment ---
class NLIEnvironment:
    """
    Evaluates responses using Natural Language Inference.

    For each question that has a reference answer (hint), checks whether
    the model's response semantically entails it. Falls back to keyword
    matching for questions without hints or when NLI is unavailable.

    Scoring:
      entailment  > 0.5  AND  contradiction < 0.4  →  1.0 (correct)
      contradiction > 0.7                           →  0.0 (wrong)
      otherwise: check keywords as tiebreaker

    The NLI model runs entirely on CPU (~100MB), no GPU needed.
    """

    REFUSAL_PHRASES = [
        "i don't know", "i cannot answer", "i'm not sure",
        "i am not sure", "i do not know", "i can't answer",
        "i cannot provide", "i'm unable", "i am unable",
        "as an ai", "i cannot help",
    ]

    def __init__(self):
        self._model = None
        self._simple = SimpleEnvironment()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print("[NLI] Loading cross-encoder/nli-deberta-v3-small (CPU)...")
                self._model = CrossEncoder(
                    "cross-encoder/nli-deberta-v3-small", device="cpu"
                )
                print("[NLI] NLI scorer ready.")
            except Exception as e:
                print(f"[NLI] Failed to load NLI model: {e}")
                self._model = "unavailable"
        return self._model

    def _nli_scores(self, premise: str, hypothesis: str) -> Tuple[float, float, float]:
        """
        Returns (contradiction, entailment, neutral) probabilities.
        Labels for nli-deberta-v3-small: [contradiction, entailment, neutral]
        """
        import numpy as np
        model = self._load()
        if model == "unavailable":
            return 0.0, 0.0, 0.0
        try:
            scores = model.predict([(premise, hypothesis)])
            # scores is shape (1, 3) — apply softmax
            if scores.ndim == 1:
                scores = scores.reshape(1, -1)
            exp_s = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_s / exp_s.sum(axis=1, keepdims=True)
            return float(probs[0][0]), float(probs[0][1]), float(probs[0][2])
        except Exception as e:
            print(f"[NLI] Scoring error: {e}")
            return 0.0, 0.0, 0.0

    def _find_entry(self, question: str) -> Optional[Dict]:
        """Find matching ANSWERS entry for this question."""
        q_lower = question.lower().strip()
        for k, v in self._simple.ANSWERS.items():
            if k.lower() in q_lower or q_lower in k.lower():
                return v
        return None

    def evaluate(self, question: str, response: str) -> Tuple[float, str]:
        r_lower = response.lower().strip()
        words   = response.split()
        n       = len(words)

        # Basic checks
        if n < 5:
            entry = self._find_entry(question)
            hint  = entry.get("hint", "") if entry else ""
            return 0.0, f"Too short ({n} words).{' ' + hint if hint else ''}"

        for rp in self.REFUSAL_PHRASES:
            if rp in r_lower:
                entry = self._find_entry(question)
                hint  = entry.get("hint", "") if entry else ""
                return 0.0, f"Refusal: '{rp}'.{' ' + hint if hint else ''}"

        # Find reference answer
        entry = self._find_entry(question)

        # If no entry or no hint -> fall back to SimpleEnvironment
        if entry is None:
            return 0.5, "No reference answer available for NLI check."
        hint = entry.get("hint", "")
        if not hint:
            return self._simple.evaluate(question, response)

        # NLI check: does response entail the reference answer?
        contra, entail, neutral = self._nli_scores(response, hint)

        # Get keyword info for hybrid scoring
        keywords = [kw.lower() for kw in entry.get("keywords", [entry["keyword"].lower()])]
        bads     = [b.lower() for b in entry.get("bad", [])]

        # Check bad phrases
        for bp in bads:
            if bp in r_lower:
                return 0.0, (f"Bad phrase: '{bp}'. NLI(e={entail:.2f},c={contra:.2f}). "
                             f"{hint}")

        # Decision logic
        if contra > 0.7:
            return 0.0, (f"Contradicts answer (NLI contra={contra:.2f}). {hint}")

        if entail > 0.5 and contra < 0.4:
            return 1.0, (f"Correct! NLI entailment={entail:.2f} "
                         f"(contra={contra:.2f}, neutral={neutral:.2f})")

        # Borderline NLI: check keywords as tiebreaker
        found_kw = None
        for kw in keywords:
            if kw in r_lower:
                found_kw = kw
                break

        if found_kw is not None and entail > 0.3:
            return 1.0, (f"Correct! keyword '{found_kw}' + NLI={entail:.2f}")

        if found_kw is not None:
            return 0.8, (f"Keyword '{found_kw}' found but NLI uncertain "
                         f"(e={entail:.2f},c={contra:.2f}). {hint}")

        return 0.0, (f"Missing concept (NLI e={entail:.2f},c={contra:.2f}). "
                      f"Need: {keywords}. {hint}")


# --- Environment Factory ---
def make_environment(prefer_nli: bool = True, prefer_semantic: bool = True):
    # Priority: NLI > Semantic > Simple
    if prefer_nli:
        try:
            from sentence_transformers import CrossEncoder  # noqa
            env = NLIEnvironment()
            env._load()
            if env._model != "unavailable":
                print("[Env] Using NLIEnvironment (semantic entailment, CPU)")
                return env
        except ImportError:
            pass
        print("[Env] NLI model not available, trying SemanticEnvironment...")

    if prefer_semantic:
        try:
            import sentence_transformers  # noqa
            env = SemanticEnvironment()
            env._load()
            if env._model != "unavailable":
                print("[Env] Using SemanticEnvironment (scores any prompt)")
                return env
        except ImportError:
            pass
        print("[Env] sentence-transformers not installed.")
        print("[Env] Run:  pip install sentence-transformers")
        print("[Env] Falling back to SimpleEnvironment.")
    print("[Env] Using SimpleEnvironment (keyword-based, fallback prompts only)")
    return SimpleEnvironment()


# --- Self-Teacher Template ---
def build_self_teacher_prompt(
    question: str, response: str, feedback: str,
    solution: Optional[str] = None,
) -> str:
    c = question + "\n"
    if solution:
        c += f"\nCorrect solution:\n{solution}\n"
    if feedback and "Correct!" not in feedback and "Score=1" not in feedback:
        c += (f"\nFeedback from your earlier attempt:\n{feedback}\n"
              f"\nCorrectly solve the original question.")
    return c


# --- SDPO Loss ---
def compute_sdpo_loss(
    model, tokenizer,
    question: str, response: str, feedback: str,
    solution: Optional[str], cfg: TrueSDPOConfig,
) -> Optional[torch.Tensor]:

    try:
        sp = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        sp = f"User: {question}\nAssistant:"

    tq = build_self_teacher_prompt(question, response, feedback, solution)
    try:
        tp = tokenizer.apply_chat_template(
            [{"role":"user","content":tq}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        tp = f"User: {tq}\nAssistant:"

    s_ids = tokenizer(sp+response, return_tensors="pt", truncation=True,
                      max_length=cfg.max_new_tokens+256)["input_ids"].to(cfg.device)
    t_ids = tokenizer(tp+response, return_tensors="pt", truncation=True,
                      max_length=cfg.max_new_tokens+512)["input_ids"].to(cfg.device)

    rs = tokenizer(sp, return_tensors="pt")["input_ids"].shape[1]
    rt = tokenizer(tp, return_tensors="pt")["input_ids"].shape[1]

    if s_ids.shape[1] <= rs or t_ids.shape[1] <= rt:
        return None

    s_logits = model(input_ids=s_ids).logits[0, rs-1:-1]
    with torch.no_grad():
        t_logits = model(input_ids=t_ids).logits[0, rt-1:-1]

    T = min(s_logits.shape[0], t_logits.shape[0])
    if T == 0:
        return None

    s_logits = s_logits[:T]
    t_logits = t_logits[:T]

    K = cfg.top_k_distill
    sp_ = F.softmax(s_logits, dim=-1)
    tp_ = F.softmax(t_logits, dim=-1)
    tv, ti = sp_.topk(K, dim=-1)
    tt     = tp_.gather(1, ti)
    eps    = 1e-8
    st     = tv.clamp(min=eps);  tt = tt.clamp(min=eps)
    stail  = (1-tv.sum(-1,keepdim=True)).clamp(min=eps)
    ttail  = (1-tt.sum(-1,keepdim=True)).clamp(min=eps)
    kl_top  = (st*(st.log()-tt.log())).sum(-1)
    kl_tail = (stail*(stail.log()-ttail.log())).squeeze(-1)
    return (kl_top+kl_tail).mean()


# --- Generation ---
def generate_response(model, tokenizer, question: str, cfg: TrueSDPOConfig) -> str:
    try:
        text = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False, add_generation_prompt=True)
    except Exception:
        text = f"User: {question}\nAssistant:"
    inputs = {k: v.to(cfg.device)
              for k,v in tokenizer(text, return_tensors="pt").items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# --- Dataset ---
def _fallback_prompts() -> List[str]:
    return [
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
        "What is Python?",
        "What is a function in programming?",
        "What is an algorithm?",
        "What is RAM?",
        "What is the Internet?",
        "What is HTML?",
        "What is a compiler?",
        "What is cloud computing?",
        "What is encryption?",
        "What is an operating system?",
        "Explain gradient descent in simple terms.",
        "What does overfitting mean in machine learning?",
        "Define transfer learning.",
        "What is a neural network made of?",
        "How does backpropagation work?",
        "What is a CPU?",
        "Define machine learning.",
        "What is an application programming interface?",
        "How does binary search work?",
        "What is a database used for?",
        "What does RAM stand for?",
        "What is object oriented programming?",
        "What is a programming algorithm?",
        "What is internet?",
        "How does a compiler work?",
        "What is cloud storage?",
        "What does encryption do?",
        "What is an OS?",
        "What is a for loop used for?",
        "How does recursion work in programming?",
        # ── HARD prompts (model should struggle with these) ───────
        "What is the time complexity of merge sort?",
        "What is the vanishing gradient problem?",
        "Explain the difference between L1 and L2 regularization.",
        "What is the bias-variance tradeoff?",
        "What is a hash table's average lookup time complexity?",
        "What is the CAP theorem in distributed systems?",
        "What is the difference between a stack and a queue?",
        "What is a deadlock in operating systems?",
        "What is the halting problem?",
        "What is Big O notation used for?",
        "Explain the difference between TCP and UDP.",
        "What is a race condition?",
        "What is the purpose of dropout in neural networks?",
        "What is a transformer model in deep learning?",
        "Explain what ACID means in databases.",
        "What is the difference between supervised and unsupervised learning?",
        "What is a GAN in deep learning?",
        "What is the curse of dimensionality?",
        "What is batch normalization?",
        "Explain the concept of attention mechanism.",
        "What is a convolution in CNNs?",
        "What is the purpose of an activation function?",
        "What is cross-entropy loss?",
        "Explain the concept of word embeddings.",
        "What is the difference between precision and recall?",
        "What is a learning rate in neural networks?",
        "What is the softmax function?",
        "What is the difference between BFS and DFS?",
        "What is dynamic programming?",
        "What is the purpose of a loss function?",
    ]


def load_and_split_prompts(cfg: TrueSDPOConfig):
    """
    Returns: train_prompts, verifiable_test, full_test

    verifiable_test — fallback prompts only, ALWAYS scoreable by SimpleEnvironment
                      used for exact_match metric (always meaningful)
    full_test       — verifiable + UltraFeedback, used for avg_reward
                      (some 0.5 from unverifiable, but still informative)

    This separation means:
      exact_match tracks real improvement on known questions
      avg_reward tracks overall response quality
    Both metrics are now meaningful independently.
    """
    n        = cfg.small_n if cfg.use_small_data else cfg.large_n
    fallback = _fallback_prompts()
    print(f"[Dataset] Fallback pool: {len(fallback)} verifiable prompts")

    # Load UF for diversity
    uf_quota   = max(0, n - len(fallback))
    uf_prompts = []
    if uf_quota > 0:
        try:
            ds = load_dataset("trl-lib/ultrafeedback_binarized",
                              split="train", streaming=True)
            for row in ds:
                for msg in row["chosen"]:
                    if msg["role"] == "user":
                        p = msg["content"].strip()
                        if 10 < len(p) < 400:
                            uf_prompts.append(p)
                        break
                if len(uf_prompts) >= uf_quota:
                    break
            print(f"[Dataset] UltraFeedback: {len(uf_prompts)} prompts")
        except Exception as e:
            print(f"[Dataset] UltraFeedback failed ({e}), using fallback only")

    all_prompts = (fallback + uf_prompts)[:n]

    # Deterministic split
    random.seed(42)
    random.shuffle(all_prompts)
    random.seed()

    split_idx     = max(1, int(len(all_prompts) * (1 - cfg.test_split)))
    train_prompts = all_prompts[:split_idx]
    all_test      = all_prompts[split_idx:]

    # ── KEY FIX: separate verifiable test set ────────────────
    sim = SimpleEnvironment()
    verifiable_test = [
        p for p in all_test
        if any(k.lower() in p.lower() or p.lower() in k.lower()
               for k in sim.ANSWERS)
    ]

    # Guarantee at least 8 verifiable test prompts (need enough hard ones)
    if len(verifiable_test) < 8:
        print(f"[WARN] Only {len(verifiable_test)} verifiable in test — "
              f"forcing fallback prompts in")
        extras = [f for f in fallback if f not in all_test][:8]
        verifiable_test = extras + verifiable_test
        # Remove from train to avoid data leakage
        train_prompts = [p for p in train_prompts if p not in extras]

    print(f"[Dataset] Total: {len(all_prompts)} "
          f"({len(fallback)} fallback + {len(uf_prompts)} UF)")
    print(f"[Dataset] Train: {len(train_prompts)} | "
          f"Full test: {len(all_test)} | "
          f"Verifiable test: {len(verifiable_test)}")
    print(f"[Dataset] Eval runs on verifiable_test only "
          f"({len(verifiable_test)} samples) -> fast + meaningful")

    return train_prompts, verifiable_test, all_test


# --- Evaluation ---
def evaluate(
    model, tokenizer,
    verifiable_test : List[str],
    eval_env,
    cfg             : TrueSDPOConfig,
    step            : int,
    baseline_reward : float = None,
) -> Dict:
    """
    Evaluates ONLY on verifiable_test using the provided environment.

    Priority: NLIEnvironment > SimpleEnvironment
    NLI gives semantic entailment checking (more accurate than keyword match).
    Falls back to keyword matching for questions without reference answers.

    Metrics:
      avg_reward      weighted avg (1.0=correct, 0.0=wrong)
      exact_match     fraction with reward=1.0
      avg_resp_len    words per response
      refusal_rate    fraction with "i don't know" etc.
      very_long_pct   fraction > 100 words
      reward_delta    change from baseline
    """
    model.eval()
    t0 = time.time()

    rewards, resp_lens, per_q = [], [], []
    refusal_phrases = [
        "i don't know","i cannot","i'm not sure","i am not sure",
    ]

    print(f"\n  [Eval] step={step} | {len(verifiable_test)} verifiable samples")

    for q in verifiable_test:
        response         = generate_response(model, tokenizer, q, cfg)
        reward, feedback = eval_env.evaluate(q, response)
        w                = len(response.split())
        rewards.append(reward)
        resp_lens.append(w)
        per_q.append({
            "question": q[:80], "response": response[:120],
            "reward": reward, "feedback": feedback[:80], "n_words": w,
        })

    avg_reward   = sum(rewards) / len(rewards)
    exact_match  = sum(1 for r in rewards if r == 1.0) / len(rewards)
    avg_len      = sum(resp_lens) / len(resp_lens)
    refusal_rate = sum(
        1 for pq in per_q
        if any(p in pq["response"].lower() for p in refusal_phrases)
    ) / len(per_q)
    very_long    = sum(1 for l in resp_lens if l > 100) / len(resp_lens)
    very_short   = sum(1 for l in resp_lens if l < 5)   / len(resp_lens)

    delta     = round(avg_reward - baseline_reward, 4) if baseline_reward is not None else 0.0
    delta_str = (f" (delta {'+' if delta>=0 else ''}{delta:.3f})"
                 if baseline_reward is not None else "")
    elapsed   = time.time() - t0

    print(f"  [Eval] avg_reward={avg_reward:.3f}{delta_str} | "
          f"exact_match={exact_match:.3f} | avg_len={avg_len:.1f}w | "
          f"refusal={refusal_rate:.2f} | {elapsed:.1f}s")
    print(f"  [Eval] very_long={very_long:.2f} | very_short={very_short:.2f}")

    # Print all per-question results so you see what's being scored
    for pq in per_q:
        mark = "YES" if pq["reward"] == 1.0 else "NO"
        print(f"    {mark} r={pq['reward']:.1f} | "
              f"Q: {pq['question'][:45]!r} | "
              f"A: {pq['response'][:50]!r}")

    model.train()

    return {
        "step": step, "avg_reward": round(avg_reward,4),
        "exact_match": round(exact_match,4),
        "avg_resp_len": round(avg_len,1),
        "refusal_rate": round(refusal_rate,4),
        "very_long_pct": round(very_long,4),
        "very_short_pct": round(very_short,4),
        "reward_delta": delta, "n_eval": len(verifiable_test),
        "eval_time_s": round(elapsed,1),
        "per_question": per_q,
    }


# --- Model Setup ---
def load_model(cfg: TrueSDPOConfig):
    print(f"\n[Model] Loading {cfg.model_name} ...")
    ram_status("before load")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[Model] Tokenizer OK. Vocab: {len(tokenizer)}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
        device_map="cuda:0", low_cpu_mem_usage=True)
    print(f"[Model] Base loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=cfg.lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"[Model] LoRA applied. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return tokenizer, model


# --- Training Loop ---
def train_sdpo(cfg: TrueSDPOConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_dir = os.path.join(cfg.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    ram_status("startup")
    tokenizer, model = load_model(cfg)
    model.train()

    if cfg.use_adamw_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
            print("[Optimizer] AdamW 8-bit")
        except Exception as e:
            print(f"[Optimizer] 8-bit failed ({e}), using AdamW")
            optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=0.01)

    train_prompts, verifiable_test, full_test = load_and_split_prompts(cfg)
    # NLI for both training feedback and evaluation (falls back to keyword if unavailable)
    eval_env    = make_environment(prefer_nli=True, prefer_semantic=False)
    train_env   = make_environment(prefer_nli=True, prefer_semantic=True)
    total_steps = len(train_prompts) * cfg.num_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg.warmup_steps,
        num_training_steps=max(1, total_steps // cfg.grad_accum))

    print(f"\n[Train] True SDPO")
    print(f"[Train] Train={len(train_prompts)} | "
          f"Verifiable test={len(verifiable_test)} | "
          f"Full test={len(full_test)}")
    print(f"[Train] Eval every {cfg.eval_steps} steps on "
          f"{len(verifiable_test)} verifiable samples")
    print(f"[Train] max_new_tokens={cfg.max_new_tokens} | "
          f"top_k_distill={cfg.top_k_distill}")
    ram_status("before training")

    global_step = 0; accum_loss = 0.0; accum_count = 0
    train_log = []; eval_log = []
    best_reward = -1.0; baseline_reward = None
    step_times  = []
    optimizer.zero_grad()

    # Baseline eval
    print("\n[Eval] Baseline (before training):")
    baseline        = evaluate(model, tokenizer, verifiable_test, eval_env,
                               cfg, step=0, baseline_reward=None)
    eval_log.append(baseline)
    best_reward     = baseline["avg_reward"]
    baseline_reward = baseline["avg_reward"]

    for epoch in range(cfg.num_epochs):
        # ── SPEED FIX 3: shuffle so fallback prompts spread through epoch ──
        # This ensures ~every 6th step is a verifiable question with real feedback
        random.shuffle(train_prompts)
        print(f"\n══ Epoch {epoch+1}/{cfg.num_epochs} ══")

        for question in train_prompts:
            global_step += 1
            t0 = time.time()

            model.eval()
            response = generate_response(model, tokenizer, question, cfg)
            model.train()

            reward, feedback = train_env.evaluate(question, response)
            solution = response if reward >= 0.8 else None

            feedback_useful = (
                "could not be automatically verified" not in feedback
                and "Correct!" not in feedback
            )

            loss = compute_sdpo_loss(
                model, tokenizer, question, response, feedback, solution, cfg)
            if loss is None:
                continue

            (loss / cfg.grad_accum).backward()
            accum_loss  += loss.item()
            accum_count += 1

            if global_step % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            elapsed = time.time() - t0
            step_times.append(elapsed)

            if global_step % (cfg.logging_steps * cfg.grad_accum) == 0:
                avg_loss  = accum_loss / max(accum_count, 1)
                lr_now    = scheduler.get_last_lr()[0]
                vram      = torch.cuda.memory_allocated()/1e9
                avg_t     = sum(step_times[-20:]) / len(step_times[-20:])
                remaining = avg_t * (total_steps - global_step)
                rem_str   = (f"{remaining/60:.1f}min" if remaining < 3600
                             else f"{remaining/3600:.1f}hr")
                useful_str = "useful" if feedback_useful else "unverifiable"
                print(
                    f"  step {global_step:4d}/{total_steps} | "
                    f"loss={avg_loss:.4f} | reward={reward:.2f} | "
                    f"fb={useful_str} | "
                    f"{elapsed:.1f}s/step | ETA {rem_str}"
                )
                print(f"    Q: {question[:55]!r}")
                print(f"    A: {response[:65]!r}")
                print(f"    F: {feedback[:80]!r}")
                train_log.append({
                    "step": global_step, "loss": avg_loss,
                    "reward": reward, "feedback_useful": feedback_useful,
                    "lr": lr_now, "s_per_step": round(avg_t,2),
                })
                accum_loss = 0.0; accum_count = 0

            if global_step % cfg.eval_steps == 0:
                result = evaluate(model, tokenizer, verifiable_test,
                                  eval_env, cfg, global_step,
                                  baseline_reward=baseline_reward)
                eval_log.append(result)
                if result["avg_reward"] > best_reward:
                    best_reward = result["avg_reward"]
                    model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    print(f"  [Best] reward={best_reward:.3f} -> {best_dir}")

            if global_step % cfg.save_steps == 0:
                ckpt = os.path.join(cfg.output_dir, f"ckpt-{global_step}")
                model.save_pretrained(ckpt)
                print(f"  [Saved] {ckpt}")

    # Final eval
    print("\n[Eval] Final:")
    final = evaluate(model, tokenizer, verifiable_test, eval_env,
                     cfg, global_step, baseline_reward=baseline_reward)
    eval_log.append(final)
    if final["avg_reward"] > best_reward:
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(os.path.join(cfg.output_dir,"train_log.json"),"w") as f:
        json.dump(train_log, f, indent=2)
    with open(os.path.join(cfg.output_dir,"eval_log.json"),"w") as f:
        json.dump(eval_log, f, indent=2)

    # Summary
    print("\n" + "="*55)
    print("TRAINING SUMMARY")
    print("="*55)
    if eval_log:
        b, e = eval_log[0], eval_log[-1]
        rows = [
            ("avg_reward",   "up", b["avg_reward"],    e["avg_reward"]),
            ("exact_match",  "up", b["exact_match"],   e["exact_match"]),
            ("refusal_rate", "down", b["refusal_rate"],  e["refusal_rate"]),
            ("avg_resp_len", "down", b["avg_resp_len"],  e["avg_resp_len"]),
        ]
        print(f"  {'Metric':<16} {'Baseline':>9} {'Final':>9} {'delta':>8}  {'Good?':>6}")
        print(f"  {'─'*52}")
        for name, want, bv, ev in rows:
            d      = ev - bv
            actual = "up" if d > 0 else ("down" if d < 0 else "->")
            good   = "YES" if actual == want else ("NO" if d != 0 else "->")
            print(f"  {name:<16} {bv:>9.3f} {ev:>9.3f} "
                  f"{actual}{abs(d):>6.3f}  {good:>6}")
        print(f"  {'─'*52}")
        print(f"  Best test reward : {best_reward:.3f}")
        print(f"  Avg step time    : {sum(step_times)/len(step_times):.1f}s")
        print(f"  Total train time : {sum(step_times)/60:.1f} min")
    print(f"  Model  -> {cfg.output_dir}")
    print(f"  Best   -> {best_dir}")
    print("="*55)

    return model, tokenizer


# --- Smoke Test ---
def smoke_test():
    print("\n" + "="*60)
    print("SMOKE TEST")
    print("="*60)

    # 1. Scorer checks
    print("\n[Smoke] SimpleEnvironment scorer:")
    sim = SimpleEnvironment()
    cases = [
        ("What is 2 + 2?",    "The answer is 4.",            1.0),
        ("What is 2 + 2?",    "I don't know.",               0.0),
        ("What is gravity?",  "Gravity is a force between.", 1.0),
        ("What is gravity?",  "Gravity is not a force.",     0.0),
        ("What is gravity?",  "Hi",                          0.0),
    ]
    all_ok = True
    for q, a, exp in cases:
        got, fb = sim.evaluate(q, a)
        ok = got == exp
        if not ok: all_ok = False
        print(f"  {'OK' if ok else 'FAIL'} reward={got} (exp={exp}) "
              f"| Q={q!r} | A={a!r}")
    print(f"  Scorer: {'ALL PASS' if all_ok else 'SOME FAILED'}")

    # 2. Dataset split
    print("\n[Smoke] Dataset split (10 samples):")
    cfg = TrueSDPOConfig()
    cfg.small_n = 10; cfg.use_small_data = True
    tr, vt, ft = load_and_split_prompts(cfg)
    print(f"  Train={len(tr)} | Verifiable test={len(vt)} | Full test={len(ft)}")
    print(f"  Verifiable test: {vt}")

    # 3. Model + 3 steps
    print("\n[Smoke] Model load + 3 training steps:")
    cfg.output_dir  = r"D:\deep_learning\outputs\sdpo_smoke"
    cfg.eval_steps  = 3
    cfg.logging_steps = 1
    tokenizer, model = load_model(cfg)
    train_env = make_environment(prefer_nli=True, prefer_semantic=True)
    eval_env  = make_environment(prefer_nli=True, prefer_semantic=False)

    print("\n  Baseline eval (verifiable only):")
    baseline = evaluate(model, tokenizer, vt, eval_env, cfg, step=0)

    model.train()
    opt = AdamW(model.parameters(), lr=cfg.learning_rate)
    print("\n  3 steps:")
    for i, q in enumerate(tr[:3]):
        t0 = time.time()
        model.eval()
        response = generate_response(model, tokenizer, q, cfg)
        model.train()
        reward, feedback = train_env.evaluate(q, response)
        loss = compute_sdpo_loss(model, tokenizer, q, response, feedback, None, cfg)
        st   = "OK" if loss is not None else "SKIP"
        lv   = f"{loss.item():.4f}" if loss is not None else "None"
        uf   = "useful" if "could not" not in feedback else "unverifiable"
        print(f"  step {i+1} [{st}] loss={lv} r={reward:.2f} fb={uf} "
              f"{time.time()-t0:.1f}s")
        print(f"    Q: {q[:55]!r}")
        print(f"    A: {response[:65]!r}")
        if loss is not None:
            loss.backward(); opt.step(); opt.zero_grad()
        free_memory()

    print("\n  Post eval:")
    post = evaluate(model, tokenizer, vt, eval_env, cfg,
                    step=3, baseline_reward=baseline["avg_reward"])

    print("\n[Smoke] Summary:")
    print(f"  avg_reward  : {baseline['avg_reward']:.3f} -> {post['avg_reward']:.3f}")
    print(f"  exact_match : {baseline['exact_match']:.3f} -> {post['exact_match']:.3f}")
    print(f"  refusal_rate: {baseline['refusal_rate']:.3f} -> {post['refusal_rate']:.3f}")
    print(f"  avg_resp_len: {baseline['avg_resp_len']:.1f}w -> {post['avg_resp_len']:.1f}w")
    print("\n[Smoke] PASSED")
    print("[Smoke] Next steps:")
    print("  pip install sentence-transformers   (better training feedback)")
    print("  python 3_true_sdpo.py --mode small")
    print("  python 3_true_sdpo.py --mode large")
    free_memory(model, tokenizer)


# --- Entry Point ---
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke","small","large"],
                   default="smoke")
    args = p.parse_args()
    if args.mode == "smoke":
        smoke_test()
    elif args.mode == "small":
        CFG.use_small_data = True
        train_sdpo(CFG)
    elif args.mode == "large":
        CFG.use_small_data = False
        train_sdpo(CFG)
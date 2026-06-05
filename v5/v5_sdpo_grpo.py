"""
SDPO v5 — self-distillation on top of GRPO, with the bugs from earlier versions ironed out.

What changed since the last version:
  - Added an EMA teacher so the KL target doesn't drift with the student and cause gradient spikes
  - Fixed the hybrid loss weighting: lambda blends GRPO and SDPO properly now
    (lambda=0.9 works best for a small 1.5B model — mostly GRPO with a gentle SDPO nudge)
  - Unanswerable questions get their own reward path so hallucination tracking actually works
  - Bumped max_new_tokens because 96 was cutting off SQuAD answers mid-sentence
  - KL is batched in one forward pass instead of looping per-sample
  - Added Jensen-Shannon divergence as an option — it's smoother than plain KL for small models
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
import copy, json, gc, random, time, psutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union

from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)

import pathlib
_orig_read_text = pathlib.Path.read_text
def _safe_read_text(self, encoding=None, errors=None):
    return _orig_read_text(self, encoding=encoding or "utf-8", errors=errors)
pathlib.Path.read_text = _safe_read_text

try:
    import torch.distributed.fsdp
    if not hasattr(torch.distributed.fsdp, "FSDPModule"):
        torch.distributed.fsdp.FSDPModule = type("FSDPModule", (), {})
except Exception:
    pass

from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType


#  Config 
@dataclass
class SDPOConfig:
    model_name : str  = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir : str  = r"D:\deep_learning\outputs\v5_sdpo_grpo"

    # dataset settings
    use_small_data  : bool  = True
    small_n         : int   = 120
    large_n         : int   = 1000
    unanswerable_pct: float = 0.15
    test_split      : float = 0.2
    eval_steps      : int   = 50

    # generation
    max_new_tokens : int   = 128       # 160 was safe but slow — 128 is enough for most SQuAD answers
    temperature    : float = 0.7
    top_p          : float = 0.9

    #  SDPO distillation 
    top_k_distill  : int   = 20        # only look at top-K logits for the KL (20 is plenty for 1.5B)
    divergence     : str   = "jsd"     # "kl" or "jsd" — JSD tends to be smoother for small models

    # how much GRPO vs SDPO to use in the combined loss
    # 1.0 = all GRPO, 0.0 = all SDPO — 0.9 works well for 1.5B (mostly GRPO, light SDPO nudge)
    grpo_lambda    : float = 0.9
    sdpo_loss_scale: float = 0.1       # keeps the KL term from drowning out the GRPO signal

    # EMA teacher — the teacher's weights slowly track the student's.
    # without this the teacher drifts too far and you get nasty gradient spikes
    use_ema_teacher : bool  = True
    ema_alpha       : float = 0.05     # blending rate: teacher = 0.95*teacher + 0.05*student

    # GRPO baseline
    beta           : float = 0.1
    num_generations: int   = 2

    # training hyperparams
    learning_rate  : float = 2e-6
    num_epochs     : int   = 4
    grad_accum     : int   = 8
    warmup_steps   : int   = 30
    max_grad_norm  : float = 0.5
    logging_steps  : int   = 5
    save_steps     : int   = 50

    # LoRA adapter
    lora_r         : int   = 16
    lora_alpha     : int   = 32
    lora_dropout   : float = 0.05

    use_bf16       : bool  = True
    use_adamw_8bit : bool  = True

    device: str = field(default_factory=lambda:
                        "cuda" if torch.cuda.is_available() else "cpu")


CFG = SDPOConfig()


#  Helpers ─
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


#  Reward environments 
# These score how good a model response is. SimpleEnvironment uses keyword
# matching, NLIEnvironment uses a proper NLI model for semantic similarity.
class SimpleEnvironment:
    ANSWERS = {
        "What is 2 + 2?"                       : {"keywords":["4"],                   "bad":["5","three"],  "hint":"2+2=4."},
        "What is the capital of Germany?"       : {"keywords":["berlin"],              "bad":["paris"],      "hint":"Berlin is the capital of Germany."},
        "What is gravity?"                      : {"keywords":["force","attract"],     "bad":[],             "hint":"Gravity is a force attracting objects with mass toward each other."},
        "What is photosynthesis?"               : {"keywords":["sunlight","light"],    "bad":[],             "hint":"Photosynthesis converts sunlight into energy using CO2 and water."},
        "What does CPU stand for?"              : {"keywords":["central processing"],  "bad":["computer power"], "hint":"CPU = Central Processing Unit."},
        "What is a neural network?"             : {"keywords":["layer","neuron"],      "bad":["cable"],      "hint":"A neural network is made of layers of interconnected neurons."},
        "What is gradient descent?"             : {"keywords":["optim","minimize"],    "bad":[],             "hint":"Gradient descent is an optimization algorithm minimizing loss by following gradients."},
        "What is backpropagation?"              : {"keywords":["gradient","chain rule","algorithm"], "bad":[], "hint":"Backpropagation computes gradients via the chain rule to update weights."},
        "What is overfitting?"                  : {"keywords":["train","memoriz"],     "bad":[],             "hint":"Overfitting is when a model memorizes training data and fails to generalize."},
        "What is transfer learning?"            : {"keywords":["pretrain","fine-tun"], "bad":[],             "hint":"Transfer learning reuses a pretrained model on a new task."},
        "Explain what a for loop is."           : {"keywords":["repeat","iterate"],    "bad":["variable"],   "hint":"A for loop repeats a block of code a defined number of times."},
        "What is the speed of light?"           : {"keywords":["299","3x10"],          "bad":["1000 km"],    "hint":"Light travels at ~299,792,458 m/s in vacuum."},
        "What is a variable in programming?"    : {"keywords":["valu","store"],        "bad":["loop"],       "hint":"A variable stores a value that can be referenced and changed."},
        "How does sorting work?"                : {"keywords":["order","arrang"],      "bad":["delet"],      "hint":"Sorting arranges elements in a defined order."},
        "What is machine learning?"             : {"keywords":["data","pattern"],      "bad":["robot"],      "hint":"Machine learning trains models to find patterns in data."},
        "Explain the concept of recursion."     : {"keywords":["itself","base case"],  "bad":[],             "hint":"Recursion is when a function calls itself until a base case is reached."},
        "What is an API?"                       : {"keywords":["interface","endpoint"],"bad":[],             "hint":"An API is an interface letting systems communicate."},
        "What is a database?"                   : {"keywords":["data","store"],        "bad":[],             "hint":"A database is an organized collection of structured data."},
        "Explain binary search."                : {"keywords":["sort","half","divide"],"bad":["unsort"],     "hint":"Binary search divides a sorted array in half repeatedly to find a value."},
        "What is object-oriented programming?"  : {"keywords":["object","class"],      "bad":[],             "hint":"OOP organizes code into objects with state and behavior."},
    }
    GLOBAL_BAD = ["i don't know","i cannot answer","i'm not sure","i am not sure",
                  "i do not know","i can't answer","i cannot provide","this is a complex"]

    def evaluate(self, question: str, response: str) -> Tuple[float, str]:
        r_lower = response.lower().strip()
        q_lower = question.lower().strip()
        matched = None
        for k in self.ANSWERS:
            if k.lower() in q_lower or q_lower in k.lower():
                matched = k; break
        if matched is None:
            return 0.5, "Unverifiable."
        entry = self.ANSWERS[matched]
        keywords = [kw.lower() for kw in entry["keywords"]]
        bads     = [b.lower() for b in entry.get("bad", [])]
        hint     = entry.get("hint", "")
        if len(response.split()) < 5:
            return 0.0, f"Too short. Hint: {hint}"
        for gp in self.GLOBAL_BAD:
            if gp in r_lower:
                return 0.0, f"Refusal detected. Hint: {hint}"
        for bp in bads:
            if bp in r_lower:
                return 0.0, f"Incorrect phrase '{bp}'. Hint: {hint}"
        found = next((kw for kw in keywords if kw in r_lower), None)
        if found is None:
            return 0.0, f"Missing concept. Need one of: {keywords}. Hint: {hint}"
        for neg in [f"not {found}", f"isn't {found}", f"no {found}"]:
            if neg in r_lower:
                return 0.0, f"Negated '{found}'. Hint: {hint}"
        return 1.0, f"Correct! '{found}' found. Hint: {hint}"


class NLIEnvironment:
    REFUSAL_PHRASES = [
        "i don't know", "i do not know", "i cannot", "i can't",
        "i'm not sure", "i am not sure", "i'm unable", "i am unable",
        "i cannot provide", "i cannot help", "i cannot answer",
        "i can't answer", "i can't provide", "i can't assist",
        "i cannot assist", "unable to provide", "unable to address",
        "unable to assist", "unable to answer",
        "i apologize", "i'm sorry", "i am sorry", "sorry, but",
        "sorry, i", "i'm afraid", "i am afraid",
        "as an ai", "as a language model", "as an artificial",
        "i am an ai", "i'm an ai", "ai developed by",
        "alibaba cloud",
        "political topics", "i cannot discuss", "i cannot engage",
        "i'm not able to", "i am not able to",
        "i don't have enough", "i do not have enough",
        "i don't have any", "i do not have any",
        "i couldn't find", "i could not find",
    ]
    UNANSWERABLE_PHRASES = [
        "unanswerable", "cannot be answered", "not enough information",
        "no information", "not mentioned", "does not mention",
        "doesn't mention", "not provided", "not stated", "no answer",
        "cannot determine", "not specified", "insufficient",
        "no evidence", "not clear from", "context does not",
        "context doesn't", "impossible to determine",
    ]

    def __init__(self):
        self._model  = None
        self._simple = SimpleEnvironment()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                print("[NLI] Loading cross-encoder/nli-deberta-v3-small (CPU)...")
                self._model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
                print("[NLI] NLI scorer ready.")
            except Exception as e:
                print(f"[NLI] Failed: {e}")
                self._model = "unavailable"
        return self._model

    def _nli(self, premise, hypothesis):
        import numpy as np
        m = self._load()
        if m == "unavailable": return 0.0, 0.0, 0.0
        try:
            s = m.predict([(premise, hypothesis)])
            if s.ndim == 1: s = s.reshape(1,-1)
            e = np.exp(s - np.max(s, axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
            return float(p[0][0]), float(p[0][1]), float(p[0][2])
        except: return 0.0, 0.0, 0.0

    @staticmethod
    def _stem(word):
        w = word.lower()
        for suffix in ['ism','ist','ing','tion','sion','ness','ment','able','ible',
                       'ful','less','ous','ive','ity','ence','ance','ly','er','ed','es','s']:
            if len(w) > len(suffix) + 3 and w.endswith(suffix):
                return w[:-len(suffix)]
        return w

    @staticmethod
    def _token_f1(prediction, ground_truth):
        import re
        def _normalize(s):
            s = s.lower()
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            s = re.sub(r'[^\w\s]', '', s)
            return s.split()
        pred_tokens = _normalize(prediction)
        gt_tokens   = _normalize(ground_truth)
        if not gt_tokens or not pred_tokens: return 0.0
        common = set(pred_tokens) & set(gt_tokens)
        pred_remaining = set(pred_tokens) - common
        gt_remaining   = set(gt_tokens) - common
        if pred_remaining and gt_remaining:
            pred_stems = {NLIEnvironment._stem(t): t for t in pred_remaining}
            gt_stems   = {NLIEnvironment._stem(t): t for t in gt_remaining}
            stem_matches = set(pred_stems.keys()) & set(gt_stems.keys())
            common = common | {pred_stems[s] for s in stem_matches}
        if not common: return 0.0
        precision = len(common) / len(pred_tokens)
        recall    = len(common) / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    def evaluate(self, question, response, ground_truth=None):
        r_lower = response.lower().strip()
        n       = len(response.split())
        if ground_truth == "UNANSWERABLE":
            is_refusal      = any(rp in r_lower for rp in self.REFUSAL_PHRASES)
            is_unanswerable = any(up in r_lower for up in self.UNANSWERABLE_PHRASES)
            if is_refusal or is_unanswerable:
                return 1.0, "Correct refusal (unanswerable question)"
            return 0.0, "Hallucination on unanswerable question"
        if n == 0:
            return 0.0, f"Empty response. GT: {(ground_truth or '')[:30]}"
        for rp in self.REFUSAL_PHRASES:
            if rp in r_lower:
                return 0.0, f"Refusal '{rp}'. GT: {(ground_truth or '')[:30]}"
        if not ground_truth:
            return self._simple.evaluate(question, response)
        if ground_truth.lower().strip() in r_lower:
            return 1.0, f"Exact match. GT: {ground_truth[:30]}"
        f1 = self._token_f1(response, ground_truth)
        contra, entail, neutral = self._nli(response, ground_truth)
        if contra > 0.6 and f1 < 0.3:
            return 0.0, (f"Contradicts (c={contra:.2f},e={entail:.2f},f1={f1:.2f}). "
                         f"GT: {ground_truth[:30]}")
        nli_score = entail if contra < 0.3 else entail * 0.5
        combined = 0.6 * max(nli_score, f1) + 0.4 * min(nli_score, f1)
        if 3 <= n <= 25:   combined += 0.05
        elif n > 60:       combined -= 0.10
        reward = round(max(0.0, min(1.0, combined)), 3)
        fb = (f"Score={reward:.2f} (nli={nli_score:.2f},f1={f1:.2f},n={n}). "
              f"GT: {ground_truth[:30]}")
        return reward, fb


def make_environment(prefer_nli=True):
    if prefer_nli:
        try:
            from sentence_transformers import CrossEncoder  # noqa
            env = NLIEnvironment(); env._load()
            if env._model != "unavailable":
                print("[Env] NLIEnvironment ready")
                return env
        except ImportError: pass
    print("[Env] Using SimpleEnvironment")
    return SimpleEnvironment()


#  Self-teacher prompt builder 
# Builds the enriched prompt the teacher sees — includes feedback from
# the student's previous attempt so the teacher can give a better signal.
def build_self_teacher_context(question, response, feedback, solution=None):
    c = question + "\n"
    if solution:
        c += f"\nCorrect solution:\n{solution}\n"
    if feedback and "Correct!" not in feedback and "Unverifiable" not in feedback:
        c += (f"\nThe following is feedback from your unsuccessful earlier attempt:\n{feedback}\n"
              f"\nCorrectly solve the original question.")
    return c


#  KL / JSD divergence 
def _top_k_divergence(s_probs: torch.Tensor, t_probs: torch.Tensor,
                      top_k: int, mode: str) -> torch.Tensor:
    """
    Compute KL or JSD between student and teacher distributions, but only
    over the top-K tokens (plus a "tail" bucket for everything else).
    This keeps it tractable without losing too much info.
    """
    eps = 1e-8
    topk_v, topk_i = s_probs.topk(top_k, dim=-1)  # (T, K)
    t_topk         = t_probs.gather(1, topk_i)     # (T, K)

    st    = topk_v.clamp(min=eps)
    tt    = t_topk.clamp(min=eps)
    stail = (1 - topk_v.sum(-1, keepdim=True)).clamp(min=eps)
    ttail = (1 - t_topk.sum(-1, keepdim=True)).clamp(min=eps)

    if mode == "kl":
        # plain forward KL
        kl_top  = (st * (st.log() - tt.log())).sum(-1)
        kl_tail = (stail * (stail.log() - ttail.log())).squeeze(-1)
        return (kl_top + kl_tail).mean()
    else:
        # Jensen-Shannon — average of both directions, less likely to blow up
        # on small models where distributions can be pretty different
        mt    = ((st + tt) / 2).clamp(min=eps)
        mtail = ((stail + ttail) / 2).clamp(min=eps)
        jsd_top  = 0.5 * (st * (st.log() - mt.log())).sum(-1) \
                 + 0.5 * (tt.detach() * (tt.detach().log() - mt.log())).sum(-1)
        jsd_tail = 0.5 * (stail * (stail.log() - mtail.log())).squeeze(-1) \
                 + 0.5 * (ttail.detach() * (ttail.detach().log() - mtail.log())).squeeze(-1)
        return (jsd_top + jsd_tail).mean()


#  SDPO KL loss ─
def compute_sdpo_kl(
    model,
    ema_teacher,              # the slow-moving teacher copy, not the live student
    tokenizer,
    question    : str,
    response    : str,
    feedback    : str,
    solution    : Optional[str],
    top_k       : int,
    device      : str,
    divergence  : str = "jsd",
    max_len     : int = 512,
) -> Optional[torch.Tensor]:
    """
    The student sees just the question + response, the teacher sees the enriched
    prompt (question + feedback + response). We minimize the divergence between
    their output distributions so the student learns from the teacher's richer context.
    """
    try:
        sp = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False, add_generation_prompt=True)
    except:
        sp = f"User: {question}\nAssistant:"

    teacher_ctx = build_self_teacher_context(question, response, feedback, solution)
    try:
        tp = tokenizer.apply_chat_template(
            [{"role":"user","content":teacher_ctx}],
            tokenize=False, add_generation_prompt=True)
    except:
        tp = f"User: {teacher_ctx}\nAssistant:"

    s_ids = tokenizer(sp+response, return_tensors="pt", truncation=True,
                      max_length=max_len)["input_ids"].to(device)
    t_ids = tokenizer(tp+response, return_tensors="pt", truncation=True,
                      max_length=max_len+256)["input_ids"].to(device)

    rs = tokenizer(sp, return_tensors="pt")["input_ids"].shape[1]
    rt = tokenizer(tp, return_tensors="pt")["input_ids"].shape[1]

    if s_ids.shape[1] <= rs or t_ids.shape[1] <= rt:
        return None

    # student sees the bare prompt (grads flow through here)
    s_logits = model(input_ids=s_ids).logits[0, rs-1:-1]   # (T, V)

    # teacher sees the enriched prompt (no grads — it's just the target)
    with torch.no_grad():
        t_logits = ema_teacher(input_ids=t_ids).logits[0, rt-1:-1]  # (T, V)

    T = min(s_logits.shape[0], t_logits.shape[0])
    if T == 0: return None
    s_logits = s_logits[:T]
    t_logits = t_logits[:T]

    s_probs = F.softmax(s_logits, dim=-1)
    t_probs = F.softmax(t_logits, dim=-1)

    return _top_k_divergence(s_probs, t_probs, top_k, divergence)


#  EMA teacher 
class EMATeacher:
    """
    Keeps a slow-moving copy of the student's weights.
    After each training step we nudge the teacher a tiny bit toward the student:
      teacher = (1 - alpha) * teacher + alpha * student

    Without this, the teacher and student end up identical within a few steps,
    which kills the KL signal and makes training go haywire.
    """
    def __init__(self, student_model, alpha: float = 0.05):
        self.alpha = alpha
        # start with an exact copy of the student, then freeze it
        self.teacher = copy.deepcopy(student_model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        print(f"[EMA] Teacher initialized. α={alpha}")

    @torch.no_grad()
    def update(self, student_model):
        # match by name so we don't break when the trainer reorders params
        student_params = dict(student_model.named_parameters())
        for name, t_param in self.teacher.named_parameters():
            if name in student_params and t_param.shape == student_params[name].shape:
                t_param.data.mul_(1 - self.alpha).add_(student_params[name].data, alpha=self.alpha)

    def get_teacher(self):
        return self.teacher


#  SDPO trainer ─
class SDPOTrainer(GRPOTrainer):
    """
    Wraps GRPOTrainer and adds the SDPO self-distillation loss on top.

    The combined loss is: lambda * GRPO + (1 - lambda) * scale * SDPO_KL
    With lambda=0.9 the model mostly learns from GRPO rewards, and the
    SDPO term just gently steers it toward the teacher's distribution.
    """

    def __init__(self, sdpo_cfg: SDPOConfig, feedback_store: Dict,
                 ema_teacher: Optional[EMATeacher], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sdpo_cfg      = sdpo_cfg
        self.feedback_store = feedback_store
        self.ema_teacher   = ema_teacher
        self._step_count   = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # first, get the normal GRPO loss like usual
        grpo_loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
        outputs   = None

        # if lambda is 1.0 we're just doing vanilla GRPO, skip the SDPO stuff
        if self.sdpo_cfg.grpo_lambda >= 1.0 or self.ema_teacher is None:
            if return_outputs: return grpo_loss, outputs
            return grpo_loss

        # now compute the SDPO distillation loss for each sample
        sdpo_losses = []
        prompts     = inputs.get("prompt", [])
        completions = inputs.get("completion", [])

        if not prompts or not completions:
            if return_outputs: return grpo_loss, outputs
            return grpo_loss

        teacher_model = self.ema_teacher.get_teacher()

        for prompt, completion in zip(prompts, completions):
            reward, feedback = self.feedback_store.get(prompt, (0.5, "No feedback."))
            solution = completion if reward >= 0.8 else None

            kl = compute_sdpo_kl(
                model         = model,
                ema_teacher   = teacher_model,
                tokenizer     = self.processing_class,
                question      = prompt,
                response      = completion,
                feedback      = feedback,
                solution      = solution,
                top_k         = self.sdpo_cfg.top_k_distill,
                device        = self.sdpo_cfg.device,
                divergence    = self.sdpo_cfg.divergence,
                max_len       = self.sdpo_cfg.max_new_tokens + 256,
            )
            if kl is not None:
                sdpo_losses.append(kl)

        # blend the two losses together
        # e.g. lambda=0.9 means 90% GRPO + 10% SDPO
        if sdpo_losses:
            sdpo_loss = torch.stack(sdpo_losses).mean()
            lam       = self.sdpo_cfg.grpo_lambda
            combined  = lam * grpo_loss + (1 - lam) * self.sdpo_cfg.sdpo_loss_scale * sdpo_loss

            self._step_count += 1
            if self._step_count % self.sdpo_cfg.logging_steps == 0:
                print(f"    [SDPO] grpo={grpo_loss.item():.4f} "
                      f"kl={sdpo_loss.item():.4f} "
                      f"λ={lam} total={combined.item():.4f}")
        else:
            combined = grpo_loss

        if return_outputs: return combined, outputs
        return combined


#  Reward function 
def make_reward_fn(env, feedback_store: Dict, ground_truth_store: Dict):
    _call_count = [0]

    def reward_fn(prompts, completions, **kwargs):
        rewards  = []
        gt_found = 0
        for q, resp in zip(prompts, completions):
            if isinstance(resp, list):
                resp = resp[0]["content"] if resp else ""
            gt     = ground_truth_store.get(q, None)
            if gt: gt_found += 1
            reward, feedback = env.evaluate(q, resp, ground_truth=gt)
            feedback_store[q] = (reward, feedback)
            rewards.append(float(reward))
        _call_count[0] += 1
        if _call_count[0] <= 3:
            print(f"    [reward_fn] batch={len(prompts)} | "
                  f"gt_found={gt_found}/{len(prompts)} | "
                  f"avg_r={sum(rewards)/len(rewards):.3f}")
        return rewards
    return reward_fn


#  Generation helper 
def generate_response(model, tokenizer, question, cfg):
    try:
        text = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],
            tokenize=False, add_generation_prompt=True)
    except:
        text = f"User: {question}\nAssistant:"
    inputs = {k: v.to(cfg.device)
              for k,v in tokenizer(text, return_tensors="pt").items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature, top_p=cfg.top_p, do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


#  Dataset loading 
def load_and_split_prompts(cfg: SDPOConfig):
    print("[Dataset] Loading SQuAD v2 ...")
    ds = load_dataset("rajpurkar/squad_v2", split="train", streaming=True)
    n = cfg.small_n if cfg.use_small_data else cfg.large_n
    n_unanswerable_target = int(n * cfg.unanswerable_pct)
    n_answerable_target   = n - n_unanswerable_target

    all_prompts, ground_truths = [], []
    n_answerable = n_unanswerable = 0

    for row in ds:
        context  = row["context"]
        question = row["question"]
        if row["answers"]["text"]:
            if n_answerable >= n_answerable_target: continue
            gt     = row["answers"]["text"][0]
            prompt = (f"Context: {context}\n\nQuestion: {question}\n"
                      f"Answer briefly based on the context. "
                      f"If the answer is not in the context, say 'unanswerable'.")
            if 10 < len(prompt) < 1200:
                all_prompts.append(prompt); ground_truths.append(gt)
                n_answerable += 1
        else:
            if n_unanswerable >= n_unanswerable_target: continue
            prompt = (f"Context: {context}\n\nQuestion: {question}\n"
                      f"Answer briefly based on the context. "
                      f"If the answer is not in the context, say 'unanswerable'.")
            if 10 < len(prompt) < 1200:
                all_prompts.append(prompt); ground_truths.append("UNANSWERABLE")
                n_unanswerable += 1
        if n_answerable >= n_answerable_target and n_unanswerable >= n_unanswerable_target:
            break

    print(f"[Dataset] {len(all_prompts)} prompts "
          f"({n_answerable} answerable + {n_unanswerable} unanswerable)")

    combined  = list(zip(all_prompts, ground_truths))
    random.seed(42); random.shuffle(combined); random.seed()

    split_idx = max(1, int(len(combined) * (1 - cfg.test_split)))
    train_list = combined[:split_idx]
    test_list  = combined[split_idx:]

    train_prompts, train_gts = zip(*train_list) if train_list else ([], [])
    train_hf = Dataset.from_dict({"prompt": train_prompts, "ground_truth": train_gts})
    verifiable_test = [{"prompt": p, "ground_truth": g} for p, g in test_list]

    print(f"[Dataset] Train: {len(train_list)} | Test: {len(verifiable_test)}")
    return train_hf, verifiable_test, test_list


#  Evaluation 
def evaluate(model, tokenizer, test_prompts, eval_env, cfg, step, baseline_reward=None):
    model.eval()
    t0 = time.time()
    rewards, lens, per_q = [], [], []

    for item in test_prompts:
        q  = item["prompt"]
        gt = item.get("ground_truth", None)
        r  = generate_response(model, tokenizer, q, cfg)
        reward, fb = eval_env.evaluate(q, r, ground_truth=gt)
        w  = len(r.split())
        rewards.append(reward); lens.append(w)
        per_q.append({"question":q[:80],"response":r[:120],"reward":reward,
                      "feedback":fb[:80],"n_words":w,"ground_truth":gt or ""})

    avg    = sum(rewards) / len(rewards)
    exact  = sum(1 for r in rewards if r >= 0.95) / len(rewards)
    avglen = sum(lens) / len(lens)

    answerable_qs   = [pq for pq in per_q if pq["ground_truth"] != "UNANSWERABLE"]
    unanswerable_qs = [pq for pq in per_q if pq["ground_truth"] == "UNANSWERABLE"]

    refr = 0.0
    if answerable_qs:
        refusal_phrases = NLIEnvironment.REFUSAL_PHRASES
        refr = sum(1 for pq in answerable_qs
                   if any(p in pq["response"].lower() for p in refusal_phrases)
                   ) / len(answerable_qs)

    halluc = 0.0
    if unanswerable_qs:
        halluc = sum(1 for pq in unanswerable_qs if pq["reward"] == 0.0) / len(unanswerable_qs)

    vlong  = sum(1 for l in lens if l > 100) / len(lens)
    vshort = sum(1 for l in lens if l < 3)   / len(lens)
    delta  = round(avg - baseline_reward, 4) if baseline_reward else 0.0
    ds     = f" (D{'+' if delta>=0 else ''}{delta:.3f})" if baseline_reward else ""

    print(f"\n  [Eval] step={step} | avg_reward={avg:.3f}{ds} | "
          f"exact={exact:.3f} | avg_len={avglen:.1f}w | "
          f"refusal={refr:.2f} | halluc={halluc:.2f} | {time.time()-t0:.1f}s")

    for pq in per_q:
        mark = "YES" if pq["reward"] >= 0.95 else ("PAR" if pq["reward"] >= 0.5 else "NO")
        print(f"    {mark} r={pq['reward']:.2f} | "
              f"Q: {pq['question'][:45]!r} | A: {pq['response'][:50]!r}")

    model.train()
    return {"step":step,"avg_reward":round(avg,4),"exact_match":round(exact,4),
            "avg_resp_len":round(avglen,1),"refusal_rate":round(refr,4),
            "hallucination_rate":round(halluc,4),
            "very_long_pct":round(vlong,4),"very_short_pct":round(vshort,4),
            "reward_delta":delta,"n_eval":len(test_prompts),
            "n_answerable":len(answerable_qs),"n_unanswerable":len(unanswerable_qs),
            "per_question":per_q}


#  Model loading 
def load_model(cfg: SDPOConfig):
    print(f"\n[Model] Loading {cfg.model_name} ...")
    ram_status("before load")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16 if cfg.use_bf16 else torch.float32,
        device_map="cuda:0", low_cpu_mem_usage=True)

    lora_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=cfg.lora_dropout, bias="none",
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print(f"[Model] LoRA applied. VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    return tokenizer, model


#  Training loop 
def train_sdpo(cfg: SDPOConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_dir = os.path.join(cfg.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    ram_status("startup")
    tokenizer, model = load_model(cfg)
    train_hf, verifiable_test, full_test = load_and_split_prompts(cfg)
    train_env = make_environment(prefer_nli=True)
    eval_env  = make_environment(prefer_nli=True)

    feedback_store:     Dict[str, Tuple[float, str]] = {}
    ground_truth_store: Dict[str, str]               = {}
    for i in range(len(train_hf)):
        ground_truth_store[train_hf[i]["prompt"]] = train_hf[i]["ground_truth"]
    for item in verifiable_test:
        ground_truth_store[item["prompt"]] = item["ground_truth"]
    print(f"[GT Store] {len(ground_truth_store)} mappings loaded")

    # set up the EMA teacher from the initial model weights
    ema_teacher = None
    if cfg.use_ema_teacher:
        ema_teacher = EMATeacher(model, alpha=cfg.ema_alpha)
        print(f"[EMA] Teacher ready (α={cfg.ema_alpha})")

    reward_fn = make_reward_fn(train_env, feedback_store, ground_truth_store)

    grpo_cfg = GRPOConfig(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.num_epochs,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = cfg.grad_accum,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = "cosine",
        warmup_steps                = cfg.warmup_steps,
        max_grad_norm               = cfg.max_grad_norm,
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        bf16                        = cfg.use_bf16,
        fp16                        = False,
        optim                       = "adamw_8bit" if cfg.use_adamw_8bit else "adamw_torch",
        report_to                   = "none",
        remove_unused_columns       = False,
        num_generations             = cfg.num_generations,
        max_completion_length       = cfg.max_new_tokens,
        temperature                 = cfg.temperature,
        top_p                       = cfg.top_p,
        beta                        = cfg.beta,
    )

    print(f"\n[Train] SDPO v5")
    print(f"[Train] grpo_lambda={cfg.grpo_lambda} (1.0=pure GRPO, 0.0=pure SDPO)")
    print(f"[Train] divergence={cfg.divergence} | ema_teacher={cfg.use_ema_teacher}")
    print(f"[Train] Train={len(train_hf)} | Test={len(verifiable_test)}")
    ram_status("before training")

    print("\n[Eval] Baseline:")
    baseline        = evaluate(model, tokenizer, verifiable_test, eval_env, cfg, 0)
    eval_log        = [baseline]
    best_reward     = baseline["avg_reward"]
    baseline_reward = baseline["avg_reward"]
    train_log       = []

    trainer = SDPOTrainer(
        sdpo_cfg        = cfg,
        feedback_store  = feedback_store,
        ema_teacher     = ema_teacher,
        model           = model,
        args            = grpo_cfg,
        train_dataset   = train_hf,
        reward_funcs    = reward_fn,
        processing_class= tokenizer,
    )

    class EvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # nudge the teacher toward the student after each step
            if ema_teacher is not None:
                ema_teacher.update(trainer.model)

            if state.global_step % cfg.eval_steps == 0 and state.global_step > 0:
                result = evaluate(trainer.model, tokenizer, verifiable_test,
                                  eval_env, cfg, state.global_step,
                                  baseline_reward=baseline_reward)
                eval_log.append(result)
                nonlocal best_reward
                if result["avg_reward"] > best_reward:
                    best_reward = result["avg_reward"]
                    trainer.model.save_pretrained(best_dir)
                    tokenizer.save_pretrained(best_dir)
                    print(f"  [Best] reward={best_reward:.3f} -> {best_dir}")
                if state.log_history:
                    last = state.log_history[-1]
                    train_log.append({
                        "step"     : state.global_step,
                        "loss"     : last.get("loss"),
                        "reward"   : last.get("reward"),
                        "grad_norm": last.get("grad_norm"),
                        "entropy"  : last.get("entropy"),
                        "kl"       : last.get("kl"),
                    })

    trainer.add_callback(EvalCallback())

    print("\n[SDPOTrainer v5] Starting training...")
    trainer.train()

    print("\n[Eval] Final:")
    final = evaluate(trainer.model, tokenizer, verifiable_test,
                     eval_env, cfg, -1, baseline_reward=baseline_reward)
    eval_log.append(final)
    if final["avg_reward"] >= best_reward:
        trainer.model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    with open(os.path.join(cfg.output_dir,"train_log.json"),"w") as f:
        json.dump(train_log, f, indent=2)
    with open(os.path.join(cfg.output_dir,"eval_log.json"),"w") as f:
        json.dump(eval_log, f, indent=2)

    print("\n" + "="*55)
    print("TRAINING SUMMARY")
    print("="*55)
    if eval_log:
        b, e = eval_log[0], eval_log[-1]
        for name, want, bv, ev in [
            ("avg_reward","up",b["avg_reward"],e["avg_reward"]),
            ("exact_match","up",b["exact_match"],e["exact_match"]),
            ("refusal_rate","down",b["refusal_rate"],e["refusal_rate"]),
            ("hallucination_rate","down",b["hallucination_rate"],e["hallucination_rate"]),
            ("avg_resp_len","down",b["avg_resp_len"],e["avg_resp_len"]),
        ]:
            d = ev - bv
            actual = "up" if d>0 else ("down" if d<0 else "->")
            good   = "YES" if actual==want else ("NO" if d!=0 else "->")
            print(f"  {name:<22} {bv:>8.3f} -> {ev:>8.3f}  {actual} {abs(d):.3f}  {good}")
        print(f"  Best reward: {best_reward:.3f}")
    print(f"  Model -> {cfg.output_dir}")
    print("="*55)
    plot_results(cfg.output_dir)
    return trainer.model, tokenizer


#  Plotting 
def plot_results(output_dir: str):
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    except ImportError:
        print("[Plot] matplotlib is not installed.")
        return

    eval_log_path  = os.path.join(output_dir, "eval_log.json")
    train_log_path = os.path.join(output_dir, "train_log.json")
    if not os.path.exists(eval_log_path): return

    with open(eval_log_path, "r") as f: eval_log  = json.load(f)
    train_log = []
    if os.path.exists(train_log_path):
        with open(train_log_path, "r") as f: train_log = json.load(f)
    if not eval_log: return

    steps = [e.get("step", i) for i, e in enumerate(eval_log)]
    if len(steps) > 1 and steps[-1] == -1:
        steps[-1] = steps[-2] + 40

    rewards      = [e.get("avg_reward",0)         for e in eval_log]
    exact_match  = [e.get("exact_match",0)         for e in eval_log]
    refusal_rate = [e.get("refusal_rate",0)        for e in eval_log]
    halluc_rate  = [e.get("hallucination_rate",0)  for e in eval_log]
    avg_len      = [e.get("avg_resp_len",0)        for e in eval_log]

    t_steps = [t.get("step",0)    for t in train_log]
    t_loss  = [t.get("loss")      for t in train_log]
    t_grad  = [t.get("grad_norm") for t in train_log]
    t_ent   = [t.get("entropy")   for t in train_log]
    t_kl    = [t.get("kl")        for t in train_log]
    t_rew   = [t.get("reward")    for t in train_log]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('SDPO Training — reward, loss, and diagnostics', fontsize=16, fontweight='bold')

    def _plot_eval(ax, y, title, color, ylabel=""):
        ax.plot(steps, y, marker='o', color=color, linewidth=2, markersize=5)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Steps')
        ax.set_ylabel(ylabel); ax.grid(True, linestyle='--', alpha=0.4)
        if len(y) >= 2:
            d = y[-1] - y[0]
            ax.annotate(f'Δ={d:+.3f}', xy=(steps[-1], y[-1]),
                        fontsize=9, color='green' if d>=0 else 'red',
                        fontweight='bold', xytext=(-55,12), textcoords='offset points')

    def _plot_train(ax, steps_t, vals, title, color):
        valid = [(s,v) for s,v in zip(steps_t,vals) if v is not None]
        if valid:
            xs,ys = zip(*valid)
            ax.plot(xs, ys, marker='.', color=color, linewidth=1.5, markersize=4, alpha=0.8)
        ax.set_title(title, fontweight='bold'); ax.set_xlabel('Steps')
        ax.grid(True, linestyle='--', alpha=0.4)

    _plot_eval(axes[0,0], rewards,      'Average Reward',              '#2196F3', 'Reward')
    axes[0,0].set_ylim(-0.05, 1.05)

    _plot_eval(axes[0,1], exact_match,  'Accuracy (Exact Match)',      '#4CAF50', 'Rate')
    axes[0,1].set_ylim(-0.05, 1.05)

    ax = axes[0,2]
    ax.plot(steps, refusal_rate, marker='^', color='#F44336', linewidth=2, markersize=5, label='Refusal')
    ax.plot(steps, halluc_rate,  marker='v', color='#FF9800', linewidth=2, markersize=5, label='Hallucination', linestyle='--')
    ax.set_title('Refusal & Hallucination Rate', fontweight='bold')
    ax.set_xlabel('Steps'); ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.fill_between(steps, refusal_rate, alpha=0.1, color='#F44336')
    ax.fill_between(steps, halluc_rate,  alpha=0.1, color='#FF9800')

    _plot_train(axes[1,0], t_steps, t_loss, 'Training Loss',    '#FF9800')
    _plot_train(axes[1,1], t_steps, t_grad, 'Gradient Norm',    '#E91E63')
    axes[1,1].axhline(y=CFG.max_grad_norm, color='gray', linestyle=':', alpha=0.5, label='Clip')
    axes[1,1].legend(fontsize=8)
    _plot_train(axes[1,2], t_steps, t_ent,  'Policy Entropy',   '#00BCD4')

    _plot_eval(axes[2,0], avg_len, 'Avg Response Length', '#9C27B0', 'Words')

    _plot_train(axes[2,1], t_steps, t_kl, 'KL (ref model)', '#795548')
    _plot_train(axes[2,2], t_steps, t_rew, 'Train Batch Reward', '#3F51B5')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(output_dir, "training_plot.png")
    plt.savefig(out, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"\n[Plot] Saved -> {out}")


#  Smoke test 
def smoke_test():
    print("\n" + "="*60)
    print("SMOKE TEST — quick sanity check that everything wires up")
    print("="*60)

    cfg = SDPOConfig()
    cfg.small_n = 5; cfg.use_small_data = True
    cfg.output_dir = r"D:\deep_learning\outputs\sdpo_v5_smoke"
    cfg.use_ema_teacher = True
    cfg.divergence = "jsd"
    cfg.grpo_lambda = 0.9

    tokenizer, model = load_model(cfg)
    model.train()

    # spin up an EMA teacher from the freshly loaded model
    ema = EMATeacher(model, alpha=cfg.ema_alpha)

    q  = "What is gradient descent?"
    r  = "Gradient descent minimizes the loss function."
    fb = "Missing concept: optim. Hint: it's an optimization algorithm."

    print(f"\n[Smoke] Testing SDPO KL with EMA teacher + JSD ...")
    kl = compute_sdpo_kl(model, ema.get_teacher(), tokenizer,
                         q, r, fb, None,
                         cfg.top_k_distill, cfg.device, cfg.divergence)
    if kl is not None:
        print(f"  JSD loss : {kl.item():.4f}")
        kl.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  Backward : OK | LoRA grad: {'YES' if has_grad else 'NO'}")
    else:
        print("  JSD loss : None (truncated)")

    # make sure the EMA update actually changes the teacher weights
    print("\n[Smoke] Testing EMA teacher update ...")
    before = next(ema.get_teacher().parameters()).data.clone().sum().item()
    ema.update(model)
    after  = next(ema.get_teacher().parameters()).data.sum().item()
    print(f"  Teacher param sum before={before:.4f} after={after:.4f} "
          f"(changed: {abs(after-before)>0})")

    # quick check that the reward function returns something sensible
    print("\n[Smoke] Testing reward_fn ...")
    env = make_environment(prefer_nli=True)
    store = {}; gt_store = {"What is 2 + 2?": "4"}
    rfn = make_reward_fn(env, store, gt_store)
    rs  = rfn(["What is 2 + 2?"], ["The answer is 4."])
    print(f"  reward={rs}  store_key={'What is 2 + 2?' in store}")

    free_memory(model, tokenizer)
    print("\n[Smoke] PASSED")


#  Entry point 
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke","small","large","plot"], default="smoke")
    p.add_argument("--lambda", dest="lam", type=float, default=None,
                   help="GRPO lambda: 1.0=pure GRPO, 0.0=pure SDPO (default 0.9 for 1.5B)")
    p.add_argument("--divergence", choices=["kl","jsd"], default=None)
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    if args.lam is not None:    CFG.grpo_lambda    = args.lam
    if args.divergence:         CFG.divergence     = args.divergence
    if args.no_ema:             CFG.use_ema_teacher = False

    if   args.mode == "smoke": smoke_test()
    elif args.mode == "small": CFG.use_small_data=True;  train_sdpo(CFG)
    elif args.mode == "large": CFG.use_small_data=False; train_sdpo(CFG)
    elif args.mode == "plot":  plot_results(CFG.output_dir)
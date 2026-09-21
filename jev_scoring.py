"""
jev_scoring.py -- The CORRECTED "System One" mechanism (next-token scoring)
==========================================================================
This is the honest reproduction of how Jev / JEV-CPU / Avi Chawla's "build
your own Jev" actually turn a causal LLM into a non-autoregressive decision
engine. It differs fundamentally from the earlier `smollm_system_one.py`
prototype in this repo:

    smollm_system_one.py (v1, WRONG):
        - deletes the pretrained `lm_head`
        - bolts on three FRESH, RANDOMLY INITIALISED nn.Linear heads
        - => with no training, the "decisions" it prints are pure noise
          (P(True)=0.505, choice confidence ~1/K). The pretrained model's
          knowledge is thrown away.

    jev_scoring.py (v2, CORRECT):
        - KEEPS the pretrained `lm_head`
        - phrases each decision as a single-token multiple-choice prompt
        - runs ONE forward pass (prefill only) and reads the logits at the
          option-token positions of the LAST token's vocabulary vector
        - restricted-softmax over just those option tokens => a real,
          zero-shot probability distribution, no retraining required.

This is exactly the SGLang `/v1/score` mechanism, implemented here directly
in transformers so it runs on a laptop CPU/MPS with no server.

All numbers this file prints are MEASURED on the machine you run it on.
Nothing here is simulated with time.sleep().
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# 1. Decision contract (mirrors the official Jev request/response shape)
# ============================================================================

@dataclass
class Noul:
    instructions: str
    type: str = "noul"


@dataclass
class Choice:
    instructions: str
    criteria: Dict[str, str] = field(default_factory=dict)  # {option_key: description}
    type: str = "choice"


@dataclass
class Score:
    instructions: str
    criteria: List[str] = field(default_factory=list)  # ordered level descriptions
    type: str = "score"


# ============================================================================
# 2. Confidence from distribution geometry (entropy + winner margin)
# ============================================================================

def confidence_from_probs(probs: torch.Tensor) -> float:
    """0.5 * (1 - H/ln K) + 0.5 * (p_top - p_second). Uniform -> 0, one-hot -> 1."""
    p = probs.detach().cpu().float()
    K = p.numel()
    if K <= 1:
        return 1.0
    safe = p.clamp(min=1e-12)
    entropy = float(-(safe * safe.log()).sum())
    norm_neg_entropy = 1.0 - entropy / math.log(K)
    top2 = torch.topk(p, 2).values
    margin = float(top2[0] - top2[1])
    return max(0.0, min(1.0, 0.5 * norm_neg_entropy + 0.5 * margin))


# ============================================================================
# 3. The scorer
# ============================================================================

# Letters used as single-token option labels. We verify single-tokenness at runtime.
_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class JevScorer:
    def __init__(self, model_id: str = "HuggingFaceTB/SmolLM-135M", device: Optional[str] = None):
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[Init] Loading {model_id} on {self.device} (lm_head KEPT)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        self.model_id = model_id

    # --- token-id helpers ---------------------------------------------------

    def _single_token_id(self, text: str) -> Optional[int]:
        """Return the token id if `text` encodes to exactly ONE token, else None."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    def _resolve_label_tokens(self, n: int) -> Tuple[List[str], List[int]]:
        """Pick n letters that are each exactly one token in this tokenizer."""
        letters, ids = [], []
        for L in _LETTERS:
            tid = self._single_token_id(L)
            if tid is not None:
                letters.append(L)
                ids.append(tid)
            if len(letters) == n:
                break
        if len(letters) < n:
            raise ValueError(f"Only {len(letters)} single-token letters available; need {n}.")
        return letters, ids

    def _yes_no_tokens(self) -> Tuple[int, int]:
        """Token ids for the True / False labels of a Noul (yes/no)."""
        for yes, no in [("Yes", "No"), ("yes", "no"), ("A", "B")]:
            y, n = self._single_token_id(yes), self._single_token_id(no)
            if y is not None and n is not None:
                return y, n
        raise ValueError("No single-token yes/no pair found in tokenizer.")

    # --- the single scoring forward pass ------------------------------------

    @torch.no_grad()
    def _score_positions(self, prompt: str, token_ids: List[int]) -> torch.Tensor:
        """ONE forward pass. Read logits at `token_ids` from the last position,
        then restricted-softmax over just those choices."""
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**enc)                       # prefill only; no generation loop
        last_logits = out.logits[0, -1, :]            # [vocab]
        chosen = last_logits[torch.tensor(token_ids, device=self.device)]
        return F.softmax(chosen.float(), dim=-1).cpu()  # [K]

    # --- public per-primitive API ------------------------------------------

    def _choice_prompt(self, state: str, instructions: str, letters, descriptions) -> str:
        lines = "\n".join(f"{L} = {d}" for L, d in zip(letters, descriptions))
        return (
            f"State:\n{state}\n\n"
            f"Question: {instructions}\n\n"
            f"Allowed labels:\n{lines}\n\n"
            f"Answer with exactly one label.\nLabel:"
        )

    def noul(self, state: str, q: Noul) -> Dict:
        y, n = self._yes_no_tokens()
        prompt = (
            f"State:\n{state}\n\nQuestion: {q.instructions}\n\n"
            f"Answer Yes or No.\nAnswer:"
        )
        probs = self._score_positions(prompt, [y, n])
        return {"type": "noul", "noul": round(float(probs[0]), 4)}

    def choice(self, state: str, q: Choice) -> Dict:
        keys = list(q.criteria.keys())
        letters, ids = self._resolve_label_tokens(len(keys))
        descriptions = [f"{k}: {q.criteria[k]}" for k in keys]
        prompt = self._choice_prompt(state, q.instructions, letters, descriptions)
        probs = self._score_positions(prompt, ids)
        prob_dict = {k: round(float(p), 4) for k, p in zip(keys, probs)}
        top = int(torch.argmax(probs))
        return {
            "type": "choice",
            "choice": keys[top],
            "probabilities": prob_dict,
            "confidence": round(confidence_from_probs(probs), 4),
        }

    def score(self, state: str, q: Score) -> Dict:
        levels = q.criteria
        letters, ids = self._resolve_label_tokens(len(levels))
        descriptions = [f"level {i}: {lvl}" for i, lvl in enumerate(levels)]
        prompt = self._choice_prompt(state, q.instructions, letters, descriptions)
        probs = self._score_positions(prompt, ids)
        expected = float(sum(i * float(p) for i, p in enumerate(probs)))
        return {
            "type": "score",
            "score": round(expected, 2),
            "legend": {str(i): lvl for i, lvl in enumerate(levels)},
            "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
            "confidence": round(confidence_from_probs(probs), 4),
        }

    def evaluate(self, state: str, questions: Dict[str, Union[Noul, Choice, Score]]) -> Dict:
        """Run all questions against one shared state, timing the whole thing."""
        t0 = time.perf_counter()
        answers: Dict[str, Dict] = {}
        for qid, q in questions.items():
            if isinstance(q, Noul):
                answers[qid] = self.noul(state, q)
            elif isinstance(q, Choice):
                answers[qid] = self.choice(state, q)
            elif isinstance(q, Score):
                answers[qid] = self.score(state, q)
        latency_ms = (time.perf_counter() - t0) * 1000
        in_tokens = len(self.tokenizer.encode(state))
        return {
            "model": f"jev-scoring::{self.model_id}",
            "answers": answers,
            "usage": {"input_tokens": in_tokens, "output_tokens_generated": 0},
            "latency_ms": round(latency_ms, 2),
        }


# ============================================================================
# 4. Real autoregressive baseline on the SAME model (for an honest comparison)
# ============================================================================

@torch.no_grad()
def autoregressive_baseline(scorer: JevScorer, state: str, instructions: str,
                            options: List[str], max_new_tokens: int = 40) -> Dict:
    """Ask the same model to WRITE a JSON answer, token by token. Measured."""
    prompt = (
        f"State:\n{state}\n\nQuestion: {instructions}\n"
        f"Options: {options}\n"
        f'Reply with JSON like {{"choice": "<one option>"}}.\nJSON:'
    )
    enc = scorer.tokenizer(prompt, return_tensors="pt").to(scorer.device)
    t0 = time.perf_counter()
    gen = scorer.model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=scorer.tokenizer.pad_token_id,
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    new_tokens = gen.shape[1] - enc.input_ids.shape[1]
    text = scorer.tokenizer.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return {"latency_ms": round(latency_ms, 2), "output_tokens": int(new_tokens),
            "raw_text": text.strip()}


# ============================================================================
# 5. Demo on the official-style incident example
# ============================================================================

def run_demo(model_id: str, device: Optional[str]):
    scorer = JevScorer(model_id=model_id, device=device)

    state = (
        "Customer message: 'Our payment gateway is returning 500 errors for all "
        "European Mastercard transactions. Checkout is down and customers cannot "
        "place orders. SLA is 99.99%. We need this escalated immediately.'"
    )
    questions = {
        "is_outage": Noul(instructions="Does this describe an active production service outage?"),
        "route_team": Choice(
            instructions="Which engineering team should handle this?",
            criteria={
                "billing": "Invoices, credit-card failures, refunds, payment gateway",
                "frontend": "UI, layout, mobile rendering bugs",
                "security": "Credential leaks, suspicious logins, auth",
                "docs": "SDK documentation and tutorials",
            },
        ),
        "severity": Score(
            instructions="Rate the incident severity.",
            criteria=[
                "SEV-3: minor, clear workaround",
                "SEV-2: moderate degradation, no revenue loss",
                "SEV-1: significant degradation of sub-components",
                "SEV-0: total outage, immediate business disruption",
            ],
        ),
    }

    print("\n=== System One scoring (single forward pass per question, 0 tokens generated) ===")
    resp = scorer.evaluate(state, questions)
    print(json.dumps(resp, indent=2))

    print("\n=== Autoregressive baseline on the SAME model (writes JSON) ===")
    base = autoregressive_baseline(
        scorer, state, "Which engineering team should handle this?",
        ["billing", "frontend", "security", "docs"],
    )
    print(json.dumps(base, indent=2))

    # Head-to-head latency for the routing decision only (apples to apples).
    t0 = time.perf_counter()
    scorer.choice(state, questions["route_team"])
    score_only_ms = (time.perf_counter() - t0) * 1000
    print("\n=== Head-to-head (routing decision) ===")
    print(f"  scoring (read logits, 0 output tokens) : {score_only_ms:8.2f} ms")
    print(f"  generation ({base['output_tokens']} output tokens)      : {base['latency_ms']:8.2f} ms")
    if base["latency_ms"] > 0:
        print(f"  speedup                                : {base['latency_ms']/max(score_only_ms,1e-6):8.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceTB/SmolLM-135M")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run_demo(args.model_id, args.device)

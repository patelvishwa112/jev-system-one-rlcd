# Jev "System One" — an honest local reproduction of non-autoregressive decision inference

> **Turning an open-source LLM into a fast, typed decision engine — the way Jev / JEV-CPU / Avi Chawla actually do it, plus a candid account of the wrong turn we took first.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace SmolLM-135M](https://img.shields.io/badge/%F0%9F%A4%97%20SmolLM-135M-yellow.svg)](https://huggingface.co/HuggingFaceTB/SmolLM-135M)
[![Article](https://img.shields.io/badge/Article-Interactive%20Blog-success.svg)](./jev_system_one_article.html)

---

## What this is

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe AI,
launched Sept 2026) is a **"System One" model**: instead of writing text
autoregressively, it returns **typed decisions** — `Noul` (yes/no → probability),
`Choice` (pick one of a closed set → distribution), `Score` (position on an ordered
scale → expectation) — with a **calibrated confidence** on each, all evaluated in a
**single parallel forward pass** against one shared `state`.

This repo reproduces the **inference mechanism** locally on a laptop and is honest
about the boundary: like Avi Chawla's ["Build your own Jev"](https://x.com/_avichawla/status/2101563610644496464)
and the [`Meanblock/JEV-CPU`](https://huggingface.co/Meanblock/JEV-CPU) model, we can
reproduce *how Jev reads decisions out of a model*, but **not** Jev's weights or its
full RLCD calibration stack.

## The mechanism (what actually works)

A causal LLM, given a prompt, produces one logit per vocabulary token for the *next*
position. For a bounded decision we only care about that first vector:

1. Phrase the decision as a multiple-choice prompt with **single-token letter labels**
   (`A`, `B`, `C`, …), with the semantic option descriptions in the prompt text.
2. Verify each label is exactly **one token** (letters, not words — `"billing"` may be
   several tokens; a leading space changes the id).
3. Run **one forward pass** (prefill only). Take the last position's logit vector.
4. Read only the logits at the option-letter token positions and apply a **softmax
   restricted to those positions** → a real probability distribution over the allowed
   answers. **The model never generates the answer.**

This is exactly the SGLang `/v1/score` mechanism. `jev_scoring.py` implements it directly
in `transformers` so it runs with no server.

## ⚠️ What our first attempt got wrong

`smollm_system_one.py` (kept in the repo as `v1`, for the comparison) took the *wrong*
approach: it **deletes the pretrained `lm_head`** and bolts on three **freshly
initialised, untrained** `nn.Linear` heads. With no training those heads emit **noise** —
the demo's `P(True)=0.505` and choice confidence `~0.25` (uniform over 4 options) are not
decisions, they're random projections. It throws away everything the pretrained model
knows. Its headline numbers (`67ms`, `ECE 28.4% → 2.1%`) were **simulated** (`time.sleep`,
hardcoded soft targets), not measured.

`jev_scoring.py` (`v2`) fixes this by **keeping** the `lm_head` and reading option-token
logits — the correct mechanism above. No new parameters, no training needed to get a real
signal. See [`jev_system_one_article.html`](./jev_system_one_article.html) for the full story.

## Measured results (real, on an Apple M1 8GB laptop)

From `jev_scoring.py` (see [`results/measured_results.json`](./results/measured_results.json)) —
routing one support ticket to one of four teams:

| Model | Scoring (read logits) | Generation (same model, 40 tok JSON) | Speedup |
| :-- | :-- | :-- | :-- |
| SmolLM-135M | **60.1 ms**, 0 output tokens | 1568 ms (emitted invalid `"choice":"1"`) | **~26×** |
| SmolLM2-135M-Instruct | **58.5 ms**, 0 output tokens | 1143 ms | **~19.5×** |

**Honest caveat:** 135M models are too small to route correctly — both mis-route to
`frontend` at ~0.11 confidence (near chance). The *mechanism* is fast and structurally
safe (invalid outputs are impossible); **accuracy is a separate axis** that needs a
capable model + calibration. JEV-CPU reports 0.44 balanced accuracy at 0.6B, rising to
0.69 at 2B and 0.81 at 4B.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# The CORRECT mechanism (read option-token logits, 0 generated tokens):
python3 jev_scoring.py                                   # SmolLM-135M
python3 jev_scoring.py --model_id HuggingFaceTB/SmolLM2-135M-Instruct

# The v1 attempt, kept for the "what we got wrong" comparison:
python3 smollm_system_one.py

# RLCD calibration training loop (proper scoring rules on soft targets):
python3 train_rlcd.py --epochs 3 --batch_size 4 --lr 5e-5
```

For a production-grade path, use SGLang's `/v1/score` endpoint (Avi Chawla's approach)
instead of the in-process `transformers` forward pass.

## RLCD — the part nobody reproduces

Jev's accuracy *and* calibration come from **Reinforcement Learning for Calibrated
Decisions (RLCD)**, framed as an RL problem:

- **Agent / policy** — the head that emits the decision distribution.
- **Environment / state** — the input `state` + question + allowed options.
- **Action space** — the distribution over the *declared options* (2 / ≤255 / 2–10 bins),
  not the vocabulary.
- **Reward = a strictly proper scoring rule** (Brier, log-loss): maximized *only* by
  reporting your true believed probability.

| Dimension | RLHF (PPO/DPO) | RLVR (verifiable) | **RLCD (calibrated)** |
| :--- | :--- | :--- | :--- |
| Target | Bradley–Terry preference | Binary ground truth | **Proper scoring rules** |
| Reward | conviction / style | `{0,1}` | **Brier / log-loss / CRPS** |
| Ambiguity | overconfidence | mode collapse | **preserves posterior** |
| ECE | 20–45% | extreme on soft edges | **minimal (<3%)** |

**"CD"** = the promise that a prediction given probability `p` is correct ≈`p` of the time.
Getting there needs **soft targets** that capture real disagreement: multi-annotator
datasets (`takala/financial_phrasebank` tiered agreement, `hatexplain`) or a synthetic
**Dirichlet consensus** from many cheap-LLM raters. This is where labeled data is
essential and where the community reproductions stop.

> **Honesty note:** the ECE numbers produced by `train_rlcd.py` on the built-in synthetic
> generator are **illustrative of the method**, not a measured calibration result on real
> annotator data.

## Directory

```
jev_scoring.py                     # v2: the CORRECT next-token scoring mechanism (run this)
jev_judge.py                       # Jev-style LLM-as-a-judge (pairwise/rubric/faithfulness)
bench_latency.py                   # real latency-vs-tokens sweep -> measured chart
smollm_system_one.py               # v1: untrained bolt-on heads (kept for comparison)
train_rlcd.py                      # RLCD training loop (proper scoring rules + ECE)
jev_poc.py                         # pure-NumPy prototype (latency here is SIMULATED)
results/measured_results.json      # real measured numbers
results/latency_benchmark.json     # real latency sweep
jev_system_one_article.html        # the full deep-dive write-up (8 parts)
jev_build_your_own_avi_style.html  # tutorial-style variant (Avi Chawla voice)
docs/model_dissection.md           # Qwen = LLM, JEV-CPU = framework (download + dissect)
docs/jev_as_judge.md               # research: Jev-style scoring as an LLM judge
assets/                            # figures (measured latency + illustrative charts)
```

### Two article variants
- **`jev_system_one_article.html`** &mdash; the full, somewhat academic deep-dive (mechanism, our SmolLM experiment, RLCD as a proper RL problem, Jev-as-a-judge).
- **`jev_build_your_own_avi_style.html`** &mdash; the same material as a warm, hands-on, first-principles tutorial in the style of Avi Chawla's "Build your own Jev".

## Credits & sources

- TypeSafe AI — [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- Avi Chawla ([@_avichawla](https://x.com/_avichawla/status/2101563610644496464)) — *Build your own Jev (100% local)* (SGLang `/v1/score`)
- [`Meanblock/JEV-CPU`](https://huggingface.co/Meanblock/JEV-CPU) — CPU SemIf engine (Qwen3-0.6B)
- Gneiting & Raftery (2007), *Strictly Proper Scoring Rules*; Murphy (1973), Brier score decomposition

MIT License. Built for open research into non-autoregressive decision intelligence.

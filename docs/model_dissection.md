# Dissection: are JEV-CPU and Qwen embedding models or generative LLMs?

**Question:** the Jev-style approach reads "logits" — so are the models it uses
embedding models (bi-encoders that output a pooled vector) or generative LLMs
(decoders with a vocabulary head)? I downloaded and inspected them.

## TL;DR
- **Qwen (0.6B and 2.5-0.5B) are generative decoder-only LLMs** — they have a
  vocabulary `lm_head`, decoder layers, and a chat template, and **no pooling
  layer**. They are NOT embedding models.
- **`Meanblock/JEV-CPU` is not a model at all — it is a code framework** ("SemIf
  Phase 1"). It ships no weights; it *drives* generative LLMs and reads their
  next-token logits.

An **embedding model** (e.g. `sentence-transformers/*`, BGE, GTE) is a bi-encoder:
it pools hidden states into one fixed vector for similarity/retrieval and has no
`lm_head`. A **generative LLM** projects the last hidden state to a
vocabulary-sized logit vector via `lm_head`. Jev-style scoring needs the second
kind, because it reads the logits of the option tokens.

## Evidence: Qwen configs (`config.json`)
| | Qwen/Qwen3-0.6B | Qwen/Qwen2.5-0.5B-Instruct |
| :-- | :-- | :-- |
| `architectures` | `["Qwen3ForCausalLM"]` | `["Qwen2ForCausalLM"]` |
| `model_type` | `qwen3` | `qwen2` |
| `vocab_size` | 151936 | 151936 |
| `hidden_size` | 1024 | 896 |
| `num_hidden_layers` | 28 | 24 |
| chat template | yes | yes |
| sentence-transformers markers (`modules.json`, `1_Pooling/`, …) | **none** | **none** |

`...ForCausalLM` is the decisive tag: it is the class with a causal decoder + a
vocabulary projection head. Embedding models report `["BertModel"]`,
`["XLMRobertaModel"]`, etc., and ship a `modules.json`/`1_Pooling/` describing a
pooling layer. Neither Qwen has those.

## Evidence: Qwen3-0.6B weights (safetensors header, no full download)
Reading only the safetensors metadata (311 tensors):
- `model.embed_tokens.weight`  → shape **[151936, 1024]** (token embedding table)
- `lm_head.weight`             → shape **[151936, 1024]** (vocabulary projection)
- decoder layers `model.layers.0..27` → **28** transformer blocks
- pooling / pooler / sentence-transformers dense tensors → **NONE**

A vocab-sized `lm_head` plus decoder layers and no pooler = a generative
decoder-only LLM, full stop.

## Evidence: `Meanblock/JEV-CPU` is a framework, not a model
The repo has **no `config.json`, no `model.safetensors`, no tokenizer**. It
contains code and results instead:
- `semif_cpu.py`, `server.py`, `src/semif_phase1/{core,direct,reranker,serial,shared,mlx_backend}.py`
- `manifests/models.json` — the generative LLMs it loads:
  `Qwen/Qwen3-0.6B`, `openbmb/MiniCPM5-2B`, `Qwen/Qwen3.5-4B`, and
  `Qwen/Qwen3-Reranker-4B` (plus GGUF quantized artifacts).
- `docs/METHOD.md`: *"The direct system prompts frozen Qwen3.5-4B … performs one
  native forward pass and applies a softmax only to the logits of fixed uppercase
  answer tokens. It does not decode a token."*
- `src/semif_phase1/direct.py`: reads `model(**inputs).logits[:, -1, :]`, gathers
  the answer-slot token ids, softmaxes — **the same mechanism as our
  `jev_scoring.py`**, including the single-token-slot verification.

So the HF model card's `pipeline("zero-shot-classification", model="Meanblock/JEV-CPU")`
snippet is misleading — there is no model config that would make that work. The
real entry points are `server.py` / the SemIf CLI, which load a generative LLM.

## Why it matters for us
Our `jev_scoring.py` is on the right track: it uses a generative causal LLM
(SmolLM) and reads its `lm_head` logits, exactly as JEV-CPU/SemIf does with Qwen.
An embedding model could not do this — it has no per-token vocabulary logits to
read. (A *reranker* like `Qwen3-Reranker-4B` is a third category: a cross-encoder
judge that scores relevance via yes/no logits — still a generative-LM readout,
not a pooled embedding. See `docs/jev_as_judge.md`.)

## Reproduce
```bash
python3 - <<'PY'
from huggingface_hub import hf_hub_download, list_repo_files
import json
for r in ["Qwen/Qwen3-0.6B","Qwen/Qwen2.5-0.5B-Instruct"]:
    d=json.load(open(hf_hub_download(r,"config.json")))
    print(r, d["architectures"], "vocab", d["vocab_size"])
print("JEV-CPU has config.json?", "config.json" in list_repo_files("Meanblock/JEV-CPU"))
PY
```

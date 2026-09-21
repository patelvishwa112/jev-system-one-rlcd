# Using Jev-style models as the LLM judge

**Thesis:** most "LLM-as-a-judge" work is a *bounded decision*, not an essay — so
you can swap the generative judge for a Jev-style scoring readout (one forward
pass, read the option-token logits, restricted softmax). You get a calibrated
probability distribution instead of a single parsed number, at 0 output tokens,
and the two biggest judge pathologies (position bias, mis-calibrated scores)
become cheap to fix.

This is not speculative: **G-Eval already does a primitive version of this**, and
the community `JEV-CPU`/SemIf engine ships a "reranker" judge built exactly this
way. See `jev_judge.py` in this repo for a working demo.

---

## 1. Judge tasks are Jev primitives

| LLM-as-judge task | Today (generation) | Jev primitive | What you read |
| :-- | :-- | :-- | :-- |
| Pairwise: is A or B better? | generate "A"/"B"/verdict + rationale | **Choice** over {A, B, tie} | P(A), P(B), P(tie) |
| Rubric rating 1–5 / 0–4 | generate an integer + rationale | **Score** over ordered levels | distribution + E[score] |
| Binary check: faithful? toxic? correct? | generate "yes"/"no" + rationale | **Noul** | calibrated P(true) |
| Retrieval / relevance rerank | generate a relevance label | **Choice / reranker yes-no** | log-odds per candidate |

Every one of these has a **closed, known output space** — the exact condition
under which reading logits beats generating tokens.

## 2. G-Eval is Jev-scoring in disguise (the strongest prior evidence)

[G-Eval](https://arxiv.org/abs/2303.16634) is a widely-used LLM-as-judge method.
Its distinctive move: instead of taking the single integer the model writes, it
weights the score by the **token probabilities** of each rating level:

> score = Σ_i p(s_i) · s_i   over rating levels s_i ∈ {1..5}

That is **exactly Jev's `Score` primitive** (expectation over an ordinal
distribution). The G-Eval authors adopted it because the raw single integer had
"low variance and poor correlation with human judgments"; the probability-weighted
score correlates with humans much better and can separate near-tied outputs
([G-Eval guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide),
[DeepEval](https://deepeval.com/docs/metrics-llm-evals)).

The difference is only *how you get the probabilities*:

- **G-Eval:** asks a hosted model for `logprobs`, or (if unavailable) samples the
  prompt many times at temperature and uses score frequencies as a stand-in.
- **Jev-style / this repo:** read the option-token logits **directly** from one
  forward pass on an open model — no sampling, no logprobs API, 0 output tokens.

So Jev-as-judge is the native, generation-free generalization of the technique
G-Eval already proved works — extended to pairwise (Choice) and binary (Noul),
not just rubric scores.

## 3. It makes the two worst judge biases cheap to fix

**Position bias.** When a judge compares A vs B, it favors whichever appears
first/last — studies find **22–30% of verdicts flip** when you swap the order
([systematic study](https://aclanthology.org/2025.ijcnlp-long.18.pdf),
[Adaline](https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias)). The
standard fix is *balanced position calibration*: run both orderings and average
([Wang et al.](https://arxiv.org/abs/2305.17926)). With a generative judge that
doubles a slow, paid call. With Jev-style scoring each judgment is **0 output
tokens (~50–60 ms on a laptop, see `results/latency_benchmark.json`)**, so
running both orderings and averaging is essentially free — and you can afford it
on *every* sample, not a subsample. `jev_judge.py::pairwise_judge` does exactly
this and reports whether the winner flipped between orderings.

**Score mis-calibration / selection bias.** Raw judge scores are overconfident
and skewed toward certain option letters
([CalibraEval](https://arxiv.org/abs/2410.15393),
[scoring-bias study](https://arxiv.org/html/2506.22316v1)). Two levers:
(1) reading a full distribution instead of a greedy pick already exposes ties and
low-confidence cases the caller can route to humans; (2) **RLCD calibration** (see
the main article, Part 5) is precisely the training step that makes P(A)=0.9 mean
"right 90% of the time." A Jev judge trained with RLCD is a *calibrated* judge —
its confidence becomes a usable routing threshold, not decoration.

## 4. The reranker variant (already shipping in JEV-CPU)

`Meanblock/JEV-CPU`'s SemIf engine has a second judge system beyond the direct
readout: it uses **Qwen3-Reranker-4B**'s native yes/no contract, computing
`logit("yes") − logit("no")` for each candidate and softmaxing those log-odds
across candidates (`docs/METHOD.md`, `src/semif_phase1/reranker.py`). A reranker
*is* a judge — it scores candidate relevance without generating — so this is a
ready-made, production-shaped pattern for "which of these is best" judging.

## 5. Benefits vs. limits

**Benefits**
- 0 output tokens → cheap/fast enough to judge every sample + debias by swapping.
- Structurally valid output (a probability simplex), never an unparseable verdict.
- Returns a distribution + confidence, not a lone number → ties and low-confidence
  cases are visible and routable.
- Position-bias and self-consistency mitigations become affordable.
- With RLCD, the confidence is calibrated and usable as a threshold.

**Limits (be honest)**
- **No written rationale.** Scoring returns no text, so you lose the judge's
  explanation. Mitigation: score for the *verdict* on every item; generate a
  rationale only for flagged/low-confidence cases (a tiny fraction).
- **Only enumerable judgments.** Works for rating/choice/binary (most rubrics);
  not for open-ended critique.
- **Accuracy needs a capable, calibrated model.** Selection/letter bias still
  exists at the logit level and must be debiased + calibrated. A 135M model is a
  fast judge but a poor one (see demo below); JEV-CPU needs 2–4B for 0.69–0.81.

## 6. Working demo (`jev_judge.py`)

Real output on `SmolLM2-135M-Instruct` (Apple M1, MPS). The 135M model is too
small to have good "taste" — verdicts are near-chance — but the **mechanism,
the G-Eval-style expectation, and the order-swap debiasing all work**, and even
here faithfulness is directionally right (supported > unsupported):

```
Pairwise (A=good axial-tilt answer, B=wrong distance-to-Sun answer), order-swap debiased:
  winner: tie   probs {A: 0.297, B: 0.290, tie: 0.414}   position_bias_flip: false   conf: 0.065

Rubric (Score 0..3) on the good answer:  E[score]=1.39   dist {0:.264, 1:.224, 2:.374, 3:.138}

Faithfulness (Noul):  supported claim P=0.722   vs   unsupported claim P=0.658
```

Run it:
```bash
python3 jev_judge.py                                         # SmolLM2-135M-Instruct
python3 jev_judge.py --model_id Qwen/Qwen2.5-0.5B-Instruct   # sharper on a bigger model
```

## 7. How you'd deploy a Jev judge for real
1. Cast the eval as Choice/Score/Noul with an explicit rubric in the prompt.
2. Read option-token logits (`jev_scoring.py`) on a capable open model.
3. Debias: average both orderings for pairwise; add an `OTHER/abstain` option.
4. Calibrate with RLCD against logged human labels or ensemble votes (Part 5).
5. Threshold on the calibrated confidence: auto-accept high-confidence verdicts,
   route the rest (or generate a rationale only for those) to humans.

## References
- G-Eval: [confident-ai guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide) · [DeepEval](https://deepeval.com/docs/metrics-llm-evals)
- Position bias: [A Systematic Study of Position Bias in LLM-as-a-Judge](https://aclanthology.org/2025.ijcnlp-long.18.pdf) · [Adaline: why frontier judges fail bias tests](https://www.adaline.ai/blog/llm-as-a-judge-reliability-bias)
- Calibration / selection bias: [CalibraEval](https://arxiv.org/pdf/2410.15393) · [Evaluating Scoring Bias in LLM-as-a-Judge](https://arxiv.org/html/2506.22316v1)
- Balanced position calibration: [Wang et al., Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926)
- SemIf/JEV-CPU reranker method: `Meanblock/JEV-CPU` `docs/METHOD.md`, `src/semif_phase1/reranker.py`

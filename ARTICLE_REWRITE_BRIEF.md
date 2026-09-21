# BRIEF: Rewrite `jev_system_one_article.html` (honest v2)

You are rewriting the body of a technical blog article. The existing file
`jev_system_one_article.html` (1375 lines) is well-styled but the CONTENT is
weak: no concrete story, it just paraphrases TypeSafe's launch blog, has no
end-to-end example, and — worst — it documents an implementation approach that
is actually WRONG. Your job is to rewrite the article body into a credible,
original, honest piece, reusing the existing HTML shell and CSS.

## DELIVERABLE
Write a complete, self-contained HTML file to `jev_system_one_article_v2.html`.
- PRESERVE verbatim the existing `<head>`, the entire `<style>...</style>`
  block, the KaTeX `<script>` includes, the top nav bar, the hero/cover, the
  author card, and the closing `<script>` (the `switchSim`/`addClap` JS).
- REUSE the existing CSS classes (`callout`, `callout-title`, `math-card`,
  `formula`, `figure-box`, `figure-caption`, `table-wrapper`, `explanation`,
  `code-badge`, `tag`, `interactive-card`, `sim-*`, etc.). Do NOT invent a new
  design system or restyle.
- Code blocks use `<pre><code class="language-python">` / `language-bash`.
- Math uses `$...$` (inline) and `$$...$$` (display) so KaTeX renders it.
- Keep the existing interactive simulator section working (the `switchSim`
  buttons + `#sim-...` elements the closing script references). You may update
  its numbers/text but do not break the element IDs the script uses.
- Single file, no external assets beyond what already loads. It must open
  correctly in a browser by double-clicking.

## VOICE
Technical, precise, a little opinionated, honest. Write for an engineer who
knows LLMs but not Jev. NO marketing fluff, NO invented benchmark numbers.
Every number is either (a) one of the MEASURED numbers below, (b) an
officially published TypeSafe/community number that you attribute, or (c)
explicitly labeled "illustrative". If you are tempted to state a number that is
none of these, don't.

## HARD FACTS (use these, attribute them)

### Real Jev (TypeSafe AI)
- Launched 2026-09-15 by Diego Almeida (co-creator of ChatGPT / RLHF, ex-OpenAI).
- First "System One" model. Non-autoregressive: outputs decisions directly as
  probabilities + calibrated confidence, not text.
- Three answer primitives: **Noul** (yes/no -> probability), **Choice** (pick
  one of a closed set -> distribution), **Score** (position on an ordered
  scale -> expectation).
- Evaluates all questions in parallel against one shared `state` in a single pass.
- Published claims (attribute to TypeSafe): 70–500ms end-to-end (most queries
  ~100ms), "40x–200x faster for the same frontier intelligence", $0.042 per
  million input tokens, output tokens "free (too cheap to meter)". Limits:
  state+questions budget ~64k tokens; Choice up to 255 options; Score 2–10
  levels; text only. Rate limits 250k tok/s, 1200 req/min.
- Trained with **RLCD (Reinforcement Learning for Calibrated Decisions)**:
  optimizes so that predictions given probability p are correct ~p of the time.

### Official API shape (reproduce this in the walkthrough section)
Request:
```json
{
  "model": "jev-latest",
  "state": { "name": "Managed Postgres", "description": "..." },
  "questions": {
    "is_sponsor_inquiry": { "type": "noul", "instructions": "Does `description` ask to sponsor the site?" },
    "product_category": {
      "type": "choice",
      "instructions": "What kind of product is described?",
      "criteria": { "dev_tool": "Developer tools, hosting, APIs, SaaS", "course": "Courses, books, training", "unrelated": "Anything not aimed at developers" }
    },
    "message_quality": {
      "type": "score",
      "instructions": "How specific is the request?",
      "criteria": ["Generic template, no reference to this site", "Mentions the site but no concrete ask", "Concrete ask with a timeframe or product named"]
    }
  }
}
```
Response:
```json
{
  "model": "jev-1.13.0",
  "answers": {
    "is_sponsor_inquiry": { "type": "noul", "noul": 0.99 },
    "product_category": { "type": "choice", "choice": "dev_tool", "probabilities": {"dev_tool":0.97,"course":0.01,"unrelated":0.02}, "confidence": 0.95 },
    "message_quality": { "type": "score", "score": 1.9, "legend": {"0":"Generic...","1":"Mentions...","2":"Concrete..."}, "probabilities": {"0":0.0,"1":0.1,"2":0.9}, "confidence": 0.86 }
  },
  "usage": { "input_tokens": 210, "output_tokens": 31 }
}
```

### The community reproductions (this is HOW it actually works)
Both the HF model `Meanblock/JEV-CPU` and Avi Chawla's "Build your own Jev (100%
local)" reproduce Jev's INFERENCE mechanism the same way (attribute both):
1. Phrase the decision as a multiple-choice prompt with SINGLE-TOKEN letter
   labels (A, B, C...), putting the semantic option descriptions in the prompt.
2. Verify each label is exactly one token in the tokenizer (letters, not words,
   because "billing" may be multiple tokens; a leading space changes the id).
3. Run ONE forward pass (prefill). Take the last position's vocabulary-sized
   logit vector.
4. Read only the logits at the option-letter token positions; apply softmax
   restricted to just those positions -> a probability distribution over the
   allowed answers. The model NEVER generates the answer.
- Avi implements this with SGLang's `/v1/score` endpoint on Qwen2.5-0.5B-Instruct,
  Qwen3-4B, etc. He is explicit: this reproduces the inference path only, NOT
  Jev's weights, RLCD training, or calibration.
- Avi's key calibration caveat (quote the idea): a score of 0.91 means billing
  got 91% of the probability mass among the three allowed choices; it does NOT
  mean the model is right 91% of the time — that requires labeled data. This is
  the calibration problem RLCD solves.
- Add an OTHER/ESCALATE option when the listed choices may not be exhaustive,
  otherwise restricted softmax forces all mass onto possibly-wrong options.
- JEV-CPU: Qwen3-0.6B fp32 (~2.4GB, runs on 8GB CPU, ~1s/decision). Balanced
  accuracy 0.44 at 0.6B, 0.686 at 2B (MiniCPM), 0.813 at 4B (Qwen3.5-4B). Same
  "read the logits from one forward pass" mechanism.

### WHAT OUR FIRST POC GOT WRONG (be honest, this is a required section)
- `smollm_system_one.py` (v1) DELETES the pretrained `lm_head` and bolts on
  THREE FRESH, RANDOMLY-INITIALIZED `nn.Linear` heads (NoulHead/ChoiceHead/
  ScoreHead). With no training, these heads output NOISE: the demo's P(True)=
  0.505 and choice confidence ~0.25 (uniform over 4) are meaningless, not
  decisions. It threw away all the pretrained model's knowledge.
- v1's headline numbers were SIMULATED: `jev_poc.py` used `time.sleep()` for
  latency and hardcoded the "autoregressive ~1200ms" comparison. The "ECE
  28.4% -> 2.1%" came from synthetically hardcoded soft targets
  (`soft[label]=0.8`), which is circular — not a real calibration measurement.
- The FIX (`jev_scoring.py`, v2) KEEPS the lm_head and reads option-token logits
  — the correct mechanism above. No new parameters, no training needed to get a
  real signal.

### OUR MEASURED NUMBERS (real, from `jev_scoring.py`; see results/measured_results.json)
Machine: Apple M1, 8GB, MPS, fp32. These are REAL, not simulated.
- SmolLM-135M (base): routing decision via scoring = **60.1 ms, 0 output
  tokens**; same model generating a 40-token JSON answer = **1568 ms** ->
  **~26x** slower. The generation output was MALFORMED: it wrote
  `"choice": "1"` (not even a valid option) then rambled.
- SmolLM2-135M-Instruct: scoring 58.5 ms vs generation 1143 ms -> **~19.5x**.
- Honest accuracy caveat: both 135M models are too small to route correctly —
  they mis-route the payment-gateway outage to "frontend" at ~0.11 confidence
  (near chance). This MATCHES JEV-CPU's finding that you need 2B–4B params for
  0.69–0.81 accuracy. The mechanism is fast and structurally safe; ACCURACY is
  a separate axis that needs a capable model + RLCD calibration.
- Nuance worth stating: on the instruct model, single-token letter-scoring
  mis-routed, yet free generation happened to answer "billing" (correct) —
  tiny models' first-token distribution at the label position isn't always
  aligned; capable models + calibration are what make scoring both fast AND
  accurate.

## REQUIRED STRUCTURE (rewrite the body to these 7 parts)
Keep the existing title/hero. New body:

1. **The decision you didn't need an essay for.** Hook with the support ticket:
   you need 3 decisions (outage? which team? severity?), not prose. Real Jev
   returns all three as calibrated probabilities in ~100ms, 0 output tokens.
   Set up System One vs System Two (Kahneman) but grounded in "known output
   space => generation is wasted work". Include the real Jev facts.

2. **End-to-end walkthrough (the big picture).** THE centerpiece. Use the
   official example request/response JSON above. Walk step by step through what
   the model does with the `state` and EACH question type, how the caller
   supplies custom `criteria`/options at inference and still gets reliable typed
   output. Show all three outcomes (Noul/Choice/Score) with their probabilities
   and confidence. Then show Avi's "two results" idea: a clear winner (0.91) vs
   a near-tie (0.46 vs 0.44) — and how the CALLER's code (thresholds) decides
   auto-route vs human review. Conceptual + practical. This section answers
   "what happens when a user enters a prompt/question with 3 options".

3. **How it actually works: next-token scoring.** How an LLM produces its first
   token (logits over the vocab). Turning a decision into a single-token MCQ;
   single-token discipline (letters vs words, leading-space gotcha); read logits
   at option positions; restricted softmax. Noul = 2-option (Yes/No tokens),
   Score = choice over ordinal levels + expectation $E[S]=\sum_i i\,p_i$.
   OTHER/ESCALATE escape hatch. Include a short `jev_scoring.py` snippet
   (`_score_positions`: one forward pass, gather option-token logits, softmax).

4. **Our SmolLM experiment: what we built, measured, and got wrong.** Honest.
   First the wrong turn (v1 untrained bolt-on heads = noise; simulated numbers).
   Then the corrected `jev_scoring.py`. Then the MEASURED numbers table (base +
   instruct: scoring ms vs generation ms, speedup, output tokens; the malformed
   generation output as evidence). Then the honest accuracy caveat (135M too
   small; JEV-CPU 0.44->0.81 by scale). Directly COMPARE to Avi Chawla's SGLang
   `/v1/score` approach: same mechanism, he used a server + continuous batching,
   we did it in-process in transformers; note that our first attempt diverged by
   inventing heads instead of reading logits.

5. **RLCD, properly (the part nobody reproduces).** Expand RLCD as a real RL
   problem. Explicitly lay out the RL components:
   - **Agent / policy**: the model's head that emits the decision distribution.
   - **Environment / state**: the input `state` + question + allowed options.
   - **Action space**: the probability distribution over the declared options
     (Noul: 2, Choice: <=255, Score: 2–10 ordered bins) — NOT the vocabulary.
   - **Reward = a strictly proper scoring rule** (this is the core): the reward
     is maximized ONLY by reporting your true believed probability. Cover Brier
     score and log loss; give the Murphy decomposition
     $\text{Brier}=\text{Reliability(ECE)}-\text{Resolution}+\text{Uncertainty}$
     and explain WHY minimizing Brier drives calibration error toward 0 while
     rewarding sharp/confident-when-right distributions.
   - **What "CD" (Calibrated Decisions) means**: p should equal the empirical
     frequency of being right; higher confidence must mean higher accuracy.
   - **How you'd gather the data**: you need targets that capture genuine
     aleatoric uncertainty / disagreement, i.e. SOFT labels, not one-hot. Ways:
     multi-annotator datasets with real disagreement (e.g. financial_phrasebank
     tiered agreement, hatexplain); or synthesize a Dirichlet "consensus" by
     sampling many cheap-LLM raters (temperature/persona perturbations) and
     using their vote distribution as the soft target. Be explicit that this is
     where labeled/soft data is essential and where Avi & JEV-CPU stop.
   - HONESTY: state clearly that our repo's earlier ECE numbers came from
     synthetic hardcoded soft targets and are ILLUSTRATIVE of the method, not a
     measured calibration result on real annotator data.

6. **When to use scoring vs generation vs structured output.** Avi's framing:
   score when the output set is known and a label+distribution suffices; generate
   when the content is unknown; structured output still decodes token-by-token so
   it's not the same as scoring. Give a short decision list and the limits
   (scoring needs an enumerable output space; needs an escape option; accuracy
   needs a capable, calibrated model).

7. **Reproduction toolkit.** The repo files and commands:
   `jev_scoring.py` (the correct mechanism, run: `python3 jev_scoring.py`),
   `smollm_system_one.py` (the v1 attempt, kept for the "what we got wrong"
   comparison), `train_rlcd.py` (RLCD training loop),
   `results/measured_results.json` (the measured numbers). Note SGLang `/v1/score`
   as the production-grade path (Avi's approach).

## RULES
- Do not delete or restyle the `<style>` block or the closing `<script>`.
- Do not invent numbers. Use only the measured/attributed numbers above.
- Keep it honest — the "what we got wrong" material is required, not optional.
- Attribute TypeSafe, Avi Chawla (@_avichawla), and JEV-CPU where their facts
  are used.
- Output ONLY the full HTML file to `jev_system_one_article_v2.html`.

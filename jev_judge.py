"""
jev_judge.py -- Using a Jev-style scoring model AS AN LLM JUDGE.
================================================================
LLM-as-a-judge tasks are almost always *bounded decisions*:
  - pairwise preference  -> "A better", "B better", or "tie"      (Choice)
  - rubric rating 1..5    -> a level on an ordered scale           (Score)
  - binary check          -> faithful? toxic? correct?  yes/no     (Noul)

So instead of asking a judge LLM to GENERATE a verdict (slow, ~$/token,
parse-fragile, and famously order-biased), we read the option-token logits in
one forward pass -- the same mechanism as jev_scoring.py. Two concrete wins
that this file demonstrates with real numbers:

1. Rubric scoring becomes G-Eval's probability-weighted score for free:
   score = sum_i i * p(level_i)   (a genuine expectation, not a single integer).
2. Position-bias debiasing becomes affordable: because a judgment costs 0 output
   tokens, we run BOTH answer orderings and average the distributions
   ("balanced position calibration"), then report whether the winner flipped.

Accuracy still needs a capable + RLCD-calibrated model; on a 135M model the
verdicts are near-chance. The point here is the MECHANISM and the debiasing,
not the quality of a tiny model's taste.
"""
import argparse, json
from jev_scoring import JevScorer, Choice, Score, Noul, confidence_from_probs
import torch, torch.nn.functional as F


def pairwise_judge(scorer: JevScorer, question: str, answer_a: str, answer_b: str,
                   criterion: str = "Which answer is more helpful, correct, and complete?") -> dict:
    """Pairwise A/B/tie judging, debiased by averaging both answer orderings."""
    def run(first, second):
        state = (f"Question:\n{question}\n\nAnswer 1:\n{first}\n\nAnswer 2:\n{second}")
        q = Choice(instructions=criterion, criteria={
            "answer_1": "Answer 1 is clearly better",
            "answer_2": "Answer 2 is clearly better",
            "tie": "Both are about equally good (or bad)",
        })
        r = scorer.choice(state, q)
        return r["probabilities"]  # {answer_1, answer_2, tie}

    # order 1: A=1, B=2
    p1 = run(answer_a, answer_b)
    pA1, pB1, pT1 = p1["answer_1"], p1["answer_2"], p1["tie"]
    # order 2: B=1, A=2  -> remap so A/B stay fixed
    p2 = run(answer_b, answer_a)
    pA2, pB2, pT2 = p2["answer_2"], p2["answer_1"], p2["tie"]

    # balanced position calibration: average the two orderings
    pA, pB, pT = (pA1 + pA2) / 2, (pB1 + pB2) / 2, (pT1 + pT2) / 2
    winner = max((("A", pA), ("B", pB), ("tie", pT)), key=lambda kv: kv[1])[0]
    # position-bias diagnostic: did the winner change between orderings?
    w1 = max((("A", pA1), ("B", pB1), ("tie", pT1)), key=lambda kv: kv[1])[0]
    w2 = max((("A", pA2), ("B", pB2), ("tie", pT2)), key=lambda kv: kv[1])[0]
    return {
        "winner": winner,
        "probs_debiased": {"A": round(pA, 4), "B": round(pB, 4), "tie": round(pT, 4)},
        "order1_winner": w1, "order2_winner": w2,
        "position_bias_flip": w1 != w2,
        "confidence": round(confidence_from_probs(torch.tensor([pA, pB, pT])), 4),
    }


def rubric_judge(scorer: JevScorer, question: str, answer: str, rubric: list[str],
                 criterion: str = "Rate the answer's quality against the rubric.") -> dict:
    """G-Eval-style probability-weighted rubric score via one forward pass."""
    state = f"Question:\n{question}\n\nAnswer:\n{answer}"
    r = scorer.score(state, Score(instructions=criterion, criteria=rubric))
    return r  # {score (expectation), legend, probabilities, confidence}


def faithfulness_check(scorer: JevScorer, context: str, answer: str) -> dict:
    """Binary judge: is the answer supported by the context? -> calibrated P(true)."""
    state = f"Context:\n{context}\n\nAnswer:\n{answer}"
    return scorer.noul(state, Noul(instructions="Is the Answer fully supported by the Context (no unsupported claims)?"))


def demo(model_id):
    s = JevScorer(model_id=model_id)
    q = "What causes the seasons on Earth?"
    a_good = "The tilt of Earth's axis (~23.5 degrees) changes how directly sunlight hits each hemisphere through the year."
    a_bad = "The seasons happen because Earth gets closer to and farther from the Sun in its orbit."

    print("\n=== Pairwise judge (A=good, B=bad), order-swap debiased ===")
    print(json.dumps(pairwise_judge(s, q, a_good, a_bad), indent=2))

    print("\n=== Rubric judge (Score 0..3) on the GOOD answer ===")
    print(json.dumps(rubric_judge(s, q, a_good, [
        "Wrong or misleading", "Partially correct", "Correct but shallow",
        "Correct, precise, and complete",
    ]), indent=2))

    print("\n=== Faithfulness (Noul) ===")
    ctx = "Earth's axial tilt is about 23.5 degrees."
    print("supported claim :", faithfulness_check(s, ctx, "Earth's axis is tilted ~23.5 degrees."))
    print("unsupported claim:", faithfulness_check(s, ctx, "Earth is the third planet and has one moon."))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model_id", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    demo(ap.parse_args().model_id)

"""
bench_latency.py -- REAL latency-vs-output-length benchmark (no simulation).
Measures, on the machine you run it on:
  - next-token SCORING latency (one forward pass, 0 output tokens) -- constant
  - autoregressive GENERATION latency at several output-token budgets
and saves both the raw numbers (results/latency_benchmark.json) and a chart
(assets/measured_latency_benchmark.png). Every point is measured with
time.perf_counter(); nothing is hardcoded.
"""
import argparse, json, time
import torch
from jev_scoring import JevScorer, Choice

STATE = ("Customer message: 'Our payment gateway is returning 500 errors for all "
         "European Mastercard transactions. Checkout is down.'")
Q = Choice(instructions="Which engineering team should handle this?",
           criteria={"billing": "payments", "frontend": "UI", "security": "auth", "docs": "docs"})

def median(xs): xs=sorted(xs); n=len(xs); return xs[n//2] if n%2 else 0.5*(xs[n//2-1]+xs[n//2])

def main(model_id):
    s = JevScorer(model_id=model_id)
    # warmup (exclude model-load / first-call graph build from measurements)
    for _ in range(2):
        s.choice(STATE, Q)
    # scoring latency (constant wrt output length; 0 tokens generated)
    score_ms = median([_timed_score(s) for _ in range(5)])

    token_budgets = [1, 10, 25, 50, 75, 100]
    gen = []
    prompt = (f"State:\n{STATE}\n\nQuestion: {Q.instructions}\n"
              f"Options: {list(Q.criteria)}\nReply with JSON.\nJSON:")
    enc = s.tokenizer(prompt, return_tensors="pt").to(s.device)
    for n in token_budgets:
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.no_grad():
                s.model.generate(**enc, max_new_tokens=n, do_sample=False,
                                 pad_token_id=s.tokenizer.pad_token_id)
            runs.append((time.perf_counter() - t0) * 1000)
        gen.append({"tokens": n, "latency_ms": round(median(runs), 1)})
        print(f"  gen {n:3d} tok -> {gen[-1]['latency_ms']:8.1f} ms")

    data = {"model_id": model_id, "device": str(s.device),
            "scoring_ms_constant": round(score_ms, 1), "scoring_output_tokens": 0,
            "generation": gen}
    with open("results/latency_benchmark.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nscoring (0 tokens): {score_ms:.1f} ms | wrote results/latency_benchmark.json")
    _plot(data)

def _timed_score(s):
    t0 = time.perf_counter(); s.choice(STATE, Q); return (time.perf_counter()-t0)*1000

def _plot(data):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping chart: {e})"); return
    toks = [g["tokens"] for g in data["generation"]]
    lat = [g["latency_ms"] for g in data["generation"]]
    sc = data["scoring_ms_constant"]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    ax.plot(toks, lat, "o-", color="#f43f5e", lw=2.5, ms=7,
            label="Autoregressive generation (prefill + sequential decode)")
    ax.plot(toks, [sc]*len(toks), "s-", color="#10b981", lw=2.5, ms=7,
            label=f"Next-token scoring (single pass, 0 output tokens): {sc:.0f} ms")
    ax.fill_between(toks, [sc]*len(toks), lat, color="#f43f5e", alpha=0.10)
    ax.set_title(f"MEASURED latency vs output length\n{data['model_id']} on {data['device']} (median of repeats)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Output tokens generated"); ax.set_ylabel("End-to-end latency (ms)")
    ax.grid(True, ls="--", alpha=0.5); ax.legend(loc="upper left")
    plt.tight_layout(); plt.savefig("assets/measured_latency_benchmark.png"); plt.close()
    print("wrote assets/measured_latency_benchmark.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model_id", default="HuggingFaceTB/SmolLM-135M")
    main(ap.parse_args().model_id)

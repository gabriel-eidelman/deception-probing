"""
Pilot inter-rater reliability experiment for the LLM-as-judge grading pipeline.

For each scenario in deception_elecitation.py:
  1. Generate one response (gpt-4o, temp=0.7).
  2. Grade that response N_RATINGS times independently (gpt-4o, temp=1.0).
  3. Compute agreement statistics across the 3x3 factorial cells.

Outputs a JSON results file and a printed summary report.
Run: modal run --detach pilot_interrater_reliability.py
"""

import os
import re
import json
import time
import sys
import math
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from deception_elecitation_tests import PROMPTS
from grading.grading import grade_response
from inference.run_llama_inference import generate_llama_response, generate_llama_responses_batch, app as llama_app

# ── Configuration ──────────────────────────────────────────────────────────────
RESPONSE_MODEL = "llama-3.3-70b-instruct"  # via Modal
GRADER_MODEL   = "gpt-4o"
N_RATINGS      = 5        # independent grader calls per example
RESPONSE_TEMP  = 0.7      # some diversity in generated responses
GRADER_TEMP    = 1.0      # non-zero so different calls can disagree (tests reliability)
MAX_RETRIES    = 3
RESULTS_PATH   = Path(__file__).parent / "pilot_interrater_results.json"

TASK_DOMAINS       = ["factual_qa", "accountability_explanation", "reporting_evaluation"]
ELICITATION_STRATS = ["instrumental_pressure", "role_based_social_dynamics", "instructed_deception"]


# ── API helpers ────────────────────────────────────────────────────────────────

def make_client() -> OpenAI:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=key)


# ── Reliability statistics ────────────────────────────────────────────────────

def pairwise_agreement(ratings: list[int]) -> float:
    """Fraction of all rater pairs that give the same score."""
    pairs = total = 0
    n = len(ratings)
    for i in range(n):
        for j in range(i + 1, n):
            pairs += int(ratings[i] == ratings[j])
            total += 1
    return pairs / total if total else float("nan")


def mean_abs_diff(ratings: list[int]) -> float:
    """Mean |r_i - r_j| across all pairs."""
    diffs = []
    n = len(ratings)
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(ratings[i] - ratings[j]))
    return sum(diffs) / len(diffs) if diffs else float("nan")


def krippendorff_alpha_ordinal(ratings_matrix: list[list[int]]) -> float:
    """
    Krippendorff's alpha for ordinal data using the standard coincidence-matrix method.
    ratings_matrix: shape (n_units, n_raters), all values 1–7.
    """
    # Build coincidence matrix (7×7 for scores 1..7)
    n_units  = len(ratings_matrix)
    n_raters = len(ratings_matrix[0])
    VALUES   = list(range(1, 8))
    idx      = {v: i for i, v in enumerate(VALUES)}
    k        = len(VALUES)

    o = [[0.0] * k for _ in range(k)]  # coincidence matrix
    n_total = 0

    for unit_ratings in ratings_matrix:
        m_u = len(unit_ratings)  # raters for this unit
        if m_u < 2:
            continue
        for a in range(m_u):
            for b in range(m_u):
                if a == b:
                    continue
                va, vb = unit_ratings[a], unit_ratings[b]
                o[idx[va]][idx[vb]] += 1.0 / (m_u - 1)
        n_total += m_u

    if n_total == 0:
        return float("nan")

    # Marginal frequencies
    n_k = [sum(o[i][j] for j in range(k)) for i in range(k)]

    # Observed disagreement (ordinal metric)
    D_o = 0.0
    for c in range(k):
        for d in range(k):
            D_o += o[c][d] * (VALUES[c] - VALUES[d]) ** 2
    D_o /= sum(sum(row) for row in o)

    # Expected disagreement (ordinal metric)
    N = sum(n_k)
    D_e = 0.0
    for c in range(k):
        for d in range(k):
            D_e += n_k[c] * n_k[d] * (VALUES[c] - VALUES[d]) ** 2
    D_e /= N * (N - 1)

    if D_e == 0:
        return 1.0
    return 1.0 - D_o / D_e


def aggregate_stats(ratings_matrix: list[list[int]]) -> dict:
    all_pairwise = []
    all_abs_diff = []
    all_std      = []
    for r in ratings_matrix:
        all_pairwise.append(pairwise_agreement(r))
        all_abs_diff.append(mean_abs_diff(r))
        mean_r = sum(r) / len(r)
        var_r  = sum((x - mean_r) ** 2 for x in r) / len(r)
        all_std.append(math.sqrt(var_r))

    exact_agree = sum(1 for r in ratings_matrix if len(set(r)) == 1) / len(ratings_matrix)

    return {
        "n_examples":              len(ratings_matrix),
        "exact_agreement_rate":    round(exact_agree, 4),
        "mean_pairwise_agreement": round(sum(all_pairwise) / len(all_pairwise), 4),
        "mean_abs_diff":           round(sum(all_abs_diff) / len(all_abs_diff), 4),
        "mean_within_std":         round(sum(all_std) / len(all_std), 4),
        "krippendorff_alpha":      round(krippendorff_alpha_ordinal(ratings_matrix), 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

@llama_app.local_entrypoint()
def main():
    client = make_client()

    print("=" * 70)
    print("PILOT INTER-RATER RELIABILITY EXPERIMENT")
    print(f"  Examples : {len(PROMPTS)}")
    print(f"  Ratings/example: {N_RATINGS}")
    print(f"  Response model: {RESPONSE_MODEL} (single Modal call)")
    print(f"  Grader model  : {GRADER_MODEL} (temp={GRADER_TEMP})")
    print("=" * 70)

    # ── Step 1: generate all responses in one Modal call (model loads once) ──────
    print(f"\nGenerating {len(PROMPTS)} responses via Modal ...", end=" ", flush=True)
    all_responses = generate_llama_responses_batch(PROMPTS)
    print(f"done.")

    records = []

    for i, (prompt, response_text) in enumerate(zip(PROMPTS, all_responses)):
        domain   = prompt["task_domain"]
        strategy = prompt["elicitation_strategy"]
        print(f"\n[{i+1:02d}/{len(PROMPTS)}] {domain} | {strategy}")
        print(f"  Response ({len(response_text)} chars): {response_text[:120].replace(chr(10), ' ')!r}")

        # ── Step 2: grade N times ────────────────────────────────────────────────
        ratings = []
        print(f"  Grading {N_RATINGS}x ...", end=" ", flush=True)
        for j in range(N_RATINGS):
            score = grade_response(client, prompt["system"], prompt["message"],
                       response_text, temperature=GRADER_TEMP)
            ratings.append(score)
            print(score, end=" ", flush=True)
            time.sleep(0.3)
        print()

        pa  = pairwise_agreement(ratings)
        mad = mean_abs_diff(ratings)
        mean_r = sum(ratings) / len(ratings)
        var_r  = sum((x - mean_r) ** 2 for x in ratings) / len(ratings)
        std_r  = math.sqrt(var_r)

        print(f"  Pairwise agreement: {pa:.2%}  |  Mean abs diff: {mad:.2f}  |  Std: {std_r:.2f}")

        records.append({
            "index":               i,
            "task_domain":         domain,
            "elicitation_strategy": strategy,
            "system_prompt":       prompt["system"],
            "question":            prompt["message"],
            "response":            response_text,
            "ratings":             ratings,
            "mean_rating":         round(mean_r, 3),
            "std_rating":          round(std_r, 3),
            "pairwise_agreement":  round(pa, 4),
            "mean_abs_diff":       round(mad, 4),
        })

    # ── Aggregate statistics ───────────────────────────────────────────────────
    ratings_matrix = [r["ratings"] for r in records]
    overall = aggregate_stats(ratings_matrix)

    print("\n" + "=" * 70)
    print("OVERALL RELIABILITY")
    print("=" * 70)
    print(f"  N examples              : {overall['n_examples']}")
    print(f"  Exact agreement rate    : {overall['exact_agreement_rate']:.2%}")
    print(f"  Mean pairwise agreement : {overall['mean_pairwise_agreement']:.2%}")
    print(f"  Mean |r_i - r_j|        : {overall['mean_abs_diff']:.3f}")
    print(f"  Mean within-example SD  : {overall['mean_within_std']:.3f}")
    print(f"  Krippendorff's α (ord.) : {overall['krippendorff_alpha']:.3f}")

    alpha = overall["krippendorff_alpha"]
    if alpha >= 0.8:
        verdict = "ACCEPTABLE — grader is highly reliable"
    elif alpha >= 0.67:
        verdict = "MARGINAL — acceptable by Krippendorff's threshold; consider rubric refinement"
    else:
        verdict = "INSUFFICIENT — refine rubric or add few-shot examples before full-scale run"
    print(f"\n  Verdict: {verdict}")

    # ── By domain ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("BY TASK DOMAIN")
    print("-" * 70)
    for domain in TASK_DOMAINS:
        subset = [r["ratings"] for r in records if r["task_domain"] == domain]
        if not subset:
            continue
        s = aggregate_stats(subset)
        print(f"  {domain:<30}  α={s['krippendorff_alpha']:.3f}  "
              f"agree={s['mean_pairwise_agreement']:.2%}  "
              f"mad={s['mean_abs_diff']:.2f}")

    # ── By elicitation strategy ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("BY ELICITATION STRATEGY")
    print("-" * 70)
    for strat in ELICITATION_STRATS:
        subset = [r["ratings"] for r in records if r["elicitation_strategy"] == strat]
        if not subset:
            continue
        s = aggregate_stats(subset)
        print(f"  {strat:<30}  α={s['krippendorff_alpha']:.3f}  "
              f"agree={s['mean_pairwise_agreement']:.2%}  "
              f"mad={s['mean_abs_diff']:.2f}")

    # ── Per-example table ──────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("PER-EXAMPLE SUMMARY")
    print("-" * 70)
    header = f"{'#':>3}  {'Domain':<28} {'Strategy':<26} {'Ratings':<18} {'Mean':>5} {'SD':>5} {'PA':>6}"
    print(header)
    print("-" * len(header))
    for r in records:
        ratings_str = "[" + " ".join(str(x) for x in r["ratings"]) + "]"
        print(f"{r['index']+1:>3}  {r['task_domain']:<28} {r['elicitation_strategy']:<26} "
              f"{ratings_str:<18} {r['mean_rating']:>5.2f} {r['std_rating']:>5.2f} "
              f"{r['pairwise_agreement']:>5.0%}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "config": {
            "response_model": RESPONSE_MODEL,
            "grader_model":   GRADER_MODEL,
            "n_ratings":      N_RATINGS,
            "response_temp":  RESPONSE_TEMP,
            "grader_temp":    GRADER_TEMP,
            "n_examples":     len(PROMPTS),
        },
        "overall": overall,
        "by_domain":   {
            d: aggregate_stats([r["ratings"] for r in records if r["task_domain"] == d])
            for d in TASK_DOMAINS if any(r["task_domain"] == d for r in records)
        },
        "by_strategy": {
            s: aggregate_stats([r["ratings"] for r in records if r["elicitation_strategy"] == s])
            for s in ELICITATION_STRATS if any(r["elicitation_strategy"] == s for r in records)
        },
        "examples": records,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

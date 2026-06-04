"""Generate on-policy responses, probe them, grade with GPT-4o, compute metrics."""
import modal

app = modal.App("probe-generate-grade")
model_volume = modal.Volume.from_name("inoculation-models")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .add_local_dir("deception-detection", remote_path="/repo", copy=True)
    .run_commands("cd /repo && pip install -e . 'transformers>=4.45,<5'")
)

MODEL_PATH = "/models/Llama-3.3-70B-Instruct"
DETECTOR_PATH = "/repo/example_results/instructed_pairs/detector.pt"


# ──────────────────────────────────────────────────────────────────────
# Scenarios: system + user only. No pre-assigned label — the grader
# determines honest/deceptive AFTER we see the actual generated response.
# ──────────────────────────────────────────────────────────────────────

@app.function(image=image, volumes={"/models": model_volume},
              gpu="A100-80GB:2", timeout=100000)
def generate_and_probe(scenarios: list) -> list:
    """On the GPU: generate responses, then probe them. Returns raw data
    (system/user/response/score) for grading + metrics on the client side."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from deception_detection.detectors import LogisticRegressionDetector
    from deception_detection.data.base import DialogueDataset
    from deception_detection.tokenized_data import TokenizedDataset
    from deception_detection.activations import Activations
    from deception_detection.types import Message, Label

    detector = LogisticRegressionDetector.load(DETECTOR_PATH)
    print(f"Probe: layers={detector.layers}, normalize={detector.normalize}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    tokenizer.name_or_path = "meta-llama/Llama-3.3-70B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto", local_files_only=True,
    ).eval()

    # --- Generate on-policy responses ---
    print("\n=== GENERATING ===\n")
    dialogues, generated = [], []
    for sc in scenarios:
        messages = [
            {"role": "system", "content": sc["system"]},
            {"role": "user", "content": sc["user"]},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                input_ids, max_new_tokens=256,
                temperature=1.0, top_p=0.95, do_sample=True,
            )
        resp = tokenizer.decode(
            out_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        ).strip()
        generated.append(resp)
        dialogues.append([
            Message("system", sc["system"], False),
            Message("user", sc["user"], False),
            Message("assistant", resp, True),
        ])
        print(f"  {sc['description']}\n    {resp[:90]}...\n")

    # --- Probe the generated responses ---
    print("=== PROBING ===\n")
    dataset = DialogueDataset(
        variant="plain", dialogues=dialogues,
        labels=[Label.HONEST] * len(dialogues),  # placeholder; real labels come from grader
        metadata=None, skip_variant_validation=True, shuffle_upon_init=False,
    )
    dataset.base_name = "novel_prompts_generated"

    toks = TokenizedDataset.from_dataset(dataset, tokenizer)
    assert toks.detection_mask is not None and toks.detection_mask.sum() > 0, \
        "Empty detection mask — check responses are non-empty"
    acts = Activations.from_model(model, toks, batch_size=4, layers=detector.layers)
    scores = detector.score(acts)
    per_dialogue = [float(s.mean()) for s in scores.scores]

    # Return raw payload for client-side grading + metrics
    return [
        {
            "description": sc["description"],
            "system": sc["system"],
            "user": sc["user"],
            "response": resp,
            "probe_score": score,
        }
        for sc, resp, score in zip(scenarios, generated, per_dialogue)
    ]


# ──────────────────────────────────────────────────────────────────────
# Local: grading + metrics (grader lives in grading/grading.py, shared
# with the inter-rater reliability study)
# ──────────────────────────────────────────────────────────────────────
import os


@app.local_entrypoint()
def main():
    import json
    from openai import OpenAI
    from pathlib import Path
    from dotenv import load_dotenv
    from sklearn.metrics import roc_auc_score
    from grading.grading import grade_response, bin_label  # local-only; not shipped to container
    from scenarios import DATASET

    load_dotenv(Path(__file__).parent.parent / ".env")
    # 1. Generate + probe on the GPU
    payload = generate_and_probe.remote(DATASET)

    # 2. Grade each ACTUAL generated response locally.
    #    temperature=0.0 for deterministic ground-truth labeling (vs the
    #    reliability study, which calls the same grader at temp=1.0).
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("\n=== GRADING ===\n")
    for item in payload:
        grade = grade_response(
            client, item["system"], item["user"], item["response"],
            temperature=0.0,
        )
        item["grade"] = grade
        item["label"] = bin_label(grade)
        print(f"  [{item['label']:9}] grade={grade}  {item['description']}")
        print(f"    {item['response'][:90]}...\n")

    # 3. Metrics — AUROC on honest vs deceptive, EXCLUDING ambiguous (Apollo convention)
    honest = [it["probe_score"] for it in payload if it["label"] == "honest"]
    decep = [it["probe_score"] for it in payload if it["label"] == "deceptive"]
    amb = [it for it in payload if it["label"] == "ambiguous"]

    print("=== SUMMARY ===")
    print(f"  n_honest={len(honest)}  n_deceptive={len(decep)}  n_ambiguous={len(amb)} (excluded)")
    if honest:
        print(f"  mean honest score:    {sum(honest)/len(honest):+.4f}")
    if decep:
        print(f"  mean deceptive score: {sum(decep)/len(decep):+.4f}")

    auroc = None
    if honest and decep:
        y = [0] * len(honest) + [1] * len(decep)
        auroc = float(roc_auc_score(y, honest + decep))
        print(f"  AUROC (honest vs deceptive): {auroc:.4f}")
    else:
        print("  AUROC: undefined (need both honest and deceptive after grading)")

    out = {
        "auroc": auroc,
        "n_honest": len(honest),
        "n_deceptive": len(decep),
        "n_ambiguous": len(amb),
        "results": payload,
    }
    print("\n=== FULL RESULTS ===")
    print(json.dumps(out, indent=2))
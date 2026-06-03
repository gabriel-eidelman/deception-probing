"""
Step-1 smoke test: load the PROVIDED probe and score `roleplaying__plain`
using the Apollo deception-detection package's own classes.

Strategy (per investigation of the repo):
  - Do NOT reimplement extraction/scoring. Call their Experiment + detector.
    Their `LogisticRegressionDetector.get_score_tensor` does:
        normalize: (acts - scaler_mean) / scaler_scale   (z-score ONLY)
        project:   einsum(acts, directions) -> per-layer
        aggregate: mean over layers
    NOTE: there is NO L2-normalization step. The earlier standalone script
    added one; that was wrong. Using their classes avoids that class of bug.
  - Do NOT use the top-level `run` entrypoint: on exception it DELETES the
    experiment folder (f.unlink(); folder.rmdir()). Pointed at the shipped
    example_results that would destroy the provided probe. Instead we copy the
    probe into a throwaway working folder and drive the Experiment object
    directly.
  - Do NOT use their model loader (`get_llama3_model_and_tokenizer`): it uses
    cache_dir=/data/huggingface + local_files_only=True + a hub id, which will
    NOT resolve a flat local checkpoint dir. We load the model ourselves from
    the local path (same as the working standalone script) and inject it.

ASSUMPTIONS TO CONFIRM ON FIRST RUN (flagged inline as [Vn]):
  [V1] Repo installs cleanly via `pip install -e .` on this image.
  [V2] Experiment exposes a settable `_model` (constructor sets self._model =
       None and reads it via a `model` property). We set it to skip their loader.
  [V3] ExperimentConfig field names match (eval_data, control_data, folder,
       method, model_name, detect_layers, etc.).
  [V4] get_eval_scores() returns {dataset_name: Scores}, and Scores exposes
       per-dialogue scores + honest/deceptive labels in a readable way. The
       extraction of (score, label) pairs at the bottom is best-effort and may
       need adjusting to the actual Scores API.
  [V5] The roleplaying rollout JSON exists and eval_on_policy routing reads it
       (so no generation happens; the model is only used for the forward pass).
"""

import modal

app = modal.App("roleplaying-probe-smoke-test")

# Volume holding the local model weights (flat checkpoint dir).
model_volume = modal.Volume.from_name("inoculation-models")

# Build the image FROM the cloned repo so the package is importable.
# `copy=True` bakes the repo into the image; then editable-install it.
image = (
    # Pinned PyTorch image whose conda Python is 3.11 (the Apollo package uses
    # typing.Self throughout its core modules, so it requires >=3.11; the older
    # 2.2.2 image shipped Python 3.10 and failed `pip install -e .`). Install
    # with the image's own conda `pip` so build-time install and runtime import
    # share one interpreter. torch 2.5.1 + CUDA 12.1 + cudnn9 baked in.
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime",
    )
    # [V1] Copy the whole repo (NOT just probe/) so `deception_detection` is a
    # package. Point this at your local clone path.
    .add_local_dir(
        "deception-detection",
        remote_path="/repo",
        copy=True,
    )
    .run_commands(
        "cd /repo && pip install -e . 'transformers>=4.45,<5' || (echo '--- pip install -e . FAILED ---' && exit 1)"
    )
)

MODEL_PATH = "/models/Llama-3.3-70B-Instruct"

# Where the shipped example probe lives inside the repo.
EXAMPLE_RESULTS = "/repo/example_results/roleplaying"


@app.function(
    image=image,
    volumes={"/models": model_volume},
    gpu="A100-80GB:2",
    timeout=3600,
)
def run_smoke_test():
    import shutil
    import tempfile
    from pathlib import Path

    import torch
    import yaml
    from transformers import AutoModelForCausalLM

    # ------------------------------------------------------------------
    # 0. Sanity: confirm the example probe + config are present
    # ------------------------------------------------------------------
    ex = Path(EXAMPLE_RESULTS)
    assert (ex / "detector.pt").exists(), f"No detector.pt in {ex}"
    assert (ex / "cfg.yaml").exists(), f"No cfg.yaml in {ex}"
    print(f"=== Found example probe at {ex} ===")

    # ------------------------------------------------------------------
    # 1. Copy probe + config into a THROWAWAY working folder.
    #    Protects the shipped example_results from any cleanup-on-error path
    #    and gives the Experiment a folder it owns.
    # ------------------------------------------------------------------
    work = Path(tempfile.mkdtemp(prefix="rp_smoke_"))
    shutil.copy(ex / "detector.pt", work / "detector.pt")
    shutil.copy(ex / "cfg.yaml", work / "cfg.yaml")
    print(f"=== Working folder: {work} ===")

    # ------------------------------------------------------------------
    # 2. Build the ExperimentConfig from the shipped cfg, overriding to:
    #      - eval ONLY roleplaying__plain
    #      - NO control datasets (avoids the 10k-sample control run and the
    #        empty-dataset edge case)
    #      - folder = our working copy (so get_detector() finds detector.pt)
    # ------------------------------------------------------------------
    from deception_detection.experiment import Experiment, ExperimentConfig

    with open(work / "cfg.yaml") as f:
        cfg_dict = yaml.safe_load(f)

    print("\n=== Shipped config ===")
    for k, v in cfg_dict.items():
        print(f"  {k}: {v}")

    # [V3] Override fields. Construct via yaml-dict spread, mirroring how their
    # `run` builds it: ExperimentConfig(**config_dict).
    cfg_dict["eval_data"] = ["roleplaying__plain"]
    cfg_dict["control_data"] = []          # skip control scoring entirely
    cfg_dict["folder"] = str(work)         # detector.pt lives here
    cfg_dict["val_fraction"] = 0           # avoid using train data in eval split

    cfg = ExperimentConfig(**cfg_dict)
    print("\n=== Effective overrides ===")
    print(f"  eval_data    : {cfg.eval_data}")
    print(f"  control_data : {cfg.control_data}")
    print(f"  folder       : {cfg.folder}")
    print(f"  method       : {cfg.method}")
    print(f"  detect_layers: {cfg.detect_layers}")

    # ------------------------------------------------------------------
    # 3. Construct the Experiment, then INJECT our locally-loaded model so
    #    their loader (which expects an HF cache layout) is never called.
    # ------------------------------------------------------------------
    print("\n=== Constructing Experiment (tokenizer + datasets) ===")
    experiment = Experiment(cfg)   # sets self._model = None, loads tokenizer/datasets

    print(f"\n=== Loading local model from {MODEL_PATH} ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    # [V2] Inject. Their Experiment stores the model in `_model` and exposes a
    # `model` property that lazily loads if None. Setting `_model` short-circuits
    # that. If the attribute name differs, adjust here.
    experiment._model = model
    print("  Model injected into experiment._model")

    # ------------------------------------------------------------------
    # 4. Load the PROVIDED probe (no training) and score.
    # ------------------------------------------------------------------
    print("\n=== Loading detector (provided probe) ===")
    detector = experiment.get_detector()   # loads work/detector.pt since it exists
    print(f"  Detector: {type(detector).__name__}")

    print("\n=== Computing eval scores on roleplaying__plain ===")
    # [V4] Returns {dataset_name: Scores}. We avoid the `run` wrapper and call
    # this directly so the control-scoring + folder-cleanup paths never trigger.
    eval_scores = experiment.get_eval_scores()

    # ------------------------------------------------------------------
    # 5. Extract (score, label) pairs and summarise separation.
    #    [V4] The exact Scores API may differ; this tries a few common shapes
    #    and prints the raw object if it can't introspect it.
    # ------------------------------------------------------------------
    import numpy as np

    scores_obj = eval_scores.get("roleplaying__plain")
    if scores_obj is None:
        # key might be the full id; grab the only entry
        scores_obj = next(iter(eval_scores.values()))

    print(f"\n=== Raw Scores object type: {type(scores_obj).__name__} ===")

    honest_scores, deceptive_scores = [], []

    # Best-effort extraction. Adjust to the real Scores API once seen.
    try:
        # Common pattern: Scores has per-dialogue scalar scores + labels.
        # e.g. scores_obj.scores (list[float]) and scores_obj.labels (list[str])
        per_dialogue = getattr(scores_obj, "scores", None)
        labels = getattr(scores_obj, "labels", None)
        if per_dialogue is not None and labels is not None:
            for s, lab in zip(per_dialogue, labels):
                val = float(s) if np.isscalar(s) else float(np.mean(np.asarray(s)))
                lab_str = str(lab).lower()
                if "decept" in lab_str:
                    deceptive_scores.append(val)
                elif "honest" in lab_str:
                    honest_scores.append(val)
        else:
            raise AttributeError("no .scores/.labels")
    except Exception as e:
        print(f"  Could not auto-extract via .scores/.labels ({e}).")
        print("  Dumping object attributes for manual inspection:")
        print("   ", [a for a in dir(scores_obj) if not a.startswith("__")])
        # Try a dict-ish dump
        if hasattr(scores_obj, "to_dict"):
            print("   to_dict() ->", scores_obj.to_dict())

    result = {
        "n_honest": len(honest_scores),
        "n_deceptive": len(deceptive_scores),
        "mean_honest": float(np.mean(honest_scores)) if honest_scores else None,
        "mean_deceptive": float(np.mean(deceptive_scores)) if deceptive_scores else None,
        "honest_scores": honest_scores,
        "deceptive_scores": deceptive_scores,
    }

    print("\n=== Separation summary ===")
    print(f"  n honest    : {result['n_honest']}")
    print(f"  n deceptive : {result['n_deceptive']}")
    print(f"  mean honest    : {result['mean_honest']}")
    print(f"  mean deceptive : {result['mean_deceptive']}")

    # Quick AUROC if we have both classes (higher score = more deceptive expected)
    if honest_scores and deceptive_scores:
        try:
            from sklearn.metrics import roc_auc_score
            y = [0] * len(honest_scores) + [1] * len(deceptive_scores)
            s = honest_scores + deceptive_scores
            auroc = float(roc_auc_score(y, s))
            result["auroc_deceptive_vs_honest"] = auroc
            print(f"  AUROC (deceptive vs honest): {auroc:.4f}")
            print("  (Paper reports ~0.964 on roleplaying. Close-ish = wired right.)")
        except Exception as e:
            print(f"  AUROC computation failed: {e}")

    return result


@app.local_entrypoint()
def main():
    import json
    import datetime
    import pathlib

    result = run_smoke_test.remote()

    print("\n=== Returned to local ===")
    print(json.dumps(
        {k: v for k, v in result.items()
         if k not in ("honest_scores", "deceptive_scores")},
        indent=2,
    ))

    out_dir = pathlib.Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"rp_smoke_{ts}.json"
    with open(path, "w") as f:
        json.dump({"timestamp": ts, "result": result}, f, indent=2)
    print(f"\nSaved: {path}")
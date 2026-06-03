"""
Modal script for extracting layer-wise residual stream activations from
Llama-3.1-Instruct on a batch of (system, user) prompts.

This is a modified version of the inference script. Key differences:
  - We do NOT generate text. We only need a forward pass on the prompt.
  - We return hidden states from every transformer layer, at the LAST
    token position of the prompt (before any generation).
  - We return one (n_layers+1, hidden_dim) tensor per prompt.
    (n_layers+1 because output_hidden_states includes the embedding layer
    at index 0, plus one entry per transformer block.)

Notes on the "last prompt token" choice:
  - This is the standard probe-target location used in Goldowsky-Dill et al.
    and most other deception-probe work. It represents the model's internal
    state at the moment it commits to a response.
  - Alternative: average activations over response tokens. Captures the
    deception "in flight" rather than the pre-decision state. We do NOT
    use this here, but it's worth being aware of as a robustness check.

Default model path is the 8B; swap for the 70B when GPU budget allows.
The 70B will need a larger GPU (A100 80GB or H100) and longer timeout.
"""
import modal

app = modal.App("llama-activation-extraction")

volume = modal.Volume.from_name("inoculation-models")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
    "numpy",
)


@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="A10G",  # bump to "A100-80GB" or "H100" for the 70B model
    timeout=1200,  # generous; activation extraction is fast but model load is slow
)
def extract_activations(
    prompts: list[dict],
    model_subdir: str = "Llama-3.1-8B-Instruct",
) -> dict:
    """Extract per-layer residual stream activations for a batch of prompts.

    Args:
        prompts: list of dicts with keys "system" and "message".
        model_subdir: name of the model directory under /models.

    Returns:
        dict with:
          "activations": np.ndarray of shape (n_prompts, n_layers+1, hidden_dim),
                         dtype float32. Activations at the last prompt token,
                         one row per (prompt, layer).
          "n_layers":    int, number of transformer blocks (excludes embedding).
          "hidden_dim":  int, residual stream width.
          "model_name":  str, the model subdirectory used.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import numpy as np

    model_path = f"/models/{model_subdir}"
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model with output_hidden_states=True...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    # We'll collect activations into a list, then stack.
    # Shape per prompt: (n_layers + 1, hidden_dim)
    all_acts = []

    n_layers = None
    hidden_dim = None

    for i, prompt in enumerate(prompts):
        if i % 10 == 0:
            print(f"  Processing prompt {i + 1}/{len(prompts)}...")

        # Build the chat-formatted prompt exactly as in inference, with
        # add_generation_prompt=True so the model is positioned to respond.
        # The "last prompt token" is then the token right before generation.
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["message"]},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )

        # outputs.hidden_states is a tuple of length (n_layers + 1).
        # Each element has shape (batch=1, seq_len, hidden_dim).
        # We take the activation at the final token position.
        hidden_states = outputs.hidden_states  # tuple
        last_idx = inputs["input_ids"].shape[1] - 1

        # Stack into (n_layers+1, hidden_dim)
        layer_acts = torch.stack(
            [h[0, last_idx, :].float().cpu() for h in hidden_states],
            dim=0,
        ).numpy()

        if n_layers is None:
            n_layers = len(hidden_states) - 1  # exclude embedding layer
            hidden_dim = layer_acts.shape[-1]

        all_acts.append(layer_acts)

        # Free GPU memory between prompts (defensive; matters more for the 70B)
        del outputs
        torch.cuda.empty_cache()

    activations = np.stack(all_acts, axis=0).astype(np.float32)
    # Final shape: (n_prompts, n_layers + 1, hidden_dim)

    print(f"Done. Activation array shape: {activations.shape}")

    return {
        "activations": activations,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "model_name": model_subdir,
    }


def extract_activations_batch(
    prompts: list[dict],
    model_subdir: str = "Llama-3.1-8B-Instruct",
) -> dict:
    """Local-side wrapper that calls the remote Modal function."""
    with app.run():
        return extract_activations.remote(prompts, model_subdir=model_subdir)
import modal

app = modal.App("llama-3.1-8b-instruct-inference")

volume = modal.Volume.from_name("inoculation-models")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
)

@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="A100-80GB:2",
    timeout=1800,
)
def run_inference(prompts, max_new_tokens: int = 400):
    """Run batch inference on a list of prompts.

    Each prompt is a dict with keys:
        "system"  – system prompt string
        "message" – user message string

    Returns a list of (user_message, response) tuples.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_path = "/models/Llama-3.3-70B-Instruct"
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="auto",
    )

    results = []
    for prompt in prompts:
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        results.append((prompt["message"], response))

    return results


def generate_llama_responses_batch(
    prompts: list[dict],
    max_new_tokens: int = 400,
) -> list[str]:
    """Generate responses for a batch of prompts in a single Modal call.

    The model is loaded once on the remote worker, amortising the cold-start
    cost across all prompts.

    Args:
        prompts:        List of dicts, each with "system" and "message" keys.
        max_new_tokens: Maximum tokens to generate per prompt (default 400).

    Returns:
        List of response strings in the same order as *prompts*.
    """
    results = run_inference.remote(prompts, max_new_tokens=max_new_tokens)
    return [response for _, response in results]


def generate_llama_response(
    system_prompt: str,
    user_message: str,
    max_new_tokens: int = 400,
) -> str:
    """Generate a single response from Llama-3.1-8B-Instruct via Modal.

    Args:
        system_prompt:  The system prompt that sets the model's context/role.
        user_message:   The user's question or instruction.
        max_new_tokens: Maximum number of tokens to generate (default 400).

    Returns:
        The model's response as a plain string.
    """
    results = run_inference.remote(
        [{"system": system_prompt, "message": user_message}],
        max_new_tokens=max_new_tokens,
    )
    _, response = results[0]
    return response


def main():
    from deception_elecitation_tests import PROMPTS
    results = run_inference.remote(PROMPTS)
    print("\n" + "=" * 60)
    for i, (message, response) in enumerate(results, 1):
        print(f"\n[{i}] Prompt: {message}")
        print(f"    Response: {response}")
        print("-" * 60)

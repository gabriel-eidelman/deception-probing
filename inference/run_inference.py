import modal

app = modal.App("test-qwen-inference")

volume = modal.Volume.from_name("inoculation-models")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch",
    "transformers",
    "accelerate",
)


@app.function(
    image=image,
    volumes={"/models": volume},
    gpu="A10G",
    timeout=300,
)
def run_inference(prompts: list[str]):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_path = "/models/qwen3.5-2b"
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    results = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append((prompt, response))

    return results


@app.local_entrypoint()
def main():
    from deception_elecitation_tests import PROMPTS
    results = run_inference.remote(PROMPTS)
    print("\n" + "=" * 60)
    for i, (prompt, response) in enumerate(results, 1):
        print(f"\n[{i}] Prompt: {prompt}")
        print(f"    Response: {response}")
        print("-" * 60)

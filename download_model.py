import modal

app = modal.App("download-llama-3.3-70b")

volume = modal.Volume.from_name("inoculation-models")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_transfer]",
)


@app.function(
    image=image,
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
)
def download_model():
    import os
    from huggingface_hub import snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    local_dir = "/models/Llama-3.3-70B-Instruct"

    print(f"Downloading {model_id} to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        token=os.environ["HF_TOKEN"],
    )

    volume.commit()
    print("Done. Model saved and volume committed.")


@app.local_entrypoint()
def main():
    download_model.remote()

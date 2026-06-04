# Decomposing Probe Generalization: A Factorial Test of Elicitation and Domain Shift in Deception Monitoring

## Environment Setup

### Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Modal (Remote Compute)

This project uses [Modal](https://modal.com) to run inference on Llama-3.3-70B-Instruct. The model is downloaded to a Modal volume and accessed remotely.

**Before downloading the model**, you must request access to `meta-llama/Llama-3.3-70B-Instruct` on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and add your HF token as a Modal secret named `huggingface-secret`.

Once access is granted:

```bash
modal run setup/download_model.py
```

### Deception Detection Package

This project depends on the [`deception_detection`](https://github.com/ApolloResearch/deception-detection/tree/main/deception_detection) package from Apollo Research. Install it directly from the GitHub repository:

```bash
pip install git+https://github.com/ApolloResearch/deception-detection.git
```

### Replicating Results

1. Create a `.env` file in the project root and add your OpenAI API key (used for llm grading):

   ```
   OPENAI_API_KEY=your_key_here
   ```

2. Run the probe evaluation pipeline:

   ```bash
   python probe_evaluation_pipeline/run_probe_pipeline.py
   ```

3. Run the G-study analysis:

   ```bash
   python g_study/run_all.py
   ```
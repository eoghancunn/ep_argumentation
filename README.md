# EP Argumentation Project

Tools for analyzing European Parliament debates using argumentation models.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
huggingface-cli login
```

## Usage

### Extract Arguments

```bash
python run_argument_extraction.py --debate data/debates/CRE-20060404-ITEM-006 --output results/
```

### Classify Relations

Results are automatically stored in separate directories based on the model used (e.g., `results/arc_results/hf_brunoyun-Llama-3.1-Amelia-AR-8B-v1/`, `results/arc_results/ollama_llama3.1/`, `results/arc_results/anthropic_claude-3-5-haiku-20241022/`).

Default (local Hugging Face model):
```bash
python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results/
```

Using Ollama:
```bash
python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results/ --use-ollama --ollama-model llama3.1
```

Using Anthropic Haiku (requires ANTHROPIC_API_KEY in .env file or env var):
```bash
python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results/ --use-anthropic --anthropic-model claude-3-5-haiku-20241022
```

With quantization (for limited memory):
```bash
python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results/ --load-in-8bit
```

## Models

- **Argument Mining**: `oberbics/llama-3.1-8B-newspaper_argument_mining`
- **Argument Relations**:
  - Default: `brunoyun/Llama-3.1-Amelia-AR-8B-v1` (local Hugging Face model)
  - Ollama: Use `--use-ollama` with `--ollama-model` (default: `llama3.1`)
  - Anthropic: Use `--use-anthropic` with `--anthropic-model` (default: `claude-3-5-haiku-20241022`, requires `ANTHROPIC_API_KEY` in .env file or env var)
  - Custom: Use `--model` to specify a different Hugging Face model

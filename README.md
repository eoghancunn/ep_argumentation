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

```bash
python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results/
```

## Models

- **Argument Mining**: `oberbics/llama-3.1-8B-newspaper_argument_mining`
- **Argument Relations**: `brunoyun/Llama-3.1-Amelia-AR-8B-v1`

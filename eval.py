"""
Script for running argument relation models on the evaluation set.
"""

import argparse
import json
import os
import sys
import time
import re
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm
import torch
import platform

# Load .env file from project root
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent
    env_path = project_root / '.env'
    load_dotenv(dotenv_path=env_path, override=False)
except ImportError:
    pass

from src.argument_models import ArgumentRelationModel, ArgumentRelationModelOllama, ArgumentRelationModelAnthropic


def get_model_identifier(args) -> str:
    """Generate a filesystem-safe identifier for the model being used."""
    if args.use_anthropic:
        model_name = args.anthropic_model
        prefix = "anthropic"
    elif args.use_ollama:
        model_name = args.ollama_model
        prefix = "ollama"
    else:
        model_name = args.model
        prefix = "hf"
    
    sanitized = re.sub(r'[^a-zA-Z0-9._-]', '_', model_name)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    
    return f"{prefix}_{sanitized}"


def main():
    parser = argparse.ArgumentParser(description="Run argument relation models on evaluation set pairs")
    
    parser.add_argument('--eval-file', type=str, default='data/evaluation_set.json',
                      help='Path to evaluation_set.json')
    parser.add_argument('--output', type=str, required=True,
                      help='Output JSON file path or directory')
    
    # Model options
    parser.add_argument('--use-ollama', action='store_true',
                      help='Use Ollama API (requires Ollama running)')
    parser.add_argument('--ollama-model', type=str, default='llama3.1',
                      help='Ollama model name (default: llama3.1)')
    parser.add_argument('--ollama-url', type=str, default=None,
                      help='Ollama API URL')
    parser.add_argument('--use-anthropic', action='store_true',
                      help='Use Anthropic API (requires ANTHROPIC_API_KEY)')
    parser.add_argument('--anthropic-model', type=str, default='claude-3-5-haiku-20241022',
                      help='Anthropic model name')
    parser.add_argument('--anthropic-api-key', type=str, default=None,
                      help='Anthropic API key')
    parser.add_argument('--model', type=str, default='brunoyun/Llama-3.1-Amelia-AR-8B-v1',
                      help='HuggingFace model name')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                      help='Device to use (auto-detect if not specified)')
    parser.add_argument('--load-in-4bit', action='store_true',
                      help='Use 4-bit quantization (CUDA only)')
    parser.add_argument('--load-in-8bit', action='store_true',
                      help='Use 8-bit quantization')
    
    parser.add_argument('--max-pairs', type=int, help='Maximum number of pairs to process')
    parser.add_argument('--max-new-tokens', type=int, default=None,
                      help='Maximum tokens to generate (default: 1024 for Anthropic, 128 for others)')
    parser.add_argument('--overwrite', action='store_true',
                      help='Overwrite existing results (default: skip pairs that already have results)')
    
    args = parser.parse_args()
    
    # Load evaluation set
    with open(args.eval_file, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]
    
    # Load model
    try:
        if args.use_anthropic:
            model = ArgumentRelationModelAnthropic(
                model_name=args.anthropic_model,
                api_key=args.anthropic_api_key
            )
        elif args.use_ollama:
            model = ArgumentRelationModelOllama(
                model_name=args.ollama_model,
                base_url=args.ollama_url
            )
        else:
            if not args.device:
                if torch.cuda.is_available():
                    args.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    args.device = "mps"
                else:
                    args.device = "cpu"
            
            if not args.load_in_4bit and not args.load_in_8bit:
                if args.device == "cuda":
                    args.load_in_4bit = True
                elif args.device == "mps" or (platform.system() == "Darwin" and args.device != "cuda"):
                    args.load_in_8bit = True
            
            model = ArgumentRelationModel(
                model_name=args.model,
                device=args.device,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    model_identifier = get_model_identifier(args)
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith('/'):
        if not output_path.endswith('/'):
            output_path = output_path + '/'
        output_path = os.path.join(output_path, f"eval_results_{model_identifier}.json")
    else:
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{model_identifier}{ext}"
    
    # Load existing results if they exist
    existing_results = {}
    if os.path.exists(output_path) and not args.overwrite:
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                # Handle both list and dict formats
                if isinstance(existing_data, list):
                    for result in existing_data:
                        pair_id = result.get('pair_id')
                        if pair_id:
                            existing_results[pair_id] = result
                elif isinstance(existing_data, dict) and 'results' in existing_data:
                    for result in existing_data['results']:
                        pair_id = result.get('pair_id')
                        if pair_id:
                            existing_results[pair_id] = result
        except Exception as e:
            tqdm.write(f"Warning: Could not load existing results: {e}")
    
    # Filter pairs: skip those that already have valid relations
    valid_relations = {'support', 'attack', 'no relation'}
    pairs_to_process = []
    
    for pair in pairs:
        source = pair.get('source', {})
        target = pair.get('target', {})
        pair_id = f"{source.get('full_identifier', 'unknown')}_vs_{target.get('full_identifier', 'unknown')}"
        
        if args.overwrite:
            # Process all pairs if overwriting
            pairs_to_process.append(pair)
        else:
            # Check if this pair already has a valid relation
            existing_result = existing_results.get(pair_id)
            if existing_result:
                existing_relation = existing_result.get('relation', '').lower().strip()
                if existing_relation in valid_relations:
                    continue  # Skip this pair, already labeled
            pairs_to_process.append(pair)
    
    if not pairs_to_process:
        print(f"All pairs already have results. Use --overwrite to regenerate.")
        sys.exit(0)
    
    skipped_count = len(pairs) - len(pairs_to_process)
    if skipped_count > 0:
        print(f"Skipping {skipped_count} pairs that already have results")
    
    # Run model on pairs
    max_tokens = args.max_new_tokens if args.max_new_tokens is not None else (1024 if args.use_anthropic else 128)
    
    new_results = []
    pbar = tqdm(pairs_to_process, desc="Processing")
    
    for pair in pbar:
        source = pair.get('source', {})
        target = pair.get('target', {})
        source_text = source.get('text', '')
        target_text = target.get('text', '')
        
        if not source_text or not target_text:
            continue
        
        topic = source.get('debate_id', '')
        
        try:
            relation_result = model.classify_relation(
                source=source_text,
                target=target_text,
                topic=topic,
                max_new_tokens=max_tokens
            )
            
            if isinstance(relation_result, dict):
                relation = relation_result.get('relation', '')
            else:
                relation = relation_result
            
            pair_id = f"{source.get('full_identifier', 'unknown')}_vs_{target.get('full_identifier', 'unknown')}"
            
            new_results.append({
                'pair_id': pair_id,
                'source': source,
                'target': target,
                'relation': relation
            })
        except Exception as e:
            tqdm.write(f"Error: {e}")
            continue
    
    pbar.close()
    
    # Merge new results with existing results
    if args.overwrite:
        # Overwrite mode: use only new results
        all_results = new_results
    else:
        # Merge mode: combine existing and new results (new results override existing)
        merged_dict = existing_results.copy()
        for result in new_results:
            merged_dict[result['pair_id']] = result
        all_results = list(merged_dict.values())
    
    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(all_results)} total results ({len(new_results)} new) to {output_path}")


if __name__ == "__main__":
    main()

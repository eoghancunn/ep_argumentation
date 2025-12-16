"""
Script for running argument extraction (mining) on debate interventions.

This script uses the oberbics/llama-3.1-8B-newspaper_argument_mining model
to extract argumentative units from text.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import platform
import torch

from src.argument_models import ArgumentMiningModel, ArgumentMiningModelAPI, load_intervention


def is_president_intervention(speaker: Optional[str]) -> bool:
    """
    Check if an intervention is made by the president/chair.
    
    Args:
        speaker: Speaker name from intervention
        
    Returns:
        True if the intervention is by a president/chair, False otherwise
    """
    if not speaker:
        return False
    
    speaker_upper = speaker.strip().upper()
    
    # Check for "President" at the start (with or without period/dash)
    if speaker_upper.startswith("PRESIDENT"):
        return True
    
    # Check for "IN THE CHAIR" pattern
    if "IN THE CHAIR" in speaker_upper:
        return True
    
    return False


def extract_from_text(model: ArgumentMiningModel, text: str, output_file: Optional[str] = None,
                     max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9,
                     use_greedy: bool = False) -> Dict:
    """
    Extract arguments from a single text.
    
    Args:
        model: Loaded ArgumentMiningModel
        text: Input text to analyze
        output_file: Optional file to save results
        
    Returns:
        Dictionary with extracted arguments
    """
    extracted = model.extract_arguments(
        text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_greedy=use_greedy
    )
    
    result = {
        "input_text": text,
        "extracted_arguments": extracted
    }
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def extract_from_intervention(model: ArgumentMiningModel, intervention_file: str, 
                             output_dir: Optional[str] = None,
                             max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9,
                             use_greedy: bool = False) -> Dict:
    """
    Extract arguments from a single intervention JSON file.
    
    Args:
        model: Loaded ArgumentMiningModel
        intervention_file: Path to intervention JSON file
        output_dir: Optional directory to save results
        
    Returns:
        Dictionary with extracted arguments
    """
    intervention = load_intervention(intervention_file)
    speaker = intervention.get('speaker', '')
    
    # Skip president interventions
    if is_president_intervention(speaker):
        return {}
    
    text = intervention.get('english') or intervention.get('original', '')
    
    if not text:
        return {}
    
    extracted = model.extract_arguments(
        text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        use_greedy=use_greedy
    )
    
    result = {
        "intervention_id": intervention.get('intervention_id'),
        "debate_id": intervention.get('debate_id'),
        "speaker": intervention.get('speaker'),
        "agenda_item": intervention.get('agenda_item'),
        "original_text": text,
        "extracted_arguments": extracted
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        intervention_id = intervention.get('intervention_id', 'unknown')
        output_file = os.path.join(output_dir, f"{intervention_id}_extracted.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result


def extract_from_debate(model: ArgumentMiningModel, debate_dir: str, 
                      output_dir: Optional[str] = None, 
                      max_interventions: Optional[int] = None,
                      max_new_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.9,
                      use_greedy: bool = False) -> List[Dict]:
    """
    Extract arguments from all interventions in a debate directory.
    
    Args:
        model: Loaded ArgumentMiningModel
        debate_dir: Path to debate directory
        output_dir: Optional directory to save results
        max_interventions: Maximum number of interventions to process (None = all)
        
    Returns:
        List of dictionaries with extracted arguments
    """
    # Find interventions directory
    interventions_dir = os.path.join(debate_dir, "interventions")
    if not os.path.exists(interventions_dir):
        interventions_dir = debate_dir
    
    if not os.path.exists(interventions_dir):
        print(f"Error: Interventions directory not found: {interventions_dir}")
        return []
    
    # Get all intervention files
    intervention_files = sorted([f for f in os.listdir(interventions_dir) 
                                if f.endswith('.json')])
    
    if not intervention_files:
        print(f"No intervention files found in {interventions_dir}")
        return []
    
    if max_interventions:
        intervention_files = intervention_files[:max_interventions]
    
    # Count president interventions to skip
    president_count = 0
    results = []
    for intervention_file in intervention_files:
        file_path = os.path.join(interventions_dir, intervention_file)
        
        # Check if this is a president intervention before processing
        try:
            intervention = load_intervention(file_path)
            speaker = intervention.get('speaker', '')
            if is_president_intervention(speaker):
                president_count += 1
                continue
        except Exception as e:
            continue
        
        try:
            result = extract_from_intervention(
                model, file_path, output_dir,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                use_greedy=use_greedy
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error processing {intervention_file}: {e}")
            continue
    
    # Save combined results if output directory specified
    if output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        debate_id = os.path.basename(debate_dir.rstrip('/'))
        combined_output = os.path.join(output_dir, f"{debate_id}_all_extracted.json")
        with open(combined_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract arguments from debate interventions using the argument mining model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from a single intervention file
  python run_argument_extraction.py --intervention data/debates/CRE-20060404-ITEM-006/interventions/2-031.json

  # Extract from all interventions in a debate
  python run_argument_extraction.py --debate data/debates/CRE-20060404-ITEM-006 --output results/

  # Extract from text input
  python run_argument_extraction.py --text "Your text here..." --output results/extracted.json

  # Use 8-bit quantization for Apple Silicon (recommended)
  python run_argument_extraction.py --debate data/debates/CRE-20060404-ITEM-006 --load-in-8bit

  # Use API instead of local model (much faster, requires internet)
  python run_argument_extraction.py --debate data/debates/CRE-20060404-ITEM-006 --use-api

  # Use API with greedy decoding for maximum speed
  python run_argument_extraction.py --debate data/debates/CRE-20060404-ITEM-006 --use-api --greedy
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--intervention', type=str,
                           help='Path to a single intervention JSON file')
    input_group.add_argument('--debate', type=str,
                           help='Path to a debate directory')
    input_group.add_argument('--text', type=str,
                           help='Text to extract arguments from')
    
    # Model options
    parser.add_argument('--use-api', action='store_true',
                       help='Use Gradio API instead of loading local model (much faster, requires internet)')
    parser.add_argument('--api-url', type=str,
                       default='oberbics/Argument-Mining',
                       help='Gradio API URL or Hugging Face Space name (default: oberbics/Argument-Mining)')
    parser.add_argument('--model', type=str,
                       default='oberbics/llama-3.1-8B-newspaper_argument_mining',
                       help='Model name (default: oberbics/llama-3.1-8B-newspaper_argument_mining, ignored if --use-api)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (auto-detect if not specified, ignored if --use-api)')
    parser.add_argument('--load-in-4bit', action='store_true',
                       help='Use 4-bit quantization (CUDA only, ignored if --use-api)')
    parser.add_argument('--load-in-8bit', action='store_true',
                       help='Use 8-bit quantization (works on all platforms, recommended for MPS, ignored if --use-api)')
    
    # Processing options
    parser.add_argument('--max-interventions', type=int,
                       help='Maximum number of interventions to process (for --debate)')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                       help='Maximum number of tokens to generate (default: 256, reduced for speed)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Sampling temperature (default: 0.05 for API, 0.7 for local, ignored if --greedy)')
    parser.add_argument('--top-p', type=float, default=0.9,
                       help='Nucleus sampling parameter (default: 0.9, ignored if --greedy)')
    parser.add_argument('--greedy', action='store_true',
                       help='Use greedy decoding instead of sampling (faster, less diverse)')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file or directory (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Set default temperature based on mode
    if args.temperature is None:
        args.temperature = 0.05 if args.use_api else 0.7
    
    # Auto-detect device if not specified (prefer CUDA for remote GPU)
    if not args.device:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    
    # Load model (API or local)
    if args.use_api:
        try:
            model = ArgumentMiningModelAPI(api_url=args.api_url)
        except Exception as e:
            print(f"Error connecting to API: {e}")
            print("\nTip: Make sure you have internet connection and gradio-client installed:")
            print("  pip install gradio-client")
            sys.exit(1)
    else:
        # Auto-detect quantization based on device
        if not args.load_in_4bit and not args.load_in_8bit:
            if args.device == "cuda":
                args.load_in_4bit = True
            elif args.device == "mps" or (platform.system() == "Darwin" and args.device != "cuda"):
                args.load_in_8bit = True
        
        try:
            model = ArgumentMiningModel(
                model_name=args.model,
                device=args.device,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            if "out of memory" in str(e).lower():
                print("\nTip: Try using --load-in-8bit, --device cpu, or --use-api for faster processing")
            sys.exit(1)
    
    try:
        import time
        start_time = time.time()
        
        if args.intervention:
            extract_from_intervention(
                model, args.intervention, args.output,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_greedy=args.greedy
            )
        
        elif args.debate:
            extract_from_debate(
                model, 
                args.debate, 
                args.output,
                max_interventions=args.max_interventions,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_greedy=args.greedy
            )
        
        elif args.text:
            extract_from_text(
                model, args.text, args.output,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                use_greedy=args.greedy
            )
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


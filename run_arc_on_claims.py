"""
Script for running Argument Relation Classification (ARC) on extracted arguments.

This script:
1. Loads extracted argument files
2. Extracts all <argument> tags from the extracted_arguments
3. Loads report statements from debate directories (core claims)
4. Classifies relations between:
   - Mined arguments vs report statements (support, attack, no relation)
   - All pairs of mined arguments (support, attack, no relation)
5. Does NOT classify relations between report statements
"""

import argparse
import json
import os
import re
import sys
from typing import List, Dict, Tuple, Optional
from itertools import combinations
import platform
import torch
from tqdm import tqdm

from src.argument_models import ArgumentRelationModel, ArgumentRelationModelOllama


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


def extract_arguments(extracted_arguments: str) -> List[str]:
    """
    Extract all arguments from the extracted_arguments text.
    
    Args:
        extracted_arguments: Text containing <argument> tags
        
    Returns:
        List of argument texts
    """
    if not extracted_arguments:
        return []
    
    # Pattern to match <argument>...</argument> tags
    pattern = r'<argument>(.*?)</argument>'
    arguments = re.findall(pattern, extracted_arguments, re.DOTALL)
    
    # Clean up arguments (strip whitespace, filter out "NA")
    arguments = [arg.strip() for arg in arguments if arg.strip() and arg.strip().upper() != "NA"]
    
    return arguments


def load_extracted_files(input_dir: str) -> List[Dict]:
    """
    Load all extracted JSON files from a directory.
    
    Args:
        input_dir: Directory containing extracted JSON files
        
    Returns:
        List of loaded extraction results
    """
    if not os.path.exists(input_dir):
        print(f"Error: Directory not found: {input_dir}")
        return []
    
    extracted_files = [f for f in os.listdir(input_dir) 
                      if f.endswith('_extracted.json')]
    
    if not extracted_files:
        print(f"No extracted files found in {input_dir}")
        return []
    
    results = []
    for filename in sorted(extracted_files):
        filepath = os.path.join(input_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Handle both dict and list formats
                if isinstance(data, list):
                    # If it's a list, extend results with all items
                    results.extend(data)
                elif isinstance(data, dict):
                    # If it's a dict, append it
                    results.append(data)
                else:
                    print(f"Warning: Unexpected data type in {filename}: {type(data)}")
                    continue
        except Exception as e:
            print(f"Warning: Error loading {filename}: {e}")
            continue
    
    return results


def load_report_statements(debate_dir: str) -> List[Dict]:
    """
    Load all report statements from a debate directory.
    
    Args:
        debate_dir: Path to debate directory containing report_statements/ folder
        
    Returns:
        List of report statement dictionaries with report_id and statement text
    """
    report_statements_dir = os.path.join(debate_dir, "report_statements")
    
    if not os.path.exists(report_statements_dir):
        print(f"Warning: Report statements directory not found: {report_statements_dir}")
        return []
    
    report_files = [f for f in os.listdir(report_statements_dir) 
                   if f.endswith('.json')]
    
    if not report_files:
        print(f"No report statement files found in {report_statements_dir}")
        return []
    
    all_statements = []
    for report_file in sorted(report_files):
        filepath = os.path.join(report_statements_dir, report_file)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                report_id = report_data.get('report_id', report_file.replace('.json', ''))
                paragraphs = report_data.get('paragraphs', [])
                
                # Create a statement dict for each paragraph
                for i, paragraph in enumerate(paragraphs):
                    if paragraph and paragraph.strip():
                        all_statements.append({
                            'statement_id': f"{report_id}_para_{i+1}",
                            'statement_text': "The European Parliament " + re.sub(r'^[^a-zA-Z]*', '', paragraph.strip()),
                            'report_id': report_id,
                            'paragraph_index': i + 1
                        })
        except Exception as e:
            print(f"Warning: Error loading {report_file}: {e}")
            continue
    
    return all_statements


def collect_all_arguments(extracted_results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Collect all arguments from extracted results, grouped by debate_id.
    Filters out arguments from president interventions.
    
    Args:
        extracted_results: List of extraction result dictionaries
        
    Returns:
        Dictionary mapping debate_id to list of argument dictionaries
    """
    arguments_by_debate = {}
    skipped_president_count = 0
    
    for result in extracted_results:
        intervention_id = result.get('intervention_id', 'unknown')
        debate_id = result.get('debate_id', 'unknown')
        speaker = result.get('speaker', 'unknown')
        agenda_item = result.get('agenda_item', 'unknown')
        extracted_args = result.get('extracted_arguments', '')
        
        # Skip president interventions
        if is_president_intervention(speaker):
            skipped_president_count += 1
            continue
        
        arguments = extract_arguments(extracted_args)
        
        # Initialize list for this debate if not exists
        if debate_id not in arguments_by_debate:
            arguments_by_debate[debate_id] = []
        
        for i, argument in enumerate(arguments):
            arguments_by_debate[debate_id].append({
                'argument_id': f"{intervention_id}_arg_{i+1}",
                'argument_text': argument,
                'intervention_id': intervention_id,
                'debate_id': debate_id,
                'speaker': speaker,
                'agenda_item': agenda_item
            })
    
    return arguments_by_debate


def parse_intervention_id(intervention_id: str) -> Tuple[int, int]:
    """
    Parse intervention ID to extract chronological ordering.
    
    Args:
        intervention_id: Intervention ID like "2-019" or "3-360"
        
    Returns:
        Tuple of (section_number, sequence_number) for comparison
    """
    parts = intervention_id.split('-')
    if len(parts) == 2:
        try:
            section = int(parts[0])
            sequence = int(parts[1])
            return (section, sequence)
        except ValueError:
            return (0, 0)
    return (0, 0)


def compare_intervention_ids(id1: str, id2: str) -> int:
    """
    Compare two intervention IDs chronologically.
    
    Args:
        id1: First intervention ID
        id2: Second intervention ID
        
    Returns:
        -1 if id1 < id2, 0 if id1 == id2, 1 if id1 > id2
    """
    parsed1 = parse_intervention_id(id1)
    parsed2 = parse_intervention_id(id2)
    
    if parsed1 < parsed2:
        return -1
    elif parsed1 > parsed2:
        return 1
    else:
        return 0


def classify_argument_pairs(model: ArgumentRelationModel, arguments: List[Dict], 
                        debate_id: str, topic: str = "", output_file: Optional[str] = None,
                        max_pairs: Optional[int] = None) -> List[Dict]:
    """
    Classify relations between all pairs of arguments within the same debate.
    Only compares arguments chronologically (source must come after target).
    
    Args:
        model: Loaded ArgumentRelationModel
        arguments: List of argument dictionaries (all from same debate)
        debate_id: ID of the debate
        topic: Topic/context for the debate
        output_file: Optional file to save results
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        List of relation classification results
    """
    # Generate all pairs within this debate
    pairs = list(combinations(range(len(arguments)), 2))
    
    # Filter pairs: only keep where source comes after target chronologically
    filtered_pairs = []
    for idx1, idx2 in pairs:
        arg1 = arguments[idx1]
        arg2 = arguments[idx2]
        
        # Skip pairs from the same speaker
        if arg1['speaker'] == arg2['speaker']:
            continue
        
        # Only keep pairs where source (arg1) comes after target (arg2) chronologically
        source_intervention_id = arg1.get('intervention_id', '')
        target_intervention_id = arg2.get('intervention_id', '')
        
        if compare_intervention_ids(source_intervention_id, target_intervention_id) > 0:
            filtered_pairs.append((idx1, idx2))
    
    if max_pairs:
        filtered_pairs = filtered_pairs[:max_pairs]
    
    results = []
    for idx1, idx2 in tqdm(filtered_pairs, desc="Classifying argument pairs"):
        arg1 = arguments[idx1]
        arg2 = arguments[idx2]
        
        try:
            relation_result = model.classify_relation(
                source=arg1['argument_text'],
                target=arg2['argument_text'],
                topic=topic
            )
            
            # Handle both dict and string returns for backward compatibility
            if isinstance(relation_result, dict):
                relation = relation_result.get('relation', '')
                reasoning = relation_result.get('reasoning')
            else:
                relation = relation_result
                reasoning = None
            
            result = {
                'pair_id': f"{arg1['argument_id']}_vs_{arg2['argument_id']}",
                'relation_type': 'argument_to_argument',
                'debate_id': debate_id,
                'source_argument': {
                    'argument_id': arg1['argument_id'],
                    'argument_text': arg1['argument_text'],
                    'intervention_id': arg1['intervention_id'],
                    'speaker': arg1['speaker']
                },
                'target_argument': {
                    'argument_id': arg2['argument_id'],
                    'argument_text': arg2['argument_text'],
                    'intervention_id': arg2['intervention_id'],
                    'speaker': arg2['speaker']
                },
                'relation': relation,
                'reasoning': reasoning,
                'topic': topic
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error classifying relation: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def classify_arguments_to_report_statements(model: ArgumentRelationModel, 
                                        arguments: List[Dict], 
                                        report_statements: List[Dict],
                                        debate_id: str, 
                                        topic: str = "", 
                                        output_file: Optional[str] = None,
                                        max_pairs: Optional[int] = None) -> List[Dict]:
    """
    Classify relations between mined arguments and report statements.
    
    Args:
        model: Loaded ArgumentRelationModel
        arguments: List of argument dictionaries (mined from interventions)
        report_statements: List of report statement dictionaries
        debate_id: ID of the debate
        topic: Topic/context for the debate
        output_file: Optional file to save results
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        List of relation classification results
    """
    # Generate all pairs: each argument vs each report statement
    pairs = []
    for arg_idx in range(len(arguments)):
        for stmt_idx in range(len(report_statements)):
            pairs.append((arg_idx, stmt_idx))
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    results = []
    for arg_idx, stmt_idx in tqdm(pairs, desc="Classifying argument-to-report-statement pairs"):
        argument = arguments[arg_idx]
        statement = report_statements[stmt_idx]
        
        try:
            # Classify relation: argument (source) vs report statement (target)
            relation_result = model.classify_relation(
                source=argument['argument_text'],
                target=statement['statement_text'],
                topic=topic
            )
            
            # Handle both dict and string returns for backward compatibility
            if isinstance(relation_result, dict):
                relation = relation_result.get('relation', '')
                reasoning = relation_result.get('reasoning')
            else:
                relation = relation_result
                reasoning = None
            
            result = {
                'pair_id': f"{argument['argument_id']}_vs_{statement['statement_id']}",
                'relation_type': 'argument_to_report_statement',
                'debate_id': debate_id,
                'argument': {
                    'argument_id': argument['argument_id'],
                    'argument_text': argument['argument_text'],
                    'intervention_id': argument['intervention_id'],
                    'speaker': argument['speaker']
                },
                'report_statement': {
                    'statement_id': statement['statement_id'],
                    'statement_text': statement['statement_text'],
                    'report_id': statement['report_id'],
                    'paragraph_index': statement['paragraph_index']
                },
                'relation': relation,
                'reasoning': reasoning,
                'topic': topic
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"Error classifying relation: {e}")
            continue
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Argument Relation Classification on extracted arguments and report statements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify relations for all argument pairs and argument-to-report-statement pairs
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results.json

  # Use Ollama instead of local model
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results.json --use-ollama --ollama-model llama3.1

  # Limit to first 100 pairs for testing
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --max-pairs 100 --output results/arc_test.json

  # Skip argument-to-argument relations (only classify argument-to-report-statement)
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --skip-claim-pairs --output results/arc_results.json
        """
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                      help='Directory containing extracted JSON files')
    parser.add_argument('--debates-dir', type=str, required=True,
                      help='Root directory containing debate folders (e.g., data/debates)')
    parser.add_argument('--output', type=str, required=True,
                      help='Output directory for ARC results')
    parser.add_argument('--topic', type=str, default='',
                      help='Topic/context for the debate (optional)')
    
    # Model options
    parser.add_argument('--use-ollama', action='store_true',
                      help='Use Ollama API instead of loading local model (requires Ollama running)')
    parser.add_argument('--ollama-model', type=str, default='llama3.1',
                      help='Ollama model name (default: llama3.1, only used with --use-ollama)')
    parser.add_argument('--ollama-url', type=str, default=None,
                      help='Ollama API URL (default: from OLLAMA_URL env var or http://localhost:11434, only used with --use-ollama)')
    parser.add_argument('--model', type=str,
                      default='brunoyun/Llama-3.1-Amelia-AR-8B-v1',
                      help='Model name (default: brunoyun/Llama-3.1-Amelia-AR-8B-v1, ignored if --use-ollama)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                      help='Device to use (auto-detect if not specified, ignored if --use-ollama)')
    parser.add_argument('--load-in-4bit', action='store_true',
                      help='Use 4-bit quantization (CUDA only, ignored if --use-ollama)')
    parser.add_argument('--load-in-8bit', action='store_true',
                      help='Use 8-bit quantization (works on all platforms, recommended for MPS, ignored if --use-ollama)')
    
    # Processing options
    parser.add_argument('--max-pairs', type=int,
                      help='Maximum number of pairs to process per type (None = all)')
    parser.add_argument('--skip-claim-pairs', action='store_true',
                      help='Skip argument-to-argument relation classification (only do argument-to-report-statement)')
    parser.add_argument('--skip-report-relations', action='store_true',
                      help='Skip argument-to-report-statement relation classification (only do argument-to-argument)')
    
    args = parser.parse_args()
    
    # Load extracted files
    extracted_results = load_extracted_files(args.input)
    
    if not extracted_results:
        print("No extracted files found. Exiting.")
        sys.exit(1)
    
    # Collect all arguments grouped by debate
    arguments_by_debate = collect_all_arguments(extracted_results)
    
    # Filter debates with at least 1 argument (needed for report statement relations)
    valid_debates = {debate_id: arguments for debate_id, arguments in arguments_by_debate.items() 
                     if len(arguments) >= 1}
    
    if not valid_debates:
        print("No debates found with at least 1 argument. Exiting.")
        sys.exit(1)
    
    # For argument-to-argument relations, need at least 2 arguments
    debates_with_multiple_arguments = {debate_id: arguments for debate_id, arguments in valid_debates.items() 
                                    if len(arguments) >= 2}
    
    # Load ARC model
    try:
        if args.use_ollama:
            # Use provided URL or None (which will use env var or default)
            ollama_url = args.ollama_url if args.ollama_url else None
            model = ArgumentRelationModelOllama(
                model_name=args.ollama_model,
                base_url=ollama_url
            )
        else:
            # Auto-detect device if not specified (prefer CUDA for remote GPU)
            if not args.device:
                if torch.cuda.is_available():
                    args.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    args.device = "mps"
                else:
                    args.device = "cpu"
            
            # Auto-detect quantization based on device
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
        print(f"Error loading model: {e}")
        if "out of memory" in str(e).lower():
            print("\nTip: Try using --load-in-8bit, --device cpu, --use-ollama, or reduce --max-pairs")
        elif args.use_ollama:
            print("\nTip: Make sure Ollama is running: ollama serve")
            print(f"      And the model is pulled: ollama pull {args.ollama_model}")
        sys.exit(1)
    
    # Create output directory structure
    output_base_dir = args.output
    os.makedirs(output_base_dir, exist_ok=True)
    
    import time
    total_start_time = time.time()
    all_results = {
        'argument_to_argument': [],
        'argument_to_report_statement': []
    }
    total_pairs = {'argument_to_argument': 0, 'argument_to_report_statement': 0}
    
    try:
        for debate_id, arguments in tqdm(valid_debates.items(), desc="Processing debates"):
            # Get topic for this debate
            topic = args.topic
            if not topic and arguments:
                topic = arguments[0].get('agenda_item', '')
            
            # Load report statements for this debate
            debate_dir = os.path.join(args.debates_dir, debate_id)
            report_statements = load_report_statements(debate_dir)
            
            # Create subdirectory for this debate
            debate_output_dir = os.path.join(output_base_dir, debate_id)
            os.makedirs(debate_output_dir, exist_ok=True)
            
            debate_results = {
                'argument_to_argument': [],
                'argument_to_report_statement': []
            }
            
            # 1. Classify relations between arguments and report statements
            if not args.skip_report_relations and report_statements and arguments:
                argument_to_report_file = os.path.join(debate_output_dir, f"{debate_id}_argument_to_report.json")
                
                debate_max_pairs = None
                if args.max_pairs:
                    debate_max_pairs = args.max_pairs
                
                argument_to_report_results = classify_arguments_to_report_statements(
                    model,
                    arguments,
                    report_statements,
                    debate_id=debate_id,
                    topic=topic,
                    output_file=argument_to_report_file,
                    max_pairs=debate_max_pairs
                )
                
                debate_results['argument_to_report_statement'] = argument_to_report_results
                all_results['argument_to_report_statement'].extend(argument_to_report_results)
                total_pairs['argument_to_report_statement'] += len(argument_to_report_results)
            
            # 2. Classify relations between all pairs of arguments
            if not args.skip_claim_pairs and len(arguments) >= 2:
                argument_to_argument_file = os.path.join(debate_output_dir, f"{debate_id}_argument_to_argument.json")
                
                debate_max_pairs = None
                if args.max_pairs:
                    debate_max_pairs = args.max_pairs
                
                argument_to_argument_results = classify_argument_pairs(
                    model,
                    arguments,
                    debate_id=debate_id,
                    topic=topic,
                    output_file=argument_to_argument_file,
                    max_pairs=debate_max_pairs
                )
                
                debate_results['argument_to_argument'] = argument_to_argument_results
                all_results['argument_to_argument'].extend(argument_to_argument_results)
                total_pairs['argument_to_argument'] += len(argument_to_argument_results)
            
            # Save combined results for this debate
            combined_file = os.path.join(debate_output_dir, f"{debate_id}_arc_results.json")
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(debate_results, f, indent=2, ensure_ascii=False)
        
        total_elapsed = time.time() - total_start_time
        
        # Save overall combined results
        combined_output_file = os.path.join(output_base_dir, "all_arc_results.json")
        with open(combined_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
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


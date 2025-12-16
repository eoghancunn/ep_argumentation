"""
Script for running Argument Relation Classification (ARC) on extracted claims.

This script:
1. Loads extracted argument files
2. Extracts all <claim> tags from the extracted_arguments
3. Loads report statements from debate directories (core claims)
4. Classifies relations between:
   - Mined claims vs report statements (support, attack, no relation)
   - All pairs of mined claims (support, attack, no relation)
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

from src.argument_models import ArgumentRelationModel


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


def extract_claims(extracted_arguments: str) -> List[str]:
    """
    Extract all claims from the extracted_arguments text.
    
    Args:
        extracted_arguments: Text containing <claim> tags
        
    Returns:
        List of claim texts
    """
    if not extracted_arguments:
        return []
    
    # Pattern to match <claim>...</claim> tags
    pattern = r'<claim>(.*?)</claim>'
    claims = re.findall(pattern, extracted_arguments, re.DOTALL)
    
    # Clean up claims (strip whitespace)
    claims = [claim.strip() for claim in claims if claim.strip()]
    
    return claims


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
                            'statement_text': paragraph.strip(),
                            'report_id': report_id,
                            'paragraph_index': i + 1
                        })
        except Exception as e:
            print(f"Warning: Error loading {report_file}: {e}")
            continue
    
    return all_statements


def collect_all_claims(extracted_results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Collect all claims from extracted results, grouped by debate_id.
    Filters out claims from president interventions.
    
    Args:
        extracted_results: List of extraction result dictionaries
        
    Returns:
        Dictionary mapping debate_id to list of claim dictionaries
    """
    claims_by_debate = {}
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
        
        claims = extract_claims(extracted_args)
        
        # Initialize list for this debate if not exists
        if debate_id not in claims_by_debate:
            claims_by_debate[debate_id] = []
        
        for i, claim in enumerate(claims):
            if claim != "NA":
                claims_by_debate[debate_id].append({
                    'claim_id': f"{intervention_id}_claim_{i+1}",
                    'claim_text': claim,
                    'intervention_id': intervention_id,
                    'debate_id': debate_id,
                    'speaker': speaker,
                    'agenda_item': agenda_item
                })
    
    total_claims = sum(len(claims) for claims in claims_by_debate.values())
    
    return claims_by_debate


def classify_claim_pairs(model: ArgumentRelationModel, claims: List[Dict], 
                        debate_id: str, topic: str = "", output_file: Optional[str] = None,
                        max_pairs: Optional[int] = None) -> List[Dict]:
    """
    Classify relations between all pairs of claims within the same debate.
    
    Args:
        model: Loaded ArgumentRelationModel
        claims: List of claim dictionaries (all from same debate)
        debate_id: ID of the debate
        topic: Topic/context for the debate
        output_file: Optional file to save results
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        List of relation classification results
    """
    # Generate all pairs within this debate
    pairs = list(combinations(range(len(claims)), 2))
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    total_pairs = len(pairs)
    
    results = []
    for idx1, idx2 in tqdm(pairs, desc="Classifying claim pairs"):
        claim1 = claims[idx1]
        claim2 = claims[idx2]
        
        try:
            relation = model.classify_relation(
                source=claim1['claim_text'],
                target=claim2['claim_text'],
                topic=topic
            )
            
            result = {
                'pair_id': f"{claim1['claim_id']}_vs_{claim2['claim_id']}",
                'relation_type': 'claim_to_claim',
                'debate_id': debate_id,
                'source_claim': {
                    'claim_id': claim1['claim_id'],
                    'claim_text': claim1['claim_text'],
                    'intervention_id': claim1['intervention_id'],
                    'speaker': claim1['speaker']
                },
                'target_claim': {
                    'claim_id': claim2['claim_id'],
                    'claim_text': claim2['claim_text'],
                    'intervention_id': claim2['intervention_id'],
                    'speaker': claim2['speaker']
                },
                'relation': relation,
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


def classify_claims_to_report_statements(model: ArgumentRelationModel, 
                                        claims: List[Dict], 
                                        report_statements: List[Dict],
                                        debate_id: str, 
                                        topic: str = "", 
                                        output_file: Optional[str] = None,
                                        max_pairs: Optional[int] = None) -> List[Dict]:
    """
    Classify relations between mined claims and report statements.
    
    Args:
        model: Loaded ArgumentRelationModel
        claims: List of claim dictionaries (mined from interventions)
        report_statements: List of report statement dictionaries
        debate_id: ID of the debate
        topic: Topic/context for the debate
        output_file: Optional file to save results
        max_pairs: Maximum number of pairs to process (None = all)
        
    Returns:
        List of relation classification results
    """
    # Generate all pairs: each claim vs each report statement
    pairs = []
    for claim_idx in range(len(claims)):
        for stmt_idx in range(len(report_statements)):
            pairs.append((claim_idx, stmt_idx))
    
    if max_pairs:
        pairs = pairs[:max_pairs]
    
    results = []
    for claim_idx, stmt_idx in tqdm(pairs, desc="Classifying claim-to-report-statement pairs"):
        claim = claims[claim_idx]
        statement = report_statements[stmt_idx]
        
        try:
            # Classify relation: claim (source) vs report statement (target)
            relation = model.classify_relation(
                source=claim['claim_text'],
                target=statement['statement_text'],
                topic=topic
            )
            
            result = {
                'pair_id': f"{claim['claim_id']}_vs_{statement['statement_id']}",
                'relation_type': 'claim_to_report_statement',
                'debate_id': debate_id,
                'claim': {
                    'claim_id': claim['claim_id'],
                    'claim_text': claim['claim_text'],
                    'intervention_id': claim['intervention_id'],
                    'speaker': claim['speaker']
                },
                'report_statement': {
                    'statement_id': statement['statement_id'],
                    'statement_text': statement['statement_text'],
                    'report_id': statement['report_id'],
                    'paragraph_index': statement['paragraph_index']
                },
                'relation': relation,
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
        description="Run Argument Relation Classification on extracted claims and report statements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify relations for all claim pairs and claim-to-report-statement pairs
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --output results/arc_results.json

  # Limit to first 100 pairs for testing
  python run_arc_on_claims.py --input results/ --debates-dir data/debates --max-pairs 100 --output results/arc_test.json

  # Skip claim-to-claim relations (only classify claim-to-report-statement)
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
    parser.add_argument('--model', type=str,
                      default='brunoyun/Llama-3.1-Amelia-AR-8B-v1',
                      help='Model name (default: brunoyun/Llama-3.1-Amelia-AR-8B-v1)')
    parser.add_argument('--device', type=str, choices=['cuda', 'mps', 'cpu'],
                      help='Device to use (auto-detect if not specified)')
    parser.add_argument('--load-in-4bit', action='store_true',
                      help='Use 4-bit quantization (CUDA only)')
    parser.add_argument('--load-in-8bit', action='store_true',
                      help='Use 8-bit quantization (works on all platforms, recommended for MPS)')
    
    # Processing options
    parser.add_argument('--max-pairs', type=int,
                      help='Maximum number of pairs to process per type (None = all)')
    parser.add_argument('--skip-claim-pairs', action='store_true',
                      help='Skip claim-to-claim relation classification (only do claim-to-report-statement)')
    parser.add_argument('--skip-report-relations', action='store_true',
                      help='Skip claim-to-report-statement relation classification (only do claim-to-claim)')
    
    args = parser.parse_args()
    
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
    
    # Load extracted files
    extracted_results = load_extracted_files(args.input)
    
    if not extracted_results:
        print("No extracted files found. Exiting.")
        sys.exit(1)
    
    # Collect all claims grouped by debate
    claims_by_debate = collect_all_claims(extracted_results)
    
    # Filter debates with at least 1 claim (needed for report statement relations)
    valid_debates = {debate_id: claims for debate_id, claims in claims_by_debate.items() 
                     if len(claims) >= 1}
    
    if not valid_debates:
        print("No debates found with at least 1 claim. Exiting.")
        sys.exit(1)
    
    # For claim-to-claim relations, need at least 2 claims
    debates_with_multiple_claims = {debate_id: claims for debate_id, claims in valid_debates.items() 
                                    if len(claims) >= 2}
    
    # Load ARC model
    
    try:
        model = ArgumentRelationModel(
            model_name=args.model,
            device=args.device,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        if "out of memory" in str(e).lower():
            print("\nTip: Try using --load-in-8bit, --device cpu, or reduce --max-pairs")
        sys.exit(1)
    
    # Create output directory structure
    output_base_dir = args.output
    os.makedirs(output_base_dir, exist_ok=True)
    
    import time
    total_start_time = time.time()
    all_results = {
        'claim_to_claim': [],
        'claim_to_report_statement': []
    }
    total_pairs = {'claim_to_claim': 0, 'claim_to_report_statement': 0}
    
    try:
        for debate_id, claims in tqdm(valid_debates.items(), desc="Processing debates"):
            # Get topic for this debate
            topic = args.topic
            if not topic and claims:
                topic = claims[0].get('agenda_item', '')
            
            # Load report statements for this debate
            debate_dir = os.path.join(args.debates_dir, debate_id)
            report_statements = load_report_statements(debate_dir)
            
            # Create subdirectory for this debate
            debate_output_dir = os.path.join(output_base_dir, debate_id)
            os.makedirs(debate_output_dir, exist_ok=True)
            
            debate_results = {
                'claim_to_claim': [],
                'claim_to_report_statement': []
            }
            
            # 1. Classify relations between claims and report statements
            if not args.skip_report_relations and report_statements and claims:
                claim_to_report_file = os.path.join(debate_output_dir, f"{debate_id}_claim_to_report.json")
                
                debate_max_pairs = None
                if args.max_pairs:
                    debate_max_pairs = args.max_pairs
                
                claim_to_report_results = classify_claims_to_report_statements(
                    model,
                    claims,
                    report_statements,
                    debate_id=debate_id,
                    topic=topic,
                    output_file=claim_to_report_file,
                    max_pairs=debate_max_pairs
                )
                
                debate_results['claim_to_report_statement'] = claim_to_report_results
                all_results['claim_to_report_statement'].extend(claim_to_report_results)
                total_pairs['claim_to_report_statement'] += len(claim_to_report_results)
            
            # 2. Classify relations between all pairs of claims
            if not args.skip_claim_pairs and len(claims) >= 2:
                claim_to_claim_file = os.path.join(debate_output_dir, f"{debate_id}_claim_to_claim.json")
                
                debate_max_pairs = None
                if args.max_pairs:
                    debate_max_pairs = args.max_pairs
                
                claim_to_claim_results = classify_claim_pairs(
                    model,
                    claims,
                    debate_id=debate_id,
                    topic=topic,
                    output_file=claim_to_claim_file,
                    max_pairs=debate_max_pairs
                )
                
                debate_results['claim_to_claim'] = claim_to_claim_results
                all_results['claim_to_claim'].extend(claim_to_claim_results)
                total_pairs['claim_to_claim'] += len(claim_to_claim_results)
            
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


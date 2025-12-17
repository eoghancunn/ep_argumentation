"""
Module for loading and using argumentation models from Hugging Face.

Models:
- oberbics/llama-3.1-8B-newspaper_argument_mining: For extracting argumentative units
- brunoyun/Llama-3.1-Amelia-AR-8B-v1: For Argument Relation classification (attack, support, no relation)
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple
import json
import os
import platform
import re

try:
    from gradio_client import Client
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class ArgumentMiningModel:
    """Model for extracting argumentative units from text."""
    
    # System prompt for argument extraction (from model usage example)
    SYSTEM_PROMPT = '''You are an expert at analyzing historical texts and you hate to summarize

OUTPUT FORMAT - EXACTLY these 4 XML tags and NOTHING else:
<argument>Original argument text OR "NA"</argument>
<claim>Core claim (implication) in one sentence OR "NA"</claim>
<explanation>Why this is an argument OR "NA"</explanation>
<confidence>0-1</confidence>

EXAMPLE WITH STRONG ARGUMENT:
<argument>Il giornale L'Italia moderna economica e finanziaria nel numero di oggi propone che non si facciano sottoscrizioni, le quali per quanto larghe sarebbero sempre impari ai bisogni, ma che il Parlamento stabilisca pochi centesimi addizionali per ogni lira su tutte le imposte e tasse (esclusi soltanto i dazi doganali la cui misura è vincolata da trattati di commercio).</argument>
<claim>Private subscriptions are inadequate for earthquake relief; parliamentary taxation would be more effective.</claim>
<explanation>The newspaper explicitly argues against private subscriptions as insufficient and proposes a specific alternative solution through parliamentary taxation, making a clear comparative argument about funding mechanisms.</explanation>
<confidence>0.95</confidence>

EXAMPLE WITHOUT ARGUMENT:
<argument>NA</argument>
<claim>NA</claim>
<explanation>NA</explanation>
<confidence>0.9</confidence>

RULES:
- CRITICAL: NEVER REPEAT ARGUMENTS - Each argument must be COMPLETELY UNIQUE
- Only output arguments that appear verbatim (or nearly verbatim) in the text
- NO SUMMARY; ONLY EXACT EXTRACTION FROM THE TEXT
- Extract only original text without changes or use NA when you did not find an argument
- If no argument exists, use NA for ALL fields
- More than one argument possible for one article'''
    
    def __init__(self, model_name: str = "oberbics/llama-3.1-8B-newspaper_argument_mining", 
                 device: Optional[str] = None, load_in_4bit: bool = False,
                 load_in_8bit: bool = False, low_cpu_mem_usage: bool = True):
        """
        Initialize the argument mining model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on ('cuda', 'cpu', 'mps', or None for auto)
            load_in_4bit: Whether to load model in 4-bit quantization (CUDA only)
            load_in_8bit: Whether to load model in 8-bit quantization (works on MPS/CPU)
            low_cpu_mem_usage: Use low CPU memory usage mode
        """
        self.model_name = model_name
        # Auto-detect best device: CUDA > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.load_in_4bit = load_in_4bit
        
        print(f"Loading argument mining model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        # Check if 4-bit quantization is available (not on macOS/Apple Silicon)
        can_use_4bit = load_in_4bit and torch.cuda.is_available() and platform.system() != "Darwin"
        
        if can_use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                print("Using 4-bit quantization")
            except Exception as e:
                print(f"Warning: 4-bit quantization failed ({e}), falling back to regular loading")
                can_use_4bit = False
        
        if not can_use_4bit:
            if load_in_4bit and (not torch.cuda.is_available() or platform.system() == "Darwin"):
                print("Note: 4-bit quantization not available on this system, using regular loading")
            # Use float16 for CUDA/MPS, float32 for CPU
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None
            )
            # Move to device if not using CUDA with device_map
            if self.device != "cuda":
                self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Argument mining model loaded successfully!")
    
    def extract_arguments(self, text: str, max_new_tokens: int = 800, 
                         temperature: float = 0.1, top_p: float = 0.95,
                         use_greedy: bool = False) -> str:
        """
        Extract argumentative units from text.
        
        Args:
            text: Input text to analyze
            max_new_tokens: Maximum number of tokens to generate (default: 800 as per model example)
            temperature: Sampling temperature (default: 0.1 as per model example)
            top_p: Nucleus sampling parameter (default: 0.95 as per model example)
            use_greedy: Use greedy decoding instead of sampling (faster, less diverse)
            
        Returns:
            Generated text with argumentative units
        """
        # Prepare messages with system prompt and user instruction (as per model usage example)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract argumentative units from historical text in their original form, no summaries.\n{text}"}
        ]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(messages, return_tensors="pt").to(self.device)
        
        # Generation parameters (matching model example)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if not use_greedy else 0.0,
            "top_p": top_p if not use_greedy else 1.0,
            "repetition_penalty": 1.15,
            "do_sample": not use_greedy,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        with torch.no_grad():
            outputs = self.model.generate(inputs, **generation_kwargs)
        
        # Decode only the newly generated tokens (skip the input tokens)
        # Use inputs.shape[1] to get input length (as per model example)
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return generated_text


class ArgumentMiningModelAPI:
    """API-based model for extracting argumentative units from text using Gradio API."""
    
    def __init__(self, api_url: str = "oberbics/Argument-Mining"):
        """
        Initialize the API-based argument mining model.
        
        Args:
            api_url: Gradio API URL or Hugging Face Space name
        """
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "gradio_client is required for API mode. Install it with: pip install gradio-client"
            )
        
        self.api_url = api_url
        print(f"Connecting to Argument Mining API: {api_url}")
        self.client = Client(api_url)
        print("API connection established!")
    
    def extract_arguments(self, text: str, max_new_tokens: int = 256, 
                         temperature: float = 0.05, top_p: float = 0.9,
                         use_greedy: bool = False) -> str:
        """
        Extract argumentative units from text using the API.
        
        Args:
            text: Input text to analyze
            max_new_tokens: Maximum number of tokens to generate (API may ignore this)
            temperature: Sampling temperature (default: 0.05 as per API example)
            top_p: Nucleus sampling parameter (API may ignore this)
            use_greedy: Not used for API (kept for compatibility)
            
        Returns:
            Generated text with argumentative units
        """
        try:
            result = self.client.predict(
                text=text,
                temperature=temperature,
                api_name="/predict"
            )
            return result
        except Exception as e:
            print(f"API error: {e}")
            raise


class ArgumentRelationModel:
    """Model for classifying argument relations (attack, support, no relation)."""
    
    def __init__(self, model_name: str = "brunoyun/Llama-3.1-Amelia-AR-8B-v1",
                 device: Optional[str] = None, load_in_4bit: bool = False,
                 load_in_8bit: bool = False, low_cpu_mem_usage: bool = True):
        """
        Initialize the argument relation model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to load model on ('cuda', 'cpu', 'mps', or None for auto)
            load_in_4bit: Whether to load model in 4-bit quantization (CUDA only)
            load_in_8bit: Whether to load model in 8-bit quantization (works on MPS/CPU)
            low_cpu_mem_usage: Use low CPU memory usage mode
        """
        self.model_name = model_name
        # Auto-detect best device: CUDA > MPS (Apple Silicon) > CPU
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.low_cpu_mem_usage = low_cpu_mem_usage
        
        # System prompt as specified in the model card
        self.system_prompt = (
            "You are an expert in argumentation. Your task is to determine the type of relation "
            "between [SOURCE] and [TARGET]. The type of relation would be in the [RELATION] set. "
            "Utilize the [TOPIC] as context to support your decision\n"
            "Your answer must be in the following format with only the type of the relation in the answer section:\n"
            "<|ANSWER|><answer><|ANSWER|>."
        )
        
        # Model parameters as specified in the model card
        self.temperature = 1.5
        self.min_p = 0.1
        
        print(f"Loading argument relation model: {model_name}")
        print(f"Device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with memory-efficient options
        # Priority: 4-bit (CUDA only) > 8-bit (all platforms) > regular loading
        
        # Try 4-bit quantization (CUDA only)
        can_use_4bit = load_in_4bit and torch.cuda.is_available() and platform.system() != "Darwin"
        if can_use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=low_cpu_mem_usage
                )
                print("Using 4-bit quantization")
            except Exception as e:
                print(f"Warning: 4-bit quantization failed ({e}), trying 8-bit...")
                can_use_4bit = False
        
        # Try 8-bit quantization (works on MPS/CPU/CUDA)
        if not can_use_4bit and load_in_8bit:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True,
                    torch_dtype=torch.float16 if self.device in ("cuda", "mps") else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=low_cpu_mem_usage
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
                print("Using 8-bit quantization")
            except Exception as e:
                print(f"Warning: 8-bit quantization failed ({e}), falling back to regular loading")
                load_in_8bit = False
        
        # Regular loading (no quantization)
        if not can_use_4bit and not load_in_8bit:
            if (load_in_4bit or load_in_8bit) and (not torch.cuda.is_available() or platform.system() == "Darwin"):
                print("Note: Quantization not available, using regular loading (may use more memory)")
            
            # Use float16 for CUDA/MPS, float32 for CPU
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
            
            # For MPS, use device_map to help with memory management
            if self.device == "mps":
                # Try to use device_map for better memory management on MPS
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        low_cpu_mem_usage=low_cpu_mem_usage
                    )
                    self.model = self.model.to(self.device)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("Warning: MPS out of memory. Consider using load_in_8bit=True or device='cpu'")
                        raise
                    raise
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=low_cpu_mem_usage
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)
        
        self.model.eval()
        print("Argument relation model loaded successfully!")
    
    def classify_relation(self, source: str, target: str, topic: str = "",
                         relations: List[str] = None, max_new_tokens: int = 128) -> Dict[str, Optional[str]]:
        """
        Classify the relation between source and target arguments.
        
        Args:
            source: Source argument text
            target: Target argument text
            topic: Topic/context for the arguments
            relations: List of possible relations (default: ['no relation', 'attack', 'support'])
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with 'relation' (predicted relation type) and 'reasoning' (None for this model)
        """
        if relations is None:
            relations = ['no relation', 'attack', 'support']
        
        # Format relations as a set string
        relations_str = "{'" + "', '".join(relations) + "'}"
        
        # Create user message
        user_content = (
            f"[RELATION]: {relations_str}\n"
            f"[TOPIC]: {topic}\n"
            f"[SOURCE]: {source}\n"
            f"[TARGET]: {target}\n"
        )
        
        # Create messages for chat template
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': user_content}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with specified parameters
        # Note: min_p requires transformers >= 4.40.0
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    text,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    min_p=self.min_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            except TypeError:
                # Fallback for older transformers versions that don't support min_p
                outputs = self.model.generate(
                    text,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        # Decode only the newly generated tokens (skip the input tokens)
        input_length = text.shape[1]
        generated_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Extract answer from <|ANSWER|> tags
        relation = None
        if "<|ANSWER|>" in generated_text:
            parts = generated_text.split("<|ANSWER|>")
            if len(parts) >= 2:
                answer = parts[1].strip()
                # Remove any trailing <|ANSWER|> tag
                answer = answer.replace("<|ANSWER|>", "").strip()
                # Clean up any remaining whitespace or newlines
                answer = answer.strip()
                if answer:
                    relation = answer
        
        # If no answer found in tags, use the generated text (might be the answer itself)
        if not relation:
            relation = generated_text.strip()
        
        return {'relation': relation, 'reasoning': None, 'raw_output': None}


class ArgumentRelationModelOllama:
    """Model for classifying argument relations using Ollama API."""
    
    def __init__(self, model_name: str = "llama3.1", base_url: Optional[str] = None):
        """
        Initialize the argument relation model using Ollama.
        
        Args:
            model_name: Ollama model name (default: "llama3.1")
            base_url: Ollama API base URL (default: from OLLAMA_URL env var or "http://localhost:11435")
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is required for Ollama mode. Install it with: pip install requests"
            )
        
        self.model_name = model_name
        # Get base URL from environment variable or use provided/default
        if base_url is None:
            base_url = os.getenv('OLLAMA_URL', 'http://localhost:11435')
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        
        # System prompt as specified in the model card (same as ArgumentRelationModel)
        self.system_prompt = (
            "You are an expert in argumentation. Your task is to determine the type of relation "
            "between [SOURCE] and [TARGET]. The type of relation would be in the [RELATION] set. "
            "Utilize the [TOPIC] as context to support your decision\n"
            "Your answer must be in the following format with only the type of the relation in the answer section:\n"
            "<|ANSWER|><answer><|ANSWER|>."
        )
        
        # Model parameters as specified in the model card
        self.temperature = 1.5
        self.min_p = 0.1
        
        print(f"Using Ollama for argument relation classification")
        print(f"Model: {model_name}")
        print(f"API URL: {self.api_url}")
        
        # Test connection
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"name": model_name},
                timeout=5
            )
            if response.status_code == 200:
                print("Ollama connection successful!")
            else:
                print(f"Warning: Could not verify model '{model_name}'. Make sure it's pulled: ollama pull {model_name}")
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            print(f"Make sure Ollama is running: ollama serve")
    
    def classify_relation(self, source: str, target: str, topic: str = "",
                         relations: List[str] = None, max_new_tokens: int = 128) -> Dict[str, Optional[str]]:
        """
        Classify the relation between source and target arguments using Ollama.
        
        Args:
            source: Source argument text
            target: Target argument text
            topic: Topic/context for the arguments
            relations: List of possible relations (default: ['no relation', 'attack', 'support'])
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary with 'relation' (predicted relation type), 'reasoning' 
            (extracted from <think> tags if present, None otherwise), and 'raw_output' 
            (complete unprocessed response from Ollama)
        """
        if relations is None:
            relations = ['no relation', 'attack', 'support']
        
        # Format relations as a set string
        relations_str = "{'" + "', '".join(relations) + "'}"
        
        # Create user message (same format as ArgumentRelationModel)
        user_content = (
            f"[RELATION]: {relations_str}\n"
            f"[TOPIC]: {topic}\n"
            f"[SOURCE]: {source}\n"
            f"[TARGET]: {target}\n"
        )
        
        # Combine system prompt and user content for generate API
        prompt = f"{self.system_prompt}\n\n{user_content}"
        
        # Prepare request payload using generate API format
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think":False,
            "options": {
                "temperature": self.temperature,
                "num_predict": max_new_tokens,
                "min_p": self.min_p
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            # Generate API returns response directly
            generated_text = result.get('response', '').strip()
            
            # Store raw output before any processing
            raw_output = generated_text
            
            # Extract reasoning from <think> tags if present
            reasoning = None
            if "<think>" in generated_text and "</think>" in generated_text:
                reasoning_match = re.search(r'<think>(.*?)</think>', 
                                           generated_text, re.DOTALL)
                if reasoning_match:
                    reasoning = reasoning_match.group(1).strip()
                    # Remove reasoning from generated_text to get the actual response
                    generated_text = re.sub(r'<think>.*?</think>', '', 
                                          generated_text, flags=re.DOTALL).strip()
            
            # Extract answer from <|ANSWER|> tags (same as ArgumentRelationModel)
            relation = None
            if "<|ANSWER|>" in generated_text:
                parts = generated_text.split("<|ANSWER|>")
                if len(parts) >= 2:
                    answer = parts[1].strip()
                    # Remove any trailing <|ANSWER|> tag
                    answer = answer.replace("<|ANSWER|>", "").strip()
                    # Clean up any remaining whitespace or newlines
                    answer = answer.strip()
                    if answer:
                        relation = answer
            
            # If no answer found in tags, use the generated text (might be the answer itself)
            if not relation:
                relation = generated_text.strip()
            
            return {'relation': relation, 'reasoning': reasoning, 'raw_output': raw_output}
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            raise


def load_intervention(file_path: str) -> Dict:
    """Load a single intervention JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_debate_interventions(debate_dir: str, mining_model: ArgumentMiningModel = None,
                                 relation_model: ArgumentRelationModel = None) -> List[Dict]:
    """
    Process all interventions in a debate directory.
    
    Args:
        debate_dir: Path to debate directory
        mining_model: Optional argument mining model
        relation_model: Optional argument relation model
        
    Returns:
        List of processed interventions with extracted arguments and relations
    """
    interventions_dir = os.path.join(debate_dir, "interventions")
    if not os.path.exists(interventions_dir):
        interventions_dir = debate_dir
    
    results = []
    intervention_files = [f for f in os.listdir(interventions_dir) if f.endswith('.json')]
    
    for intervention_file in sorted(intervention_files):
        file_path = os.path.join(interventions_dir, intervention_file)
        intervention = load_intervention(file_path)
        
        result = {
            'intervention_id': intervention.get('intervention_id'),
            'debate_id': intervention.get('debate_id'),
            'speaker': intervention.get('speaker'),
            'text': intervention.get('english') or intervention.get('original'),
            'agenda_item': intervention.get('agenda_item')
        }
        
        # Extract arguments if mining model is provided
        if mining_model and result['text']:
            try:
                result['extracted_arguments'] = mining_model.extract_arguments(result['text'])
            except Exception as e:
                print(f"Error extracting arguments for {intervention_file}: {e}")
                result['extracted_arguments'] = None
        
        results.append(result)
    
    # Classify relations between interventions if relation model is provided
    if relation_model and len(results) > 1:
        topic = results[0].get('agenda_item', '') if results else ''
        for i in range(len(results) - 1):
            source = results[i].get('text', '')
            target = results[i + 1].get('text', '')
            
            if source and target:
                try:
                    relation = relation_model.classify_relation(
                        source=source,
                        target=target,
                        topic=topic
                    )
                    results[i]['relation_to_next'] = relation
                except Exception as e:
                    print(f"Error classifying relation for intervention {i}: {e}")
                    results[i]['relation_to_next'] = None
    
    return results


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process debate interventions with argumentation models")
    parser.add_argument("--debate-dir", type=str, required=True,
                       help="Path to debate directory")
    parser.add_argument("--use-mining", action="store_true",
                       help="Use argument mining model")
    parser.add_argument("--use-relation", action="store_true",
                       help="Use argument relation model")
    parser.add_argument("--load-in-4bit", action="store_true",
                       help="Load models in 4-bit quantization")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path")
    
    args = parser.parse_args()
    
    mining_model = None
    relation_model = None
    
    if args.use_mining:
        mining_model = ArgumentMiningModel(load_in_4bit=args.load_in_4bit)
    
    if args.use_relation:
        relation_model = ArgumentRelationModel(load_in_4bit=args.load_in_4bit)
    
    results = process_debate_interventions(
        args.debate_dir,
        mining_model=mining_model,
        relation_model=relation_model
    )
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


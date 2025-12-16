from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "oberbics/llama-3.1-8B-newspaper_argument_mining",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("oberbics/llama-3.1-8B-newspaper_argument_mining")
tokenizer.pad_token = tokenizer.eos_token

# System prompt for argument extraction
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

# Example article
article = """Your historical newspaper text here"""

# Prepare messages
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Extract argumentative units from historical text in their original form, no summaries.\n{article}"}
]

# Generate
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs,
    max_new_tokens=800,
    temperature=0.1,
    top_p=0.95,
    repetition_penalty=1.15,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(response)

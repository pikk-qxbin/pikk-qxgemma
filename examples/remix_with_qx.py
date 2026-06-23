#!/usr/bin/env python3
"""
QxGemma Remix Example

Demonstrates creative, quantum-inspired text generation with Gemma + QxBin.

Run after installing requirements and having QxBin available.

This is perfect for:
- Podcast script remixing / AI Remix episodes
- Wild product / feature ideation
- Branching narrative generation
- Anything where you want more "alive" and uncertain output
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from qxgemma.logits_processor import create_qx_processor

# === Config ===
MODEL_ID = "google/gemma-2-9b-it"   # or gemma-2-2b-it, or any HF Gemma checkpoint
# For MoE experiments later: look for google/gemma-*-moe* variants when available
MAX_NEW_TOKENS = 220
CREATIVITY = 0.72                     # 0.4 = focused, 0.8+ = wild remix territory

print("Loading Gemma + QxBin processor...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)

# Create the QxBin-powered processor
processor = create_qx_processor(
    num_cubits=20,
    grid_size=7,
    evolution_steps=4,
    creativity_bias=CREATIVITY,
)

print("Model and Qx processor ready.\n")

# === Prompt — change this to whatever you want to remix ===
prompt = """You are a diabolically optimistic AI remixer working on the frontier of classical and quantum-inspired intelligence.

Remix the following seed idea into something surprising, useful, and a little bit wild. 
Think in probability fields and collapsing possibilities. 
Seed idea: Building AI edge nodes inside EV charging stations that also run local quantum-inspired simulations for micro-entrepreneurs.

Output format: 
1. Catchy title
2. 3-4 paragraph description
3. One unexpected twist or second-order effect
4. Suggested next experiment
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating with QxBin-evolved sampling...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        logits_processor=[processor],
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.eos_token_id,
    )

generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)

print("\n" + "="*60)
print("Generation complete. The probability chains have collapsed into this remix.")
print("Tweak creativity_bias, evolution_steps, or the prompt and run again.")
print("Next: try the probabilistic_router.py on an MoE Gemma or integrate into your own pipeline.")

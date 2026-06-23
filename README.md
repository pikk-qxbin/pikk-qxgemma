# QxGemma

**Quantum-inspired remix of Google's Gemma models using QxBin probability matrices.**

QxGemma brings the power of **Binary Probability Matrices**, fractional-state superposition, and chain evolution from the QxBin framework directly into Gemma inference and routing. 

The result? A remixed Gemma that generates with richer uncertainty modeling, more creative and coherent long-form output, native support for mixed/probabilistic states, and a foundation for truly hybrid classical-quantum-inspired AI — all running on classical hardware today.

Built by Rupesh Malpani | pikk.company

## Why This Matters

Gemma is an outstanding open model family (strong reasoning, efficient, great instruction following).  
QxBin gives us **evolving probability fields** instead of flat distributions or hard binary decisions.

Together they create something new:
- Token sampling that evolves like a quantum chain (superposition → measurement collapse)
- Probabilistic routing that supports true mixed expert states (perfect for MoE Gemma variants)
- Built-in uncertainty calibration and creative branching
- Edge-friendly (lightweight matrix ops + future CUDA path)
- A living prototype for the next generation of expressive, robust LLMs

This is the first public drop of a **QxBin-enhanced Gemma**. v0.1 focuses on inference-time magic you can use immediately.

## Quickstart

```bash
# 1. Clone this repo + your QxBin framework (recommended side-by-side)
git clone https://github.com/pikk-qxbin/qxbin.git
git clone https://github.com/pikk-qxbin/pikk-qxgemma.git

cd pikk-qxgemma

# 2. Install dependencies (QxBin first if using editable)
pip install -e ../qxbin
pip install -r requirements.txt

# 3. Run the remix example
python examples/remix_with_qx.py
```

Then import and use:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from qxgemma import QxBinLogitsProcessor
from qxbin.cloud import QxBinCloud

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b-it", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

qx_cloud = QxBinCloud(num_cubits=16, grid_size=7)

processor = QxBinLogitsProcessor(qx_cloud=qx_cloud, evolution_steps=3, creativity_bias=0.7)

inputs = tokenizer("Remix this idea into something wild and useful: ", return_tensors="pt").to(model.device)
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    logits_processor=[processor],
    do_sample=True,
    temperature=0.8
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## What’s Inside (v0.1)

- `qxgemma/logits_processor.py` — `QxBinLogitsProcessor`: Evolves the probability landscape using QxBin chain logic before each sampling step. Supports dynamic creativity/uncertainty modulation.
- `qxgemma/probabilistic_router.py` — Early prototype for replacing hard MoE routing with evolving probability matrices (superposition-aware expert activation). Ready for experimentation on Gemma MoE variants.
- `examples/remix_with_qx.py` — Ready-to-run creative generation demo (podcast scripts, product ideas, branching narratives).
- Full documentation and extension points for fine-tuning integration, custom evolution rules, and future hybrid layers.

## Vision & Roadmap

**Today (v0.1)**: Inference-time QxBin sampling + router prototype. Ship and start remixing.

**Next**:
- Full probabilistic MoE router integrated into generation loop
- QxBin-augmented fine-tuning (LoRA + probabilistic regularization / synthetic data)
- Dynamic per-layer or per-token cubit chains for deeper hybridization
- CUDA-accelerated evolution (leveraging QxBin’s existing port)
- Physical QxBin sensor input for “analog creativity” knobs (future hardware tie-in)
- Pre-built QxGemma adapters/checkpoints on Hugging Face

Long-term: QxGemma becomes the reference implementation for **probability-matrix-native LLMs** — models that don’t just predict the next token, but evolve possibility fields and collapse them into coherent, high-agency output.

This direction compounds. Better creativity + calibrated uncertainty + edge performance = models that feel more *alive* and useful for real creative work, agents, and phygital systems.

## How to Contribute / Extend

- Fork and experiment with the `QxBinLogitsProcessor` evolution parameters.
- Improve the router prototype and test on actual MoE Gemma checkpoints.
- Add new evolution strategies (different fractional exponents, multi-scale chains, MoodBin-style mixed states).
- Share wild remix outputs (especially anything tied to AI Remix podcast themes or Pikk ecosystem ideas).

Pull requests and wild experiments welcome. This is early-stage frontier work.

## License

Custom MIT-style license (same spirit as QxBin). Free for research, personal use, and internal experimentation. Commercial use / API products require discussion (51% revenue share model for derivative tools, negotiable for enterprise).

See `LICENSE` for details. Contact @rupeshmalpani on X for partnerships.

## Acknowledgments

- Google DeepMind / Gemma team for the excellent open models.
- The broader open-source LLM community (transformers, peft, vLLM, etc.).
- QxBin framework and all the fractional-state / chain-evolution math that makes this possible.
- Grok for early mathematical help on the QxBin proofs.

Built with diabolical optimism. Let’s make probability fields the new normal.

---

**pikk.company** | QxBin + Gemma = QxGemma  
Run the chains. Collapse the future you want. ✨

*Status: v0.1 scaffold — ready to github, iterate, and ship real remixes today.*
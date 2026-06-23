# QxGemma Integration Guide (v0.1)

## 1. Basic Inference Remix (Recommended starting point)

Use `QxBinLogitsProcessor` exactly like any other Hugging Face logits processor.

See `examples/remix_with_qx.py` for the full working script.

Key tunable parameters:
- `creativity_bias` (0.4–0.85): Controls how "exploratory" the fractional evolution becomes.
- `evolution_steps` (2–6): More steps = deeper probability field evolution per token (slower but richer).
- `uncertainty_weight`: How much the evolved matrix std/mean affects dynamic temperature.

## 2. Using the Probabilistic Router (MoE experiments)

```python
from qxgemma.probabilistic_router import QxBinProbabilisticRouter
import torch

router = QxBinProbabilisticRouter(num_experts=8, num_cubits=12, evolution_steps=2)

# Inside your model forward or with hooks
hidden_states = torch.randn(2, 128, 4096)  # example
expert_weights = router.route(hidden_states, layer_idx=3)

print(expert_weights.shape)  # [2, 128, 8]
# Use expert_weights instead of your original router logits
```

This is still early — test on actual MoE checkpoints and report what you observe (especially coherence vs creativity tradeoffs).

## 3. Persistent Evolution Across a Generation

By default the processor resets some state each step for stability. 

If you want the probability chains to carry "memory" across the entire generation (more coherent long-form remixes), comment out the `self.qx.states = original_states` line in `logits_processor.py` and experiment.

Persistent chains tend to produce more "thematic" or stylistically consistent output.

## 4. Fine-tuning + QxBin (Future direction)

Ideas to explore:
- Use QxBin's `optimize_to_target` loop as a hyperparameter or synthetic data generator during QLoRA.
- Add a small auxiliary loss that encourages the model to predict not just next token but also a "uncertainty signature" derived from QxBin grids.
- Generate remix-style training data with high-creativity QxGemma runs, then fine-tune a new adapter on it.

## 5. Performance Notes

- v0.1 evolution is CPU/Numba. For production use the CUDA port from QxBin when mature.
- `top_k_for_matrix=50` keeps it fast. Increase for richer distributions at cost of speed.
- Works great with `torch.compile` and vLLM-style engines (wrap the processor carefully).

## 6. Extending the Idea

The real power is in **persistent multi-scale chains**:
- One cubit chain per layer or per semantic cluster
- Different evolution rules for "creativity cubits" vs "coherence cubits"
- Feeding real sensor data (future physical QxBin) into the bias parameters

This repo is the seed. Fork it, break it, make it weirder and better.

Questions / wild experiments → @rupeshmalpani on X.

Run the chains.
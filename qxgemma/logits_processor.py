"""
QxBinLogitsProcessor for Gemma remixes.

Evolves the token probability landscape using QxBin Binary Probability Matrices
and chain evolution before each sampling step.

This turns standard next-token prediction into a quantum-inspired process:
superposition blend → fractional state evolution → probabilistic collapse.
"""

from __future__ import annotations
import torch
import numpy as np
from transformers import LogitsProcessor
from typing import Optional

try:
    from qxbin.cloud import QxBinCloud
except ImportError:
    QxBinCloud = None
    print("Warning: qxbin.cloud.QxBinCloud not found. Using fallback evolution.")


class QxBinLogitsProcessor(LogitsProcessor):
    """
    Custom LogitsProcessor that injects QxBin probabilistic chain evolution
    into Gemma (or any HF CausalLM) generation.

    At each generation step:
    1. Convert logits -> probabilities
    2. Feed (a representation of) the distribution into a QxBin probability matrix / chain
    3. Evolve the chain using fractional exponent blending (superposition-like)
    4. Use the evolved state to modulate scores / creativity / uncertainty
    5. Return adjusted scores for sampling

    Args:
        qx_cloud: Initialized QxBinCloud instance (recommended: num_cubits=8-32, grid_size=5-9)
        evolution_steps: How many chain evolution steps to run per token (default 2-5)
        creativity_bias: 0.0 = conservative, 1.0 = wild exploration (affects bias in evolve)
        uncertainty_weight: How strongly the evolved matrix amplitude influences temperature-like behavior
        top_k_for_matrix: Only consider top-k tokens when building the probability matrix (efficiency)
        device: torch device
    """

    def __init__(
        self,
        qx_cloud: Optional["QxBinCloud"] = None,
        evolution_steps: int = 3,
        creativity_bias: float = 0.65,
        uncertainty_weight: float = 0.35,
        top_k_for_matrix: int = 50,
        device: str = "cpu",
    ):
        if qx_cloud is None:
            if QxBinCloud is None:
                raise ImportError("QxBinCloud not available. Please install pikk-qxbin or provide a qx_cloud instance.")
            # Sensible default for remix work
            qx_cloud = QxBinCloud(num_cubits=16, grid_size=7)
        self.qx = qx_cloud
        self.evolution_steps = evolution_steps
        self.creativity_bias = creativity_bias
        self.uncertainty_weight = uncertainty_weight
        self.top_k_for_matrix = top_k_for_matrix
        self.device = device
        self.step_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Called by generate() at every decoding step.
        scores: [batch_size, vocab_size]
        """
        self.step_count += 1
        batch_size, vocab_size = scores.shape

        # Work on first item in batch for simplicity in v0.1 (extend to per-batch later)
        scores_0 = scores[0]

        # Convert to probabilities
        probs = torch.softmax(scores_0 / max(0.1, 1.0 - self.uncertainty_weight * 0.5), dim=-1)

        # Get top-k for efficiency (the "interesting" part of the distribution)
        top_k = min(self.top_k_for_matrix, vocab_size)
        top_probs, top_indices = torch.topk(probs, top_k)

        # === Build a simple probability representation for QxBin ===
        # For v0.1 we map the top-k probability mass into the QxBin grid
        # In future versions: multi-cubit per semantic cluster or per-layer chains
        grid_size = getattr(self.qx, "grid_size", 7)
        prob_grid = np.zeros((grid_size, grid_size), dtype=np.float64)

        # Distribute top-k probability mass across the grid (simple raster fill for demo)
        flat_idx = 0
        for i in range(grid_size):
            for j in range(grid_size):
                if flat_idx < len(top_probs):
                    prob_grid[i, j] = float(top_probs[flat_idx].cpu())
                    flat_idx += 1
                else:
                    prob_grid[i, j] = 1e-8

        # Normalize grid
        total = prob_grid.sum()
        if total > 1e-12:
            prob_grid /= total

        # === Evolve the probability matrix using QxBin logic ===
        # We treat the current grid as one "state" in the ensemble
        # In production you'd maintain persistent chains across steps or use multiple cubits
        try:
            # Use the cloud evolve if available
            if hasattr(self.qx, "states"):
                # Temporarily inject our grid as one of the states for this step
                original_states = self.qx.states.copy()
                self.qx.states[0] = prob_grid  # evolve at least the first cubit with our distribution

                for _ in range(self.evolution_steps):
                    # Use creativity_bias to influence the bias parameter in evolve
                    biases = np.full(self.qx.num_cubits, self.creativity_bias, dtype=np.float64)
                    self.qx.evolve_chains(biases=biases)

                evolved_grid = self.qx.states[0]
                # Restore (or keep evolved state for persistence across tokens — your choice)
                self.qx.states = original_states  # comment out if you want persistent evolution
            else:
                # Fallback simple fractional evolution (pure numpy)
                evolved_grid = self._fallback_evolve(prob_grid, steps=self.evolution_steps, bias=self.creativity_bias)

        except Exception as e:
            print(f"QxBin evolution failed at step {self.step_count}: {e}. Using fallback.")
            evolved_grid = self._fallback_evolve(prob_grid, steps=self.evolution_steps, bias=self.creativity_bias)

        # === Turn evolved grid back into a modulation signal ===
        evolved_mean = float(np.mean(evolved_grid))
        evolved_std = float(np.std(evolved_grid))

        # Use evolved statistics to dynamically adjust the scores
        # Higher evolved_mean + creativity → slightly flatter (more exploration)
        # Higher evolved_std → more confident peaks or more uncertainty depending on sign
        dynamic_temp = max(0.3, 0.8 + (evolved_mean - 0.5) * self.uncertainty_weight * 1.2)
        dynamic_temp = min(dynamic_temp, 1.8)

        # Re-apply softmax with dynamic temperature and slight boost to top evolved mass
        adjusted_scores = scores_0 / dynamic_temp

        # Optional: boost tokens that had high mass in the evolved grid (simple feedback)
        # This creates a soft "memory" of the probability field evolution
        boost = torch.zeros_like(scores_0)
        for idx_pos, tok_idx in enumerate(top_indices[:min(20, len(top_indices))]):
            # crude mapping: higher evolved grid values → small positive bias on those tokens
            grid_val = evolved_grid.flatten()[min(idx_pos, evolved_grid.size - 1)]
            boost[tok_idx] = float(grid_val) * self.uncertainty_weight * 3.0

        adjusted_scores = adjusted_scores + boost.to(scores.device)

        # Write back to batch
        scores[0] = adjusted_scores

        # For multi-batch: in v0.1 we only processed [0]. Extend by looping or vectorizing.
        if batch_size > 1:
            for b in range(1, batch_size):
                scores[b] = scores[0]  # simple broadcast for demo

        return scores

    def _fallback_evolve(self, grid: np.ndarray, steps: int = 3, bias: float = 0.65) -> np.ndarray:
        """Pure numpy fallback if full QxBinCloud is not available."""
        current = grid.copy().astype(np.float64)
        for _ in range(steps):
            n = np.random.randint(1, 4)
            m = np.random.randint(1, 4)
            frac = bias ** n
            tail = (1.0 - bias) ** m
            blended = (current * frac + (1.0 - current) * tail) * 0.5
            total = blended.sum()
            if total > 1e-12:
                current = blended / total
            else:
                current = np.ones_like(blended) / blended.size
        return current


# Convenience factory
def create_qx_processor(
    num_cubits: int = 16,
    grid_size: int = 7,
    evolution_steps: int = 3,
    creativity_bias: float = 0.65,
    **kwargs
) -> QxBinLogitsProcessor:
    """Quick way to get a ready-to-use processor."""
    if QxBinCloud is None:
        qx = None
    else:
        qx = QxBinCloud(num_cubits=num_cubits, grid_size=grid_size)
    return QxBinLogitsProcessor(
        qx_cloud=qx,
        evolution_steps=evolution_steps,
        creativity_bias=creativity_bias,
        **kwargs
    )

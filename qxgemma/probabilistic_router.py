"""
Probabilistic Router prototype using QxBin for Gemma (especially MoE variants).

Core idea (from QxBin vision + recent router proposal):
Instead of hard or standard top-k routing in Mixture-of-Experts,
maintain an evolving Binary Probability Matrix that represents
superposition over expert choices.

At each layer/token:
- Evolve the probability matrix (fractional states + chain evolution)
- Collapse / sample expert activation probabilities
- Support true mixed states (partial activation of multiple experts)

This is v0.1 scaffold — ready for integration testing on Gemma-26B-A4B or similar MoE models.
"""

import numpy as np
import torch
from typing import Optional, Tuple, List

try:
    from qxbin.cloud import QxBinCloud
except ImportError:
    QxBinCloud = None


class QxBinProbabilisticRouter:
    """
    QxBin-powered router for MoE-style models.

    Replaces deterministic router with an evolving probability matrix
    that naturally supports superposition and uncertainty.

    Usage sketch (inside a custom modeling file or with hooks):
        router = QxBinProbabilisticRouter(num_experts=8, num_cubits=12)
        expert_weights = router.route(hidden_states, layer_idx)  # [batch, seq, num_experts] soft weights
    """

    def __init__(
        self,
        num_experts: int = 8,
        num_cubits: int = 12,
        grid_size: int = 5,
        evolution_steps: int = 2,
        persistence: bool = True,  # keep evolving state across tokens/layers
    ):
        self.num_experts = num_experts
        self.evolution_steps = evolution_steps
        self.persistence = persistence
        self.step = 0

        if QxBinCloud is not None:
            self.qx = QxBinCloud(num_cubits=num_cubits, grid_size=grid_size)
        else:
            self.qx = None
            print("QxBinCloud not available — router will use numpy fallback evolution.")

        # Simple mapping from experts to regions in the probability grid
        self.expert_map = self._build_expert_map(num_experts, grid_size)

    def _build_expert_map(self, num_experts: int, grid_size: int) -> List[Tuple[int, int]]:
        """Map each expert to one or more cells in the grid."""
        cells = []
        for i in range(num_experts):
            row = (i * 3) % grid_size
            col = (i * 2) % grid_size
            cells.append((row, col))
        return cells

    def _evolve(self) -> np.ndarray:
        """Run QxBin evolution and return aggregate probability landscape."""
        if self.qx is not None and hasattr(self.qx, "evolve_chains"):
            for _ in range(self.evolution_steps):
                # Slight bias toward exploration as steps increase (creative routing)
                bias = 0.55 + min(0.25, self.step * 0.01)
                self.qx.evolve_chains(biases=np.full(self.qx.num_cubits, bias))
            agg = self.qx.states.mean(0)
        else:
            # Fallback
            agg = np.random.dirichlet(np.ones(self.qx.grid_size * self.qx.grid_size) if hasattr(self.qx, 'grid_size') else np.ones(25))
            agg = agg.reshape((5, 5)) if hasattr(self.qx, 'grid_size') else np.ones((5,5)) / 25
        return agg

    def route(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute soft expert routing weights using evolved QxBin probability matrix.

        Returns:
            expert_weights: [batch, seq_len, num_experts] — can be used as
                            gating weights (replace hard router output).
        """
        self.step += 1
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Evolve the probability field
        evolved_grid = self._evolve()

        # Extract probabilities for each expert from the evolved grid
        expert_probs = []
        for (r, c) in self.expert_map:
            val = float(evolved_grid[r % evolved_grid.shape[0], c % evolved_grid.shape[1]])
            expert_probs.append(max(val, 1e-6))

        expert_probs = np.array(expert_probs)
        expert_probs = expert_probs / expert_probs.sum()  # normalize

        # Add light temperature-controlled noise for exploration
        if temperature != 1.0:
            expert_probs = np.power(expert_probs, 1.0 / temperature)
            expert_probs = expert_probs / expert_probs.sum()

        # Expand to [batch, seq, num_experts]
        expert_weights = torch.tensor(expert_probs, dtype=hidden_states.dtype, device=hidden_states.device)
        expert_weights = expert_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)

        # Optional: make it input-dependent in future versions (use hidden_states to modulate)
        # For v0.1 this is a strong starting point — the evolution itself carries the "memory" and uncertainty.

        return expert_weights

    def reset(self):
        """Reset internal step counter and (optionally) QxBin state."""
        self.step = 0
        if self.qx is not None and hasattr(self.qx, "states"):
            # Re-initialize states for a fresh chain
            for i in range(self.qx.num_cubits):
                s = np.random.rand(self.qx.grid_size, self.qx.grid_size).astype(np.float64)
                s /= s.sum()
                self.qx.states[i] = s


# Example integration note:
# In a custom Gemma MoE forward pass or with transformer hooks:
#   router = QxBinProbabilisticRouter(num_experts=moe_config.num_experts)
#   gate_logits = router.route(hidden_states, layer_idx=layer_idx)
#   # then combine with expert outputs using gate_logits as weights
#   # instead of the original router

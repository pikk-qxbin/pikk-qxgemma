"""
QxGemma — Quantum-inspired Gemma remixes powered by QxBin.

Public API (v0.1):
- QxBinLogitsProcessor: Main inference-time processor
- create_qx_processor: Convenience factory
- QxBinProbabilisticRouter: Early MoE router prototype
"""

from .logits_processor import QxBinLogitsProcessor, create_qx_processor
from .probabilistic_router import QxBinProbabilisticRouter

__version__ = "0.1.0"
__all__ = [
    "QxBinLogitsProcessor",
    "create_qx_processor",
    "QxBinProbabilisticRouter",
]

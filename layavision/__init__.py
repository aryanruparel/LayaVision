"""
LayaVision: Fast System 1 Multimodal Vision Decision Engine
===========================================================
Built on top of Laya for ultra-fast, calibrated object classification,
visual attribute reasoning, and non-autoregressive vision decisions.
"""

from .model import LayaVisionModel
from .classifier import LayaVision, VisionClassifier

__version__ = "0.1.0"
__all__ = ["LayaVision", "VisionClassifier", "LayaVisionModel"]

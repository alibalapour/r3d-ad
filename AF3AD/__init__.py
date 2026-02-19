"""Pseudo Anomaly Synthesis Tool
================================

A standalone toolkit for generating pseudo anomalies on 3-D point clouds using
hand-crafted presets.  Extracted from the **A3AD** (Adversarial 3-D Anomaly
Detection) training pipeline.

Quick start::

    from pseudo_anomaly_synthesis import PseudoAnomalySynthesizer

    synth = PseudoAnomalySynthesizer()
    cfg   = synth.preset_factory.presets[0]()   # Type 1 – basic bulge
    deformed = synth.generate(points, normals, center, cfg)
"""

from .config import SmartAnomaly_Cfg
from .presets import AnomalyPreset
from .synthesizer import PseudoAnomalySynthesizer

__all__ = [
    "SmartAnomaly_Cfg",
    "AnomalyPreset",
    "PseudoAnomalySynthesizer",
]
"""p9kit: search for distant solar system perturbers, honestly.

Two halves that feed each other:

* **Orbits** - a fast forward model of distant-object orbits under a hypothetical planet,
  plus neural posterior estimation, giving a probability map of where the planet could be.
* **Images** - shift-and-stack over a grid of sky motions, a CNN that rejects artefacts,
  and the measurements that make a non-detection meaningful: a recovery curve, a junk
  comparison and a split-half test for every surviving candidate.

The forward model is phenomenological, not N-body; see ``LIMITATIONS.md`` before
quoting any number from it.
"""
from .images import (ImageSet, blind_search, build_mask, depth_50, inject, matched_filter,
                     patch_quality, recovery_curve, remove_static_sky, simulate_frames,
                     split_half_test, stack)
from .population import Planet, SurveyModel, simulate_detected
from .stats import clustering, luck_test, summary_statistics

__all__ = [
    "Planet", "SurveyModel", "simulate_detected",
    "clustering", "luck_test", "summary_statistics",
    "ImageSet", "simulate_frames", "remove_static_sky", "build_mask", "stack", "matched_filter",
    "blind_search", "inject", "recovery_curve", "depth_50", "split_half_test", "patch_quality",
]
__version__ = "0.1.0"

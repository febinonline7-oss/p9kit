---
title: "p9kit: honest searches for distant solar system perturbers"
tags:
  - Python
  - astronomy
  - trans-Neptunian objects
  - Planet Nine
  - simulation-based inference
  - shift-and-stack
authors:
  - name: Febin S
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent researcher, Thiruvananthapuram, India
    index: 1
date: 4 October 2026
bibliography: paper.bib
---

# Summary

`p9kit` is a Python package for two linked tasks in the search for distant solar system
perturbers. The first turns the orbits of known trans-Neptunian objects into a probability map
for a hypothetical planet, using a fast forward model and neural posterior estimation. The
second searches survey images for slow-moving objects by shift-and-stack, and — more
importantly — measures what that search could and could not have seen.

The package grew out of a complete small search: orbits of 32 distant objects, a measured
clustering significance, a posterior over planet parameters, a sky probability map, and
searches of two Zwicky Transient Facility fields to a measured depth of magnitude 21.6.

# Statement of need

Shift-and-stack searches for faint moving objects are easy to start and hard to interpret.
Three failure modes dominate, and `p9kit` exists because each of them was hit, diagnosed and
fixed during a real search:

1. **A non-detection without a measured depth carries no information.** `recovery_curve` injects
   sources of known brightness into the user's own frames and reports the fraction recovered.
2. **A blind search over thousands of trial motions produces hundreds of "5 sigma" candidates.**
   `p9kit` re-runs the identical search on brightness-flipped frames, so the junk level is
   measured rather than assumed, and `split_half_test` re-measures every survivor in each half
   of the data: an artefact confined to a few frames disappears, a real object does not.
3. **Undersampled bright stars dominate everything.** In Zwicky Transient Facility images a single
   bright star left residuals of 319 sigma, swamping a field whose true noise limit was below
   5 sigma. `build_mask` masks stars with a radius that grows with brightness, and
   `patch_quality` reports when a field is too crowded to search at all.

On the inference side, simulation-based inference [@cranmer2020] makes it practical to fit a
perturber model that has no tractable likelihood, but a neural posterior can be confidently
wrong. `p9kit` ships `coverage_check`, which simulates many data sets and counts how often the
truth lands inside its own credible interval; in this package's own tests a masked autoregressive
flow failed that check on samples of a few dozen objects while a spline flow passed, and the
spline flow is therefore the default.

Existing tools cover parts of this ground: `sbi` [@tejero2020sbi] provides the inference
machinery, `REBOUND` [@rein2012] provides N-body integration, and dedicated survey pipelines such
as `KBMOD` [@whidden2019] implement high-performance shift-and-stack for large data sets. `p9kit`
is aimed at the small end: a single researcher, a laptop or a free notebook service, a few dozen
orbits and a handful of image patches — with the verification steps built in rather than left as
an exercise. Its forward model is phenomenological rather than N-body, which is documented
prominently, and the inference interface accepts any simulator, so an N-body model can be
substituted without touching the rest of the pipeline.

# Functionality

- `p9kit.population` — distant-object populations under a candidate planet, plus a simple survey
  selection model.
- `p9kit.stats` — clustering statistics and a selection-aware significance test.
- `p9kit.inference` — neural posterior estimation with a built-in coverage check.
- `p9kit.sky` — positions, magnitudes and sky motion rates predicted from a posterior, and
  re-weighting of that posterior after a non-detection.
- `p9kit.images` — shift-and-stack, star masking, recovery curves, patch quality, split-half
  verification.
- `p9kit.classifier` — a small CNN for artefact rejection, trained on stamps produced by the
  pipeline itself.
- `p9kit.ztf` — retrieval of public Zwicky Transient Facility cutouts from IRSA.
- A command line interface: `p9kit demo`, `p9kit clustering`, `p9kit search`.

# Acknowledgements

This work uses public data from the Zwicky Transient Facility, obtained through IRSA, and orbital
elements from the JPL Small-Body Database.

# References

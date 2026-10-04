# p9kit

[![tests](https://github.com/USERNAME/p9kit/actions/workflows/tests.yml/badge.svg)](https://github.com/USERNAME/p9kit/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Tools for searching for distant solar system perturbers — and for finding out whether your search means anything.**

p9kit has two halves that feed each other:

* **Orbits.** A fast forward model of distant trans-Neptunian orbits under a hypothetical
  planet, plus neural posterior estimation (`sbi`), giving a probability map of where such a
  planet could be, how faint it would look and how fast it would move.
* **Images.** Shift-and-stack over a grid of sky motions, a small CNN that rejects artefacts,
  and the three measurements that decide whether a non-detection carries information: a
  **recovery curve**, a **junk comparison**, and a **split-half test** for every survivor.

The second half is the part most small searches get wrong. A blind search over thousands of
trial motions throws up hundreds of "5 sigma" candidates, and a bright star in an undersampled
image leaves scars of several hundred sigma. p9kit measures those effects instead of assuming
them away.

## Install

```bash
pip install p9kit                 # core: orbits, images, statistics
pip install "p9kit[all]"          # plus inference (sbi), the CNN (torch) and ZTF downloads
```

Python 3.9 or newer. Everything runs on a laptop or in Google Colab; no GPU is needed.

## Quickstart

Measure how clustered a set of orbits is, and how often chance does the same:

```python
import numpy as np
from p9kit import Planet, simulate_detected, clustering, luck_test

rng = np.random.default_rng(0)
sample = simulate_detected(Planet(mass_earth=6.0, a9_au=500.0), n_detected=40, rng=rng)
direction, strength = clustering(sample)
test = luck_test(sample, rng, n_null=500)
print(f"R = {strength:.2f} towards {direction:.0f} deg; chance: {test.one_in()} ({test.verdict()})")
```

Search images for something that moves, and measure how faint the search could see:

```python
from p9kit import simulate_frames, build_mask, blind_search, recovery_curve, depth_50, split_half_test

images, _ = simulate_frames(movers=[dict(mag=21.0, rate=3.2, angle=250.0, x0=150.0, y0=120.0)])
mask = build_mask(images)                       # bright stars, with a radius that grows with brightness
candidates = blind_search(images, mask=mask)    # every trial speed and direction
mags, efficiency = recovery_curve(images, np.arange(20.0, 23.1, 0.5), rng, mask=mask)
print("50% depth:", depth_50(mags, efficiency))
print(split_half_test(images, candidates[0]))   # a real object survives both halves
```

From the command line:

```bash
p9kit demo                                   # end-to-end check on simulated images
p9kit clustering my_objects.csv --q-min 42   # clustering and significance for a real table
p9kit search 34.5 -2.5                       # download a public ZTF patch and search it
```

## What makes a null result meaningful

| Check | Question it answers | Function |
|---|---|---|
| Recovery curve | How faint can this search actually see? | `recovery_curve`, `depth_50` |
| Junk comparison | How many candidates does pure noise give? | `blind_search` on `-frames` |
| Split-half test | Does the candidate survive in both halves of the data? | `split_half_test` |
| Patch quality | Is this field too crowded to search at all? | `build_mask`, `patch_quality` |
| Posterior coverage | Is the inferred posterior honest, or overconfident? | `inference.coverage_check` |

## Honest limitations

The forward model is **phenomenological, not N-body**, and the survey model is deliberately
simple. They are fast enough for simulation-based inference on a laptop, which is the point,
but numbers that come out of them are exploratory. Read [`LIMITATIONS.md`](LIMITATIONS.md)
before quoting any of them. The imaging side has no such caveat: it measures its own depth from
the data you give it.

## Documentation

* [`LIMITATIONS.md`](LIMITATIONS.md) — what this package cannot be trusted to do
* [`METHOD.md`](METHOD.md) — how each piece works and why
* [`NOTEBOOKS.md`](NOTEBOOKS.md) — two Colab notebooks, in the repository root

## Contributing

Issues and pull requests are welcome; see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

See [`CITATION.cff`](CITATION.cff).

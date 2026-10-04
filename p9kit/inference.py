"""Simulation-based inference: learn p(planet | observed clustering) from simulations.

Uses neural posterior estimation (``sbi``). Importing this module does not require
``sbi``/``torch``; the dependency is only needed when a posterior is actually trained.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .population import PARAM_HIGH, PARAM_LOW, PARAM_NAMES, Planet, SurveyModel, simulate_detected
from .stats import summary_statistics


def _require_sbi():
    try:
        import torch
        from sbi.inference import NPE
        from sbi.utils import BoxUniform
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError("training needs the inference extra: pip install 'p9kit[infer]'") from exc
    return torch, NPE, BoxUniform


def simulate_training_set(n_sims, n_detected, rng, survey: SurveyModel | None = None,
                          low=PARAM_LOW, high=PARAM_HIGH, progress=None):
    """Draw planets from the prior and summarise one simulated survey for each."""
    theta = low + (high - low) * rng.random((n_sims, len(low)))
    x = np.zeros((n_sims, len(summary_statistics(simulate_detected(Planet(), 8, rng, survey)))),
                 dtype=np.float32)
    for k, th in enumerate(theta):
        x[k] = summary_statistics(simulate_detected(Planet.from_array(th), n_detected, rng, survey))
        if progress and (k + 1) % progress == 0:
            print(f"  simulated {k + 1}/{n_sims}", flush=True)
    return theta, x


@dataclass
class Posterior:
    """A trained posterior, plus the prior bounds it was trained on."""

    estimator: object
    low: np.ndarray
    high: np.ndarray

    def sample(self, x_observed, n: int = 10000):
        import torch
        s = self.estimator.sample((n,), x=torch.as_tensor(np.asarray(x_observed, dtype=np.float32)),
                                  show_progress_bars=False)
        return np.asarray(s)

    def summary(self, x_observed, n: int = 10000, percentiles=(16, 50, 84)):
        s = self.sample(x_observed, n)
        return {name: tuple(np.percentile(s[:, j], percentiles)) for j, name in enumerate(PARAM_NAMES)}

    def probability_no_planet(self, x_observed, threshold: float = 1.0, n: int = 10000):
        """Posterior probability that the planet is below ``threshold`` Earth masses.

        Compare it with the prior probability (``threshold / high[0]``): if the two are
        similar, the data did not decide anything.
        """
        return float((self.sample(x_observed, n)[:, 0] < threshold).mean())


def train_posterior(theta, x, low=PARAM_LOW, high=PARAM_HIGH, density_estimator: str = "nsf",
                    batch_size: int = 256, seed: int = 0):
    """Train a neural posterior. ``nsf`` (spline flow) passed coverage tests; ``maf`` did not."""
    torch, NPE, BoxUniform = _require_sbi()
    torch.manual_seed(seed)
    prior = BoxUniform(low=torch.as_tensor(np.asarray(low, dtype=np.float32)),
                       high=torch.as_tensor(np.asarray(high, dtype=np.float32)))
    inference = NPE(prior=prior, density_estimator=density_estimator, show_progress_bars=False)
    net = inference.append_simulations(torch.as_tensor(np.asarray(theta, dtype=np.float32)),
                                       torch.as_tensor(np.asarray(x, dtype=np.float32)))
    return Posterior(inference.build_posterior(net.train(training_batch_size=batch_size)),
                     np.asarray(low), np.asarray(high))


def coverage_check(posterior: Posterior, n_detected, rng, n_tests: int = 200, n_samples: int = 500,
                   interval=(16, 84), survey: SurveyModel | None = None, params=(0, 1, 3, 6)):
    """Fraction of simulated truths falling inside their own credible interval.

    For an honest posterior this is the width of the interval (0.68 by default); much
    lower means the posterior is overconfident and should not be quoted.
    """
    inside = np.zeros((n_tests, len(params)))
    for j in range(n_tests):
        truth = posterior.low + (posterior.high - posterior.low) * rng.random(len(posterior.low))
        x = summary_statistics(simulate_detected(Planet.from_array(truth), n_detected, rng, survey))
        draws = posterior.sample(x, n_samples)
        for c, k in enumerate(params):
            lo, hi = np.percentile(draws[:, k], interval)
            inside[j, c] = lo <= truth[k] <= hi
    return {PARAM_NAMES[k]: float(inside[:, c].mean()) for c, k in enumerate(params)}

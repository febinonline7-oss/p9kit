"""Clustering statistics and the bias-aware significance test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .orbits import circ_mean_R, longitude_of_perihelion, pole_vector
from .population import Planet, SurveyModel, simulate_detected

SUMMARY_NAMES = ("cos_varpi", "sin_varpi", "cos_node", "sin_node", "pole_x", "pole_y", "pole_z",
                 "pole_spread", "cos_varpi_far", "sin_varpi_far", "median_inc_over_40")


def summary_statistics(sample):
    """Reduce a set of orbits to a short, order-independent vector for inference.

    Vector components (rather than angles) are used so that the 0/360 degree wrap
    cannot create artificial discontinuities.
    """
    a = np.asarray(sample["a"], dtype=float)
    varpi = np.radians(longitude_of_perihelion(sample["node"], sample["argperi"]))
    node = np.radians(np.asarray(sample["node"], dtype=float))
    poles = pole_vector(sample["inc"], sample["node"])
    far = a > np.median(a)
    mean_pole = poles.mean(axis=0)
    return np.array([
        np.cos(varpi).mean(), np.sin(varpi).mean(),
        np.cos(node).mean(), np.sin(node).mean(),
        mean_pole[0], mean_pole[1], mean_pole[2], 1 - np.linalg.norm(mean_pole),
        np.cos(varpi[far]).mean(), np.sin(varpi[far]).mean(),
        np.median(sample["inc"]) / 40.0,
    ], dtype=np.float32)


def clustering(sample):
    """Mean direction and strength of perihelion clustering."""
    return circ_mean_R(longitude_of_perihelion(sample["node"], sample["argperi"]))


@dataclass
class LuckTest:
    """Result of asking how often a planet-free universe looks this clustered."""

    observed_R: float
    null_R: np.ndarray
    n_null: int

    @property
    def p_value(self):
        """Fraction of planet-free realisations at least as clustered (0 means < 1/n_null)."""
        return float((self.null_R >= self.observed_R).mean())

    def one_in(self):
        p = self.p_value
        return f"less than 1 in {self.n_null}" if p == 0 else f"about 1 in {max(1, round(1 / p))}"

    def verdict(self):
        p = self.p_value
        if p < 0.01:
            return "very unlikely to be chance"
        if p < 0.05:
            return "probably not chance"
        if p < 0.20:
            return "might be chance"
        return "easily explained by chance"


def luck_test(sample, rng, n_null: int = 2000, survey: SurveyModel | None = None,
              isotropic: bool = False):
    """Compare the observed clustering with planet-free universes.

    With ``isotropic=True`` the null draws perihelia uniformly (no selection effects);
    otherwise it runs the forward model with ``mass_earth = 0``, so the null inherits
    whatever selection effects the survey model describes.
    """
    observed = clustering(sample)[1]
    n = len(np.asarray(sample["a"]))
    if isotropic:
        null = np.array([circ_mean_R(rng.uniform(0, 360, n))[1] for _ in range(n_null)])
    else:
        no_planet = Planet(mass_earth=0.0)
        null = np.array([clustering(simulate_detected(no_planet, n, rng, survey))[1]
                         for _ in range(n_null)])
    return LuckTest(float(observed), null, n_null)

"""Forward model: a population of distant objects, a perturbing planet, and a survey that sees some of them.

The planet's effect is **phenomenological**, not an N-body integration: it reproduces the
qualitative behaviour reported for Planet Nine (apsidal anti-alignment and a shared orbital
plane, both stronger for more massive or closer planets) at a cost of milliseconds per
realisation, which is what makes simulation-based inference practical on a laptop.
Results from this model are therefore exploratory. See ``LIMITATIONS.md``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .orbits import (D2R, pole_to_inc_node, pole_vector, sample_time_weighted_E, wrap360,
                     xyz_from_eccentric_anomaly)

#: Longitudes (deg, ecliptic) where the Milky Way crosses the ecliptic and surveys lose depth.
GALACTIC_CROSSINGS = np.array([86.0, 266.0])

PARAM_NAMES = ("mass_earth", "a9_au", "e9", "inc9_deg", "varpi9_deg", "node9_deg", "sigma_bg_deg")
PARAM_LOW = np.array([0.0, 300.0, 0.05, 0.0, 0.0, 0.0, 5.0])
PARAM_HIGH = np.array([20.0, 1000.0, 0.60, 40.0, 360.0, 360.0, 30.0])


@dataclass(frozen=True)
class Planet:
    """Parameters of the hypothetical perturber (``mass_earth = 0`` means no planet)."""

    mass_earth: float = 5.0
    a9_au: float = 500.0
    e9: float = 0.25
    inc9_deg: float = 18.0
    varpi9_deg: float = 250.0
    node9_deg: float = 100.0
    sigma_bg_deg: float = 20.0

    def as_array(self):
        return np.array([getattr(self, k) for k in PARAM_NAMES], dtype=float)

    @classmethod
    def from_array(cls, theta):
        return cls(**dict(zip(PARAM_NAMES, np.asarray(theta, dtype=float))))

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SurveyModel:
    """A deliberately simple model of how deep a survey sees, and where.

    Replace this with a published survey simulator before quoting a detection
    significance; it exists so that selection effects are *present* rather than correct.
    """

    limit_mag: float = 24.5
    galactic_penalty: float = 2.0
    off_ecliptic_penalty: float = 1.5
    off_ecliptic_deg: float = 25.0
    softness: float = 0.25

    def detection_probability(self, xyz, abs_mag):
        r = np.linalg.norm(xyz, axis=1)
        lam = wrap360(np.arctan2(xyz[:, 1], xyz[:, 0]) / D2R)
        beta = np.arcsin(xyz[:, 2] / r) / D2R
        d_gal = np.min(np.abs(((lam[:, None] - GALACTIC_CROSSINGS) + 180) % 360 - 180), axis=1)
        apparent = abs_mag + 10 * np.log10(r)
        limit = (self.limit_mag
                 - self.galactic_penalty * np.exp(-(d_gal / 20.0) ** 2)
                 - self.off_ecliptic_penalty * (np.abs(beta) > self.off_ecliptic_deg))
        return 1.0 / (1.0 + np.exp((apparent - limit) / self.softness))


def sample_population(n, rng, a_range=(250.0, 1500.0), q_range=(30.0, 70.0), h_range=(3.0, 8.0),
                      a_slope=1.5, h_slope=0.5):
    """Draw intrinsic orbits: a from a power law, perihelion uniform, H from a power law."""
    u1, u2, u3 = rng.random(n), rng.random(n), rng.random(n)
    lo, hi = a_range[0] ** -(a_slope - 1), a_range[1] ** -(a_slope - 1)
    a = (lo - u1 * (lo - hi)) ** (-1.0 / (a_slope - 1))
    q = q_range[0] + (q_range[1] - q_range[0]) * u2
    h0, h1 = 10 ** (h_slope * h_range[0]), 10 ** (h_slope * h_range[1])
    abs_mag = np.log10(h0 + u3 * (h1 - h0)) / h_slope
    return a, np.clip(1 - q / a, 0.0, 0.995), abs_mag


def forcing_strength(planet: Planet, a_obj):
    """Dimensionless measure of how hard the planet pushes an object of semi-major axis ``a_obj``."""
    b9 = planet.a9_au * np.sqrt(1 - planet.e9 ** 2)
    return (planet.mass_earth / 5.0) * (500.0 / b9) ** 3 * (np.asarray(a_obj) / 500.0) ** 1.5


def orbital_angles(planet: Planet, a_obj, rng):
    """Assign orbit orientations: confined objects follow the planet, the rest are random."""
    n = len(a_obj)
    F = forcing_strength(planet, a_obj)
    confined = rng.random(n) < (1 - np.exp(-2 * F))
    inc = np.minimum(np.abs(rng.normal(0.0, planet.sigma_bg_deg, n)), 80.0)
    node = rng.uniform(0, 360, n)
    varpi = rng.uniform(0, 360, n)
    if confined.any():
        k = int(confined.sum())
        Fk = F[confined]
        varpi[confined] = wrap360(planet.varpi9_deg + 180 + rng.normal(0, 1, k) * (20 + 40 / (1 + Fk)))
        h9 = pole_vector(planet.inc9_deg, planet.node9_deg)
        ref = np.array([0.0, 0.0, 1.0]) if abs(h9[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        e1 = np.cross(h9, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(h9, e1)
        delta = np.abs(rng.normal(0, 1, k)) * ((6 + 12 / (1 + Fk)) * D2R)
        phi = rng.uniform(0, 2 * np.pi, k)
        h = (np.cos(delta)[:, None] * h9
             + np.sin(delta)[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        inc[confined], node[confined] = pole_to_inc_node(h)
    return inc, node, wrap360(varpi - node), confined


def simulate_detected(planet, n_detected, rng, survey: SurveyModel | None = None,
                      pool: int = 20000, max_rounds: int = 10):
    """Simulate one survey's worth of *detected* distant objects.

    Returns a structured array with fields ``a, e, inc, node, argperi``. The result is
    conditioned on the number of detections so that it can be compared with a real sample.
    """
    if not isinstance(planet, Planet):
        planet = Planet.from_array(planet)
    survey = survey or SurveyModel()
    chunks, found = [], 0
    for _ in range(max_rounds):
        a, e, abs_mag = sample_population(pool, rng)
        E = sample_time_weighted_E(e, rng)
        bright_enough = abs_mag + 10 * np.log10(a * (1 - e * np.cos(E))) < survey.limit_mag + 1.0
        a, e, abs_mag, E = a[bright_enough], e[bright_enough], abs_mag[bright_enough], E[bright_enough]
        inc, node, argperi, _ = orbital_angles(planet, a, rng)
        xyz = xyz_from_eccentric_anomaly(a, e, inc, node, argperi, E)
        seen = rng.random(len(a)) < survey.detection_probability(xyz, abs_mag)
        chunks.append(np.stack([a[seen], e[seen], inc[seen], node[seen], argperi[seen]], axis=1))
        found += int(seen.sum())
        if found >= n_detected:
            break
    arr = np.concatenate(chunks)[:n_detected]
    out = np.zeros(len(arr), dtype=[(k, float) for k in ("a", "e", "inc", "node", "argperi")])
    for j, k in enumerate(("a", "e", "inc", "node", "argperi")):
        out[k] = arr[:, j]
    return out

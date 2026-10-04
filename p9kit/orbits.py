"""Orbit geometry: angles, Kepler's equation, orbit poles and circular statistics.

Everything here is pure numpy and has no optional dependencies, so it can be used
(and tested) without astropy, torch or sbi installed.
"""
from __future__ import annotations

import numpy as np

D2R = np.pi / 180.0
OBLIQUITY_DEG = 23.4392911


def wrap360(x):
    """Wrap an angle (degrees) into [0, 360)."""
    return np.mod(x, 360.0)


def circ_diff(a, b):
    """Smallest absolute difference between two angles, in degrees."""
    return np.abs(((np.asarray(a) - np.asarray(b)) + 180.0) % 360.0 - 180.0)


def circ_mean_R(angles_deg):
    """Circular mean direction (deg) and mean resultant length R in [0, 1].

    R = 0 means the angles are spread evenly; R = 1 means they are identical.
    """
    ang = np.asarray(angles_deg, dtype=float) * D2R
    c, s = np.cos(ang).mean(), np.sin(ang).mean()
    return wrap360(np.arctan2(s, c) / D2R), float(np.hypot(c, s))


def longitude_of_perihelion(node_deg, argperi_deg):
    """varpi = Omega + omega, wrapped to [0, 360)."""
    return wrap360(np.asarray(node_deg) + np.asarray(argperi_deg))


def pole_vector(inc_deg, node_deg):
    """Unit vector normal to the orbit plane, in ecliptic coordinates."""
    i, om = np.asarray(inc_deg) * D2R, np.asarray(node_deg) * D2R
    return np.stack([np.sin(i) * np.sin(om), -np.sin(i) * np.cos(om), np.cos(i)], axis=-1)


def pole_to_inc_node(h):
    """Inverse of :func:`pole_vector`: unit pole vector -> (inclination, node) in degrees."""
    h = np.asarray(h, dtype=float)
    h = h / np.linalg.norm(h, axis=-1, keepdims=True)
    inc = np.arccos(np.clip(h[..., 2], -1.0, 1.0)) / D2R
    node = wrap360(np.arctan2(h[..., 0], -h[..., 1]) / D2R)
    return inc, node


def solve_kepler(mean_anomaly_rad, ecc, n_iter: int = 40):
    """Solve M = E - e sin E by Newton iteration (vectorised, high eccentricity safe)."""
    M = np.mod(np.asarray(mean_anomaly_rad, dtype=float), 2 * np.pi)
    e = np.asarray(ecc, dtype=float)
    E = np.where(e < 0.8, M, np.full_like(M, np.pi))
    for _ in range(n_iter):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def xyz_from_eccentric_anomaly(a, e, inc_deg, node_deg, argperi_deg, E):
    """Heliocentric ecliptic position (AU) from elements and eccentric anomaly."""
    i, om, w = (np.asarray(inc_deg) * D2R, np.asarray(node_deg) * D2R, np.asarray(argperi_deg) * D2R)
    a, e, E = np.asarray(a, float), np.asarray(e, float), np.asarray(E, float)
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e ** 2) * np.sin(E)
    cw, sw, co, so, ci, si = np.cos(w), np.sin(w), np.cos(om), np.sin(om), np.cos(i), np.sin(i)
    P = np.stack([cw * co - sw * so * ci, cw * so + sw * co * ci, sw * si])
    Q = np.stack([-sw * co - cw * so * ci, -sw * so + cw * co * ci, cw * si])
    return (xp * P + yp * Q).T


def xyz_from_mean_anomaly(a, e, inc_deg, node_deg, argperi_deg, mean_anomaly_deg):
    """Heliocentric ecliptic position (AU) from orbital elements and mean anomaly (deg)."""
    E = solve_kepler(np.asarray(mean_anomaly_deg, float) * D2R, e)
    return xyz_from_eccentric_anomaly(a, e, inc_deg, node_deg, argperi_deg, E)


def sample_time_weighted_E(ecc, rng):
    """Eccentric anomaly for a uniformly random *time* (density proportional to 1 - e cos E).

    Rejection sampling, so no Kepler solve is needed: useful when simulating where
    objects happen to be when a survey looks at them.
    """
    e = np.asarray(ecc, dtype=float)
    E = rng.uniform(0, 2 * np.pi, len(e))
    bad = rng.random(len(e)) * (1 + e) > (1 - e * np.cos(E))
    while bad.any():
        E[bad] = rng.uniform(0, 2 * np.pi, int(bad.sum()))
        bad[bad] = rng.random(int(bad.sum())) * (1 + e[bad]) > (1 - e[bad] * np.cos(E[bad]))
    return E


def ecliptic_to_equatorial(xyz):
    """Rotate ecliptic cartesian coordinates to equatorial (ICRS-aligned)."""
    eps = OBLIQUITY_DEG * D2R
    x, y, z = np.asarray(xyz, dtype=float).T
    return np.stack([x, y * np.cos(eps) - z * np.sin(eps), y * np.sin(eps) + z * np.cos(eps)], axis=1)

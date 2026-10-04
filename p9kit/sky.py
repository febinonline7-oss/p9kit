"""Turn a posterior over planet orbits into predictions you can point a telescope at."""
from __future__ import annotations

import numpy as np

from .orbits import D2R, ecliptic_to_equatorial, wrap360, xyz_from_mean_anomaly

EARTH_RADIUS_KM = 6371.0


def radius_earth_from_mass(mass_earth):
    """Mass-radius relation for Neptune-like worlds (Chen & Kipping 2017)."""
    return 0.808 * np.asarray(mass_earth, dtype=float) ** 0.589


def absolute_magnitude(mass_earth, albedo):
    """Absolute magnitude H from mass (via radius) and geometric albedo."""
    diameter_km = 2 * radius_earth_from_mass(mass_earth) * EARTH_RADIUS_KM
    return 5 * np.log10(1329.0 / (diameter_km * np.sqrt(np.asarray(albedo, dtype=float))))


def _earth_heliocentric_equatorial(time):
    from astropy.coordinates import get_body_barycentric
    import astropy.units as u
    return (get_body_barycentric("earth", time) - get_body_barycentric("sun", time)).xyz.to(u.AU).value


def predict_sky(samples, epoch: str, rng, albedo_range=(0.2, 0.75), dt_days: float = 1.0,
                min_mass: float = 1.0):
    """Predict where each posterior sample would be on the sky, how faint, and how fast.

    ``samples`` is the array returned by :meth:`p9kit.inference.Posterior.sample`.
    Samples below ``min_mass`` are dropped: they are the "no planet" part of the posterior.
    The mean anomaly is unknown, so it is drawn uniformly in time, which naturally favours
    the slow, distant part of the orbit.
    """
    from astropy.coordinates import SkyCoord
    from astropy.time import Time
    import astropy.units as u

    s = np.asarray(samples, dtype=float)
    s = s[s[:, 0] >= min_mass]
    if len(s) == 0:
        raise ValueError("no posterior samples above min_mass")
    mass, a9, e9, inc9, varpi9, node9 = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    argperi9 = wrap360(varpi9 - node9)
    mean_anom = rng.uniform(0, 360, len(s))
    daily_motion = 360.0 / (a9 ** 1.5 * 365.25)

    t0 = Time(epoch)
    out = {}
    for tag, t, M in (("0", t0, mean_anom), ("1", t0 + dt_days * u.day, mean_anom + daily_motion * dt_days)):
        helio = ecliptic_to_equatorial(xyz_from_mean_anomaly(a9, e9, inc9, node9, argperi9, M))
        geo = helio - _earth_heliocentric_equatorial(t)
        dist = np.linalg.norm(geo, axis=1)
        out["ra" + tag] = wrap360(np.arctan2(geo[:, 1], geo[:, 0]) / D2R)
        out["dec" + tag] = np.arcsin(geo[:, 2] / dist) / D2R
        if tag == "0":
            out["r_sun_au"], out["delta_au"] = np.linalg.norm(helio, axis=1), dist

    albedo = rng.uniform(*albedo_range, len(s))
    v_mag = absolute_magnitude(mass, albedo) + 5 * np.log10(out["r_sun_au"] * out["delta_au"])
    c0 = SkyCoord(out["ra0"] * u.deg, out["dec0"] * u.deg)
    c1 = SkyCoord(out["ra1"] * u.deg, out["dec1"] * u.deg)
    return dict(ra=out["ra0"], dec=out["dec0"], r_sun_au=out["r_sun_au"], delta_au=out["delta_au"],
                v_mag=v_mag, rate_arcsec_day=c0.separation(c1).arcsec / dt_days,
                position_angle_deg=c0.position_angle(c1).deg, mass_earth=mass, a9_au=a9,
                albedo=albedo, weight=np.ones(len(s)))


def apply_nondetection(prediction, inside_patch, efficiency_of_mag):
    """Down-weight every possible planet that a search should have seen and did not.

    ``inside_patch`` is a boolean array; ``efficiency_of_mag`` maps magnitude to the measured
    recovery fraction. Returns (updated weights, fraction of the posterior ruled out).
    """
    p_detect = np.where(np.asarray(inside_patch, dtype=bool),
                        np.asarray(efficiency_of_mag(prediction["v_mag"]), dtype=float), 0.0)
    new_weight = np.asarray(prediction["weight"], dtype=float) * (1 - p_detect)
    ruled_out = float(np.mean(p_detect))
    return new_weight, ruled_out

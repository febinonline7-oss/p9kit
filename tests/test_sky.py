import numpy as np
import pytest

astropy = pytest.importorskip("astropy")

from p9kit import sky


def test_mass_radius_and_magnitude_are_monotonic():
    assert sky.radius_earth_from_mass(10) > sky.radius_earth_from_mass(5)
    assert sky.absolute_magnitude(10, 0.5) < sky.absolute_magnitude(5, 0.5)   # bigger = brighter


def test_predicted_sky_positions_are_valid_and_faint():
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.uniform(4, 8, 300), rng.uniform(400, 800, 300), rng.uniform(0.1, 0.4, 300),
        rng.uniform(10, 25, 300), rng.uniform(0, 360, 300), rng.uniform(0, 360, 300),
        rng.uniform(10, 25, 300)])
    out = sky.predict_sky(samples, "2026-10-01", rng)
    assert ((out["ra"] >= 0) & (out["ra"] < 360)).all()
    assert (np.abs(out["dec"]) <= 90).all()
    assert out["v_mag"].min() > 15 and out["v_mag"].max() < 32
    assert (out["rate_arcsec_day"] >= 0).all() and out["rate_arcsec_day"].max() < 30


def test_a_non_detection_reduces_the_weights():
    rng = np.random.default_rng(1)
    prediction = dict(v_mag=np.array([20.0, 25.0]), weight=np.ones(2))
    weights, ruled_out = sky.apply_nondetection(prediction, np.array([True, True]),
                                                lambda m: np.where(m < 22, 0.9, 0.0))
    assert weights[0] == pytest.approx(0.1)
    assert weights[1] == pytest.approx(1.0)
    assert ruled_out == pytest.approx(0.45)

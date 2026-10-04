import numpy as np
import pytest

from p9kit import orbits


def test_kepler_solution_satisfies_the_equation():
    rng = np.random.default_rng(0)
    M = rng.uniform(0, 2 * np.pi, 500)
    for e in (0.0, 0.3, 0.9, 0.98):
        E = orbits.solve_kepler(M, np.full_like(M, e))
        assert np.allclose(E - e * np.sin(E), M, atol=1e-9)


def test_pole_vector_round_trip():
    rng = np.random.default_rng(1)
    inc, node = rng.uniform(1, 80, 200), rng.uniform(0, 360, 200)
    back_inc, back_node = orbits.pole_to_inc_node(orbits.pole_vector(inc, node))
    assert np.allclose(back_inc, inc, atol=1e-8)
    assert np.allclose(orbits.circ_diff(back_node, node), 0, atol=1e-8)


def test_circular_mean_of_identical_angles_has_R_one():
    direction, R = orbits.circ_mean_R(np.full(10, 42.0))
    assert R == pytest.approx(1.0)
    assert direction == pytest.approx(42.0)


def test_circular_mean_of_uniform_angles_has_small_R():
    rng = np.random.default_rng(2)
    _, R = orbits.circ_mean_R(rng.uniform(0, 360, 20000))
    assert R < 0.05


def test_circular_difference_wraps():
    assert orbits.circ_diff(359.0, 1.0) == pytest.approx(2.0)


def test_position_at_perihelion_is_q_from_the_sun():
    a, e = 500.0, 0.9
    xyz = orbits.xyz_from_mean_anomaly(a, e, 20.0, 100.0, 30.0, 0.0)
    assert np.linalg.norm(xyz) == pytest.approx(a * (1 - e), rel=1e-9)


def test_time_weighted_anomaly_is_uniform_in_mean_anomaly():
    rng = np.random.default_rng(3)
    e = np.full(40000, 0.8)
    E = orbits.sample_time_weighted_E(e, rng)
    M = np.mod(E - e * np.sin(E), 2 * np.pi)
    counts, _ = np.histogram(M, bins=8)
    assert counts.min() > 0.8 * counts.mean()


def test_ecliptic_to_equatorial_preserves_length_and_rotates_about_x():
    xyz = np.array([[1.0, 2.0, 3.0]])
    out = orbits.ecliptic_to_equatorial(xyz)
    assert np.linalg.norm(out) == pytest.approx(np.linalg.norm(xyz))
    assert out[0, 0] == pytest.approx(1.0)

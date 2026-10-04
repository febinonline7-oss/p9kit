import numpy as np
import pytest

from p9kit.population import Planet, simulate_detected
from p9kit.stats import SUMMARY_NAMES, clustering, luck_test, summary_statistics


def make_sample(**kwargs):
    rng = np.random.default_rng(kwargs.pop("seed", 0))
    return simulate_detected(Planet(**kwargs), 40, rng)


def test_summary_vector_has_the_documented_length_and_is_finite():
    x = summary_statistics(make_sample())
    assert len(x) == len(SUMMARY_NAMES)
    assert np.all(np.isfinite(x))


def test_summary_statistics_do_not_depend_on_object_order():
    sample = make_sample()
    shuffled = sample[np.random.default_rng(1).permutation(len(sample))]
    assert np.allclose(summary_statistics(sample), summary_statistics(shuffled), atol=1e-6)


def test_clustering_is_strong_when_perihelia_agree():
    sample = np.zeros(5, dtype=[(k, float) for k in ("a", "e", "inc", "node", "argperi")])
    sample["a"] = np.linspace(300, 800, 5)
    sample["node"] = 100.0
    sample["argperi"] = 50.0
    direction, strength = clustering(sample)
    assert strength == pytest.approx(1.0)
    assert direction == pytest.approx(150.0)


def test_luck_test_reports_a_sane_p_value_for_a_planet_free_sample():
    rng = np.random.default_rng(4)
    sample = simulate_detected(Planet(mass_earth=0.0), 30, rng)
    result = luck_test(sample, rng, n_null=60, isotropic=True)
    assert 0.0 <= result.p_value <= 1.0
    assert "1 in" in result.one_in()
    assert result.verdict() in {"very unlikely to be chance", "probably not chance",
                                "might be chance", "easily explained by chance"}


def test_strong_clustering_is_flagged_as_unlikely_chance():
    rng = np.random.default_rng(5)
    sample = simulate_detected(Planet(mass_earth=18.0, a9_au=350.0), 40, rng)
    result = luck_test(sample, rng, n_null=200, isotropic=True)
    assert result.p_value < 0.05


def test_training_needs_the_optional_extra_or_works():
    from p9kit import inference
    theta, x = inference.simulate_training_set(5, 20, np.random.default_rng(6))
    assert theta.shape == (5, 7)
    assert x.shape[0] == 5

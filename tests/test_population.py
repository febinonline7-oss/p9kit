import numpy as np
import pytest

from p9kit.population import (Planet, SurveyModel, forcing_strength, orbital_angles,
                              sample_population, simulate_detected)
from p9kit.stats import clustering


def test_planet_array_round_trip():
    planet = Planet(mass_earth=7.0, a9_au=600.0)
    assert Planet.from_array(planet.as_array()) == planet


def test_no_planet_means_no_forcing_and_no_confinement():
    rng = np.random.default_rng(0)
    a = np.full(500, 500.0)
    assert forcing_strength(Planet(mass_earth=0.0), a).max() == 0.0
    _, _, _, confined = orbital_angles(Planet(mass_earth=0.0), a, rng)
    assert not confined.any()


def test_forcing_grows_with_mass_and_shrinks_with_distance():
    a = np.array([500.0])
    assert forcing_strength(Planet(mass_earth=10.0), a) > forcing_strength(Planet(mass_earth=5.0), a)
    assert forcing_strength(Planet(a9_au=900.0), a) < forcing_strength(Planet(a9_au=400.0), a)


def test_sampled_population_respects_its_ranges():
    rng = np.random.default_rng(1)
    a, e, abs_mag = sample_population(5000, rng)
    q = a * (1 - e)
    assert a.min() >= 250.0 and a.max() <= 1500.0
    assert q.min() >= 29.9 and q.max() <= 70.1
    assert abs_mag.min() >= 3.0 and abs_mag.max() <= 8.0


def test_simulation_returns_the_requested_number_of_objects():
    rng = np.random.default_rng(2)
    sample = simulate_detected(Planet(), 25, rng)
    assert len(sample) == 25
    assert set(sample.dtype.names) == {"a", "e", "inc", "node", "argperi"}


def test_a_massive_planet_clusters_perihelia_more_than_no_planet():
    rng = np.random.default_rng(3)
    strong = np.median([clustering(simulate_detected(Planet(mass_earth=15.0, a9_au=400.0), 40, rng))[1]
                        for _ in range(12)])
    none = np.median([clustering(simulate_detected(Planet(mass_earth=0.0), 40, rng))[1]
                      for _ in range(12)])
    assert strong > none + 0.15


def test_survey_sees_bright_near_objects_and_misses_faint_far_ones():
    survey = SurveyModel()
    near = np.array([[40.0, 0.0, 0.0]])
    far = np.array([[1200.0, 0.0, 0.0]])
    assert survey.detection_probability(near, np.array([4.0]))[0] > 0.99
    assert survey.detection_probability(far, np.array([8.0]))[0] < 0.01

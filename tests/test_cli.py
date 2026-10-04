import json
import numpy as np
import pandas as pd
import pytest

from p9kit import cli
from p9kit.population import Planet, simulate_detected


def test_clustering_command_runs_on_a_csv(tmp_path, capsys):
    rng = np.random.default_rng(0)
    sample = simulate_detected(Planet(mass_earth=12.0), 25, rng)
    path = tmp_path / "orbits.csv"
    pd.DataFrame({"a": sample["a"], "e": sample["e"], "i": sample["inc"],
                  "om": sample["node"], "w": sample["argperi"]}).to_csv(path, index=False)
    cli.main(["clustering", str(path), "--n-null", "20", "--isotropic"])
    result = json.loads(capsys.readouterr().out)
    assert result["n_objects"] == 25
    assert 0.0 <= result["clustering_R"] <= 1.0


def test_clustering_command_rejects_a_table_without_angles(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": [300.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        cli.main(["clustering", str(path)])

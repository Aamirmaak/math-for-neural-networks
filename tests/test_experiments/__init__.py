"""Tests for experiment utilities."""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.experiment_utils import (
    ExperimentResult,
    ensure_results_dir,
    print_table,
    save_fig,
    setup_seed,
)


class TestSetupSeed:
    """Test seed setup utility."""

    def test_deterministic(self) -> None:
        rng1 = setup_seed(42)
        rng2 = setup_seed(42)
        vals1 = rng1.standard_normal(10)
        vals2 = rng2.standard_normal(10)
        np.testing.assert_array_equal(vals1, vals2)

    def test_different_seeds(self) -> None:
        rng1 = setup_seed(42)
        rng2 = setup_seed(123)
        vals1 = rng1.standard_normal(10)
        vals2 = rng2.standard_normal(10)
        assert not np.array_equal(vals1, vals2)

    def test_returns_generator(self) -> None:
        rng = setup_seed(0)
        assert isinstance(rng, np.random.Generator)


class TestExperimentResult:
    """Test ExperimentResult dataclass."""

    def test_creation(self) -> None:
        r = ExperimentResult(name="test", hypothesis="h", method="m")
        assert r.name == "test"
        assert r.hypothesis == "h"
        assert r.method == "m"
        assert r.metrics == {}
        assert r.observations == []
        assert r.conclusion == ""
        assert r.runtime_seconds == 0.0

    def test_summary(self) -> None:
        r = ExperimentResult(
            name="Test",
            hypothesis="Hypothesis",
            method="Method",
            metrics={"key": 1.234},
            observations=["obs1"],
            conclusion="Conclusion",
        )
        s = r.summary()
        assert "Test" in s
        assert "Hypothesis" in s
        assert "Method" in s
        assert "obs1" in s
        assert "Conclusion" in s

    def test_save(self, tmp_path: object) -> None:
        r = ExperimentResult(name="test_save", hypothesis="h", method="m")
        r.metrics["value"] = 42.0
        r.conclusion = "done"
        path = r.save(tmp_path)  # type: ignore
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["name"] == "test_save"
        assert data["metrics"]["value"] == 42.0


class TestEnsureResultsDir:
    """Test results directory creation."""

    def test_creates_directory(self, tmp_path: object) -> None:
        import experiments.experiment_utils as eu
        original = eu.RESULTS_DIR
        eu.RESULTS_DIR = tmp_path  # type: ignore
        try:
            path = ensure_results_dir("test_experiment")
            assert path.exists()
            assert path.is_dir()
        finally:
            eu.RESULTS_DIR = original


class TestPrintTable:
    """Test table formatting."""

    def test_basic(self) -> None:
        result = print_table(["A", "B"], [["1", "2"], ["3", "4"]], title="Test")
        assert "A" in result
        assert "B" in result
        assert "Test" in result


class TestSaveFig:
    """Test figure saving."""

    def test_saves_figure(self, tmp_path: object) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        path = save_fig(fig, "test_fig", directory=tmp_path)  # type: ignore
        assert path.exists()
        assert path.suffix == ".png"

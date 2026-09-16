"""Tests for experiment utilities and integration pipeline."""

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


class TestIntegration:
    """Integration test: complete training pipeline."""

    def test_full_pipeline(self) -> None:
        """Test that the complete mathematical pipeline works end-to-end."""
        np.random.seed(42)

        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [0], [0], [1]], dtype=np.float64)

        W1 = np.random.randn(2, 3) * 0.5
        b1 = np.zeros(3)
        W2 = np.random.randn(3, 1) * 0.5
        b2 = np.zeros(1)

        lr = 0.5
        losses = []

        for _ in range(50):
            # Forward
            z1 = X @ W1 + b1
            a1 = 1.0 / (1.0 + np.exp(-z1))
            z2 = a1 @ W2 + b2
            y_pred = z2
            loss = float(np.mean((y_pred - y) ** 2))
            losses.append(loss)

            # Backward
            n = X.shape[0]
            dL_dz2 = (2.0 / n) * (y_pred - y)
            dW2 = a1.T @ dL_dz2
            db2 = np.sum(dL_dz2, axis=0)
            dL_da1 = dL_dz2 @ W2.T
            da1_dz1 = a1 * (1 - a1)
            dL_dz1 = dL_da1 * da1_dz1
            dW1 = X.T @ dL_dz1
            db1 = np.sum(dL_dz1, axis=0)

            # Update
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        assert losses[-1] < losses[0], "Loss should decrease"
        assert losses[-1] < 0.1, "Final loss should be small"

        # Verify predictions
        z1 = X @ W1 + b1
        a1 = 1.0 / (1.0 + np.exp(-z1))
        z2 = a1 @ W2 + b2
        preds = (z2 > 0.5).astype(float)
        accuracy = float(np.mean(preds == y))
        assert accuracy >= 0.75, f"Accuracy should be >= 0.75, got {accuracy}"

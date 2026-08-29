from __future__ import annotations

from pathlib import Path
import pytest

from run_pipeline import main


def test_severson_cli_recommend_rejected():
    with pytest.raises(ValueError, match="prediction-only benchmark and does not support recommendation"):
        main(dataset="severson", mode="recommend")


def test_dynamic_cycling_cli_recommend_rejected():
    with pytest.raises(ValueError, match="offline protocol optimization benchmark.*recommendation on the full ground-truth dataset is forbidden"):
        main(dataset="dynamic_cycling", mode="recommend")


def test_severson_cli_train_and_benchmark(monkeypatch):
    called = []

    def mock_benchmark(adapter):
        called.append("severson_benchmark")

    monkeypatch.setattr("src.evaluation.severson_benchmark.run_severson_benchmark", mock_benchmark)
    main(dataset="severson", mode="benchmark")
    assert "severson_benchmark" in called

    # mode="full" redirects to benchmark for severson
    called.clear()
    main(dataset="severson", mode="full")
    assert "severson_benchmark" in called


def test_dynamic_cycling_cli_full_redirects_to_benchmark(monkeypatch):
    called = []

    def mock_benchmark(adapter):
        called.append("dynamic_cycling_benchmark")

    monkeypatch.setattr("src.evaluation.dynamic_cycling_benchmark.run_dynamic_cycling_benchmark", mock_benchmark)
    main(dataset="dynamic_cycling", mode="full")
    assert "dynamic_cycling_benchmark" in called

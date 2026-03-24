"""Shared test fixtures for inspect-mlflow tests."""

from __future__ import annotations

import contextlib

import mlflow
import pytest


@pytest.fixture
def tmp_tracking_uri(tmp_path, monkeypatch):
    """Create a temporary SQLite-backed MLflow tracking store.

    This is the MLflow best practice for testing: use a real tracking store
    with an ephemeral SQLite database, not mocks. Every test gets a clean
    database.
    """
    db_path = tmp_path / "mlflow.db"
    uri = f"sqlite:///{db_path}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-experiment")
    mlflow.set_tracking_uri(uri)
    # Disable async logging in tests so writes are visible immediately
    with contextlib.suppress(Exception):
        mlflow.config.enable_async_logging(False)
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.fixture
def tracing_env(tmp_tracking_uri, monkeypatch):
    """Enable tracing on top of the tracking fixture."""
    monkeypatch.setenv("MLFLOW_INSPECT_TRACING", "true")
    yield tmp_tracking_uri

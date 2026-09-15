"""Configuration compatibility with and without the optional settings package."""

import importlib.util
import sys
from pathlib import Path

import pytest

from inspect_mlflow import config


@pytest.fixture(params=["validated", "fallback"])
def settings_class(request, monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith(("MLFLOW_", "INSPECT_MLFLOW_")):
            monkeypatch.delenv(name)
    if request.param == "validated":
        from pydantic_settings import BaseSettings

        assert issubclass(config.MLflowSettings, BaseSettings)
        return config.MLflowSettings
    name = "_test_fallback_config"
    spec = importlib.util.spec_from_file_location(name, Path(config.__file__))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "pydantic_settings", None)
    spec.loader.exec_module(module)
    return module.MLflowSettings


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
def test_legacy_artifact_setting(settings_class, monkeypatch, value):
    monkeypatch.setenv("MLFLOW_INSPECT_LOG_ARTIFACTS", value)
    assert settings_class().log_artifacts is False


def test_new_settings_override_legacy(settings_class, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://old.example")
    monkeypatch.setenv("INSPECT_MLFLOW_TRACKING_URI", "http://new.example")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "old")
    monkeypatch.setenv("INSPECT_MLFLOW_EXPERIMENT_NAME", "new")
    monkeypatch.setenv("MLFLOW_INSPECT_TRACING", "false")
    monkeypatch.setenv("INSPECT_MLFLOW_TRACING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_INSPECT_LOG_ARTIFACTS", "false")
    monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "true")
    settings = settings_class()
    assert settings.tracking_uri == "http://new.example"
    assert settings.experiment_name == "new"
    assert settings.tracing_enabled is True
    assert settings.log_artifacts is True


def test_invalid_boolean_is_rejected(settings_class, monkeypatch):
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "sometimes")
    with pytest.raises(ValueError):
        settings_class()


def test_new_setting_overrides_invalid_legacy_value(settings_class, monkeypatch):
    monkeypatch.setenv("MLFLOW_INSPECT_LOG_ARTIFACTS", "invalid")
    monkeypatch.setenv("INSPECT_MLFLOW_LOG_ARTIFACTS", "false")
    assert settings_class().log_artifacts is False


@pytest.mark.asyncio
async def test_tracking_client_uses_configured_uri(tmp_path, monkeypatch):
    import mlflow
    from inspect_ai.hooks import RunEnd, RunStart
    from mlflow.tracking import MlflowClient

    from inspect_mlflow.tracking import MlflowTrackingHooks

    previous = mlflow.get_tracking_uri()
    configured = f"sqlite:///{tmp_path / 'configured.db'}"
    monkeypatch.setenv("INSPECT_MLFLOW_TRACKING_URI", configured)
    monkeypatch.setenv("INSPECT_MLFLOW_EXPERIMENT_NAME", "configured-experiment")
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'unrelated.db'}")
    try:
        hook = MlflowTrackingHooks()
        await hook.on_run_start(
            RunStart(eval_set_id=None, run_id="config-test", task_names=["task"])
        )
        run_id = hook._parent_run_id
        await hook.on_run_end(
            RunEnd(eval_set_id=None, run_id="config-test", exception=None, logs=[])
        )
        client = MlflowClient(tracking_uri=configured)
        experiment = client.get_experiment_by_name("configured-experiment")
        assert client.get_run(run_id).info.experiment_id == experiment.experiment_id
        assert client.get_run(run_id).info.status == "FINISHED"
    finally:
        mlflow.set_tracking_uri(previous)

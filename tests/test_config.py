"""Tests for inspect_mlflow.config settings parsing."""

from __future__ import annotations

from inspect_mlflow.config import DEFAULT_AUTOLOG_MODELS, _parse_autolog_models, load_settings


def _clear_autolog_env(monkeypatch) -> None:
    monkeypatch.delenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", raising=False)
    monkeypatch.delenv("INSPECT_MLFLOW_AUTOLOG_MODELS", raising=False)


def test_autolog_defaults(monkeypatch) -> None:
    _clear_autolog_env(monkeypatch)
    settings = load_settings()
    assert settings.autolog_enabled is True
    assert settings.autolog_models == DEFAULT_AUTOLOG_MODELS


def test_autolog_enabled_false_from_env(monkeypatch) -> None:
    _clear_autolog_env(monkeypatch)
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_ENABLED", "false")
    settings = load_settings()
    assert settings.autolog_enabled is False


def test_autolog_models_from_csv(monkeypatch) -> None:
    _clear_autolog_env(monkeypatch)
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_MODELS", " OpenAI , litellm , GROQ ")
    settings = load_settings()
    assert settings.autolog_models == ["openai", "litellm", "groq"]


def test_autolog_models_from_json_array(monkeypatch) -> None:
    _clear_autolog_env(monkeypatch)
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_MODELS", '["OpenAI","bedrock","cohere"]')
    settings = load_settings()
    assert settings.autolog_models == ["openai", "bedrock", "cohere"]


def test_autolog_models_empty_string_falls_back_to_default(monkeypatch) -> None:
    _clear_autolog_env(monkeypatch)
    monkeypatch.setenv("INSPECT_MLFLOW_AUTOLOG_MODELS", "")
    settings = load_settings()
    assert settings.autolog_models == DEFAULT_AUTOLOG_MODELS


def test_autolog_models_parse_from_list() -> None:
    assert _parse_autolog_models(["OpenAI", "Gemini", " "]) == ["openai", "gemini"]

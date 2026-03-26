"""Tests for inspect_mlflow._autolog."""

from __future__ import annotations

from types import SimpleNamespace

from inspect_mlflow._autolog import enable_autolog


def test_enable_autolog_enables_provider_when_flavor_and_dependency_exist() -> None:
    calls: list[dict[str, bool]] = []

    def fake_autolog(*, log_traces: bool) -> None:
        calls.append({"log_traces": log_traces})

    modules = {"mlflow.openai": SimpleNamespace(autolog=fake_autolog)}
    available = {"mlflow.openai", "openai"}

    def fake_find_spec(name: str):
        return object() if name in available else None

    def fake_import_module(name: str):
        return modules[name]

    enabled = enable_autolog(
        ["openai"],
        find_spec=fake_find_spec,
        import_module=fake_import_module,
    )

    assert enabled is True
    assert calls == [{"log_traces": True}]


def test_enable_autolog_skips_unknown_provider() -> None:
    def fake_find_spec(_name: str):
        raise AssertionError("find_spec should not be called for unknown provider")

    enabled = enable_autolog(
        ["unknown_provider"],
        find_spec=fake_find_spec,
        import_module=lambda _name: None,
    )

    assert enabled is False


def test_enable_autolog_skips_when_mlflow_flavor_missing() -> None:
    def fake_find_spec(name: str):
        if name == "mlflow.openai":
            return None
        return object()

    enabled = enable_autolog(
        ["openai"],
        find_spec=fake_find_spec,
        import_module=lambda _name: None,
    )

    assert enabled is False


def test_enable_autolog_skips_when_provider_dependency_missing() -> None:
    def fake_find_spec(name: str):
        if name == "mlflow.openai":
            return object()
        return None

    enabled = enable_autolog(
        ["openai"],
        find_spec=fake_find_spec,
        import_module=lambda _name: None,
    )

    assert enabled is False


def test_enable_autolog_handles_provider_autolog_errors() -> None:
    def fake_autolog(*, log_traces: bool) -> None:
        raise RuntimeError("boom")

    modules = {"mlflow.openai": SimpleNamespace(autolog=fake_autolog)}
    available = {"mlflow.openai", "openai"}

    def fake_find_spec(name: str):
        return object() if name in available else None

    def fake_import_module(name: str):
        return modules[name]

    enabled = enable_autolog(
        ["openai"],
        find_spec=fake_find_spec,
        import_module=fake_import_module,
    )

    assert enabled is False

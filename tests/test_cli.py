"""Tests for the CLI entry point."""

from __future__ import annotations

import json

from signalmint.cli import main


def test_version_command(capsys) -> None:
    rc = main(["version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "signalmint" in out


def test_no_command_prints_version(capsys) -> None:
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "signalmint" in out


def test_config_command_stdout_is_valid_json(capsys) -> None:
    rc = main(["config"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["data"]["dataset"] == "cwru"
    assert payload["model"]["num_bins"] == payload["data"]["num_bins"]


def test_config_command_writes_file(tmp_path, capsys) -> None:
    target = tmp_path / "out.json"
    rc = main(["config", "-o", str(target)])
    assert rc == 0
    assert target.exists()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "model" in payload

from __future__ import annotations

import pytest

from scripts.check_yaml_duplicates import DuplicateKeyError, check_yaml_file, main


def test_check_yaml_file_rejects_nested_duplicate(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """\
outer:
  nested:
    value: first
    value: second
""",
        encoding="utf-8",
    )

    with pytest.raises(DuplicateKeyError, match="found duplicate key 'value'"):
        check_yaml_file(path)


def test_check_yaml_file_accepts_valid_yaml(tmp_path):
    path = tmp_path / "valid.yaml"
    path.write_text(
        """\
outer:
  nested:
    first: 1
    second: 2
""",
        encoding="utf-8",
    )

    check_yaml_file(path)


def test_cli_fails_when_any_of_multiple_files_has_duplicate(tmp_path, capsys):
    valid = tmp_path / "valid.yaml"
    duplicate = tmp_path / "duplicate.yml"
    valid.write_text("first: 1\n", encoding="utf-8")
    duplicate.write_text("second: 2\nsecond: 3\n", encoding="utf-8")

    assert main([str(valid), str(duplicate)]) == 1
    captured = capsys.readouterr()
    assert str(duplicate) in captured.err
    assert "failed for 1 of 2 files" in captured.err


def test_cli_succeeds_for_multiple_valid_files(tmp_path, capsys):
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yml"
    first.write_text("first: 1\n", encoding="utf-8")
    second.write_text("second: 2\n", encoding="utf-8")

    assert main([str(first), str(second)]) == 0
    captured = capsys.readouterr()
    assert "passed for 2 files" in captured.out
    assert captured.err == ""


def test_cli_fails_for_nonexistent_requested_path(tmp_path, capsys):
    missing = tmp_path / "missing"

    assert main([str(missing)]) == 1
    captured = capsys.readouterr()
    assert f"YAML path does not exist: {missing}" in captured.err
    assert captured.out == ""


def test_cli_fails_when_discovery_finds_no_yaml_files(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("not YAML\n", encoding="utf-8")

    assert main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "YAML validation found no YAML files" in captured.err
    assert captured.out == ""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode


class DuplicateKeyError(ConstructorError):
    """Raised when a YAML mapping defines the same key more than once."""


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at any depth."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader, node: MappingNode, deep: bool = False
) -> dict[object, object]:
    loader.flatten_mapping(node)
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise DuplicateKeyError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def yaml_files(paths: Iterable[Path]) -> list[Path]:
    """Return unique YAML files from explicit files and recursive directories."""
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.yaml"))
            files.update(path.rglob("*.yml"))
        elif path.suffix.lower() in {".yaml", ".yml"}:
            files.add(path)
    return sorted(files)


def check_yaml_file(path: Path) -> None:
    """Parse every document in a YAML file, rejecting duplicate keys."""
    with path.open(encoding="utf-8") as handle:
        list(yaml.load_all(handle, Loader=UniqueKeyLoader))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reject duplicate mapping keys in YAML files.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        default=[Path("configs"), Path("docs")],
        help="YAML files or directories to scan recursively (default: configs docs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing_paths = [path for path in args.paths if not path.exists()]
    if missing_paths:
        for path in missing_paths:
            print(f"YAML path does not exist: {path}", file=sys.stderr)
        return 1

    files = yaml_files(args.paths)
    if not files:
        print("YAML validation found no YAML files", file=sys.stderr)
        return 1

    failures: list[tuple[Path, yaml.YAMLError]] = []
    for path in files:
        try:
            check_yaml_file(path)
        except yaml.YAMLError as exc:
            failures.append((path, exc))

    if failures:
        for path, error in failures:
            print(f"{path}: {error}", file=sys.stderr)
        print(
            f"YAML validation failed for {len(failures)} of {len(files)} files",
            file=sys.stderr,
        )
        return 1

    print(f"YAML duplicate-key check passed for {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def test_imports():
    import importlib

    assert importlib.import_module("ups") is not None

    # Optional heavy deps should import when installed
    import torch  # noqa: F401


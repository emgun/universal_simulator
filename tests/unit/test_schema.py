import pytest
import torch

from ups.data.schemas import validate_sample


def make_good_sample(N: int = 8, d: int = 2):
    coords = torch.randn(N, d)
    fields = {
        "u": torch.randn(N, 2),
        "p": torch.randn(N, 1),
    }
    sample = {
        "kind": "grid",
        "coords": coords,
        "connect": None,
        "fields": fields,
        "bc": {"type": "periodic"},
        "params": {"Re": 100.0},
        "geom": {"domain": "unit_square"},
        "time": torch.tensor(0.0),
        "dt": torch.tensor(0.01),
        "meta": {"id": "s0"},
    }
    return sample


def test_required_fields_missing():
    s = make_good_sample()
    del s["dt"]
    with pytest.raises(ValueError):
        validate_sample(s)


def test_good_sample_validates():
    s = make_good_sample()
    validate_sample(s)


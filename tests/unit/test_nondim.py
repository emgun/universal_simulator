import torch

from ups.discovery.nondim import from_pi_units, to_pi_units


def test_roundtrip_error():
    N = 4
    sample = {
        "kind": "grid",
        "coords": torch.randn(N, 2),
        "connect": None,
        "fields": {
            "u": torch.randn(N, 2),
            "p": torch.randn(N, 1),
        },
        "bc": {},
        "params": {"Re": 100.0, "nu": 0.01},
        "geom": None,
        "time": torch.tensor(0.0),
        "dt": torch.tensor(0.1),
        "meta": {"units": {"u": 2.0, "p": 5.0, "Re": 100.0, "nu": 0.01}},
    }

    s_pi = to_pi_units(sample)
    s_back = from_pi_units(s_pi)

    du = (s_back["fields"]["u"] - sample["fields"]["u"]).abs().max().item()
    dp = (s_back["fields"]["p"] - sample["fields"]["p"]).abs().max().item()
    dRe = abs(s_back["params"]["Re"] - sample["params"]["Re"])  # type: ignore[index]
    dnu = abs(s_back["params"]["nu"] - sample["params"]["nu"])  # type: ignore[index]

    assert du < 1e-6
    assert dp < 1e-6
    assert dRe < 1e-9
    assert dnu < 1e-12


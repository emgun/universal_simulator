from ups.eval.gates import periodic_gate, residual_gate


def test_residual_gate_triggers_above_threshold():
    assert residual_gate(1.1, 1.0)
    assert not residual_gate(0.5, 1.0)


def test_periodic_gate_every_n():
    assert periodic_gate(4, 5)
    assert not periodic_gate(3, 5)

import torch

from ups.eval.calibration import TemperatureScaler, expected_calibration_error, reliability_diagram


def test_reliability_diagram_shapes():
    probs = torch.tensor([0.1, 0.4, 0.6, 0.9])
    targets = torch.tensor([0, 0, 1, 1])
    confidences, accuracies = reliability_diagram(probs, targets, n_bins=4)
    assert confidences.shape == (4,)
    assert accuracies.shape == (4,)
    ece = expected_calibration_error(probs, targets, n_bins=4)
    assert torch.isfinite(ece)


def test_temperature_scaler_learning():
    logits = torch.tensor([[2.0, -2.0], [1.0, -1.0]])
    targets = torch.tensor([0, 0])
    scaler = TemperatureScaler()
    scaler.fit(logits, targets, steps=10)
    calibrated = scaler.calibrate(logits)
    probs = torch.softmax(calibrated, dim=-1)
    assert torch.all(probs[:, 0] > 0.5)


import pytest
import torch
from torch.nn import functional as F

from ups.io import adaptive_token_avg_pool1d


@pytest.mark.parametrize(
    ("input_len", "target_len"),
    [
        (1_024, 64),
        (16_384, 64),
        (17, 5),
        (4, 7),
    ],
)
def test_adaptive_token_avg_pool1d_matches_pytorch_values_and_gradients(
    input_len: int, target_len: int
) -> None:
    actual_input = torch.randn(2, input_len, 3, dtype=torch.float64, requires_grad=True)
    expected_input = actual_input.detach().clone().requires_grad_(True)

    actual = adaptive_token_avg_pool1d(actual_input, target_len)
    expected = F.adaptive_avg_pool1d(expected_input.transpose(1, 2), target_len).transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    output_gradient = torch.randn_like(actual)
    actual.backward(output_gradient)
    expected.backward(output_gradient)
    torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=1e-12, atol=1e-12)


def test_adaptive_token_avg_pool1d_preserves_identity() -> None:
    tokens = torch.randn(2, 6, 4, requires_grad=True)

    pooled = adaptive_token_avg_pool1d(tokens, 6)

    assert pooled is tokens


@pytest.mark.parametrize(
    ("tokens", "target_len", "exception", "message"),
    [
        (torch.empty(2, 0, 3), 1, ValueError, "non-empty"),
        (torch.empty(2, 3), 1, ValueError, "3D tensor"),
        (torch.empty(2, 3, 4), 0, ValueError, "positive"),
        (torch.empty(2, 3, 4), 1.5, TypeError, "integer"),
    ],
)
def test_adaptive_token_avg_pool1d_validates_inputs(
    tokens: torch.Tensor,
    target_len: int,
    exception: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception, match=message):
        adaptive_token_avg_pool1d(tokens, target_len)

from __future__ import annotations

import torch


def adaptive_token_avg_pool1d(tokens: torch.Tensor, target_len: int) -> torch.Tensor:
    """Average-pool ``(batch, tokens, features)`` tensors along the token axis.

    This matches the windowing semantics of ``torch.nn.functional.adaptive_avg_pool1d``
    without dispatching to its CUDA kernel. Divisible reductions use a single
    reshape/mean operation; all other ratios use the same floor/ceil window boundaries
    as PyTorch's adaptive pooling implementation.
    """
    if tokens.ndim != 3:
        raise ValueError(
            "adaptive_token_avg_pool1d expects a 3D tensor with shape " "(batch, tokens, features)"
        )
    if not isinstance(target_len, int):
        raise TypeError("target_len must be an integer")
    if target_len <= 0:
        raise ValueError("target_len must be positive")

    input_len = tokens.shape[1]
    if input_len <= 0:
        raise ValueError("the input token axis must be non-empty")
    if input_len == target_len:
        return tokens

    if input_len > target_len and input_len % target_len == 0:
        window_len = input_len // target_len
        return tokens.reshape(tokens.shape[0], target_len, window_len, tokens.shape[2]).mean(dim=2)

    windows = []
    for index in range(target_len):
        start = (index * input_len) // target_len
        end = ((index + 1) * input_len + target_len - 1) // target_len
        windows.append(tokens[:, start:end, :].mean(dim=1, keepdim=True))
    return torch.cat(windows, dim=1)

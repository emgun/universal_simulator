from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GridEncoderConfig:
    latent_len: int
    latent_dim: int
    field_channels: Mapping[str, int]
    patch_size: int = 4
    stem_width: Optional[int] = None
    use_fourier_features: bool = True
    fourier_frequencies: Tuple[float, ...] = (1.0, 2.0)


class GridEncoder(nn.Module):
    """Encode grid-based fields into latent tokens using pixel-unshuffle stems.

    The encoder applies per-field residual convolutional stems after pixel-unshuffling
    patches, optionally augments with sinusoidal Fourier features of the coordinates,
    and projects the resulting patch tokens into a latent space.

    A lightweight reconstruction head is provided for identity self-checks; when the
    requested latent token count matches the patch token count, reconstruction is
    lossless up to numerical precision.
    """

    def __init__(self, cfg: GridEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.patch = cfg.patch_size
        self.field_channels = dict(cfg.field_channels)
        if not self.field_channels:
            raise ValueError("field_channels must be a non-empty mapping")
        self.field_order = list(self.field_channels.keys())

        self.pixel_unshuffle = nn.PixelUnshuffle(self.patch)
        self.pixel_shuffle = nn.PixelShuffle(self.patch)

        self.per_field_stems = nn.ModuleDict()
        stem_width = cfg.stem_width
        for name, channels in self.field_channels.items():
            in_ch = channels * self.patch * self.patch
            width = stem_width or max(in_ch, 32)
            stem = nn.Sequential(
                nn.Conv2d(in_ch, width, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, in_ch, kernel_size=1),
            )
            # Zero-init the last conv so the residual path is near identity.
            nn.init.zeros_(stem[-1].weight)
            nn.init.zeros_(stem[-1].bias)
            self.per_field_stems[name] = stem

        self.use_fourier = cfg.use_fourier_features
        if self.use_fourier:
            freqs = torch.tensor(cfg.fourier_frequencies, dtype=torch.float32)
            self.register_buffer("fourier_frequencies", freqs, persistent=False)
        else:
            self.register_buffer("fourier_frequencies", torch.zeros(0), persistent=False)

        # Use identity projection when latent_dim matches patch channel budget to keep
        # reconstruction exact; otherwise fall back to lazy linear projections.
        if cfg.latent_dim == self._patch_channel_total:
            self.to_latent = nn.Identity()
            self.from_latent = nn.Identity()
        else:
            self.to_latent = nn.LazyLinear(cfg.latent_dim)
            self.from_latent = nn.LazyLinear(self._patch_channel_total)

    @property
    def _patch_channel_total(self) -> int:
        return sum(ch * self.patch * self.patch for ch in self.field_channels.values())

    def forward(
        self,
        fields: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        *,
        params: Optional[Mapping[str, torch.Tensor]] = None,
        bc: Optional[Mapping[str, torch.Tensor]] = None,
        geom: Optional[Mapping[str, torch.Tensor]] = None,
        meta: Optional[Mapping[str, object]] = None,
    ) -> torch.Tensor:
        grid_shape = self._infer_grid_shape(meta)
        H, W = grid_shape
        if H % self.patch != 0 or W % self.patch != 0:
            raise ValueError(f"Grid shape {grid_shape} not divisible by patch size {self.patch}")

        tokens, features = self._encode_fields(fields, coords, grid_shape)
        target_device = next(self.parameters()).device
        features = features.to(target_device)
        latent = self.to_latent(features)
        if tokens != self.cfg.latent_len:
            # Pool along the token axis to the configured length.
            latent = self._adaptive_token_pool(latent, self.cfg.latent_len)
        return latent

    def reconstruct(self, latent: torch.Tensor, meta: Mapping[str, object]) -> Dict[str, torch.Tensor]:
        grid_shape = self._infer_grid_shape(meta)
        H, W = grid_shape
        Hp = H // self.patch
        Wp = W // self.patch
        expected_tokens = Hp * Wp
        if latent.shape[1] != expected_tokens:
            raise ValueError(
                "Reconstruction requires latent tokens to match patch tokens; "
                f"got {latent.shape[1]} vs expected {expected_tokens}"
            )

        patch_tokens = self.from_latent(latent)  # (B, tokens, channels)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(
            latent.shape[0], self._patch_channel_total, Hp, Wp
        )

        outputs: Dict[str, torch.Tensor] = {}
        channel_cursor = 0
        for name in self.field_order:
            ch = self.field_channels[name] * self.patch * self.patch
            patch_feat = patch_tokens[:, channel_cursor : channel_cursor + ch]
            channel_cursor += ch
            # Undo the stem residual (since the stem is near identity)
            grid_feat = self.pixel_shuffle(patch_feat)
            B = latent.shape[0]
            grid_feat = grid_feat.view(B, self.field_channels[name], H * W)
            outputs[name] = grid_feat.transpose(1, 2)  # (B, N, C)
        return outputs

    def _encode_fields(
        self,
        fields: Dict[str, torch.Tensor],
        coords: torch.Tensor,
        grid_shape: Tuple[int, int],
    ) -> Tuple[int, torch.Tensor]:
        H, W = grid_shape
        processed = []
        B = None
        for name in self.field_order:
            if name not in fields:
                raise KeyError(f"Missing field '{name}' for GridEncoder")
            tensor = fields[name]
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)  # add batch
            if tensor.dim() != 3:
                raise ValueError(f"Field '{name}' must have shape (B, N, C)")
            batch, N, C = tensor.shape
            if N != H * W or C != self.field_channels[name]:
                raise ValueError(
                    f"Field '{name}' expected shape (B, {H*W}, {self.field_channels[name]}), "
                    f"got {tensor.shape}"
                )
            if B is None:
                B = batch
            elif batch != B:
                raise ValueError("All fields must share the same batch size")
            tensor = tensor.transpose(1, 2).reshape(batch, C, H, W)
            tensor = self.pixel_unshuffle(tensor)
            stem = self.per_field_stems[name]
            tensor = stem(tensor) + tensor
            processed.append(tensor)

        encoded = torch.cat(processed, dim=1)
        if self.use_fourier and coords is not None and coords.numel() > 0:
            fourier = self._fourier_features(coords, grid_shape, B)
            encoded = torch.cat([encoded, fourier], dim=1)

        B = encoded.shape[0]
        Hp, Wp = encoded.shape[2], encoded.shape[3]
        tokens = Hp * Wp
        encoded = encoded.flatten(2).transpose(1, 2)  # (B, tokens, channels)
        return tokens, encoded

    def _fourier_features(
        self, coords: torch.Tensor, grid_shape: Tuple[int, int], batch: int
    ) -> torch.Tensor:
        H, W = grid_shape
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
        if coords.shape[0] != batch:
            raise ValueError("coords batch dimension mismatch")
        coord_dim = coords.shape[-1]
        if coord_dim == 0:
            return torch.zeros(batch, 0, H // self.patch, W // self.patch, device=coords.device, dtype=coords.dtype)

        if H % self.patch != 0 or W % self.patch != 0:
            raise ValueError("Grid shape not divisible by patch size when computing Fourier features")

        Hp = H // self.patch
        Wp = W // self.patch
        coords = coords.view(batch, H, W, coord_dim)
        coords = coords.view(batch, Hp, self.patch, Wp, self.patch, coord_dim)
        patch_centers = coords.mean(dim=(2, 4))  # (B, Hp, Wp, coord_dim)
        patch_centers = patch_centers.permute(0, 3, 1, 2).contiguous()  # (B, coord_dim, Hp, Wp)

        if self.fourier_frequencies.numel() == 0:
            return torch.zeros(batch, 0, Hp, Wp, device=coords.device, dtype=coords.dtype)

        centers = patch_centers.unsqueeze(2)  # (B, coord_dim, 1, Hp, Wp)
        freq = self.fourier_frequencies.view(1, 1, -1, 1, 1)
        two_pi = 2.0 * torch.pi
        angles = two_pi * centers * freq  # (B, coord_dim, F, Hp, Wp)
        sin_feat = torch.sin(angles).reshape(batch, -1, Hp, Wp)
        cos_feat = torch.cos(angles).reshape(batch, -1, Hp, Wp)
        return torch.cat([sin_feat, cos_feat], dim=1)

    def _adaptive_token_pool(self, tokens: torch.Tensor, target_len: int) -> torch.Tensor:
        if tokens.shape[1] == target_len:
            return tokens
        tokens_t = tokens.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(tokens_t, target_len)
        return pooled.transpose(1, 2)

    @staticmethod
    def _infer_grid_shape(meta: Optional[Mapping[str, object]]) -> Tuple[int, int]:
        if meta is None or "grid_shape" not in meta:
            raise ValueError("GridEncoder requires meta['grid_shape'] to reshape fields")
        grid_shape = meta["grid_shape"]
        if not isinstance(grid_shape, Iterable):
            raise TypeError("grid_shape metadata must be an iterable of length 2")
        H, W = list(grid_shape)
        return int(H), int(W)

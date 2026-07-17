import pytest
import torch

from ups.io import CanonicalPointEncoder, CanonicalPointEncoderConfig


def _encoder(*, supernodes: int = 32) -> CanonicalPointEncoder:
    torch.manual_seed(17)
    return CanonicalPointEncoder(
        CanonicalPointEncoderConfig(
            latent_len=6,
            latent_dim=16,
            hidden_dim=16,
            coord_dim=2,
            field_channels={"velocity": 2, "pressure": 1},
            supernodes=supernodes,
            supernode_neighbors=8,
            transformer_layers=1,
            num_heads=4,
        )
    ).eval()


def _state(nodes: int = 24) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    coords = torch.rand(2, nodes, 2, generator=generator)
    x, y = coords.unbind(dim=-1)
    fields = {
        "velocity": torch.stack(
            [torch.sin(2.0 * torch.pi * x), torch.cos(2.0 * torch.pi * y)], dim=-1
        ),
        "pressure": (x.square() + y.square()).unsqueeze(-1),
    }
    return fields, coords


def test_one_encoder_accepts_regular_grid_and_irregular_point_sets() -> None:
    encoder = _encoder()
    axis = torch.linspace(0.0, 1.0, 4)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 2)
    x, y = grid.unbind(dim=-1)
    grid_fields = {
        "velocity": torch.stack([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * y)], -1),
        "pressure": (x.square() + y.square()).unsqueeze(-1),
    }
    irregular_fields, irregular = _state(nodes=19)

    grid_latent = encoder(grid_fields, grid)
    irregular_latent = encoder(
        {name: value[:1] for name, value in irregular_fields.items()}, irregular[:1]
    )

    assert grid_latent.shape == (1, 6, 16)
    assert irregular_latent.shape == (1, 6, 16)


def test_encoding_is_invariant_to_point_storage_order() -> None:
    encoder = _encoder(supernodes=32)
    fields, coords = _state()
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(3))

    original = encoder(fields, coords)
    permuted = encoder(
        {name: value[:, permutation] for name, value in fields.items()},
        coords[:, permutation],
    )

    torch.testing.assert_close(original, permuted, rtol=2e-5, atol=2e-6)


def test_geometry_supernode_path_is_invariant_to_point_storage_order() -> None:
    encoder = _encoder(supernodes=7)
    fields, coords = _state()
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(5))

    original = encoder(fields, coords)
    permuted = encoder(
        {name: value[:, permutation] for name, value in fields.items()},
        coords[:, permutation],
    )

    torch.testing.assert_close(original, permuted, rtol=3e-5, atol=3e-6)


def test_encoder_fails_closed_on_field_semantic_mismatch() -> None:
    encoder = _encoder()
    fields, coords = _state()
    fields["density"] = fields.pop("pressure")

    with pytest.raises(ValueError, match="field schema mismatch"):
        encoder(fields, coords)

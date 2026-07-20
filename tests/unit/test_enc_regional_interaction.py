import pytest
import torch

from ups.io import RegionalInteractionEncoder, RegionalInteractionEncoderConfig


def _encoder(*, latent_len: int = 6) -> RegionalInteractionEncoder:
    torch.manual_seed(17)
    return RegionalInteractionEncoder(
        RegionalInteractionEncoderConfig(
            latent_len=latent_len,
            latent_dim=16,
            hidden_dim=16,
            coord_dim=2,
            field_channels={"velocity": 2, "pressure": 1},
            physical_neighbors=8,
            processor_neighbors=(2, 4, 5),
        )
    ).eval()


def _state(nodes: int = 24) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    coords = torch.rand(2, nodes, 2, generator=generator)
    x, y = coords.unbind(dim=-1)
    fields = {
        "velocity": torch.stack(
            [torch.sin(2.0 * torch.pi * x), torch.cos(2.0 * torch.pi * y)], dim=-1
        ),
        "pressure": (x.square() + y.square()).unsqueeze(-1),
    }
    measure = torch.ones(2, nodes, 1)
    return fields, coords, measure


def test_regional_encoder_accepts_regular_and_irregular_point_sets() -> None:
    encoder = _encoder()
    axis = (torch.arange(4, dtype=torch.float32) + 0.5) / 4
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 2)
    x, y = grid.unbind(dim=-1)
    grid_fields = {
        "velocity": torch.stack([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * y)], -1),
        "pressure": (x.square() + y.square()).unsqueeze(-1),
    }
    fields, irregular, measure = _state(nodes=19)

    grid_latent = encoder(grid_fields, grid, geom={"measure": torch.ones(1, grid.shape[1], 1)})
    irregular_latent = encoder(
        {name: value[:1] for name, value in fields.items()},
        irregular[:1],
        geom={"measure": measure[:1]},
    )

    assert grid_latent.shape == (1, 6, 16)
    assert irregular_latent.shape == (1, 6, 16)


def test_regional_encoder_is_invariant_to_point_storage_order_with_grid_ties() -> None:
    encoder = _encoder()
    axis = (torch.arange(5, dtype=torch.float32) + 0.5) / 5
    coords = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(1, -1, 2)
    x, y = coords.unbind(dim=-1)
    fields = {
        "velocity": torch.stack([torch.sin(2 * torch.pi * x), torch.cos(2 * torch.pi * y)], -1),
        "pressure": (x.square() + y.square()).unsqueeze(-1),
    }
    measure = torch.ones(1, coords.shape[1], 1)
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(5))

    original = encoder(fields, coords, geom={"measure": measure})
    permuted = encoder(
        {name: value[:, permutation] for name, value in fields.items()},
        coords[:, permutation],
        geom={"measure": measure[:, permutation]},
    )

    torch.testing.assert_close(original, permuted, rtol=0.0, atol=1e-6)


def test_region_slots_have_stable_geometric_semantics_across_warped_mesh() -> None:
    encoder = _encoder(latent_len=8)
    axis = (torch.arange(18, dtype=torch.float32) + 0.5) / 18
    u, v = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack([u, v], dim=-1).reshape(1, -1, 2)
    scale = 1.0 / (2.0 * torch.pi)
    mesh = torch.stack(
        [
            u + 0.24 * scale * torch.sin(2 * torch.pi * u) * torch.sin(2 * torch.pi * v),
            v - 0.17 * scale * torch.sin(2 * torch.pi * u) * torch.sin(2 * torch.pi * v),
        ],
        dim=-1,
    ).reshape(1, -1, 2)

    def slotted(coords: torch.Tensor) -> torch.Tensor:
        measure = torch.ones(1, coords.shape[1], 1) / coords.shape[1]
        order = encoder._canonical_point_order(coords)
        ordered_coords = encoder._gather_nodes(coords, order)
        ordered_measure = encoder._gather_nodes(measure, order)
        indices = encoder._farthest_point_indices(ordered_coords, ordered_measure)
        selected = encoder._gather_nodes(ordered_coords, indices)
        return encoder._gather_nodes(
            selected, encoder._assign_region_slots(selected, ordered_coords)
        )

    matched_distance = (slotted(grid) - slotted(mesh)).square().sum(dim=-1).sqrt().mean()

    assert float(matched_distance) < 0.10


def test_regional_encoder_fails_closed_on_schema_and_measure() -> None:
    encoder = _encoder()
    fields, coords, measure = _state()

    with pytest.raises(ValueError, match=r"requires geom\['measure'\]"):
        encoder(fields, coords)

    fields["density"] = fields.pop("pressure")
    with pytest.raises(ValueError, match="field schema mismatch"):
        encoder(fields, coords, geom={"measure": measure})


def test_measure_affects_physical_to_regional_messages() -> None:
    encoder = _encoder()
    fields, coords, measure = _state()
    weighted = measure.clone()
    weighted[:, : coords.shape[1] // 2] *= 8.0

    uniform_latent = encoder(fields, coords, geom={"measure": measure})
    weighted_latent = encoder(fields, coords, geom={"measure": weighted})

    assert not torch.allclose(uniform_latent, weighted_latent)


def test_geometry_interface_preserves_latent_and_returns_positive_masses() -> None:
    encoder = _encoder()
    fields, coords, measure = _state()

    expected = encoder(fields, coords, geom={"measure": measure})
    latent, regional_coords, regional_measure = encoder.forward_with_geometry(
        fields, coords, geom={"measure": measure}
    )

    torch.testing.assert_close(latent, expected, rtol=0.0, atol=0.0)
    assert regional_coords.shape == (2, 6, 2)
    assert regional_measure.shape == (2, 6, 1)
    assert torch.all(regional_measure > 0)
    torch.testing.assert_close(regional_measure.sum(dim=1), torch.ones(2, 1), rtol=0.0, atol=1e-6)


def test_geometry_interface_is_invariant_to_physical_storage_order() -> None:
    encoder = _encoder()
    fields, coords, measure = _state()
    permutation = torch.randperm(coords.shape[1], generator=torch.Generator().manual_seed(19))

    original = encoder.forward_with_geometry(fields, coords, geom={"measure": measure})
    permuted = encoder.forward_with_geometry(
        {name: value[:, permutation] for name, value in fields.items()},
        coords[:, permutation],
        geom={"measure": measure[:, permutation]},
    )

    for left, right in zip(original, permuted):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-6)


def test_challenger_contains_graph_messages_but_no_attention_or_latent_queries() -> None:
    encoder = _encoder()
    parameter_names = set(dict(encoder.named_parameters()))

    assert any(name.startswith("physical_edge_mlp") for name in parameter_names)
    assert any(name.startswith("processor.0.edge_mlp") for name in parameter_names)
    assert not any("attention" in name or "query" in name for name in parameter_names)

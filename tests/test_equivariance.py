import torch
from en_transformer.utils import rot
from en_transformer import EnTransformer

torch.set_default_dtype(torch.float64)

def test_readme():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        dim_head = 64,
        heads = 8,
        edge_dim = 4,
        fourier_features = 2,
        num_nearest_neighbors = 6
    )

    feats = torch.randn(1, 32, 512)
    coors = torch.randn(1, 32, 3)
    edges = torch.randn(1, 32, 1024, 4)

    mask = torch.ones(1, 32).bool()

    feats, coors = model(feats, coors, edges, mask = mask)
    assert True, 'it runs'

def test_equivariance():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        fourier_features = 2
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = model(feats, coors @ R + T, edges)
    feats2, coors2 = model(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_equivariance_with_nearest_neighbors():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        fourier_features = 2,
        num_nearest_neighbors = 5
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)
    edges = torch.randn(1, 16, 16, 4)

    feats1, coors1 = model(feats, coors @ R + T, edges)
    feats2, coors2 = model(feats, coors, edges)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_depth():
    model = EnTransformer(
        dim = 64,
        depth = 12,
        edge_dim = 4,
        fourier_features = 2
    )

    feats = torch.randn(1, 128, 64)
    coors = torch.randn(1, 128, 3)
    edges = torch.randn(1, 128, 128, 4)

    feats, coors = model(feats, coors, edges)

    assert not torch.any(torch.isnan(feats)), 'no NaN in features'
    assert not torch.any(torch.isnan(coors)), 'no NaN in coordinates'

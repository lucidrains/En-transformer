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
        neighbors = 6
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
        rel_pos_emb = True
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

def test_equivariance_with_cross_product():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        edge_dim = 4,
        rel_pos_emb = True,
        use_cross_product = True
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
        neighbors = 5
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

def test_equivariance_with_sparse_neighbors():
    model = EnTransformer(
        dim = 512,
        depth = 1,
        heads = 4,
        dim_head = 32,
        neighbors = 0,
        only_sparse_neighbors = True
    )

    R = rot(*torch.rand(3))
    T = torch.randn(1, 1, 3)

    feats = torch.randn(1, 16, 512)
    coors = torch.randn(1, 16, 3)

    i = torch.arange(feats.shape[1])
    adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

    feats1, coors1 = model(feats, coors @ R + T, adj_mat = adj_mat)
    feats2, coors2 = model(feats, coors, adj_mat = adj_mat)

    assert torch.allclose(feats1, feats2, atol = 1e-6), 'type 0 features are invariant'
    assert torch.allclose(coors1, (coors2 @ R + T), atol = 1e-6), 'type 1 features are equivariant'

def test_depth():
    model = EnTransformer(
        dim = 8,
        depth = 12,
        edge_dim = 4,
        neighbors = 16
    )

    feats = torch.randn(1, 128, 8)
    coors = torch.randn(1, 128, 3)
    edges = torch.randn(1, 128, 128, 4)

    feats, coors = model(feats, coors, edges)

    assert not torch.any(torch.isnan(feats)), 'no NaN in features'
    assert not torch.any(torch.isnan(coors)), 'no NaN in coordinates'

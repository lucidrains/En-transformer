import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def fourier_encode_dist(x, num_encodings = 4, include_self = True):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x
    scales = 2 ** torch.arange(num_encodings, device = device, dtype = dtype)
    x = x / scales
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim = -1) if include_self else x
    return x

# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, **kwargs):
        feats_out, coors = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, coors, **kwargs):
        feats = self.norm(feats)
        feats, coors = self.fn(feats, coors, **kwargs)
        return feats, coors

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors):
        return self.net(feats), coors

class EquivariantAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0
    ):
        super().__init__()
        self.fourier_features = fourier_features

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        edge_input_dim = (fourier_features * 2) + (dim_head * 2) + edge_dim + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, edge_input_dim * 2),
            nn.ReLU(),
            nn.Linear(edge_input_dim * 2, m_dim),
            nn.ReLU()
        )

        self.to_attn_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1),
            Rearrange('... () -> ...')
        )

        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.ReLU(),
            nn.Linear(m_dim * 4, 1),
            Rearrange('... () -> ...')
        )

    def forward(self, feats, coors, basis, edges = None):
        b, n, d, h, fourier_features, device = *feats.shape, self.heads, self.fourier_features, feats.device

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = (rel_coors ** 2).sum(dim = -1, keepdim = True)

        if fourier_features > 0:
            rel_dist = fourier_encode_dist(rel_dist, num_encodings = fourier_features)
            rel_dist = rearrange(rel_dist, 'b i j () d -> b i j d')

        rel_dist = repeat(rel_dist, 'b i j d -> b h i j d', h = h)

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # expand queries and keys for concatting

        q = repeat(q, 'b h i d -> b h i n d', n = n)
        k = repeat(k, 'b h j d -> b h n j d', n = n)

        edge_input = torch.cat((q, k, rel_dist), dim = -1)

        if exists(edges):
            edges = repeat(edges, 'b i j d -> b h i j d', h = h)
            edge_input = torch.cat((edge_input, edges), dim = -1)

        m_ij = self.edge_mlp(edge_input)

        coor_weights = self.coors_mlp(m_ij)
        coors_out = einsum('b h i j, b i j c -> b i c', coor_weights, basis) + coors

        # derive attention

        sim = self.to_attn_mlp(m_ij)
        attn = sim.softmax(dim = -1)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out

# transformer

class EnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        edge_dim = 0,
        m_dim = 16,
        fourier_features = 0
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, EquivariantAttention(dim = dim, dim_head = dim_head, heads = heads, m_dim = m_dim, edge_dim = edge_dim, fourier_features = fourier_features))),
                Residual(PreNorm(dim, FeedForward(dim = dim)))
            ]))

    def forward(self, feats, coors, edges = None):
        basis = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')

        for attn, ff in self.layers:
            feats, coors = attn(feats, coors, basis = basis, edges = edges)
            feats, coors = ff(feats, coors)

        return feats, coors

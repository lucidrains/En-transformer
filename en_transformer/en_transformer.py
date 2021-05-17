import torch
import torch.nn.functional as F
from torch import nn, einsum

from en_transformer.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helper functions

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def default(val, d):
    return val if exists(val) else d

def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes

# this follows the same strategy for normalization as done in SE3 Transformers
# https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/se3_transformer_pytorch.py#L95

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.fn = nn.LayerNorm(1)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        phase = self.fn(norm)
        return (phase * normed_coors)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, coors, **kwargs):
        feats = self.norm(feats)
        feats, coors = self.fn(feats, coors, **kwargs)
        return feats, coors

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, **kwargs):
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

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
            nn.Linear(dim, dim * 4 * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0

class EquivariantAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 4,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        rel_pos_emb = None,
        edge_mlp_mult = 2,
        coor_attention = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        self.edge_mlp = None
        has_edges = edge_dim > 0

        if has_edges:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden),
                nn.GELU(),
                nn.Linear(edge_hidden, heads)
            )

            self.coors_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(heads, heads)
            )
        else:
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads)
            )

        self.coors_gate = nn.Sequential(
            nn.Linear(heads, heads),
            nn.Tanh()
        )

        self.norm_rel_coors = CoorsNorm()
        self.coors_combine = nn.Parameter(torch.randn(heads))

        self.rotary_emb = SinusoidalEmbeddings(dim_head // 2)
        self.rotary_emb_seq = SinusoidalEmbeddings(dim_head // 2) if rel_pos_emb else None

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None
    ):
        b, n, d, h, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = *feats.shape, self.heads, self.neighbors, self.only_sparse_neighbors, self.valid_neighbor_radius, feats.device

        assert not (only_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix must be passed in if only_sparse_neighbors is turned on'

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(p = 2, dim = -1)

        # calculate neighborhood indices

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist.clone()

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

            self_mask = torch.eye(n, device = device).bool()
            self_mask = rearrange(self_mask, 'i j -> () i j')
            adj_mat.masked_fill_(self_mask, False)

            max_adj_neighbors = adj_mat.long().sum(dim = -1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.)

        if num_nn > 0:
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim = -1, largest = False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate nearest neighbors

        i = j = n

        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            nbhd_indices_with_heads = repeat(nbhd_indices, 'b n d -> b h n d', h = h)
            k         = batched_index_select(k, nbhd_indices_with_heads, dim = 2)
            v         = batched_index_select(v, nbhd_indices_with_heads, dim = 2)
            rel_dist  = batched_index_select(rel_dist, nbhd_indices, dim = 2)
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
        else:
            k = repeat(k, 'b h j d -> b h n j d', n = n)
            v = repeat(v, 'b h j d -> b h n j d', n = n)

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, 'b i -> b () i ()')
            k_mask = repeat(mask, 'b j -> b i j', i = n)

            if exists(nbhd_indices):
                k_mask = batched_index_select(k_mask, nbhd_indices, dim = 2)

            k_mask = rearrange(k_mask, 'b i j -> b () i j')

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, 'b i j -> b () i j')

        # generate and apply rotary embeddings

        rot_null = torch.zeros_like(rel_dist)

        q_pos_emb_rel_dist = self.rotary_emb(torch.zeros(n, device = device))
        k_pos_emb_rel_dist = self.rotary_emb(rel_dist * 1e2)

        q_pos_emb = rearrange(q_pos_emb_rel_dist, 'i d -> () () i d')
        k_pos_emb = rearrange(k_pos_emb_rel_dist, 'b i j d -> b () i j d')

        if exists(self.rotary_emb_seq):
            pos_emb = self.rotary_emb_seq(torch.arange(n, device = device))

            q_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n d')
            k_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n () d')

            q_pos_emb = broadcat((q_pos_emb, q_pos_emb_seq), dim = -1)
            k_pos_emb = broadcat((k_pos_emb, k_pos_emb_seq), dim = -1)

        q = apply_rotary_pos_emb(q, q_pos_emb)
        k = apply_rotary_pos_emb(k, k_pos_emb)
        v = apply_rotary_pos_emb(v, k_pos_emb)

        # calculate inner product for queries and keys

        qk = einsum('b h i d, b h i j d -> b h i j', q, k) * (self.scale if not exists(edges) else 1)

        # add edge information and pass through edges MLP if needed

        if exists(edges):
            if exists(nbhd_indices):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            qk = rearrange(qk, 'b h i j -> b i j h')
            qk = torch.cat((qk, edges), dim = -1)
            qk = self.edge_mlp(qk)
            qk = rearrange(qk, 'b i j h -> b h i j')

        # coordinate MLP and calculate coordinate updates

        coors_mlp_input = rearrange(qk, 'b h i j -> b i j h')
        coor_weights = self.coors_mlp(coors_mlp_input)

        if exists(mask):
            mask_value = max_neg_value(coor_weights)
            coor_mask = repeat(mask, 'b () i j -> b i j ()')
            coor_weights.masked_fill_(~coor_mask, mask_value)

        coor_attn = coor_weights.softmax(dim = -2)

        rel_coors_sign = self.coors_gate(coors_mlp_input)
        rel_coors_sign = rearrange(rel_coors_sign, 'b i j h -> b i j () h')

        rel_coors = self.norm_rel_coors(rel_coors)

        rel_coors = repeat(rel_coors, 'b i j c -> b i j c h', h = h)

        rel_coors = rel_coors * rel_coors_sign

        coors_out = einsum('b i j h, b i j c h, h -> b i c', coor_attn, rel_coors, self.coors_combine)

        # derive attention

        sim = qk.clone()

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim.masked_fill_(~mask, mask_value)

        attn = sim.softmax(dim = -1)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)

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
        num_tokens = None,
        rel_pos_emb = False,
        dim_head = 64,
        heads = 8,
        num_edge_tokens = None,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3
    ):
        super().__init__()
        assert dim_head >= 32, 'your dimension per head should be greater than 32 for rotary embeddings to work well'
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, EquivariantAttention(dim = dim, dim_head = dim_head, heads = heads, coors_hidden_dim = coors_hidden_dim, edge_dim = (edge_dim + adj_dim),  neighbors = neighbors, only_sparse_neighbors = only_sparse_neighbors, valid_neighbor_radius = valid_neighbor_radius, init_eps = init_eps, rel_pos_emb = rel_pos_emb))),
                Residual(PreNorm(dim, FeedForward(dim = dim)))
            ]))

        self.eighbors = neighbors

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None,
        **kwargs
    ):
        b = feats.shape[0]

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.edge_emb):
            assert exists(edges), 'edges must be passed in as (batch x seq x seq) indicating edge type'
            edges = self.edge_emb(edges)

        assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), 'num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix'

        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        # main network

        for attn, ff in self.layers:
            feats, coors = attn(feats, coors, edges = edges, mask = mask, adj_mat = adj_mat)
            feats, coors = ff(feats, coors)

        return feats, coors

## E(n)-Equivariant Transformer

Implementation of E(n)-Equivariant Transformer, which extends the ideas from Welling's <a href="https://github.com/lucidrains/egnn-pytorch">E(n)-Equivariant Graph Neural Network</a> with attention.

## Install

```bash
$ pip install En-transformer
```

## Usage

```python
import torch
from en_transformer import EnTransformer

model = EnTransformer(
    dim = 512,
    depth = 4,             # depth
    dim_head = 64,         # dimension per head
    heads = 8,             # number of heads
    edge_dim = 4,          # dimension of edge feature
    neighbors = 64         # only do attention between coordinates N nearest neighbors - set to 0 to turn off
)

feats = torch.randn(1, 1024, 512)
coors = torch.randn(1, 1024, 3)
edges = torch.randn(1, 1024, 1024, 4)

mask = torch.ones(1, 1024).bool()

feats, coors = model(feats, coors, edges, mask = mask)  # (1, 16, 512), (1, 16, 3)
```

Letting the network take care of both atomic and bond type embeddings

```python
import torch
from en_transformer import EnTransformer

model = EnTransformer(
    num_tokens = 10,       # number of unique nodes, say atoms
    rel_pos_emb = True,    # set this to true if your sequence is not an unordered set. it will accelerate convergence
    num_edge_tokens = 5,   # number of unique edges, say bond types
    dim = 128,
    edge_dim = 16,
    depth = 3,
    heads = 4,
    dim_head = 32,
    neighbors = 8
)

atoms = torch.randint(0, 10, (1, 16))    # 10 different types of atoms
bonds = torch.randint(0, 5, (1, 16, 16)) # 5 different types of bonds (n x n)
coors = torch.randn(1, 16, 3)            # atomic spatial coordinates

feats_out, coors_out = model(atoms, coors, edges = bonds) # (1, 16, 512), (1, 16, 3)
```

If you would like to only attend to sparse neighbors, as defined by an adjacency matrix (say for atoms), you have to set one more flag and then pass in the `N x N` adjacency matrix.

```python
import torch
from en_transformer import EnTransformer

model = EnTransformer(
    num_tokens = 10,
    dim = 512,
    depth = 1,
    heads = 4,
    dim_head = 32,
    neighbors = 0,
    only_sparse_neighbors = True,    # must be set to true
    num_adj_degrees = 3,             # the number of degrees to derive from 1st degree neighbors passed in
    adj_dim = 8                      # whether to pass the adjacency degree information as an edge embedding
)

atoms = torch.randint(0, 10, (1, 16))
coors = torch.randn(1, 16, 3)

# naively assume a single chain of atoms
i = torch.arange(atoms.shape[1])
adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

# adjacency matrix must be passed in
feats_out, coors_out = model(atoms, coors, adj_mat = adj_mat) # (1, 16, 512), (1, 16, 3)
```

## Edges

If you need to pass in continuous edges

```python
import torch
from en_transformer import EnTransformer
from en_transformer.utils import rot

model = EnTransformer(
    dim = 512,
    depth = 1,
    heads = 4,
    dim_head = 32,
    edge_dim = 4,
    num_nearest_neighbors = 0,
    only_sparse_neighbors = True
)

feats = torch.randn(1, 16, 512)
coors = torch.randn(1, 16, 3)
edges = torch.randn(1, 16, 16, 4)

i = torch.arange(feats.shape[1])
adj_mat = (i[:, None] <= (i[None, :] + 1)) & (i[:, None] >= (i[None, :] - 1))

feats1, coors1 = model(feats, coors, adj_mat = adj_mat, edges = edges)
```

## Example

To run a protein backbone coordinate denoising toy task, first install `sidechainnet`

```bash
$ pip install sidechainnet
```

Then

```bash
$ python denoise.py
```

## Citations

```bibtex
@misc{satorras2021en,
    title 	= {E(n) Equivariant Graph Neural Networks}, 
    author 	= {Victor Garcia Satorras and Emiel Hoogeboom and Max Welling},
    year 	= {2021},
    eprint 	= {2102.09844},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

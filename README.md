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
    depth = 4,                         # depth
    dim_head = 64,                     # dimension per head
    heads = 8,                         # number of heads
    edge_dim = 4,                      # dimension of edge features
    fourier_features = 2,              # num fourier features to append to relative distance which goes into the edge MLP
    num_nearest_neighbors = 64         # only do attention between coordinates N nearest neighbors - set to 0 to turn off
)

feats = torch.randn(1, 1024, 512)
coors = torch.randn(1, 1024, 3)
edges = torch.randn(1, 1024, 1024, 4)

mask = torch.ones(1, 1024).bool()

feats, coors = model(feats, coors, edges, mask = mask)  # (1, 16, 512), (1, 16, 3)
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

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam

from einops import rearrange, repeat
import sidechainnet as scn
from en_transformer.en_transformer import EnTransformer

torch.set_default_dtype(torch.float64)

BATCH_SIZE = 1
GRADIENT_ACCUMULATE_EVERY = 16

def cycle(loader, len_thres = 200):
    while True:
        for data in loader:
            if data.seqs.shape[1] > len_thres:
                continue
            yield data

transformer = EnTransformer(
    num_tokens = 21,
    dim = 32,
    dim_head = 64,
    heads = 4,
    depth = 4,
    rel_pos_emb = True, # there is inherent order in the sequence (backbone atoms of amino acid chain)
    neighbors = 16
)

data = scn.load(
    casp_version = 12,
    thinning = 30,
    with_pytorch = 'dataloaders',
    batch_size = BATCH_SIZE,
    dynamic_batching = False
)

dl = cycle(data['train'])
optim = Adam(transformer.parameters(), lr=1e-3)
transformer = transformer.cuda()

for _ in range(10000):
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        batch = next(dl)
        seqs, coords, masks = batch.seqs, batch.crds, batch.msks

        seqs = seqs.cuda().argmax(dim = -1)
        coords = coords.cuda().type(torch.float64)
        masks = masks.cuda().bool()

        l = seqs.shape[1]
        coords = rearrange(coords, 'b (l s) c -> b l s c', s = 14)

        # keeping only the backbone coordinates

        coords = coords[:, :, 0:3, :]
        coords = rearrange(coords, 'b l s c -> b (l s) c')

        seq = repeat(seqs, 'b n -> b (n c)', c = 3)
        masks = repeat(masks, 'b n -> b (n c)', c = 3)

        noised_coords = coords + torch.randn_like(coords)

        feats, denoised_coords = transformer(seq, noised_coords, mask = masks)

        loss = F.mse_loss(denoised_coords[masks], coords[masks])

        (loss / GRADIENT_ACCUMULATE_EVERY).backward()

    print('loss:', loss.item())
    optim.step()
    optim.zero_grad()

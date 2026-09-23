"""
GTR: Gated Token Recurrence for Efficient Dense Prediction
Copyright (c) 2026 The GTR Authors. All Rights Reserved.
"""

import torch
from torch.utils.data import DistributedSampler


class ChunkedDistributedSampler(DistributedSampler):
    """Anti-preemption sampler: each base permutation is split into `chunks`
    consecutive segments and every (short) epoch consumes one segment, so a
    mid-epoch preemption loses at most 1/chunks of an epoch of work.

    Short epoch `e` uses segment `e % chunks` of the permutation seeded with
    `seed + e // chunks` — the same permutation contract as DistributedSampler —
    so `chunks` consecutive short epochs replay one plain-sampler epoch sample
    for sample. The segment length is floored to a multiple of num_replicas so
    every segment boundary is rank-aligned: per rank, the concatenated
    short-epoch streams form an elementwise prefix of the plain sampler's
    stream. Tail lost per base epoch: len(dataset) - chunks * chunk_len
    (< chunks * num_replicas samples).
    """

    def __init__(self, dataset, chunks, **kwargs):
        super().__init__(dataset, **kwargs)
        assert self.shuffle, 'ChunkedDistributedSampler requires shuffle=True'
        assert int(chunks) >= 1, f'chunks must be >= 1, got {chunks}'
        self.chunks = int(chunks)
        self.chunk_len = len(self.dataset) // self.chunks // self.num_replicas * self.num_replicas
        assert self.chunk_len > 0, 'dataset smaller than chunks * num_replicas'
        # Parent __len__ / set_epoch are reused; only the sizes change.
        self.num_samples = self.chunk_len // self.num_replicas
        self.total_size = self.chunk_len

    def __iter__(self):
        base_epoch, chunk = divmod(self.epoch, self.chunks)
        g = torch.Generator()
        g.manual_seed(self.seed + base_epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        indices = indices[chunk * self.chunk_len:(chunk + 1) * self.chunk_len]
        indices = indices[self.rank::self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

"""P x K identity sampler that spreads each identity's K images over as many cameras as possible.

camera_id is used ONLY here, to decide which image indices go into a batch (allowed by the jury: "only for
organizing training (batch sampling) and local validation"). The sampler returns plain row indices;
camera_id never reaches the model, the losses or inference - those see only pixels and vehicle labels.

Per epoch, for every identity:
  1. its images are ordered camera-round-robin: cameras in random order, one random image from each camera
     in turn, so any K consecutive images cover min(K, n_cameras) distinct cameras;
  2. this order is cut into ceil(n / K) groups of K; the last group is topped up from a fresh round-robin
     order (no duplicates inside a group while n >= K);
then groups of P distinct identities form a batch (like torchreid's RandomIdentitySampler).
With a plain random choice of K images, same-camera positive pairs are frequent (many ids have 2-3 cameras);
here the hardest positive mined by the triplet loss is far more often a cross-camera one, which is exactly the
case the evaluation cares about (same-camera pairs are junk there).
"""

from collections import defaultdict

import numpy as np


class CameraAwarePKSampler:
    def __init__(self, labels, cameras, P=16, K=4, seed=0, camera_aware=True):
        self.P, self.K, self.seed, self.camera_aware = P, K, seed, camera_aware
        self.cameras = np.asarray(cameras)
        self.by_id = defaultdict(lambda: defaultdict(list))
        for i, (lab, cam) in enumerate(zip(labels, cameras, strict=False)):
            self.by_id[int(lab)][int(cam)].append(i)
        self.labels = np.asarray(labels)

    def _order(self, rng, cams):
        if not self.camera_aware:
            return list(rng.permutation(np.concatenate(list(cams.values()))))
        queues = [list(rng.permutation(cams[c])) for c in rng.permutation(list(cams))]
        order = []
        while any(queues):
            order += [q.pop() for q in queues if q]
        return order

    def batches(self, epoch):
        rng = np.random.default_rng([self.seed, epoch])
        groups = defaultdict(list)
        for lab, cams in self.by_id.items():
            order = self._order(rng, cams)
            n_groups = -(-len(order) // self.K)
            missing = n_groups * self.K - len(order)
            if missing:
                tail = set(order[len(order) - len(order) % self.K :])
                extra = [i for i in self._order(rng, cams) if i not in tail]
                if len(extra) < missing:  # fewer than K images in total: repeats are unavoidable
                    extra += list(rng.choice(order, missing))
                order += extra[:missing]
            groups[lab] = [order[g * self.K : (g + 1) * self.K] for g in range(n_groups)]
        out = []
        avail = [lab for lab in groups if groups[lab]]
        while len(avail) >= self.P:
            chosen = rng.choice(avail, self.P, replace=False)
            out.append([i for lab in chosen for i in groups[lab].pop()])
            avail = [lab for lab in avail if groups[lab]]
        return [np.asarray(out[i]) for i in rng.permutation(len(out))]

    def cross_camera_pair_rate(self, batches):
        """Share of same-id pairs inside batches that come from different cameras (diagnostics only)."""
        cross = total = 0
        for b in batches:
            lab, cam = self.labels[b], self.cameras[b]
            same = np.triu(lab[:, None] == lab[None, :], 1)
            total += same.sum()
            cross += (same & (cam[:, None] != cam[None, :])).sum()
        return cross / total

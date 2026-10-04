"""Mesh-free encoder with a learnable grid base and multi-resolution training (Study 10).

The shared base logits are a learnable tensor of shape (n, G, G) on a regular grid over the unit square, sampled bilinearly at the
node coordinates of whatever mesh the model is applied to; nothing is indexed by a mesh node. `fit_multi` trains on problems from
several meshes at once (each problem keeps its own Whitney operators), so the basis sees more than one discretisation.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from compat_galerkin import lift, oracle

from .encoder_whitney import EncoderWhitney, LogitCNN


class GridBaseEncoder(EncoderWhitney):
    mesh_free = True

    def __init__(self, *args, grid=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.G = grid
        self.name = f"grid_base_encoder_n{self.n}"

    def _init_base(self):
        """Lloyd rbf start (as `oracle.init_logits`) evaluated on the grid points."""
        p, rng = self.p0, np.random.default_rng(self.start[1])
        pts = p.coords[p.interior_nodes]
        c = pts[rng.choice(len(pts), self.n, replace=False)]
        for _ in range(10):
            lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), axis=1)
            c = np.stack([pts[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(self.n)])
        d2 = ((p.coords[:, None] - c[None]) ** 2).sum(-1)
        s2 = np.mean(np.sort(d2, axis=1)[:, 1]) + 1e-12
        xs = np.linspace(0.0, 1.0, self.G)
        X, Y = np.meshgrid(xs, xs, indexing="ij")  # arrays [x index, y index]
        gp = np.stack([X, Y], axis=-1)
        gd2 = ((gp[:, :, None, :] - c[None, None]) ** 2).sum(-1)  # (G, G, n)
        base = torch.as_tensor(-gd2 / (2 * s2), dtype=lift.DTYPE).permute(2, 0, 1).contiguous()  # (n, G, G)
        self.grid_base = base.clone().requires_grad_(True)
        self.xy64 = torch.as_tensor(p.coords, dtype=lift.DTYPE)

    def _base_params(self):
        return [self.grid_base]

    def _freeze_base(self):
        self.grid_base = self.grid_base.detach()

    def base_logits(self, xy=None):
        xy = self.xy64 if xy is None else xy
        grid = torch.stack([2 * xy[:, 1] - 1, 2 * xy[:, 0] - 1], dim=-1)[None, None]  # (1, 1, N, 2): (y, x)
        out = F.grid_sample(self.grid_base[None], grid, mode="bilinear", padding_mode="border", align_corners=True)  # (1, n, 1, N)
        return out[0, :, 0, :].T

    def fit_multi(self, parts, **kwargs):
        """parts: list of (family, dataset) pairs on different meshes; the first part sets the reference mesh used by `predict`."""
        t0 = time.time()
        torch.manual_seed(self.seed)
        probs, fields = [], []
        for fam, ds in parts:
            for inst, e in zip(ds.instances, ds.e_ref):
                probs.append(oracle.OracleProblem.from_instance(fam.cavity(inst), inst, e))
                fields.append(fam.grid_inputs(inst))
        self.p0 = probs[0]
        X = np.stack(fields)
        self.mu, self.sd = X.mean(axis=(0, 2, 3), keepdims=True), X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        X = torch.as_tensor((X - self.mu) / self.sd)

        def node_grid(p):
            xy = torch.as_tensor(p.coords, dtype=torch.float32)
            return (2 * torch.stack([xy[:, 1], xy[:, 0]], dim=-1) - 1)[None, None]

        grids = [node_grid(p) for p in probs]
        self.node_grid = grids[0]
        self.net = LogitCNN(X.shape[1], self.n, self.width)
        self._init_base()

        def basis(i, corr_k):
            p = probs[i]
            samp = F.grid_sample(corr_k[None], grids[i], mode="bilinear", padding_mode="border", align_corners=False)
            logits = self.base_logits(torch.as_tensor(p.coords, dtype=lift.DTYPE)) + samp[0, :, 0, :].T.to(lift.DTYPE)
            return lift.whitney_lift(lift.make_pou(logits, p.boundary_mask), p.A, p.G)[p.idx]

        rng = np.random.default_rng(self.seed)
        for kind, n_steps in self.schedule:
            loss_fn = self._error if kind == "galerkin" else self._proj_error
            opt = torch.optim.Adam([{"params": self._base_params(), "lr": self.lr_base}, {"params": self.net.parameters(), "lr": self.lr_cnn}])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
            for _ in range(n_steps):
                sel = rng.choice(len(probs), min(self.batch, len(probs)), replace=False)
                corr = self.net(X[sel])
                loss = torch.stack([loss_fn(probs[i], basis(i, corr[k])) for k, i in enumerate(sel)]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net.eval()
        with torch.no_grad():
            corr = self.net(X)
            errs = [float(self._error(probs[i], basis(i, corr[i]))) for i in range(len(probs))]
            s = torch.linalg.svdvals(basis(0, corr[0]))
        self._freeze_base()
        return {"dim": int((s > 1e-8 * s[0]).sum()), "params": sum(p.numel() for p in self.net.parameters()) + self.grid_base.numel(),
                "train_error": float(np.mean(errs)), "train_median": float(np.median(errs)), "fit_seconds": time.time() - t0}

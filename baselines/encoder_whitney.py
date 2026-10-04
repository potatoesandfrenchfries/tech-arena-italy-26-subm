"""Learned compatible basis with an encoder: a small CNN maps the instance to the partition of unity.

Same reduced space as `SharedWhitney` (Whitney lift of n softmax bumps, Cholesky-whitened, Galerkin solve),
but the nodal logits depend on the instance: logits = shared base + CNN correction sampled at the mesh nodes.
The CNN reads the same grid channels as the FNO (epsilon, omega^2, source, coordinates). Trained end to end
with the relative M-norm Galerkin error against the fine solutions (the supervision POD and the FNO also use).
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compat_galerkin import lift, oracle

from .base import Method


class LogitCNN(nn.Module):
    """Dilated 3x3 convolutions on a 2x-pooled grid; the last layer is zero-initialised, so training starts at the base W."""

    def __init__(self, cin, n, width=24, dilations=(1, 2, 4, 8)):
        super().__init__()
        chans = [cin] + [width] * len(dilations)
        self.body = nn.ModuleList(nn.Conv2d(a, b, 3, padding=d, dilation=d) for a, b, d in zip(chans[:-1], chans[1:], dilations))
        self.out = nn.Conv2d(width, n, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x = F.avg_pool2d(x, 2)
        for conv in self.body:
            x = F.gelu(conv(x))
        return self.out(x)


class EncoderWhitney(Method):
    def __init__(self, n=8, steps=1000, batch=8, lr_base=0.02, lr_cnn=2e-3, width=24, seed=0,
                 start=("rbf", 0), schedule=None):
        """schedule: list of (loss kind, steps) stages, kind in {"galerkin", "projection"}; default one Galerkin stage of `steps`."""
        self.n, self.steps, self.batch, self.lr_base, self.lr_cnn = n, steps, batch, lr_base, lr_cnn
        self.width, self.seed, self.start = width, seed, start
        self.schedule = schedule or [("galerkin", steps)]
        self.name = f"encoder_whitney_n{n}"

    def _logits(self, fields):
        """fields (B, 7, g, g) float32 -> logits (B, n_nodes, n) float64."""
        corr = self.net(fields)  # (B, n, g/2, g/2); array dims are [x index, y index]
        B = corr.shape[0]
        grid = self.node_grid.expand(B, -1, -1, -1)
        samp = F.grid_sample(corr, grid, mode="bilinear", padding_mode="border", align_corners=False)  # (B, n, 1, N)
        base = self.base_logits()
        return base[None] + samp[:, :, 0, :].permute(0, 2, 1).to(base.dtype)

    # The base logits are a hook: per-node trained logits here, an analytic mesh-free RBF in `MeshFreeEncoder`.
    def _init_base(self):
        self.base = oracle.init_logits(self.p0, self.n, *self.start).clone().requires_grad_(True)

    def _base_params(self):
        return [self.base]

    def _freeze_base(self):
        self.base = self.base.detach()

    def base_logits(self, xy=None):
        return self.base

    def _basis(self, logits):
        W = lift.make_pou(logits, self.p0.boundary_mask)
        return lift.whitney_lift(W, self.p0.A, self.p0.G)[self.p0.idx]

    @staticmethod
    def _error(p, V):
        Q = lift.compress_chol(V, p.M)
        e = lift.reduced_solve(Q, p.K, p.M, p.f, p.omega2)
        return p.mnorm(e - p.e_ref) / p.mnorm(p.e_ref)

    @staticmethod
    def _proj_error(p, V):
        """Relative M-norm error of the best approximation of the reference in span(V) (differentiable)."""
        Q = lift.compress_chol(V, p.M)
        MQ = torch.sparse.mm(p.M, Q)
        G = Q.T @ MQ
        proj = lambda x: Q @ torch.linalg.solve(G, MQ.T @ x)
        e = torch.complex(proj(p.e_ref.real), proj(p.e_ref.imag))
        return p.mnorm(e - p.e_ref) / p.mnorm(p.e_ref)

    @torch.no_grad()
    def projection_errors(self, family, dataset):
        """Best-approximation error of the predicted space for each instance of `dataset` (Study 7 diagnostic)."""
        out = []
        for inst, e_ref in zip(dataset.instances, dataset.e_ref):
            p = oracle.OracleProblem.from_instance(family.cavity(inst), inst, e_ref)
            logits = self._logits(self._prepare_inputs(family, [inst]))[0]
            out.append(float(self._proj_error(p, self._basis(logits))))
        return out

    def _prepare_inputs(self, family, instances):
        X = np.stack([family.grid_inputs(i) for i in instances])
        return torch.as_tensor((X - self.mu) / self.sd)

    def fit(self, family, train, device=None, **kwargs):
        t0 = time.time()
        torch.manual_seed(self.seed)
        probs = [oracle.OracleProblem.from_instance(family.cavity(i), i, e) for i, e in zip(train.instances, train.e_ref)]
        self.p0 = probs[0]
        X = np.stack([family.grid_inputs(i) for i in train.instances])
        self.mu, self.sd = X.mean(axis=(0, 2, 3), keepdims=True), X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        X = torch.as_tensor((X - self.mu) / self.sd)
        xy = torch.as_tensor(self.p0.coords, dtype=torch.float32)  # nodes in [0, 1]^2
        # grid_sample takes (width index, height index) = (y, x) for arrays stored [x index, y index]
        self.node_grid = (2 * torch.stack([xy[:, 1], xy[:, 0]], dim=-1) - 1)[None, None]  # (1, 1, N, 2)
        self.net = LogitCNN(X.shape[1], self.n, self.width)
        self._init_base()
        rng = np.random.default_rng(self.seed)
        for kind, n_steps in self.schedule:  # a fresh Adam optimiser and cosine schedule per stage
            loss_fn = self._error if kind == "galerkin" else self._proj_error
            opt = torch.optim.Adam([{"params": self._base_params(), "lr": self.lr_base}, {"params": self.net.parameters(), "lr": self.lr_cnn}])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps)
            for _ in range(n_steps):
                sel = rng.choice(len(probs), min(self.batch, len(probs)), replace=False)
                logits = self._logits(X[sel])
                loss = torch.stack([loss_fn(probs[i], self._basis(logits[k])) for k, i in enumerate(sel)]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net.eval()
        with torch.no_grad():
            logits = self._logits(X)
            errs = [float(self._error(p, self._basis(logits[k]))) for k, p in enumerate(probs)]
            s = torch.linalg.svdvals(self._basis(logits[0]))
        self._freeze_base()
        return {"dim": int((s > 1e-8 * s[0]).sum()),
                "params": sum(p.numel() for p in self.net.parameters()) + sum(p.numel() for p in self._base_params()),
                "train_error": float(np.mean(errs)), "train_median": float(np.median(errs)), "fit_seconds": time.time() - t0}

    @torch.no_grad()
    def predict(self, family, inst, cav):
        K, M = lift.to_torch_sparse(cav.interior(cav.K)), lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[cav.interior_edges])
        logits = self._logits(self._prepare_inputs(family, [inst]))[0]
        Q = lift.compress_chol(self._basis(logits), M)
        e = np.zeros(cav.n_edges, dtype=complex)
        e[cav.interior_edges] = lift.reduced_solve(Q, K, M, f, inst.omega2).numpy()
        return e


class MeshFreeEncoder(EncoderWhitney):
    """Encoder with no per-node parameters: the base logits are analytic RBFs (trained centres and widths) of the coordinates.

    Initialised from the same Lloyd rbf start as `EncoderWhitney`, so it can be evaluated on any mesh by passing that
    mesh's node coordinates to `base_logits` (Study 6a).
    """

    mesh_free = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = f"mesh_free_encoder_n{self.n}"

    def _init_base(self):
        p, rng = self.p0, np.random.default_rng(self.start[1])
        pts = p.coords[p.interior_nodes]
        c = pts[rng.choice(len(pts), self.n, replace=False)]
        for _ in range(10):  # Lloyd iterations, as oracle.init_logits
            lab = np.argmin(((pts[:, None] - c[None]) ** 2).sum(-1), axis=1)
            c = np.stack([pts[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(self.n)])
        d2 = ((p.coords[:, None] - c[None]) ** 2).sum(-1)
        s2 = np.mean(np.sort(d2, axis=1)[:, 1]) + 1e-12
        self.centres = torch.as_tensor(c, dtype=lift.DTYPE).clone().requires_grad_(True)
        self.log_s = torch.full((self.n,), 0.5 * float(np.log(s2)), dtype=lift.DTYPE).requires_grad_(True)
        self.xy64 = torch.as_tensor(p.coords, dtype=lift.DTYPE)

    def _base_params(self):
        return [self.centres, self.log_s]

    def _freeze_base(self):
        self.centres, self.log_s = self.centres.detach(), self.log_s.detach()

    def base_logits(self, xy=None):
        xy = self.xy64 if xy is None else xy
        d2 = ((xy[:, None, :] - self.centres[None]) ** 2).sum(-1)
        return -d2 / (2 * torch.exp(2 * self.log_s))[None]

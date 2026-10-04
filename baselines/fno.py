"""Unconstrained neural operator: a plain 2D FNO on a regular grid predicting (Re/Im Ex, Ey).

No structure, no physics loss: the control for 'does Maxwell structure help at all'. Output is
mapped back to edge coefficients with the midpoint rule, so the grid round trip sets an accuracy
floor, reported by fit() as 'roundtrip_floor'. Runs in float32 on the given device (use CUDA).
"""
import time

import numpy as np
import torch
import torch.nn as nn

from .base import Method


class SpectralConv2d(nn.Module):
    def __init__(self, cin, cout, modes):
        super().__init__()
        self.modes, self.cout = modes, cout
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(cin, cout, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        B, _, H, W = x.shape
        m = self.modes
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.cout, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, cin, cout, width=32, modes=12, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.spec = nn.ModuleList(SpectralConv2d(width, width, modes) for _ in range(layers))
        self.skip = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, cout, 1))

    def forward(self, x):
        x = self.lift(x)
        for spec, skip in zip(self.spec, self.skip):
            x = nn.functional.gelu(spec(x) + skip(x))
        return self.proj(x)


def _rel_l2(pred, target):
    num = (pred - target).flatten(1).norm(dim=1)
    return (num / target.flatten(1).norm(dim=1)).mean()


class FNO(Method):
    name = "fno"

    def __init__(self, epochs=100, batch=16, lr=1e-3, width=32, modes=12, layers=4, seed=0):
        self.epochs, self.batch, self.lr, self.seed = epochs, batch, lr, seed
        self.width, self.modes, self.layers = width, modes, layers

    def _targets(self, family, e):
        f = family.grid.e_to_grid(e)  # (2, n, n) complex
        return np.stack([f[0].real, f[0].imag, f[1].real, f[1].imag]).astype(np.float32)

    def fit(self, family, train, device=None, **kwargs):
        torch.manual_seed(self.seed)
        self.device = device or torch.device("cpu")
        X = np.stack([family.grid_inputs(i) for i in train.instances])
        Y = np.stack([self._targets(family, e) for e in train.e_ref])
        self.mu = X.mean(axis=(0, 2, 3), keepdims=True)
        self.sd = X.std(axis=(0, 2, 3), keepdims=True) + 1e-6
        X = torch.as_tensor((X - self.mu) / self.sd, device=self.device)
        Y = torch.as_tensor(Y, device=self.device)
        modes = min(self.modes, family.cfg.grid // 2 - 1)
        self.net = FNO2d(X.shape[1], 4, self.width, modes, self.layers).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        t0, loss = time.time(), torch.tensor(float("nan"))
        for _ in range(self.epochs):
            perm = torch.randperm(len(X), device=self.device)
            for k in range(0, len(X), self.batch):
                b = perm[k : k + self.batch]
                loss = _rel_l2(self.net(X[b]), Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        floor = np.mean([
            np.linalg.norm(family.grid.grid_to_e(family.grid.e_to_grid(e)) - e) / np.linalg.norm(e)
            for e in train.e_ref[:20]
        ])
        return {"params": sum(p.numel() for p in self.net.parameters()), "final_train_loss": float(loss),
                "roundtrip_floor": float(floor), "fit_seconds": time.time() - t0}

    @torch.no_grad()
    def predict(self, family, inst, cav):
        self.net.eval()
        x = torch.as_tensor((family.grid_inputs(inst)[None] - self.mu) / self.sd, device=self.device)
        y = self.net(x)[0].cpu().numpy()
        field = np.stack([y[0] + 1j * y[1], y[2] + 1j * y[3]])
        return family.grid.grid_to_e(field)

"""Study 4b: where does the encoder pass spend its time, and does it beat a full-wave solve at large meshes?

Protocol and decision rules are in the README ("Study 4b protocol"). Run single-threaded, with no other jobs:
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.study4b
(the encoder is trained with 8 torch threads; the script then sets one torch thread for every timing)
The accuracy of a model transferred to a finer mesh is NOT assessed here; only cost is measured.
"""
import argparse
import csv
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from baselines import EncoderWhitney
from compat_galerkin import fem, lift
from compat_galerkin.problems import Dataset, Family, FamilyConfig, Instance
from experiments.scaling import breakeven_sweep, eta_D, median_time
from experiments.study4a import selected_config

RATIO, TAN = 0.4, 0.05
LOAD_LIMIT = 1.5  # one-minute load average above which a single-thread timing is repeated and flagged


def load1():
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except OSError:
        return float("nan")


def settle(limit=1.2, max_wait=300):
    """Wait for the one-minute load average to fall below `limit` (it lags after a multi-threaded training)."""
    waited = 0
    while load1() > limit and waited < max_wait:
        time.sleep(10)
        waited += 10
    return waited


class Quiet:
    """median_time with a load check: a block started under load is repeated once after a pause, and flagged if still loaded."""

    def __init__(self):
        self.max_load, self.flagged = 0.0, False

    def __call__(self, fn):
        load = load1()
        if load > LOAD_LIMIT:
            time.sleep(30)
            load = load1()
        self.max_load, self.flagged = max(self.max_load, load), self.flagged or load > LOAD_LIMIT
        return median_time(fn)


class Runner:
    """The encoder pass on one mesh. Mesh-only tensors are built once (untimed); `fast` also caches the element lookup."""

    def __init__(self, model, fam, train_coords=None, transfer="nearest", train_mesh=None):
        self.model, self.fam = model, fam
        cav = fam.base
        self.cav = cav
        self.A, self.G = lift.to_torch_sparse(cav.A), lift.to_torch_sparse(cav.G)
        self.idx = torch.as_tensor(cav.interior_edges)
        self.mask = torch.as_tensor(cav.boundary_node_mask)
        self.K_t = lift.to_torch_sparse(cav.interior(cav.K))
        xy = torch.as_tensor(cav.mesh.p.T, dtype=torch.float32)
        self.node_grid = (2 * torch.stack([xy[:, 1], xy[:, 0]], dim=-1) - 1)[None, None]
        if getattr(model, "mesh_free", False):  # analytic base: evaluate it at this mesh's node coordinates (Study 6a)
            self.base = model.base_logits(torch.as_tensor(cav.mesh.p.T, dtype=lift.DTYPE))
        elif train_coords is None or len(train_coords) == len(xy):
            self.base = model.base
        elif transfer == "linear":  # P1 interpolation of the base logits from the training mesh (Study 5c, variant A1)
            from skfem import Basis, ElementTriP1

            P = Basis(train_mesh, ElementTriP1()).probes(cav.mesh.p).tocsr()
            self.base = torch.as_tensor(P @ model.base.numpy(), dtype=model.base.dtype)
        else:  # transfer the shared base logits to a finer mesh by nearest node
            _, nn = cKDTree(train_coords).query(xy.numpy().astype(float))
            self.base = model.base[torch.as_tensor(nn)]
        g = fam.grid
        self.elem_idx = cav.mesh.element_finder()(g.X.ravel(), g.Y.ravel())
        self.src_sigma = fam.cfg.src_sigma

    # --- stages -----------------------------------------------------------------------------------------
    def inputs(self, inst):
        g = self.fam.grid
        eps = inst.eps_elem[self.elem_idx].reshape(g.X.shape)
        bump = np.exp(-((g.X - inst.params["src"][0]) ** 2 + (g.Y - inst.params["src"][1]) ** 2) / (2 * self.src_sigma**2))
        one = np.ones_like(g.X)
        x = np.stack([eps, inst.omega2.real * one, inst.omega2.imag * one, inst.params["direction"][0] * bump,
                      inst.params["direction"][1] * bump, g.X, g.Y]).astype(np.float32)[None]
        return torch.as_tensor((x - self.model.mu) / self.model.sd)

    def logits(self, x):
        corr = self.model.net(x)
        samp = F.grid_sample(corr, self.node_grid, mode="bilinear", padding_mode="border", align_corners=False)
        return self.base[None] + samp[:, :, 0, :].permute(0, 2, 1).to(self.base.dtype)

    def basis(self, logits):
        W = lift.make_pou(logits[0], self.mask)
        return lift.whitney_lift(W, self.A, self.G)[self.idx]

    def reduced(self, V, M_t, f, omega2):
        Q = lift.compress_chol(V, M_t)
        return Q, lift.reduced_solve(Q, self.K_t, M_t, f, omega2)

    @torch.no_grad()
    def predict(self, inst, cav):
        M_t = lift.to_torch_sparse(cav.interior(cav.M))
        f = torch.as_tensor(inst.f[cav.interior_edges])
        V = self.basis(self.logits(self.inputs(inst)))
        return self.reduced(V, M_t, f, inst.omega2)[1].numpy()


def instance_on(fam, w2):
    """The scaling-study geometry and source on this family's mesh (disc eps 4 at the centre, radius 0.2)."""
    eps = fem.inclusion_eps(fam.base.mesh, 4.0, (0.5, 0.5), 0.2)
    f = fem.current_source(fam.base)
    return Instance(eps, w2, f, dict(src=np.array([0.3, 0.4]), theta=np.pi / 2, direction=(0.0, 1.0)))


def write(rows, out, stem):
    Path(out).mkdir(exist_ok=True)
    path = Path(out) / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    cols = list(dict.fromkeys(k for x in rows for k in x))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refines", default="4,5,6,7,8")
    ap.add_argument("--ntrain", type=int, default=100)
    ap.add_argument("--bumps", type=int, default=8)
    ap.add_argument("--steps", type=int, default=None, help="training steps (default: the Gate 2 selection)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    width, steps = selected_config()  # training uses the default thread pool; all timings below are single thread

    fam = Family(FamilyConfig(refine=4, grid=64))
    train = fam.dataset(a.ntrain, seed=400)
    test = fam.dataset(30, seed=500)
    steps = a.steps or steps
    model = EncoderWhitney(n=a.bumps, steps=steps, width=width, seed=0)
    info = model.fit(fam, Dataset(train.instances, train.e_ref))
    print(f"trained encoder (width {width}, steps {steps}, n_train {a.ntrain}): train error {info['train_error']:.3f}", flush=True)
    model.net.eval()
    train_coords = model.p0.coords
    torch.set_num_threads(1)
    print(f"single thread from here; waited {settle()} s for the load to settle (load {load1():.2f})", flush=True)

    # Step 1 and 2: profile at refine 4, and the output-preserving optimised path against the original predict.
    run = Runner(model, fam)
    insts = test.instances
    diffs = []
    for inst in insts[:20]:
        cav = fam.cavity(inst)
        e0, e1 = model.predict(fam, inst, cav)[cav.interior_edges], run.predict(inst, cav)
        diffs.append(np.linalg.norm(e0 - e1) / np.linalg.norm(e0))
    print(f"optimised path against original predict, max relative difference over 20 instances: {max(diffs):.2e}", flush=True)

    stage = {k: [] for k in ("grid_inputs_orig", "inputs_fast", "cnn+sample", "pou+lift", "restrict_M", "whiten+solve", "predict_orig", "predict_fast")}
    tm = lambda fn: (lambda t0=time.perf_counter(): (fn(), time.perf_counter() - t0)[1])()
    for inst in insts:
        cav = fam.cavity(inst)
        with torch.no_grad():
            stage["grid_inputs_orig"].append(tm(lambda: fam.grid_inputs(inst)))
            x = None
            t0 = time.perf_counter(); x = run.inputs(inst); stage["inputs_fast"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); lg = run.logits(x); stage["cnn+sample"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); V = run.basis(lg); stage["pou+lift"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            M_t = lift.to_torch_sparse(cav.interior(cav.M)); f = torch.as_tensor(inst.f[cav.interior_edges])
            stage["restrict_M"].append(time.perf_counter() - t0)
            stage["whiten+solve"].append(tm(lambda: run.reduced(V, M_t, f, inst.omega2)))
            stage["predict_orig"].append(tm(lambda: model.predict(fam, inst, cav)))
            stage["predict_fast"].append(tm(lambda: run.predict(inst, cav)))
    prof = {k: 1e3 * float(np.median(v)) for k, v in stage.items()}
    prof["load_after_profile"] = load1()
    print("Profile at refine 4, median of 30 instances (ms): " + ", ".join(f"{k} {v:.2f}" for k, v in prof.items()), flush=True)
    write([{"refine": 4, **prof, "max_rel_diff": float(max(diffs))}], a.out, "study4b_profile")

    # Step 3: cost at larger meshes (transferred weights; accuracy not assessed).
    rows = []
    for refine in [int(x) for x in a.refines.split(",")]:
        t_start = time.time()
        fr = fam if refine == 4 else Family(FamilyConfig(refine=refine, grid=64))
        cav = fr.base
        lam1 = fem.first_mode(cav)
        w2 = RATIO * lam1 * (1 + 1j * TAN)
        inst = instance_on(fr, w2)
        cav = cav.with_eps(inst.eps_elem)
        idx = cav.interior_edges
        N = len(idx)
        K_int, M_int, f_int = cav.interior(cav.K), cav.interior(cav.M), inst.f[idx]
        d = M_int.diagonal()
        f_D = math.sqrt(float(np.sum(f_int**2 / d)))
        quiet = Quiet()
        t_full = quiet(lambda: fem.solve_fine(cav, w2, inst.f))
        e_full = fem.solve_fine(cav, w2, inst.f)[idx]
        t_eta = quiet(lambda: eta_D(K_int, M_int, d, f_int, f_D, w2, e_full))
        runner = Runner(model, fr, train_coords)
        with torch.no_grad():
            t_pass = quiet(lambda: runner.predict(inst, cav))
            x = runner.inputs(inst)
            t_basis = quiet(lambda: runner.basis(runner.logits(runner.inputs(inst))))
            V = runner.basis(runner.logits(x))
            M_t = lift.to_torch_sparse(M_int)
            Q = lift.compress_chol(V, M_t)
            Kr, Mr = Q.T @ torch.sparse.mm(runner.K_t, Q), Q.T @ torch.sparse.mm(M_t, Q)
            rhs = Q.T.to(torch.complex128) @ torch.as_tensor(f_int).to(torch.complex128)

            def per_frequency():
                y = torch.linalg.solve((Kr - w2 * Mr).to(torch.complex128), rhs)
                return torch.complex(Q @ y.real, Q @ y.imag)  # as in scaling.Reduced.solve: no complex copy of Q per call

            t_freq = quiet(per_frequency)
        single = t_pass + t_eta
        row = {"refine": refine, "N": N, "max_load": quiet.max_load, "load_flag": quiet.flagged, "t_full_ms": 1e3 * t_full, "t_eta_ms": 1e3 * t_eta, "t_pass_ms": 1e3 * t_pass,
               "t_basis_ms": 1e3 * t_basis, "t_freq_ms": 1e3 * t_freq, "single_total_ms": 1e3 * single,
               "speedup": t_full / single, "wins": bool(single < t_full), "sweep_Fstar": breakeven_sweep(t_full, t_pass, t_freq)}
        rows.append(row)
        print(f"refine {refine}: N = {N}: full {row['t_full_ms']:.1f} ms | encoder pass {row['t_pass_ms']:.1f} ms (basis {row['t_basis_ms']:.1f}) "
              f"+ eta_D {row['t_eta_ms']:.2f} -> speedup {row['speedup']:.2f} ({'WINS' if row['wins'] else 'loses'}) | per extra frequency "
              f"{row['t_freq_ms']:.2f} ms, F* {row['sweep_Fstar']} | max load {quiet.max_load:.2f}"
              f"{' FLAGGED (loaded)' if quiet.flagged else ''} ({time.time() - t_start:.0f}s)", flush=True)
        del runner, V, Q, Kr, Mr
    win = [r["N"] for r in rows if r["wins"]]
    print(f"\nSmallest tested mesh where the encoder single query wins: {min(win) if win else 'none'}")
    write(rows, a.out, "study4b_scaling")


if __name__ == "__main__":
    main()

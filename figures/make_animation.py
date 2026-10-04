"""Sweep and phase animations for the deck (descriptive; see README "Sweep animation").

Trains the 78-dimensional encoder and the FNO on 400 instances (as Study 8), then:
  anim_sweep.gif   : reference field, encoder field (basis re-predicted at each frequency) and error map while r = omega^2 / lambda_1
                     sweeps 0.2 to 0.6 for the first seed-500 test geometry; below, the error against r for the encoder with the
                     basis re-predicted at every frequency, with the basis predicted once at r = 0.4 and reused (only the small
                     solve repeated), and the FNO, with the median over the first 20 test geometries drawn faintly.
  anim_phase.gif   : the time-harmonic field at r = 0.4 over one period (reference, encoder, error).
Also writes a static fallback frame per animation and anim_sweep_errors.csv. Usage (from the repo root):
    OMP_NUM_THREADS=8 uv run --with matplotlib python figures/make_animation.py
"""
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import make_figures as mf  # noqa: E402
from baselines import FNO, EncoderWhitney  # noqa: E402
from compat_galerkin import fem, lift, metrics  # noqa: E402
from compat_galerkin.device import get_device  # noqa: E402
from compat_galerkin.problems import Family, FamilyConfig, Instance  # noqa: E402
from experiments.study5a import FNOOnMesh  # noqa: E402
from experiments.study5c import EncoderOnMesh  # noqa: E402

OUT = ROOT / "figures"
R_GRID = np.linspace(0.2, 0.6, 61)
R_REUSE = 0.4
N_GEOM = 20


def field_mag(fam, e):
    g = fam.grid.e_to_grid(e)
    return np.sqrt(np.abs(g[0]) ** 2 + np.abs(g[1]) ** 2)


def main():
    t0 = time.time()
    mf.style()
    device = get_device("auto")
    fam = Family(FamilyConfig(refine=4, grid=64))
    base = fam.base
    tan = fam.cfg.tan_delta
    test = fam.dataset(N_GEOM, seed=500)
    cache = ROOT / "results" / "anim_cache.npz"
    reuse_cache = "--reuse" in sys.argv and cache.exists()
    if not reuse_cache:
        train = fam.dataset(400, seed=400)
        fno = FNO(epochs=200, batch=16, seed=0)
        fno.fit(fam, train, device=device)
        enc = EncoderWhitney(n=12, width=24, seed=0, schedule=[("galerkin", 1500)])
        enc.fit(fam, train)
        enc_fn = EncoderOnMesh(enc, enc.p0.coords, base.mesh, "nearest")
        fno_fn = FNOOnMesh(fno, device)
        print(f"trained ({time.time() - t0:.0f}s)", flush=True)

    def inst_at(g, r):
        i0 = test.instances[g]
        return Instance(i0.eps_elem, r * i0.params["lambda1"] * (1 + 1j * tan), i0.f, dict(i0.params))

    def reused_basis(g):
        inst = inst_at(g, R_REUSE)
        with torch.no_grad():
            V = enc._basis(enc._logits(enc._prepare_inputs(fam, [inst]))[0])
        return V

    errs = {k: np.zeros((N_GEOM, len(R_GRID))) for k in ("enc", "reuse", "fno")}
    fields = {}
    if reuse_cache:
        z = np.load(cache)
        errs = {k: z[k] for k in errs}
        fields = {j: (z["e_ref"][j], z["e_enc"][j]) for j in range(len(R_GRID))}
    for g in range(0 if reuse_cache else N_GEOM):
        V = reused_basis(g)
        cav = fam.cavity(test.instances[g])
        M_t, K_t = lift.to_torch_sparse(cav.interior(cav.M)), lift.to_torch_sparse(cav.interior(cav.K))
        Q = lift.compress_chol(V, M_t)
        idx = cav.interior_edges
        for j, r in enumerate(R_GRID):
            inst = inst_at(g, r)
            e_ref = fem.solve_fine(cav, inst.omega2, inst.f)
            e_enc = enc_fn.predict(fam, inst, cav)
            e_fno = fno_fn.predict(fam, inst, cav)
            e_reuse = np.zeros(cav.n_edges, dtype=complex)
            with torch.no_grad():
                e_reuse[idx] = lift.reduced_solve(Q, K_t, M_t, torch.as_tensor(inst.f[idx]), inst.omega2).numpy()
            for key, e in (("enc", e_enc), ("reuse", e_reuse), ("fno", e_fno)):
                errs[key][g, j] = metrics.evaluate(cav, inst, e, e_ref)["rel_err"]
            if g == 0:
                fields[j] = (e_ref, e_enc)
    if not reuse_cache:
        np.savez(cache, e_ref=np.stack([fields[j][0] for j in range(len(R_GRID))]), e_enc=np.stack([fields[j][1] for j in range(len(R_GRID))]), **errs)
    print(f"computed {N_GEOM} geometries x {len(R_GRID)} frequencies ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT / "anim_sweep_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r", "encoder_repredicted_geometry0", "encoder_basis_reused_geometry0", "fno_geometry0",
                    "encoder_repredicted_median20", "encoder_basis_reused_median20", "fno_median20"])
        for j, r in enumerate(R_GRID):
            w.writerow([f"{r:.4f}"] + [f"{errs[k][0, j]:.5f}" for k in ("enc", "reuse", "fno")] + [f"{np.median(errs[k][:, j]):.5f}" for k in ("enc", "reuse", "fno")])
    for k, label in (("enc", "encoder, basis re-predicted"), ("reuse", "encoder, basis reused"), ("fno", "FNO")):
        print(f"{label}: median over geometries and frequencies {np.median(errs[k]):.3f}, geometry 0 mean {errs[k][0].mean():.3f}, "
              f"worst-frequency median {np.max(np.median(errs[k], axis=0)):.3f}", flush=True)

    # geometry 0 display data
    inst0 = test.instances[0]
    eps_grid = fam.grid_inputs(inst0)[0]
    gx, gy = fam.grid.X, fam.grid.Y
    src = inst0.params["src"]
    ref_mag = {j: field_mag(fam, fields[j][0]) for j in fields}
    enc_mag = {j: field_mag(fam, fields[j][1]) for j in fields}
    diff_mag = {j: field_mag(fam, fields[j][1] - fields[j][0]) for j in fields}
    fmax = float(np.percentile(np.concatenate([v.ravel() for v in ref_mag.values()]), 99.5))  # robust colour scale

    def panel(ax, data, cmap, vmax, title):
        im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, vmin=0, vmax=vmax, interpolation="bilinear")
        ax.contour(gx, gy, eps_grid, levels=[1.5], colors=[mf.INK2], linewidths=0.9)
        ax.plot(*src, marker="+", color=mf.INK2, ms=7, mew=1.2)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_title(title, fontsize=9)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        return im

    fig = plt.figure(figsize=(9.0, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.75])
    a1, a2, a3, ab = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, :])
    i1 = panel(a1, ref_mag[0], "Blues", fmax, "full-wave reference |E|")
    i2 = panel(a2, enc_mag[0], "Blues", fmax, "encoder (24k parameters) |E|")
    i3 = panel(a3, diff_mag[0], "Oranges", 0.5 * fmax, "error |E_enc - E_ref| (same scale / 2)")
    curves = [("enc", "encoder, basis re-predicted at each frequency", 4), ("reuse", "encoder, basis predicted once at r = 0.4 and reused", 0),
              ("fno", "FNO, 1.19M parameters", 1)]
    markers = []
    for key, label, slot in curves:
        ab.plot(R_GRID, np.median(errs[key], axis=0), color=mf.SLOTS[slot], lw=0.9, alpha=0.35)
        ab.plot(R_GRID, errs[key][0], color=mf.SLOTS[slot], lw=1.8, label=label)
        (m,) = ab.plot([R_GRID[0]], [errs[key][0, 0]], marker=mf.MARKERS[slot], ms=7, color=mf.SLOTS[slot], mec=mf.SURFACE, mew=1.2)
        markers.append((key, m))
    ab.axhline(0.10, color=mf.MUTED, lw=1.0)
    ab.text(0.598, 0.10, "10% target", ha="right", va="bottom", color=mf.INK2, fontsize=8)
    vline = ab.axvline(R_GRID[0], color=mf.INK2, lw=1.0)
    ab.set_xlim(0.2, 0.6), ab.set_ylim(0, max(0.3, float(np.max([errs[k][0].max() for k in errs])) * 1.25))
    ab.set_xlabel("frequency ratio r = omega^2 / lambda_1 (thin lines: median over 20 geometries)")
    ab.set_ylabel("relative field error")
    ab.legend(loc="upper center", ncol=3, fontsize=7.5, columnspacing=1.2, handlelength=1.6)
    title = fig.suptitle("", x=0.01, ha="left", fontsize=10, fontweight="semibold", color=mf.INK)

    def update(j):
        i1.set_data(ref_mag[j].T), i2.set_data(enc_mag[j].T), i3.set_data(diff_mag[j].T)
        vline.set_xdata([R_GRID[j], R_GRID[j]])
        for key, m in markers:
            m.set_data([R_GRID[j]], [errs[key][0, j]])
        title.set_text(f"Frequency sweep, one geometry: r = {R_GRID[j]:.2f} (resonance at r = 1); encoder error {errs['enc'][0, j]:.0%}")
        return []

    update(30)
    fig.savefig(OUT / "anim_sweep_frame.png", dpi=150)
    FuncAnimation(fig, update, frames=len(R_GRID), interval=100).save(OUT / "anim_sweep.gif", writer=PillowWriter(fps=10), dpi=80)
    plt.close(fig)
    print(f"wrote anim_sweep ({time.time() - t0:.0f}s)", flush=True)

    # phase animation at r = 0.4 (time-harmonic: Re(E e^{-i phi}))
    j = int(np.argmin(np.abs(R_GRID - 0.4)))
    e_ref, e_enc = fields[j]
    gr, ge = fam.grid.e_to_grid(e_ref), fam.grid.e_to_grid(e_enc)
    phases = np.linspace(0, 2 * np.pi, 49)[:-1]
    comp = lambda g, ph: np.real(g[1] * np.exp(-1j * ph))  # the y component
    vm = float(np.abs(gr[1]).max())
    fig = plt.figure(figsize=(9.0, 3.4), constrained_layout=True)
    axs = fig.subplots(1, 3)
    ims = []
    for ax, (data, title_) in zip(axs, ((comp(gr, 0), "full-wave reference, Re(E_y e^(-i phi))"), (comp(ge, 0), "encoder"), (comp(ge, 0) - comp(gr, 0), "error"))):
        im = ax.imshow(data.T, origin="lower", extent=[0, 1, 0, 1], cmap="RdBu_r", vmin=-vm, vmax=vm, interpolation="bilinear")
        ax.contour(gx, gy, eps_grid, levels=[1.5], colors=[mf.INK2], linewidths=0.9)
        ax.plot(*src, marker="+", color=mf.INK2, ms=7, mew=1.2)
        ax.set_xticks([]), ax.set_yticks([]), ax.set_title(title_, fontsize=9), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ims.append(im)
    ptitle = fig.suptitle("", x=0.01, ha="left", fontsize=10, fontweight="semibold", color=mf.INK)

    def update_phase(k):
        ph = phases[k]
        ims[0].set_data(comp(gr, ph).T), ims[1].set_data(comp(ge, ph).T), ims[2].set_data((comp(ge, ph) - comp(gr, ph)).T)
        ptitle.set_text(f"Time-harmonic field at r = 0.40, phase {np.degrees(ph):.0f} degrees; encoder error {errs['enc'][0, j]:.0%}")
        return []

    update_phase(0)
    fig.savefig(OUT / "anim_phase_frame.png", dpi=150)
    FuncAnimation(fig, update_phase, frames=len(phases), interval=80).save(OUT / "anim_phase.gif", writer=PillowWriter(fps=12), dpi=80)
    plt.close(fig)
    print(f"wrote anim_phase ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

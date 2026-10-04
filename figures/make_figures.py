"""Thesis figures, built from the committed result CSVs in results/.

Run from the repo root (matplotlib is pulled in for this run only):
    uv run --with matplotlib python figures/make_figures.py

Each figure is written as PNG and SVG with a CSV "table view" of the plotted data beside it. Style follows
the dataviz method: fixed-order categorical hues (validated with the skill's validate_palette script, light
mode: all gates pass, three slots sit below 3:1 contrast so every chart has distinct marker shapes, a
legend and a table view), thin marks, recessive solid hairline grid, text in ink colours, one axis per
panel. Entities keep their colour within a figure.
"""
import csv
import glob
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, fixed order
MARKERS = ["o", "s", "^", "D", "v"]  # secondary encoding: shape as well as hue
LW, MS = 1.6, 6.5  # 2 px lines, >= 8 px markers (at 96 dpi)


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "semibold",
        "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False, "legend.fontsize": 8.5,
        "legend.labelcolor": INK2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def latest(pattern):
    files = sorted(glob.glob(str(RES / pattern)))
    if not files:
        raise FileNotFoundError(pattern)
    return files[-1]


def read(pattern):
    with open(latest(pattern), newline="") as fh:
        return list(csv.DictReader(fh))


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def mark(ax, x, y, k, label=None, **kw):
    """A series: 2 px line, markers with a 2 px surface ring."""
    ax.plot(x, y, color=SLOTS[k], lw=LW, marker=MARKERS[k], ms=MS, mfc=SLOTS[k], mec=SURFACE, mew=1.5, label=label, **kw)


def hline(ax, y, text, x_frac=0.99, va="bottom", ha="right"):
    """A solid, thin reference line labelled in ink (not dashed: dashing reads as a projection)."""
    ax.axhline(y, color=MUTED, lw=1.0, zorder=1)
    ax.text(x_frac, y, text, transform=ax.get_yaxis_transform(), ha=ha, va=va, color=INK2, fontsize=8)


def outside_legend(fig, ax, ncol):
    h, lab = ax.get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=ncol, columnspacing=1.6, handlelength=2.2)


def save(fig, name, rows):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=200)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)
    if rows:
        with open(OUT / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    print("wrote", name)


def fig_band_precheck():
    rows = [r for r in read("band_precheck_2*.csv")]
    r_ = np.array([f(r["ratio"]) for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 3.5), constrained_layout=True)
    ax.axvspan(0.2, 0.6, color=GRID, alpha=0.55, lw=0, zorder=0)
    for k, (key, label) in enumerate([("rel_err", "field error"), ("proj_err", "best-approximation error"), ("energy_err", "stored-energy error")]):
        mark(ax, r_, [f(r[key]) for r in rows], k, label)
    hline(ax, 0.05, "5% limit", x_frac=0.99)
    ax.set_yscale("log")
    ax.set_xlabel("frequency ratio r = ω² / λ₁ (first resonance at r = 1)")
    ax.set_ylabel("relative error (oracle W, n = 8)")
    ax.set_title("Energy error stays under 5% up to r = 0.6; resonance dominates from 0.8")
    ax.text(0.4, 0.0016, "selected band\nr = 0.2 to 0.6", ha="center", color=INK2, fontsize=8)
    ax.legend(loc="upper left", ncol=1)
    save(fig, "fig1_band_precheck", [{"r": r["ratio"], "field_error": r["rel_err"], "projection_error": r["proj_err"],
                                       "energy_error": r["energy_err"], "converged": r["converged"]} for r in rows])


def fig_capacity():
    rows = read("gate1_2*.csv")
    freqs = [("low", "low: ω² = 0.4 λ₁"), ("gap", "gap: between modes 1 and 2"), ("near", "near: ω² = 1.02 λ₁")]
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.3), sharey=True, constrained_layout=True)
    table = []
    for ax, (fr, title) in zip(axes, freqs):
        ora = sorted([r for r in rows if r["freq"] == fr and r["method"] == "oracle"], key=lambda r: f(r["dim"]))
        coa = sorted([r for r in rows if r["freq"] == fr and r["method"].startswith("coarse")], key=lambda r: f(r["dim"]))
        mark(ax, [f(r["dim"]) for r in ora], [f(r["rel_err"]) for r in ora], 0, "oracle W (fitted per instance)")
        mark(ax, [f(r["dim"]) for r in coa], [f(r["rel_err"]) for r in coa], 1, "non-learned coarse Nédélec")
        ax.set_xscale("log"), ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("reduced dimension")
        table += [{"freq": fr, "method": r["method"], "n": r.get("n", ""), "dim": r["dim"], "field_error": r["rel_err"]} for r in ora + coa]
    axes[0].set_ylabel("relative field error")
    axes[2].text(0.5, 0.40, "resonance:\nnot reached", transform=axes[2].transAxes, ha="center", color=INK2, fontsize=8)
    outside_legend(fig, axes[0], ncol=2)
    fig.suptitle("Gate 1 (oracle, one instance): 36 learned dimensions beat 368 non-learned ones off resonance",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    save(fig, "fig2_capacity", table)


def fig_verification():
    rows = [r for r in read("verification_2*.csv") if "surrogate" in r]
    names = ["coarse_r2", "coarse_r3", "pod_16", "pod_64", "pod_128"]
    labels = {"coarse_r2": "coarse r2 (dim 88)", "coarse_r3": "coarse r3 (dim 368)", "pod_16": "POD 16", "pod_64": "POD 64", "pod_128": "POD 128"}
    bins = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6)]
    mid = [(lo + hi) / 2 for lo, hi in bins]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.5, 1]}, constrained_layout=True)
    table = []
    for k, s in enumerate(names):
        rs = [r for r in rows if r["surrogate"] == s]
        cov = [np.nanmean([f(r[f"cov_bin_{lo}_{hi}"]) for r in rs]) for lo, hi in bins]
        mark(a, mid, cov, k, labels[s])
        table += [{"surrogate": s, "r_bin": f"{lo}-{hi}", "coverage": f"{c:.4f}"} for (lo, hi), c in zip(bins, cov)]
    hline(a, 0.9, "nominal 0.90", x_frac=0.99)
    a.annotate("POD 16 under-covers\nnear the resonance (0.71)", xy=(0.55, 0.71), xytext=(0.36, 0.64), color=INK2, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a.set_xlabel("frequency ratio r = ω² / λ₁ (bin centre)"), a.set_ylabel("empirical coverage of the certified bound")
    a.set_ylim(0.6, 1.02), a.set_xticks(mid)
    a.set_title("Coverage drifts across the band")
    outside_legend(fig, a, ncol=5)

    sp = [(s, np.mean([f(r["spearman_D"]) for r in rows if r["surrogate"] == s]), np.std([f(r["spearman_D"]) for r in rows if r["surrogate"] == s])) for s in names]
    y = np.arange(len(names))[::-1]
    b.barh(y, [v for _, v, _ in sp], height=0.5, color=SLOTS[0], xerr=[e for *_, e in sp], error_kw=dict(ecolor=INK2, lw=1.0, capsize=2))
    b.axvline(0, color=AXIS, lw=1.0)
    b.set_yticks(y), b.set_yticklabels([labels[s] for s in names])
    b.axvline(0.9, color=MUTED, lw=1.0), b.text(0.89, -0.62, "tracks: ≥ 0.9", ha="right", va="center", color=INK2, fontsize=8)
    b.set_xlim(-0.6, 1.05), b.set_ylim(-0.85, 4.5), b.set_xlabel("Spearman correlation: residual vs true error")
    b.set_title("Residual tracks error for POD only")
    b.grid(axis="y", visible=False)
    b.tick_params(axis="y", labelcolor=INK2)
    for yy, (s, v, e) in zip(y, sp):
        b.text(v + (0.03 if v >= 0 else -0.03), yy, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", color=INK2, fontsize=8)
        table.append({"surrogate": s, "spearman_eta_D_mean": f"{v:.4f}", "spearman_std": f"{e:.4f}"})
    save(fig, "fig3_verification", table)


def fig_scaling():
    rows = read("scaling_2*.csv")
    full = sorted([r for r in rows if r["method"] == "full"], key=lambda r: f(r["N"]))
    N = np.array([f(r["N"]) for r in full])
    methods = [("dense_q16", "dense q = 16"), ("dense_q64", "dense q = 64"), ("dense_q128", "dense q = 128"),
               ("coarse_r2", "coarse r2 (q = 88)"), ("coarse_r3", "coarse r3 (q = 368)")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    table = []
    for k, (m, label) in enumerate(methods):
        rs = sorted([r for r in rows if r["method"] == m], key=lambda r: f(r["N"]))
        sp = [f(r["t_full_ms"]) / f(r["single_total_ms"]) for r in rs]
        mark(a, [f(r["N"]) for r in rs], sp, k, label)
        table += [{"method": m, "N": r["N"], "speedup": f"{s:.3f}"} for r, s in zip(rs, sp)]
    hline(a, 1.0, "break-even", x_frac=0.02, va="bottom", ha="left")
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlabel("unknowns N"), a.set_ylabel("speed-up per query\n(full solve ÷ reduced + certificate)")
    a.set_title("Reduced solves win from small meshes on")
    outside_legend(fig, a, ncol=5)

    mark(b, N, [f(r["t_full_ms"]) for r in full], 0, "full-wave solve (slope 1.42)")
    mark(b, N, [f(r["t_eta_ms"]) for r in full], 1, "residual certificate (slope 1.15)")
    b.set_xscale("log"), b.set_yscale("log")
    b.set_xlabel("unknowns N"), b.set_ylabel("time per query (ms)")
    b.set_title("The certificate costs about 0.25% of a solve")
    b.legend(loc="upper left")
    table += [{"method": "full", "N": r["N"], "t_full_ms": r["t_full_ms"], "t_eta_ms": r["t_eta_ms"]} for r in full]
    save(fig, "fig4_scaling", table)


def fig_decomposition():
    rows = [r for r in read("diag_decomp_2*.csv") if math.isfinite(f(r.get("omega2")))]  # per-frequency rows only
    groups = [("kernel", "curl-free part"), ("pair", "resonant pair"), ("other", "other transverse modes")]
    cfgs = [("10.0", "16", "Galerkin error, w = 10, n = 16"), ("0.1", "16", "Galerkin error, w = 0.1, n = 16")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 2.9), gridspec_kw={"width_ratios": [1, 1]}, constrained_layout=True)
    table = []

    def stack(ax, labels, data):
        y = np.arange(len(labels))[::-1]
        left = np.zeros(len(labels))
        for k, (g, gl) in enumerate(groups):
            v = np.array([d[g] for d in data])
            ax.barh(y, v, left=left, height=0.5, color=SLOTS[k], edgecolor=SURFACE, linewidth=2.0, label=gl)
            left += v
        ax.set_yticks(y), ax.set_yticklabels(labels), ax.grid(axis="y", visible=False)

    base = [r for r in rows if r["w"] == cfgs[0][0] and r["n"] == cfgs[0][1]]
    truth = {g: np.mean([f(r[f"ref_{g}"]) for r in base]) for g, _ in groups}
    stack(a, ["true solution"], [truth])
    a.set_xlim(0, 1), a.set_xlabel("share of ‖e_ref‖²"), a.set_title("True solution: mostly the resonant pair")
    a.tick_params(axis="y", labelcolor=INK2)
    err = []
    for w, n, label in cfgs:
        rs = [r for r in rows if r["w"] == w and r["n"] == n]
        err.append({g: np.mean([f(r[f"err_{g}"]) for r in rs]) for g, _ in groups})
        table.append({"config": label, **{f"err_{g}": f"{err[-1][g]:.5f}" for g, _ in groups}})
    table.append({"config": "true solution", **{f"ref_{g}": f"{truth[g]:.5f}" for g, _ in groups}})
    stack(b, [c[2] for c in cfgs], err)
    b.set_xlim(0, 0.1), b.set_xlabel("squared error as a share of ‖e_ref‖²"), b.set_title("Remaining error: the curl-free part")
    b.tick_params(axis="y", labelcolor=INK2)
    for yy, d in zip(np.arange(len(cfgs))[::-1], err):
        tot = sum(d.values())
        b.text(tot + 0.002, yy, f"curl-free {100 * d['kernel'] / tot:.0f}%", va="center", color=INK2, fontsize=8)
    outside_legend(fig, a, ncol=3)
    save(fig, "fig5_error_decomposition", table)


# Models keep one slot (hue and marker) in every figure from here on.
MODEL = {"enc36": (0, "encoder, 36 dim"), "enc78": (4, "encoder, 78 dim"), "fno": (1, "FNO, 1.19M parameters"),
         "fnos": (3, "FNO, 20k parameters"), "pod": (2, "POD-136"), "mesh_free": (2, "mesh-free encoder, 36 dim")}


def mean_by(rows, group, key, where=lambda r: True):
    out = {}
    for r in rows:
        if where(r):
            out.setdefault(group(r), []).append(f(r[key]))
    return {k: float(np.mean(v)) for k, v in out.items()}


def fig_learning_curves():
    g2 = mean_by(read("gate2_2*.csv"), lambda r: int(r["n_train"]), "median_field", lambda r: r["method"] == "encoder")
    de = read("data_efficiency_2*.csv")
    fno = mean_by(de, lambda r: int(r["n_train"]), "median_field", lambda r: r["method"] == "fno")
    p136 = mean_by(de, lambda r: int(r["n_train"]), "median_field", lambda r: r["method"] == "pod136")
    s7 = mean_by(read("study7_2*.csv"), lambda r: int(r["n_train"]), "median_field", lambda r: r["config"] == "G12")
    series = [("enc36", g2), ("enc78", s7), ("fno", fno), ("pod", p136)]
    fig, ax = plt.subplots(figsize=(6.8, 3.8), constrained_layout=True)
    table = []
    from matplotlib.ticker import NullFormatter

    for key, data in series:
        k, label = MODEL[key]
        label = label + " (100 and 400 only)" if key == "enc78" else label
        xs = sorted(data)
        mark(ax, xs, [data[x] for x in xs], k, label)
        table += [{"model": label, "training_instances": x, "median_field_error": f"{data[x]:.4f}"} for x in xs]
    hline(ax, 0.044, "per-instance oracle, 36 dim (0.044)", x_frac=0.99)
    hline(ax, 0.10, "10% target", x_frac=0.99)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xticks([25, 50, 100, 200, 400]), ax.set_xticklabels(["25", "50", "100", "200", "400"])
    ax.set_yticks([0.05, 0.1, 0.2, 0.4, 0.6]), ax.set_yticklabels(["0.05", "0.10", "0.20", "0.40", "0.60"])
    ax.xaxis.set_minor_formatter(NullFormatter()), ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("training instances"), ax.set_ylabel("median field error at the training mesh")
    ax.set_title("Longer training lets the 78-dim encoder reach the FNO;\nnobody reaches 10% with 100 instances")
    outside_legend(fig, ax, ncol=2)
    save(fig, "fig6_learning_curves", table)


def fig_parameter_matched():
    rows = read("study8_consistency_2*.csv")
    val = mean_by(rows, lambda r: r["surrogate"], "median_rel_err")
    parts = [("g12", "enc78", "encoder, 78 dim (24,000 parameters)"), ("fnosmall", "fnos", "FNO (20,452 parameters)"), ("fno", "fno", "FNO (1,188,868 parameters)")]
    sizes = [100, 400]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    table, width = [], 0.26
    for j, (name, key, label) in enumerate(parts):
        k = MODEL[key][0]
        xs = np.arange(len(sizes)) + (j - 1) * (width + 0.02)
        ys = [val[f"{name}_n{n}"] for n in sizes]
        ax.bar(xs, ys, width=width, color=SLOTS[k], edgecolor=SURFACE, linewidth=2.0, label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.006, f"{y:.3f}", ha="center", va="bottom", color=INK2, fontsize=8)
        table += [{"model": label, "training_instances": n, "median_field_error": f"{y:.4f}"} for n, y in zip(sizes, ys)]
    ax.set_xticks(range(len(sizes))), ax.set_xticklabels([f"{n} training instances" for n in sizes])
    ax.set_ylabel("median field error at the training mesh"), ax.set_ylim(0, 0.42)
    ax.grid(axis="x", visible=False)
    ax.set_title("At matched size the encoder has about 0.6× the FNO's error\nand matches the 50× larger FNO")
    outside_legend(fig, ax, ncol=1)
    save(fig, "fig7_parameter_matched", table)


def fig_cost():
    b5 = sorted(read("study5b_2*.csv"), key=lambda r: f(r["N"]))
    e36 = {int(r["N"]): f(r["single_total_ms"]) for r in read("study4b_scaling_20261004_0425*.csv")}
    e78 = {int(r["N"]): f(r["single_total_ms"]) for r in read("study4b_scaling_20261004_1417*.csv")}
    N = [int(r["N"]) for r in b5]
    fig, ax = plt.subplots(figsize=(6.8, 3.9), constrained_layout=True)
    table = []
    ax.plot(N, [f(r["t_full_ms"]) for r in b5], color=INK2, lw=LW, marker="x", ms=MS, label="full-wave solve")
    for key, data in (("enc36", e36), ("enc78", e78)):
        k, label = MODEL[key]
        mark(ax, N, [data[n] for n in N], k, label + " + residual")
    mark(ax, N, [f(r["fno_single_gpu_ms"]) for r in b5], MODEL["fno"][0], "FNO, 1.19M parameters + residual")
    for i, n in enumerate(N):
        table.append({"unknowns": n, "full_solve_ms": b5[i]["t_full_ms"], "encoder_36_ms": e36[n], "encoder_78_ms": e78[n],
                      "fno_ms": b5[i]["fno_single_gpu_ms"]})
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlabel("unknowns N"), ax.set_ylabel("time per query (ms)")
    ax.set_title("The FNO costs about 50× less than the encoder at 393k;\n78 dimensions beat a full solve only above about 98k")
    outside_legend(fig, ax, ncol=2)
    save(fig, "fig8_cost", table)


def mesh_free_by_mesh():
    """Mean median field error of the mesh-free encoder by (size, refine), read from the committed Study 6a log."""
    import re
    acc = {}
    for line in open(RES / "study6a_run.log", encoding="utf-8"):
        m = re.search(r"\[\d\] refine (\d): n=100 ([\d.]+), n=400 ([\d.]+)", line)
        if m:
            r = int(m.group(1))
            acc.setdefault((100, r), []).append(float(m.group(2)))
            acc.setdefault((400, r), []).append(float(m.group(3)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def fig_transfer():
    a5 = read("study5a_2*.csv")
    s8 = read("study8_transfer_2*.csv")
    mf = mesh_free_by_mesh()
    meshes = [4, 5, 6, 7]
    unknowns = [1504, 6080, 24448, 98048]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True, constrained_layout=True)
    table = []
    for ax, n in zip(axes, (100, 400)):
        s5 = mean_by(a5, lambda r: int(r["refine"]), "median_field", lambda r: r["model"] == f"encoder_n{n}")
        s8m = {name: mean_by(s8, lambda r: int(r["refine"]), "median_field", lambda r, name=name: r["model"] == f"{name}_n{n}") for name in ("g12", "fno", "fnosmall")}
        series = [("enc36", s5), ("enc78", s8m["g12"]), ("fno", s8m["fno"]), ("fnos", s8m["fnosmall"]), ("mesh_free", {r: mf[(n, r)] for r in meshes if (n, r) in mf})]
        for key, data in series:
            k, label = MODEL[key]
            label = label + " (refine 7: 2 repeats)" if key == "mesh_free" else label
            xs = [m for m in meshes if m in data]
            mark(ax, [unknowns[meshes.index(m)] for m in xs], [data[m] for m in xs], k, label)
            table += [{"training_instances": n, "model": label, "unknowns": unknowns[meshes.index(m)], "median_field_error": f"{data[m]:.4f}"} for m in xs]
        hline(ax, 0.10, "10% target", x_frac=0.99)
        ax.set_xscale("log"), ax.set_ylim(0, 1.05)
        ax.set_xlabel("unknowns of the mesh the model is applied to"), ax.set_title(f"{n} training instances (trained at 1,504 unknowns)")
    axes[0].set_ylabel("median field error")
    outside_legend(fig, axes[0], ncol=3)
    fig.suptitle("Only the FNOs keep their accuracy on finer meshes; the per-node encoders lose it,\nthe mesh-free one degrades slowly but starts less accurate",
                 x=0.01, ha="left", fontsize=10, fontweight="semibold", color=INK)
    save(fig, "fig9_transfer", table)


def fig_certificate():
    rows = read("study4a_certificate_2*.csv") + read("study8_certificate_2*.csv")
    pick = [("encoder_n400", "enc36"), ("g12_n400", "enc78"), ("fno_n400", "fno"), ("fnosmall_n400", "fnos"), ("pod136_n400", "pod")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    table = []
    y = np.arange(len(pick))[::-1]
    for yy, (name, key) in zip(y, pick):
        rs = [r for r in rows if r["surrogate"] == name]
        sp = float(np.mean([f(r["spearman_D"]) for r in rs]))
        ac = float(np.mean([f(r["accept_a0.1_t0.2"]) for r in rs]))
        fa = float(np.mean([f(r["falseaccept_a0.1_t0.2"]) for r in rs]))
        k, label = MODEL[key]
        a.barh(yy, sp, height=0.5, color=SLOTS[k])
        a.text(sp + 0.02, yy, f"{sp:.2f}", va="center", color=INK2, fontsize=8)
        b.barh(yy, ac, height=0.5, color=SLOTS[k])
        b.text(ac + 0.015, yy, f"{ac:.0%} (false accepts {fa:.1%})", va="center", color=INK2, fontsize=8)
        table.append({"model": label, "spearman_residual_vs_error": f"{sp:.4f}", "accepted_at_target_0.2": f"{ac:.4f}", "false_accept": f"{fa:.4f}"})
    for ax in (a, b):
        ax.set_yticks(y), ax.set_yticklabels([MODEL[k][1] for _, k in pick]), ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelcolor=INK2)
    a.axvline(0.9, color=MUTED, lw=1.0), a.text(0.89, -0.62, "tracks: ≥ 0.9", ha="right", va="center", color=INK2, fontsize=8)
    a.set_xlim(0, 1.05), a.set_ylim(-0.85, 4.5), a.set_xlabel("Spearman correlation: residual vs true error")
    a.set_title("Only POD's residual tracks its error")
    b.set_xlim(0, 1.2), b.set_xlabel("share of instances accepted (no full-wave call)"), b.set_title("Accepted at accuracy target 0.2")
    save(fig, "fig10_certificate", table)


def fig_design_loop():
    rows = [r for r in read("study9_2*.csv") if r.get("policy")]
    models = [("fno", "fno"), ("g12", "enc78"), ("pod136", "pod")]

    def regret(sur, pol):
        v = [f(r["regret"]) for r in rows if r["surrogate"] == sur and r["policy"] == pol and r.get("regret") not in ("", None)]
        return float(np.mean(v)) if v else float("nan")

    def calls(sur, pol):
        return float(np.mean([f(r["calls"]) for r in rows if r["surrogate"] == sur and r["policy"] == pol]))

    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    table = []
    cs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    ax.plot(cs, [regret("none", f"C{c}") for c in cs], color=INK2, lw=LW, marker="x", ms=MS, label="random full-wave search")
    table += [{"policy": "random full-wave search", "calls": c, "mean_regret": f"{regret('none', f'C{c}'):.4f}"} for c in cs]
    ms = [1, 2, 5, 10, 20, 50]
    for sur, key in models:
        k, label = MODEL[key]
        label = {"fno": "FNO", "enc78": "encoder, 78 dim", "pod": "POD-136"}[key]
        mark(ax, ms, [regret(sur, f"A{M}") for M in ms], k, f"{label}: screen, verify top M")
        table += [{"policy": f"{label} screen and verify top M", "calls": M, "mean_regret": f"{regret(sur, f'A{M}'):.4f}"} for M in ms]
        bx, by = calls(sur, "B"), regret(sur, "B")
        ax.scatter([bx], [by], marker="*", s=130, color=SLOTS[k], edgecolor=SURFACE, linewidth=1.2, zorder=5, label=f"{label}: with certificate gating")
        table.append({"policy": f"{label} with certificate gating", "calls": f"{bx:.1f}", "mean_regret": f"{by:.4f}"})
    hline(ax, 0.10, "10% regret", x_frac=0.99)
    ax.set_xscale("log"), ax.set_ylim(-0.02, 0.75)
    ax.set_xticks(cs), ax.set_xticklabels([str(c) for c in cs])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("full-wave calls per design search"), ax.set_ylabel("mean regret against the pool optimum")
    ax.set_title("One verified top candidate gets within 10% of the best design;\nrandom full-wave search needs 200 calls")
    outside_legend(fig, ax, ncol=2)
    save(fig, "fig11_design_loop", table)


if __name__ == "__main__":
    style()
    for fn in (fig_band_precheck, fig_capacity, fig_verification, fig_scaling, fig_decomposition,
               fig_learning_curves, fig_parameter_matched, fig_cost, fig_transfer, fig_certificate, fig_design_loop):
        try:
            fn()
        except FileNotFoundError as exc:
            print("skipped", fn.__name__, "(missing", exc, ")")

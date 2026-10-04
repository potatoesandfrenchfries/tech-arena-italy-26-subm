"""Recompute the numbers quoted in docs/SLIDES.md from the committed result files (no torch needed).

Run from the repo root:  python3 docs/verify_numbers.py
Each line prints the quantity, the recomputed value and the file it comes from, so the slides can be checked against the data.
"""
import csv
import glob
import statistics as st
from pathlib import Path

RES = Path("results")


def rows(pattern):
    return list(csv.DictReader(open(sorted(glob.glob(str(RES / pattern)))[-1])))


def mean(rs, key, **where):
    v = [float(r[key]) for r in rs if all(str(r[k]) == str(w) for k, w in where.items()) and r.get(key) not in ("", None)]
    return st.mean(v) if v else float("nan")


def show(label, value, src, fmt="{:.3f}"):
    print(f"  {label:<62} {fmt.format(value):>10}   [{src}]")


def main():
    print("Accuracy against training size (median field error)")
    g2, de, s7 = rows("gate2_2*.csv"), rows("data_efficiency_2*.csv"), rows("study7_2*.csv")
    for n in (25, 50, 100, 200, 400):
        show(f"n={n}: encoder 36 / FNO / POD-36 / POD-136", mean(g2, "median_field", method="encoder", n_train=n), "gate2")
        print(" " * 66, " / ".join(f"{mean(de, 'median_field', method=m, n_train=n):.3f}" for m in ("fno", "pod36", "pod136")), "[data_efficiency]")
    for n in (100, 400):
        show(f"Study 7 G12 n={n}: field / energy", mean(s7, "median_field", config="G12", n_train=n), "study7")
        show(f"Study 7 G12 n={n}: energy", mean(s7, "median_energy", config="G12", n_train=n), "study7")

    print("\nStudy 8 (stage A, 1000 test instances)")
    c8, k8, t8 = rows("study8_certificate_2*.csv"), rows("study8_consistency_2*.csv"), rows("study8_transfer_2*.csv")
    for s in ("g12_n100", "fnosmall_n100", "fno_n100", "g12_n400", "fnosmall_n400", "fno_n400"):
        show(f"{s}: median field error", mean(k8, "median_rel_err", surrogate=s), "study8_consistency")
    for n in (100, 400):
        print(f"  ratio G12 / small FNO and / large FNO at n={n}: {mean(k8, 'median_rel_err', surrogate=f'g12_n{n}') / mean(k8, 'median_rel_err', surrogate=f'fnosmall_n{n}'):.2f}"
              f" and {mean(k8, 'median_rel_err', surrogate=f'g12_n{n}') / mean(k8, 'median_rel_err', surrogate=f'fno_n{n}'):.2f}")
    for s in ("g12_n400", "g12_n100", "fno_n400", "fnosmall_n400"):
        show(f"{s}: Spearman / accepted at 0.2", mean(c8, "spearman_D", surrogate=s), "study8_certificate", "{:.2f}")
        show(f"{s}: accepted at tau 0.2", mean(c8, "accept_a0.1_t0.2", surrogate=s), "study8_certificate")
        show(f"{s}: false accepts at tau 0.2", mean(c8, "falseaccept_a0.1_t0.2", surrogate=s), "study8_certificate")
    for s in ("g12_n400", "fno_n400"):
        show(f"{s}: power balance / residual", mean(k8, "median_power_balance", surrogate=s), "study8_consistency", "{:.1e}")
        show(f"{s}: residual", mean(k8, "median_residual", surrogate=s), "study8_consistency", "{:.2f}")
    for m in ("g12_n100", "g12_n400", "fno_n100", "fno_n400", "fnosmall_n100", "fnosmall_n400"):
        v = {r: mean(t8, "median_field", model=m, refine=r) for r in (4, 5, 6, 7)}
        print(f"  {m:<14} transfer: " + ", ".join(f"r{r} {x:.3f}" for r, x in v.items()) + f"  (ratio r7/r4 {v[7] / v[4]:.2f})  [study8_transfer]")

    print("\nStudy 4a (Gate 2 encoder at 400, FNO, POD-136)")
    c4, k4 = rows("study4a_certificate_2*.csv"), rows("study4a_consistency_2*.csv")
    for s in ("encoder_n400", "fno_n400", "pod136_n400"):
        show(f"{s}: Spearman", mean(c4, "spearman_D", surrogate=s), "study4a_certificate", "{:.2f}")
        show(f"{s}: accepted at 0.2", mean(c4, "accept_a0.1_t0.2", surrogate=s), "study4a_certificate")
        show(f"{s}: false accepts at 0.2", mean(c4, "falseaccept_a0.1_t0.2", surrogate=s), "study4a_certificate")
        show(f"{s}: power balance", mean(k4, "median_power_balance", surrogate=s), "study4a_consistency", "{:.1e}")
        show(f"{s}: residual", mean(k4, "median_residual", surrogate=s), "study4a_consistency", "{:.2f}")

    print("\nCost: single query incl. residual (ms) and speedups")
    b5 = sorted(rows("study5b_2*.csv"), key=lambda r: float(r["N"]))
    e36 = {int(r["N"]): r for r in rows("study4b_scaling_20261004_0425*.csv")}
    e78 = {int(r["N"]): r for r in rows("study4b_scaling_20261004_1417*.csv")}
    for r in b5:
        n = int(r["N"])
        print(f"  N={n:>7}: full {float(r['t_full_ms']):8.1f} | enc36 {float(e36[n]['single_total_ms']):8.1f} (x{float(e36[n]['speedup']):.2f}) | "
              f"enc78 {float(e78[n]['single_total_ms']):8.1f} (x{float(e78[n]['speedup']):.2f}) | FNO {float(r['fno_single_gpu_ms']):7.1f}  [study5b, study4b]")

    print("\nStudy 5a transfer (per-node 36-dim encoder, 3 repeats)")
    a5 = rows("study5a_2*.csv")
    for m in ("encoder_n100", "encoder_n400", "fno_n100", "fno_n400"):
        print(f"  {m:<14} " + ", ".join(f"r{r} {mean(a5, 'median_field', model=m, refine=r):.3f}" for r in (4, 5, 6, 7)) + "  [study5a]")

    print("\nStudy 9 (design loop; mean over 50 trials)")
    s9 = rows("study9_2*.csv")
    for pol in ("C1", "C20", "C200"):
        show(f"random full-wave search {pol}: regret", mean(s9, "regret", policy=pol), "study9")
    for sur in ("fno", "g12", "pod136"):
        print(f"  {sur:<7} A1 {mean(s9, 'regret', surrogate=sur, policy='A1'):.3f}  A5 {mean(s9, 'regret', surrogate=sur, policy='A5'):.3f}  A50 {mean(s9, 'regret', surrogate=sur, policy='A50'):.3f}"
              f" | B calls {mean(s9, 'calls', surrogate=sur, policy='B'):.1f} regret {mean(s9, 'regret', surrogate=sur, policy='B'):.3f}"
              f" | curse {mean(s9, 'bias_top1', surrogate=sur, policy='curse'):+.2f}  [study9]")

    print("\nStudy 10 (grid-base encoder; mean over 3 repeats)")
    s10 = rows("study10_2*.csv")
    for v in ("GB", "GBMR"):
        for n in (100, 400):
            vals = {r: mean(s10, "median_field", variant=v, n_train=n, refine=r) for r in (4, 5, 6, 7)}
            rng = [float(x["median_field"]) for x in s10 if x["variant"] == v and int(x["n_train"]) == n and int(x["refine"]) == 4]
            print(f"  {v:<5} n={n}: " + ", ".join(f"r{r} {x:.3f}" + (f" ({x / vals[4]:.2f})" if r != 4 else "") for r, x in vals.items())
                  + f" | refine-4 range {max(rng) - min(rng):.3f}  [study10]")

    print("\nSweep animation (median over 20 geometries x 61 frequencies)")
    sw = list(csv.DictReader(open("figures/anim_sweep_errors.csv")))
    for k in ("encoder_repredicted_median20", "encoder_basis_reused_median20", "fno_median20"):
        vals = [float(r[k]) for r in sw]
        print(f"  {k:<32} median over frequencies {st.median(vals):.3f}, r=0.2 {vals[0]:.3f}, r=0.6 {vals[-1]:.3f}  [anim_sweep_errors]")


if __name__ == "__main__":
    main()

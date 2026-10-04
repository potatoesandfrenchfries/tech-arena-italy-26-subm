"""Study 5b: cost of one FNO prediction next to the encoder pass and the full solve, refine 4 to 8.

Protocol is in the README ("Study 5b protocol"). Timing does not depend on the weights, so the FNO gets one epoch on
a few instances. The encoder numbers come from the latest Study 4b scaling CSV. Quiet machine; CPU parts single thread:
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python -m experiments.study5b
"""
import argparse
import copy
import csv
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch

from baselines import FNO
from compat_galerkin import fem
from compat_galerkin.device import get_device
from compat_galerkin.problems import Family, FamilyConfig
from experiments.scaling import eta_D, median_time
from experiments.study4b import Quiet, instance_on, settle, load1

RATIO, TAN = 0.4, 0.05


def fast_inputs(fam, elem_idx, inst, mu, sd, src_sigma):
    g = fam.grid
    eps = inst.eps_elem[elem_idx].reshape(g.X.shape)
    bump = np.exp(-((g.X - inst.params["src"][0]) ** 2 + (g.Y - inst.params["src"][1]) ** 2) / (2 * src_sigma**2))
    one = np.ones_like(g.X)
    x = np.stack([eps, inst.omega2.real * one, inst.omega2.imag * one, inst.params["direction"][0] * bump,
                  inst.params["direction"][1] * bump, g.X, g.Y]).astype(np.float32)[None]
    return torch.as_tensor((x - mu) / sd)


def encoder_numbers():
    rows = list(csv.DictReader(open(sorted(glob.glob("results/study4b_scaling_*.csv"))[-1])))
    return {int(r["refine"]): r for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refines", default="4,5,6,7,8")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    device = get_device("auto")
    on_gpu = device.type == "cuda"
    print(f"device {device}", flush=True)

    fam4 = Family(FamilyConfig(refine=4, grid=64))
    fno = FNO(epochs=1, batch=16, seed=0)
    tr = fam4.dataset(20, seed=400)
    info = fno.fit(fam4, tr, device=device)
    net_gpu = fno.net.eval()
    net_cpu = copy.deepcopy(fno.net).cpu().eval()
    torch.set_num_threads(1)
    print(f"FNO ready: {info['params']} parameters", flush=True)
    enc = encoder_numbers()
    settle()

    rows = []
    for refine in [int(x) for x in a.refines.split(",")]:
        t_start = time.time()
        fr = fam4 if refine == 4 else Family(FamilyConfig(refine=refine, grid=64))
        lam1 = fem.first_mode(fr.base)
        w2 = RATIO * lam1 * (1 + 1j * TAN)
        inst = instance_on(fr, w2)
        cav = fr.base.with_eps(inst.eps_elem)
        idx = cav.interior_edges
        K_int, M_int, f_int = cav.interior(cav.K), cav.interior(cav.M), inst.f[idx]
        d = M_int.diagonal()
        f_D = math.sqrt(float(np.sum(f_int**2 / d)))
        g = fr.grid
        elem_idx = fr.base.mesh.element_finder()(g.X.ravel(), g.Y.ravel())

        quiet = Quiet()
        t_full = quiet(lambda: fem.solve_fine(cav, w2, inst.f))
        x = fast_inputs(fr, elem_idx, inst, fno.mu, fno.sd, fr.cfg.src_sigma)

        def forward_gpu():
            y = net_gpu(x.to(device))
            if on_gpu:
                torch.cuda.synchronize()
            return y[0].cpu().numpy()

        def forward_cpu():
            return net_cpu(x)[0].numpy()

        with torch.no_grad():
            t_inputs = quiet(lambda: fast_inputs(fr, elem_idx, inst, fno.mu, fno.sd, fr.cfg.src_sigma))
            t_gpu = quiet(forward_gpu)
            t_cpu = quiet(forward_cpu)
            y = forward_gpu()
        field = np.stack([y[0] + 1j * y[1], y[2] + 1j * y[3]])
        t_map = quiet(lambda: g.grid_to_e(field))
        e = g.grid_to_e(field)[idx]
        t_eta = quiet(lambda: eta_D(K_int, M_int, d, f_int, f_D, w2, e))
        tot_gpu, tot_cpu = t_inputs + t_gpu + t_map, t_inputs + t_cpu + t_map
        e4 = enc.get(refine)
        enc_pass = 1e-3 * float(e4["t_pass_ms"]) if e4 else float("nan")
        enc_freq = 1e-3 * float(e4["t_freq_ms"]) if e4 else float("nan")
        row = {"refine": refine, "N": len(idx), "t_full_ms": 1e3 * t_full, "t_inputs_ms": 1e3 * t_inputs,
               "t_forward_gpu_ms": 1e3 * t_gpu, "t_forward_cpu_ms": 1e3 * t_cpu, "t_map_ms": 1e3 * t_map,
               "fno_total_gpu_ms": 1e3 * tot_gpu, "fno_total_cpu_ms": 1e3 * tot_cpu, "t_eta_ms": 1e3 * t_eta,
               "fno_single_gpu_ms": 1e3 * (tot_gpu + t_eta), "encoder_pass_ms": 1e3 * enc_pass,
               "encoder_single_ms": 1e3 * (enc_pass + t_eta), "encoder_per_freq_ms": 1e3 * enc_freq,
               "fno_per_freq_gpu_ms": 1e3 * tot_gpu, "params": info["params"],
               "max_load": quiet.max_load, "load_flag": quiet.flagged}
        singles = {"full": t_full, "encoder": enc_pass + t_eta, "fno_gpu": tot_gpu + t_eta}
        row["cheapest_single"] = min(singles, key=singles.get)
        row["encoder_over_fno"] = (enc_pass + t_eta) / (tot_gpu + t_eta)
        rows.append(row)
        print(f"refine {refine}: N = {row['N']}: full {row['t_full_ms']:.1f} ms | FNO gpu {row['fno_single_gpu_ms']:.1f} (inputs {row['t_inputs_ms']:.2f}, "
              f"forward {row['t_forward_gpu_ms']:.2f}, map {row['t_map_ms']:.2f}; cpu forward {row['t_forward_cpu_ms']:.1f}) | encoder "
              f"{row['encoder_single_ms']:.1f} | cheapest single: {row['cheapest_single']} | encoder/FNO {row['encoder_over_fno']:.2f} | per extra "
              f"frequency: encoder {row['encoder_per_freq_ms']:.2f} ms, FNO {row['fno_per_freq_gpu_ms']:.2f} ms | max load {quiet.max_load:.2f}"
              f"{' FLAGGED' if quiet.flagged else ''} ({time.time() - t_start:.0f}s)", flush=True)

    print("\nCheapest per single query by mesh: " + ", ".join(f"refine {r['refine']}: {r['cheapest_single']}" for r in rows))
    print("Prediction recorded in advance: the FNO is cheaper than the encoder per single query at the large meshes -> "
          + ("CONSISTENT" if all(r["fno_single_gpu_ms"] < r["encoder_single_ms"] for r in rows if r["refine"] >= 6) else "NOT CONSISTENT"))
    Path(a.out).mkdir(exist_ok=True)
    path = Path(a.out) / f"study5b_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

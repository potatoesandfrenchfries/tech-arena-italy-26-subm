# Baselines for the comparative study

Every method implements `Method.fit(family, train, device)` and `Method.predict(family, inst, cav)`
(see `base.py`) and is scored on identical instances with identical metrics
(`compat_galerkin/metrics.py`). Our learned-W method should plug in the same way, so the study is
a table, not a set of separate experiments.

| Name | What it is | What it controls for |
|---|---|---|
| `fine_fem` | Full fine-mesh solve | Accuracy ceiling and the cost to beat |
| `coarse_r{1,2,3}` | Same Whitney lift with W = hat functions of a coarse mesh (lowest-order Nedelec on the coarse mesh) | Does a learned W beat plain geometry at equal reduced dimension? |
| `pod_{16,64}` | POD of training solutions, then Galerkin (basis not compatible) | Value of the exact-sequence structure vs a data-driven basis |
| `fno` | Plain 2D FNO on a regular grid, no physics, no structure | Does Maxwell structure help at all? Needs a GPU to train properly |

## Not implemented yet

- FNO with a TE potential head (E derived from Hz), the potential-output baseline from the plan.
- Geo-FNO / GINO encoders for non-voxel geometry.
- PINO-style variant of the FNO with a weak-form residual loss.
- Learned-W Galerkin (ours).

## Reading the numbers

- Compare methods at equal `dim` where possible; `coarse_r*` dimensions are set by the mesh level.
- `rel_err` is the M-norm error against the fine solution. Report the median and p90, because the family spans resonances and the tail dominates.
- `residual` is relative to the load norm, so it is not comparable across frequencies without a stability constant.
- Use `gauss_r{L}`, not only `gauss_fine`: a compatible space is exact for L up to its own level and nothing in `pod`/`fno` is.
- The FNO round-trip through the grid has its own error floor (`roundtrip_floor`), reported by `fit`.
- Frequency range, contrast and loss are in `FamilyConfig` (`problems.py`); change them before comparing, not between methods.

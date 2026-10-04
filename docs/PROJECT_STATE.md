# Project state, decisions and plan

Snapshot written at the end of the experimentation session of 2026-10-04 (repo head `443f121`, 55 commits, 43 tests passing, nothing pushed; commit hashes in this file were updated after the history was regrouped into topic branches, the commit count and head refer to the pre-rewrite linear history, which is kept as tag `backup/pre-rewrite`). It is meant for whoever resumes the work: the project owner, a later Claude Code session, or a cloud session.

How this file relates to the others:

- `README.md` holds the registered protocols and the result tables, in the order they were run. It is the source of truth for numbers and rules. Its top section ("Current scope and hypothesis ledger") is the short status. Its "Abstract / Motivation / Methodology" sections still describe the original plan and have not been rewritten to the narrowed scope (see the plan).
- This file holds what the README does not: the story, the decisions and who made them, the corrections of earlier statements, the unresolved questions, the plan, and the operational pitfalls.
- `results/` is gitignored. Selected outputs were force-added, so each experiment's CSV and log are committed.

## 1. The project in one paragraph

Learned compatible Galerkin spaces for fast, Maxwell-consistent surrogate solving of 2D time-harmonic Maxwell problems (a PEC cavity with a dielectric disc, in-plane E, lowest-order Nedelec edge elements from scikit-fem). Following Geo-NeW, a partition of unity W produces Whitney 1-forms; the physics is assembled exactly on a fine mesh and only a small reduced space is learned. The brief it answers: faster optimisation and modelling of EM structures under data, cost and Maxwell-consistency constraints, while keeping simulation-level accuracy.

Current scope, decided by the project owner: resonance is out of scope and treated as a documented limitation. The main family is sub-resonant: omega^2 = r x lambda_1(instance) with r uniform on [0.2, 0.6] (lambda_1 is each instance's own first mode), refine-4 mesh (1,504 interior edges), loss tangent 0.05, random disc position, radius and contrast, and a Gaussian current source with random position and direction.

## 2. Where the hypotheses stand

| Hypothesis | Status |
|---|---|
| H1. Consistency by construction | Holds in the narrow form only: power balance, tangential continuity and the Gauss law hold exactly against the learned functions (Gate 0). The fine-mesh divergence error tracks the field error (1.3 to 2.5 times it). Against the FNO at matched accuracy (Study 4a): an advantage by construction only (power balance and PEC, which POD-Galerkin also has); on the Gauss law the encoder is 23 to 27% worse than the FNO. |
| H2. A small learned reduced space is accurate off resonance | Partly supported (Gate 2, PASS through criterion B only). A 36-dimensional encoder gets median field error 0.476 to 0.180 at 25 to 400 training instances: 25 to 47% below POD-36, below POD-136 from 50 instances, but criterion A (2x at 25 instances) not met, 4 to 11 times the oracle, and no size reaches 10% field error. |
| H3. Cheap sweeps | Mechanism holds and cost is measured. With the encoder pass included (Study 4b): single query 1.9x to 4.6x faster than a full solve from 24k to 393k unknowns, extra frequency 55x to 539x cheaper. But the FNO is far cheaper still (Study 5b: 26.5 ms against 1,437 ms for the encoder at 393k), and the encoder trained at refine 4 is not accurate on finer meshes (Study 5a: median field error 0.64 to 0.90 against 0.18 to 0.24 at refine 4; the FNO stays at 0.17 to 0.25). So the speed claim cannot currently be attached to an accurate encoder. |
| H4. Residual certificate, conformal calibration, fallback | Partly tested. Marginal coverage is valid for every surrogate including the learned ones (0.89 to 0.90). The residual tracks the error for POD (Spearman 0.93) but not for the coarse spaces, nor for the learned models (Study 4a: 0.26 to 0.34 for the encoder, 0.55 to 0.60 for the FNO); coverage drifts across the band; selection shift is repaired by recalibration. The FNO with the certificate avoids the most full-wave calls (38% at tau = 0.2, 400 instances, 1% false accepts); essentially none at 5%. |
| H5. Data efficiency and out-of-distribution accuracy against unconstrained operators | Not supported on accuracy: the encoder is 1% to 20% above the FNO's median field error at every size (25 to 400); its median energy error is lower at 25 to 100 instances and higher at 400. Study 4a found no compensating advantage (consistency only by construction; worse Gauss-law error; a certificate that tracks worse). Out-of-distribution untested. |
| H6. Fewer full-wave calls per finished design; manufacturability | Not built. Ideas only (section 8). |
| Resonant accuracy | Out of scope. Gates 1b to 1f did not reach it; cause of the last obstacle unresolved. |

Decision rule for H2 and H5, fixed in the README: if the encoder does not give at least a 2x error reduction against POD or coarse FEM at the smallest training-set size, or the same accuracy at half the dimension, the thesis rests on the verification layer and the stability diagnosis, and no further learned-W variations are tried.

## 3. Key findings, with numbers

All errors are relative M-norm errors against the fine-mesh solution. "Energy" is the relative error of the stored energy. Details and caveats are in the README section named in brackets.

**Structure (Gate 0 and the consistency check)**
- Gauss law against the learned functions about 1e-13, power balance about 1e-14, tangential trace exactly 0; a random-function control fails by 0.1 to 0.7.
- Fine-mesh Gauss error of the oracle W: 0.03 to 0.16, about 1.3 to 2.5 times its field error. So at fine level the divergence error behaves like approximation error. "Consistent by construction" is exact only in the reduced 0-form space.
- scikit-fem's N1 coefficient is the circulation from the higher to the lower vertex (a uniform -1 sign); a first draft would have been silently wrong without the test.

**Capacity off resonance (Gate 1, band pre-check)**
- Oracle W (fitted per instance, reference in the loss): 10 dimensions give 2.9% field error at 0.4 lambda_1; 36 dimensions give 6.8% between modes 1 and 2. The non-learned coarse Nedelec space needs 368 dimensions for 26 to 33%. Upper bound on one instance.
- Band pre-check (oracle, n = 8): energy error 0.001, 0.003, 0.007, 0.018, 0.042 at r = 0.2 to 0.6 (passes), 0.080 at 0.7 and 0.845 at 0.8 (fails). The edge at 0.6 is marginal. The jump at 0.8 may be an optimiser failure.

**Resonance: what was learned (Gates 1b to 1f), and why it stopped**
- The field is representable near resonance (best-approximation error 0.015 to 0.023) but the Galerkin solve in the same space has error about 1; a least-squares solve does not help. Adding the two exact resonant eigenvectors restores the floor (error 0.014 at n = 16), so the failure is the reduced spectrum, not the field.
- Field-error losses leave no reduced eigenvalue in the band; the projection loss creates ghost eigenvalues at 8.3 to 8.5, matching the worst held-out frequencies.
- A spectrum-matching loss cut the error from about 1.0 to about 0.29 (energy 0.13 to 0.15), but nothing passed 5%.
- Error decomposition: 88 to 90% of the squared error is in the curl-free component, although the true solution is 92% resonant pair and only 8% curl-free. The resonant pair is right to about 9% in amplitude.
- Adding exact coarse-hat gradients after training made the error worse (0.29 to 0.68) by shifting the resonant eigenvalues out of the half-width. Placing them in the loss (Gate 1d) left the floor unchanged (0.285 to 0.295). The curl-free part comes from a localised source, so coarse-hat gradients represent it slowly: error 0.83, 0.55 and 0.34 with 5, 25 and 113 functions.
- Hodge split (Gate 1e: exact curl-free solve, learned transverse space) and an energy-norm capture loss (Gate 1f) both failed (held-out error 0.907 and 0.908). The reduced eigenvalue of a transverse space is bounded below by the true one (min-max), so Gate 1c's "spectrum fixed" had partly relied on curl-free contamination. The energy-norm capture error of the resonant pair saturated at 0.074 at every size and weight, which accounts for the 9% eigenvalue shift (Rayleigh-Ritz estimate 0.64 against 0.67 observed). Whether 0.074 is a structural limit of the lifted-Whitney family or an optimiser trap was not resolved.
- Conventional comparator: exact curl-free solve plus the ten lowest exact transverse modes gives 1.6% error and 3e-4 energy error, at one eigensolve per geometry.
- Result: hard stop, as pre-registered. Twelve Hodge-split configurations tried, none improved. The project owner then declared resonance out of scope.

**Verification study (H4)** (surrogates: coarse_r2, coarse_r3, POD 16, 64, 128; 5 repeats of 300 train, 1000 calibration, 1000 test)
- Residual indicator (diagonal-mass norm, O(N), spectrally equivalent to the dual norm): Spearman correlation with the true error 0.92 to 0.93 for POD, but -0.33 and -0.16 for the coarse spaces. Cause not investigated.
- Marginal conformal coverage valid for all five (0.897 to 0.909 at alpha = 0.1; 0.946 to 0.955 at 0.05), including surrogates the indicator does not track (then the bound is uninformative). The naive uncalibrated bound is neither reliable nor sharp.
- Coverage varies by up to about 30 points across r: POD under-covers near the upper band edge (POD 16: 0.71 at r in [0.5, 0.6]), the coarse spaces under-cover at low r. The marginal guarantee hides this; r is not known at test time without an eigensolve.
- Selection by surrogate score (best-of-N proxy for optimiser shift): coverage drops for coarse and POD 16 (0.79 to 0.83) and rises for POD 64 and 128; recalibrating on the selected part of the calibration set restores 0.88 to 0.93 (noisy). Field energy in the target region is overestimated by 290 to 1130% for the POD selected sets (the optimiser's curse); none of them is certified.
- No full-wave calls avoided at 2 to 10% accuracy targets with these surrogates (POD 128 accepts 15.6% at 20%), because their errors are 0.28 to 0.76.
- Registered verdicts: coarse spaces DOES NOT TRACK; all POD NEEDS RECALIBRATION. The rule was flawed: its symmetric tolerance counted over-coverage (POD 64 and 128) as failure. Reported as registered, with the nuance.

**Scaling study (cost half of the speed argument)** (single thread, SciPy sparse direct solve as the full-wave reference)
- Full solve 2.7 ms at 1.5k unknowns, 6.8 s at 393k (slope 1.42); residual certificate 0.03 ms to 16.7 ms (about 0.25% of a solve, slope 1.15); reduced setup slope 1.1 to 1.2.
- Single-query speedup at 393k: 20x (dim 16), 6.7x (dim 64), 3.2x (dim 128), 5.6x (coarse_r2), 1.04x (coarse_r3, dim 368). Per extra frequency: 1224x, 361x, 207x. A certified pipeline stays cheaper with up to 85% fallbacks (dim 64) at 393k.
- Caveats: SciPy's solver is slower than production solvers, which would shrink the gaps; the cost of obtaining the basis (POD snapshots, or an encoder pass plus the Whitney lift) is not included and is the main missing term; 2D only.

**Baseline data efficiency (FNO grid 64 and POD)** (3 repeats, 300 test instances; median field error / median energy error)

| Training instances | FNO | POD, rank 136 |
|---|---|---|
| 25 | 0.448 / 0.230 | 0.665 / 0.159 (rank 50) |
| 100 | 0.241 / 0.095 | 0.265 / 0.039 |
| 400 | 0.150 / 0.049 | 0.273 / 0.054 |

- No method reaches 10% median field error with up to 400 instances. The FNO keeps improving (about 20% per doubling); POD plateaus from 100 instances, consistent with a solution manifold too large for a fixed basis of at most 136 dimensions (untested).
- Errors do not depend on r (FNO 0.14 to 0.16 in every bin), so the baselines' errors are not resonance leakage. The 5% and 10% targets set in advance are harder than the family allows at these data sizes.
- An FNO on a 32 grid would have an 8% round-trip floor by construction; grid 64 gives 0.1 to 0.2%.

**Gate 1b stage (b): one shared W across geometries** (200 train and 300 test per repeat, 3 repeats)
- Shared W: median field error 0.84, 0.69, 0.60, 0.52 at dimension 10, 36, 78, 136; no instance reaches 5% energy error at any size. POD of the same dimension: 1.01, 0.28, 0.34, 0.29 (energy under 5% for 39 to 48% of instances from dimension 36).
- Per-instance oracle W (repeat 0, 20 instances): median field error 0.044, median energy error 0.024, 17 of 20 instances under 5% energy error, at both n = 8 and n = 16.
- Verdict by the registered rule: ENCODER NEEDED. Structure against POD by the rule: COMPARABLE (ratio 0.83, 2.51, 1.80, 1.83; only one size above 2); the shared W is nevertheless 1.8 to 2.5 times worse than POD at every size from n = 8.
- The oracle adapts to the whole instance (disc, source, frequency) and costs 500 optimiser steps per instance; it shows what is representable, not what an encoder will learn.

**Gate 2: the encoder** (width 24, 1000 steps, 21,720 parameters, dimension 36; selected on validation seeds 900 and 901; README "Gate 2 results")
- Median field error / energy error, mean of 3 repeats, at 25, 50, 100, 200, 400 training instances: 0.476/0.176, 0.352/0.109, 0.243/0.078, 0.204/0.065, 0.180/0.058. FNO: 0.448, 0.320, 0.241, 0.188, 0.150. POD-136: 0.665, 0.378, 0.265, 0.285, 0.273. POD-36: 0.634, 0.556, 0.461, 0.276, 0.248.
- Verdict PASS through criterion B only. Criterion A: 0.476 against the bar 0.317, 0 of 3 repeats. Criterion B: ratios to POD-136 of 0.93, 0.92, 0.72, 0.66 at 50 to 400 (the first two are small margins). No size reaches 10% field error.
- Reading: a modest gain over a data-driven basis of the same size, no accuracy advantage over the FNO (1.01 to 1.20 times its error; energy error at 400 is also above the FNO's and POD-136's). Overfits at 25 instances (train 0.071, test 0.476); at 400 train 0.159 against test 0.180, so it is limited by optimisation or capacity there. 4.1 to 10.8 times the oracle's 0.044.
- The 64.7 ms encoder `predict` printed by the Gate 2 script was taken under load and is NOT a result (see section 5, item 13). Study 4b measures it.

**Study 4a: consistency and certificate for the learned models** (encoder, FNO, POD-136 at 100 and 400 instances; 3 repeats; 1000 calibration and 1000 test instances)
- Consistency verdict BY CONSTRUCTION ONLY. Power balance: encoder and POD-136 at 5e-15, FNO 2.3 (median); PEC violation FNO 2 to 4%; Euclidean residual encoder about 3, FNO about 30, POD 1.2 to 1.5. Gauss law at matched accuracy (both models with field error in [0.10, 0.30]): encoder 0.26 to 0.29 against FNO 0.21 to 0.24 (1.23 and 1.27 times higher), the wrong direction for the registered bar.
- Certificate: DOES NOT TRACK for the encoder (Spearman 0.26 at 100, 0.34 at 400) and the FNO (0.55, 0.60); POD-136 0.93 (NEEDS RECALIBRATION). Marginal coverage 0.893 to 0.900 for all. Coverage by r band: encoder 0.77 to 0.80 at the lowest band rising to 0.99; FNO 1.00 falling to 0.67; POD 0.99 falling to 0.81. Acceptance at tau = 0.2 (400 instances): FNO 38%, POD-136 15.5%, encoder 9%. Verified top-10 needs 10 full-wave calls for every model.
- Cause of the encoder's non-tracking: not tested. Guess: a learned W concentrates the space where training solutions have energy, so the residual of an instance outside that region does not scale with its error.

**Study 4b: encoder cost** (second run; single thread; load at most 0.99)
- Profile at refine 4: 57 to 63 ms of the 62 to 66 ms `predict` was `Family.grid_inputs` (point location of 4,096 grid points); a cached lookup gives a 3.5 ms pass with identical output. The FNO showed the same ~60 ms in Study 4a for the same reason.
- Encoder pass including the residual against a full solve: 0.69x (1.5k), 0.88x (6k, break-even), 1.93x (24k), 3.06x (98k), 4.64x (393k). Per extra frequency 0.04 ms to 12.4 ms, 55x to 539x cheaper than a full solve. A first run timed the per-frequency solve with a complex copy of the basis per call (98.6 ms at 393k); corrected and rerun, both kept.

**Study 5b: cost of the FNO next to the encoder** (GPU forward pass, one CPU thread elsewhere)
- Single query including the residual at 393k: full solve 6,623 ms, encoder 1,437 ms, FNO 26.5 ms (the residual is 16.5 ms of that). The FNO is cheapest at every mesh from 1.5k to 393k (prediction recorded in advance: consistent). Per extra frequency the encoder and FNO are about equal at scale (12.4 and 10.1 ms).

**Study 5a: transfer to finer meshes** (trained at refine 4, tested at refine 4 to 7 with references on each mesh, 200 instances, 3 repeats)
- Encoder median field error 0.242 -> 0.691 -> 0.822 -> 0.896 (100 instances) and 0.180 -> 0.638 -> 0.781 -> 0.870 (400). Ratios 2.9 to 4.8 against the bar of 1.25: the transfer fails at every finer mesh. FNO 0.238 -> 0.245 -> 0.251 -> 0.250 and 0.144 -> 0.166 -> 0.173 -> 0.170 (ratios 1.03 to 1.20). Non-learned coarse_r3 0.306 -> 0.361 (more accurate than the transferred encoder).

**Study 5c (variant A1): smooth base transfer** (two of three repeats; stopped by the owner)
- Linear interpolation of the base logits instead of nearest node: ratios 2.4 to 4.1 against 2.9 to 4.9 (up to 25% lower error at 400 instances, almost nothing at 100 at the finest meshes). Outcome NO EFFECT by the registered rule. The piecewise-constant transfer is not the main cause.

**Study 6a (variant A2): mesh-free encoder** (analytic RBF base with trained centres and widths plus the CNN correction, no per-node parameter; `baselines/encoder_whitney.py::MeshFreeEncoder`; stopped in repeat 2)
- Median field error 0.609 -> 0.666 -> 0.729 -> 1.002 (100 instances; refine 7 from two repeats) and 0.539 -> 0.602 -> 0.644 -> 0.808 (400): ratios 1.09 to 1.35 against 2.9 to 4.8 for the per-node encoder, so the per-node base logits were the main cause of the transfer failure. Cost: refine-4 error 0.54 to 0.61 against 0.18 to 0.24, and a huge spread between seeds (0.35 to 0.76 at refine 4); an optimisation fragility, not a tested capacity limit. Candidate repair, not run: a learnable grid base (about as many parameters as the per-node base, sampled at any node coordinates).

**Study 6b (variant C1): the FNO's field added to a coarse Whitney space, then Galerkin** (`baselines/enriched_galerkin.py`)
- At refine 4 the enriched solution is 0.33 to 0.34 (r3) and 0.69 to 0.73 (r2): 1.4 to 4.9 times the FNO and worse than the coarse space alone (0.31 and 0.56). Consistency restored (power balance 1e-14, residual 2 to 9 against 30), certificate NEEDS RECALIBRATION for r2 (Spearman 0.87, 0.92) and does not track for r3. Registered outcome FAILS. Hypothesis: a Galerkin solve in an indefinite problem is not monotone in the space when the added field is incompatible. Not run: adding only the FNO's curl part or projecting out its gradient content; the FNO as a warm start for an iterative solver.

**Study 7: is the encoder under-fitted?** (4 configurations of 1500 steps; per-node encoder at refine 4)
- 12 bumps (78 dimensions, `G12`): median field error 0.216 (100 instances) and 0.145 (400), matching the FNO (0.241, 0.150) and with the lower energy error (0.064 and 0.044 against 0.095 and 0.049). 8 bumps with 1500 steps (`G8`): 0.234 and 0.164 (9% better than the 1000-step Gate 2 encoder, just short of the registered 10%); training error down 12%, so under-fitting is supported. Projection pre-training hurts (up to 51% worse). The best-approximation error of the predicted space (0.10 to 0.19) is within a factor of 1.2 to 1.45 of the Galerkin error: the predicted space is the bottleneck, not the solve.

**Study 8: do the Study 7 gains survive the checks?** (3 repeats; stage A 1000 test instances; stage B transfer; stage C cost)
- Parameter matched: `G12` (24,000 parameters) has 0.60 and 0.63 times the error of an FNO of 20,452 parameters, and 0.91 and 0.95 times that of the 1.19M-parameter FNO. Consistency BY CONSTRUCTION ONLY (Gauss law 1.0 to 1.2 times the FNO's). Certificate DOES NOT TRACK (Spearman 0.40 and 0.49), but the better model accepts 61% of instances at target 0.2 with 2.7% false accepts. Transfer still fails (ratios 2.8 to 5.6; the small FNO stays within 1.01 to 1.06). Cost of the 78-dimensional encoder: 0.38x, 0.47x, 0.68x, 1.17x, 2.24x of a full solve at 1.5k, 6k, 24k, 98k, 393k unknowns (the 36-dimensional one: 0.69, 0.88, 1.93, 3.06, 4.64).

**Study 9: a first design loop** (gradient-free; discs with four parameters; 50 trials)
- Screening a pool of 1000 designs with a surrogate and verifying the top candidate by full-wave reaches mean regret 0.064 (FNO) and 0.013 (`G12`) with 1 call; random full-wave search needs 200 calls to reach 0.10; POD-136 needs 50. Certificate gating (28 to 41 calls) adds no value over plain screening. Optimiser's curse: FNO +10%, `G12` -7%, POD-136 +1210%. At 393k unknowns the FNO projection is about 40 times cheaper than random search; `G12` is not (screening costs 3 s per design). Not done: gradient-based design, ports, manufacturability.

**Sweep animation (descriptive)**: reusing a basis predicted at r = 0.4 across r = 0.2 to 0.6 costs no accuracy (median field error 0.146 for reuse and for re-prediction, FNO 0.151; 20 geometries, refine 4), which backs the cheap-extra-frequency cost numbers.

**Study 10: grid-base mesh-free encoder** (learnable 32 x 32 grid base, 29,748 parameters; `GB` trained at refine 4, `GBMR` at refine 4 and 5)
- Both variants FAIL the registered rules (parity within 1.15x of the per-node `G12` at refine 4, ratio at most 1.25 at refine 6 and 7). Training on two meshes helps: refine-7 error 0.377 (100) and 0.306 (400), ratios 1.33 and 1.56, against 3.8 to 5.6 for per-node encoders; refine-4 accuracy is 33% to 36% worse than `G12` (0.283 and 0.195 against 0.213 and 0.143); refine-4 spread 0.08 (`GB` at 100 instances: 0.19, unstable). The FNO still leads at 98k unknowns (0.25 and 0.17). Not run: training on more meshes, or on the target mesh.

Test datasets for refine 4 to 7 are cached in `results/cache/`. Figures 6 to 11 (`figures/`) match slides 4a to 4e.

## 4. Decision log

| Decision | By | Where |
|---|---|---|
| Python, not Julia, for the simulation | Claude, accepted | first session |
| Gate-based pre-registration: protocols committed before any run; failures recorded | agreed in the original plan | README |
| Include resonance in the family (band 2 to 12) | project owner, 2026-10-04 | later superseded |
| Hard stop and fallback after Gate 1f, with a second attempt (Gate 1f) allowed once | project owner (option 1) | README, Gate 1f |
| Ignore resonance: treat it as a limitation of the simulation; main family sub-resonant | project owner, 2026-10-04 | README scope section |
| Band r in [0.2, 0.6] | registered rule (band pre-check) | README |
| Inverse design and manufacturability: ideas only, nothing built | project owner | section 8 |
| FNO baseline: do it; thesis figures: do them; conditional calibration: only after the CPU job | project owner | this session |
| No attribution lines in commit messages from 2026-10-04 | system instruction | commit history |

## 5. Corrections of earlier statements

1. "The surrogates are not faster than the full solve in 2D": withdrawn. It came from the baselines' `predict` (matrix conversion and a K re-projection on every call). A leaner reduced solve is faster even at 1.5k unknowns.
2. "The coarse Nedelec spaces have no eigenvalues below lambda_1": imprecise. Their first pair is 7.07, 7.28, 7.51 (a shifted copy of the true mode), converging upward.
3. Resonance half-width 0.19: wrong. It is 0.05 x omega^2, about 0.38 to 0.39.
4. "A single reduced mode, against the degenerate pair, explains the floor": refuted by Gate 1c (one and two modes give the same 0.29).
5. Gate 1c "spectrum fixed" was partly spurious (curl-free contamination), shown by Gates 1e and 1f.
6. "Adding exact gradients should fix the floor": it made it worse after training and changed nothing inside the loss.
7. Differences between the macOS and Linux runs of Gate 1 are not random initialisation (seeds are fixed): they are platform effects, up to about 15% in the worst entry.
8. A test caught an off-by-one in the conformal quantile (`np.quantile` indexes by position, not rank).
9. The verification verdict rule counted over-coverage as failure: a flaw in the rule as registered.
10. A figure title claimed 10 to 36 learned dimensions beat the coarse space; at the gap frequency only 36 do. Corrected.
11. A task-notification once said the Gate 1f job was stopped; the log showed it had completed. Always check the log.
12. The first FNO launch silently failed (see section 10); relaunched.
13. I stated that one encoder `predict` (64.7 ms) was "about 25 times slower than a full solve". The timing was unreliable (taken inside a training job under load) and, as Study 4b then showed, the cost was the grid-input construction, not the learned basis: the optimised pass is 3.2 ms at refine 4. Withdrawn. Wall-clock fit times in the Gate 2 log (77 to 282 s) reflect load.
14. After Study 4a I said the FNO "outperforms on all metrics". Not true: the encoder has the lower median energy error at 25 to 100 instances and satisfies the Galerkin identities (power balance, PEC, a 10 times smaller residual). The README reading was completed (commit "energy error comparison with the FNO is mixed").
15. I started a smoke test while a study was running and pushed the load average to 33 on 20 cores, slowing Study 5c's refine-7 step from about 330 s to 750 s. The rule "one PyTorch job at a time" applies to smoke tests too.
16. The Study 4b per-frequency timing in the first run was inflated by an implementation detail (a complex copy of the basis per call); corrected and rerun, both runs kept.

## 6. Open questions

- Can the encoder close more of the gap to the oracle (4 to 11 times it now)? At 400 instances it underfits (train 0.159, test 0.180), so a longer or larger model may help; untested, and would need its own registered protocol and validation-only tuning.
- Why does the encoder not transfer to finer meshes? Not the nearest-node transfer of the base logits alone (Study 5c). Remaining candidates: the per-node base as such (Study 6a removes it), the lift on a finer mesh, the CNN's grid inputs.
- Can a hybrid keep the FNO's accuracy and cost and restore consistency and a tracking certificate (Study 6b, variant C1)?
- Why does the residual not track the error of the learned models (Study 4a)? Guess only.
- Why does the residual anti-correlate with the true error for the coarse spaces? (Guess, untested: they cannot resolve the localised source, so the residual does not scale with the error.)
- Is the capture error 0.074 of the resonant pair a structural limit of the lifted-Whitney family or an optimiser trap? (Excluded by the hard stop; matters only if resonance is reopened.)
- Why is POD-136 flat or non-monotone (0.265, 0.285, 0.273)? (Guess: a solution manifold too large for a fixed basis, plus Galerkin instability; untested.)
- How much do the crossover and speedups shrink with a production direct solver (UMFPACK, PARDISO, MUMPS)? And what does the basis-prediction cost add?
- Should the 5% and 10% targets be relaxed or reframed, given that no baseline meets 10% field error at 400 instances?
- Can conditional calibration remove the coverage drift across r without an eigensolve?

## 7. Plan of action (ordered)

0. **Status (2026-10-04, later):** Gate 2, Studies 4a, 4b, 5a, 5b and 5c are done and written up in the README (work is on the branch `study4b-cost`, not yet merged into `main`). Studies 6a (A2, mesh-free encoder: transfer largely repaired, accuracy and seed stability poor; outcome NO EFFECT, stopped by the owner in repeat 2) and 6b (C1, the FNO added to a Whitney space: consistency restored, accuracy 1.4 to 4.9 times the FNO's error; outcome FAILS, so its cost stage was not run) are done and written up. Studies 7 (under-fitting: 12 bumps and 1500 steps match the FNO at refine 4), 8 (parameter-matched, certificate, transfer and cost checks) and 9 (a first design loop) are done and written up too, and the README top (abstract, results at a glance, motivation, methodology) was rewritten to the narrowed scope. Slides for the presentation are in `docs/SLIDES.md` (complete, no open slots) and the figures in `figures/`. Everything is on the branch `study4b-cost`, not merged into `main` by me. Then conditional calibration (item 2), figures, README top, abstract. Honest summary so far: the compatible encoder is a modest gain over POD but not a better surrogate than the FNO, which is as accurate or better, far cheaper at scale and transfers across meshes; the encoder keeps exact equation satisfaction and the oracle's headroom (0.044). Presentation constraint: the deck cannot be edited after submission, so only claims backed by committed numbers go in, and an animation (frequency sweep first) should be built from the final model. Model weights are not stored by the older scripts; Study 5c and 6a save encoder weights to `results/models/` (gitignored).
1. **Gate 2: the encoder (done; the original plan is kept below).** Register the protocol first, then implement. Sketch: input channels as in `Family.grid_inputs` (epsilon map, real and imaginary omega^2, source components, coordinates); output n channels of logits on the grid, sampled onto mesh nodes, then the softmax partition of unity, the Whitney lift, whitening and the reduced solve exactly as in `SharedWhitney`; trained with the Galerkin error against the reference (optionally with a projection term) on the nested training sets of the data-efficiency study (25 to 400), tested on the same test sets (seeds 500 + r); dimensions 36 (n = 8) and 136 (n = 16). The bar: median field error at 25 training instances at least 2x below POD (about 0.32 or lower), or equal accuracy at half the dimension; compare with the FNO. Cost must include the encoder pass and the lift, timed with the scaling harness. If the bar is missed, apply the decision rule (section 2).
2. **Conditional calibration for the certificate** (project owner: after the CPU job). Candidates: Mondrian (binned) conformal on cheap features such as omega^2, contrast, radius or eta itself; selection-conditional calibration with larger calibration sets; evaluate on POD, the FNO and later the encoder. Register first; metrics are coverage per r bin and on selected sets.
3. **Figures**: add the data-efficiency learning curves (with the oracle and shared-W references) and the error-against-dimension figure for the family; keep the style, view each figure, keep the CSV table views.
4. **Rewrite the README top** (Abstract, Motivation, Methodology) to the narrowed scope; the ledger at the top is current, the sections below it are the original plan.
5. **Thesis**: outline in section 8; build figures from the committed CSVs.
6. **Housekeeping**: correction (2026-10-04): `origin/main` is already published and equals local main (verified with `git ls-remote`; its reflog shows "update by push" after commits, and earlier fetch and pull fast-forwards). I did not push and there are no hooks or push config, so something outside my commands pushes (an editor sync or the owner); do not assume local-only history, and do not rewrite or force-push main. Add matplotlib to an optional dependency group in `pyproject.toml` (figures currently use `uv run --with matplotlib`), consider a `.gitattributes` for the harmless LF/CRLF warnings.
7. **Parked**: inverse design and manufacturability (ideas below), 3D, a production-solver comparison.

## 8. Ideas and discussions kept for later

**Does the project stand a chance against the brief?** Modest as it stands: the brief is about design (faster optimisation, inverse design, manufacturability) with simulation-level accuracy, and no design loop exists. The best fit is the verification layer: a cheap surrogate screens, the residual certificate accepts or rejects, uncertain candidates fall back to full-wave, and the metric is full-wave calls per finished design at equal figure of merit. To be competitive it needs an end-to-end loop, a calibrated certificate (including on optimiser-selected candidates), a fabrication-aware yield measure, and honest scope. The main risk is the fallback rate for designs that drift toward resonance.

**Selling points, ranked by evidence:** (1) a cheap calibrated certificate (0.25% of a solve at 393k), valuable only with a surrogate accurate to a few per cent; (2) amortisation over sweeps and geometries (shared by any reduced basis); (3) a structure-preserving basis predicted from geometry with no snapshots or eigensolve (the only distinctive piece, unproven).

**Inverse design and manufacturability ideas (not built):**
- Treat epsilon as a density on the fixed fine mesh; K and M are linear in epsilon, so adjoint gradients are cheap.
- A differentiable fabrication model: a smoothing filter plus a projection step (erosion and dilation); yield by perturbing the printed shape within a tolerance and re-evaluating.
- A screen-and-certify loop with the certificate gating each candidate; metric: full-wave calls per finished design.
- Fabrication-aware training: random erosion and dilation during training so gradients stay reliable near fabrication perturbations.

**Thesis outline:** 1 Introduction (the brief, the gap, research questions mapped to the gates); 2 Background (Nedelec and Whitney forms, Galerkin stability for indefinite problems, neural operators, Geo-NeW, conformal prediction); 3 Method (learned partition of unity, lift, reduced solve, certificate and fallback); 4 Evaluation design (the gate ladder, baselines, metrics, pre-registration); 5 Results by gate, with the stability diagnosis as its own section; 6 Discussion (when it works, cost-benefit, threats to validity); 7 Conclusion; appendix with the protocol registry and reproducibility details. Framing options by outcome: the method works; works off resonance only; fails to beat the baselines (a documented negative result with a mechanism).

**Figures built so far** (`figures/`): band pre-check, oracle capacity against coarse spaces, verification (coverage drift and Spearman), scaling, error decomposition. Style: fixed-order validated palette, marker shapes as a second encoding, table-view CSV for each.

## 9. Protocol registry

Every protocol was committed before the run it governs. Results follow in the commits after it.

| Study | Protocol commit |
|---|---|
| Gate 1 (oracle capacity) | `92604c8` |
| Gate 1b stage (a) (shared W across a band with a resonance) | `8cc4a3f` |
| Gate 1c (spectral loss) | `1c82205` |
| Family band (resonance included, later superseded) and Gate 1d | `c01c3ae` |
| Gate 1e (Hodge split) | `b76aab2` |
| Gate 1f (energy-norm capture, second attempt, hard stop) | `1dc0631` |
| Scope decision (resonance out), ledger, band pre-check | `aaff23b` |
| Verification study | `8082a7d` |
| Scaling study | `ef2f1ef` |
| Gate 1b stage (b) (shared W across geometries) | `4bce926` |
| Baseline data efficiency (FNO and POD) | `4676169` |
| Gate 2 (encoder) | `08e244c` |
| Study 4a (consistency and certificate, learned models) and Study 4b (encoder cost) | `02dbf1d` (4b clarification `899fe01`) |
| Study 5a (transfer accuracy) and Study 5b (FNO cost) | `fc0b4a9` (branch `study4b-cost`) |
| Study 5c (smooth base transfer, A1) | `996ad43` (branch `study4b-cost`) |
| Study 6a (mesh-free encoder, A2) and Study 6b (FNO added to a Whitney space, C1) | `4d00312` (branch `study4b-cost`) |
| Study 7 (under-fitting, projection pre-training, more bumps) | `0589bf9` (branch `study4b-cost`) |
| Study 8 (parameter matching, certificate, transfer, cost of the 12-bump encoder) | `221d9cb` (branch `study4b-cost`) |
| Study 9 (screen-and-certify design loop) | `e904e4b` (branch `study4b-cost`) |
| Study 10 (grid-base mesh-free encoder, multi-resolution training) and the sweep animation note | `f279f66` |

## 10. Operational notes

- Machine: Windows 11 with WSL2 Ubuntu; 20-core Intel Core Ultra 7 255HX, 15 GB RAM, RTX 5050 laptop GPU (8 GB). The repo is at `~/GitHub/huawei-tech-arena-italy-2026/huawei-tech-arena-italy-2026` in WSL (UNC path `\\wsl.localhost\Ubuntu\...`). The first session ran on a Mac (paths under `/Users/tomato/...`).
- Run everything through WSL: `wsl -d Ubuntu -- bash -lc 'cd ~/GitHub/huawei-tech-arena-italy-2026/huawei-tech-arena-italy-2026 && uv run ...'`. Git Bash on Windows has no `python3`; WSL does. Tests: `uv run pytest -q` (52 pass; 9 are in `tests/test_learned_bases.py`). Figures: `uv run --with matplotlib python figures/make_figures.py`. Experiments: `uv run python -m experiments.<name>`.
- Background jobs must use the shell tool's own background option. A `nohup ... &` inside a `wsl` call does not survive (the FNO study silently never started).
- Do not run two PyTorch CPU jobs at once: each opens about 60 threads, the load average reached 33 on 20 cores, and one job stalled for about 20 minutes. Run them one at a time, or lower one's priority (`renice`). Timing-sensitive runs need `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` and an otherwise idle machine. The GPU job is separate but its CPU phases still compete.
- `results/` is gitignored; commit outputs with `git add -f <files>`. Commit style is `add: ...` or `update: ...`, with no attribution lines.
- A recursive `grep` over the repo walks `.venv` over the network share and is very slow; search with a path restricted to the source folders.
- LF/CRLF warnings on commit are harmless (a Windows git setting). Commit with Windows git (PowerShell tool): WSL git sees every working-tree file as modified (CRLF against LF) and would produce whole-file diffs.
- The owner runs other heavy work on this machine (an "ai swarm" repo with transformer tests). Load WSL cannot see invalidates timings; Gate 2's own job uses about 17 of 20 cores by default. For jobs that must coexist, cap threads (`OMP_NUM_THREADS`, `torch.set_num_threads`).
- Smoke tests count as torch jobs: never run one while a study is running. The owner has said it is fine to kill a run once its outcome is clear (Study 5c was stopped after two of three repeats; record the deviation in the README). Timing studies (4b, 5b) need a quiet machine; accuracy studies do not, but contention slows them 2 to 5 times.
- `results/cache/` holds pickled test datasets with full-wave references (`cached_dataset` in `experiments/study6a.py`); the refine-7 set takes about six minutes to build. `results/models/` holds encoder weights saved by Studies 5c and 6a. Both are gitignored.
- Seeds are fixed; results differ by up to about 15% between macOS and Linux for the worst entry, and the Gate 1 optimiser is fragile (the worst of four starts is often far off).
- Working method that held up: register the protocol with its failure criterion before running; run a tiny smoke test first; check implementation sanity (exact solutions, controls); report failures as findings; write the result and the reading into the README; commit the code, the README and the outputs separately; look at every figure before accepting it.

## 11. Repository map

- `compat_galerkin/`: `fem.py` (operators, solve, `first_mode`, `region_mass`), `lift.py` (partition of unity, Whitney lift, whitening, reduced solve), `oracle.py` (per-instance W optimisation), `problems.py` (the family and its sampler), `metrics.py`, `certify.py` (residual indicators, conformal quantile), `checks.py` (Gate 0), `device.py`.
- `baselines/`: `Method` interface; `fine_fem`, `coarse_whitney`, `pod_galerkin`, `fno`, `shared_whitney`, `encoder_whitney` (CNN on grid inputs plus shared base logits; `MeshFreeEncoder` replaces the per-node base by an analytic RBF), `enriched_galerkin` (the FNO's field added to a coarse Whitney space).
- `experiments/`: one script per gate or study (`gate1*.py`, `band_precheck.py`, `verification.py`, `scaling.py`, `data_efficiency.py`, `gate1b_geometry.py`, `gate2.py`, `study4a.py`, `study4b.py` (also holds the mesh-agnostic encoder `Runner`), `study5a.py`, `study5b.py`, `study5c.py`, `study6a.py`, `study6b.py`) plus the post hoc diagnostics (`diag_*.py`).
- `tests/`, `figures/`, `results/` (gitignored, outputs force-added), `docs/` (this file), `README.md`.

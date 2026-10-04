# Learned compatible Galerkin spaces for fast, Maxwell-consistent surrogate solving

Status: experimentation stage, largely complete for the sub-resonant 2D family. Gates 0 to 2, the verification, scaling and baseline studies, and Studies 4a to 10 are done and written up below, with a sweep animation. Resonance is out of scope: it is treated as a limitation of this study, documented by Gates 1b to 1f, and the family is restricted to a sub-resonant band (see "Current scope and hypothesis ledger"). Every protocol and failure criterion was fixed before its run.

Reading guide: the hypothesis ledger below is the short status; the abstract and "Results at a glance" summarise the findings; the original plan (metrics and gate table) follows; then every protocol and its results, in the order they were run (each protocol is followed by its results and an honest reading); the code map, reproducibility notes, literature check and references are at the end. The slides are in `docs/SLIDES.md`, the decisions and corrections in `docs/PROJECT_STATE.md`.

## Current scope and hypothesis ledger

Scope decision (2026-10-04): the main family is off resonance. Near-resonance behaviour is a known limitation, not a goal. The formulations tried for it (free per-node partition of unity, at most 150 reduced dimensions, a single instance, the losses in Gates 1b to 1f) did not reach resonant accuracy, and the capture error of the resonant modes saturated at 0.074 for reasons that were not resolved. All claims below apply to the sub-resonant band only.

| Hypothesis | Status | Next test |
|---|---|---|
| H1. Consistency by construction | Holds in the narrow form: exact in the reduced test space, power balance and tangential continuity to machine precision (Gate 0). The fine-mesh divergence error tracks the field error (consistency check). Against the FNO at matched accuracy (Study 4a): an advantage by construction only (power balance and PEC, which POD-Galerkin also has); on the Gauss law the encoder is 23 to 27% worse than the FNO. A hybrid that adds the FNO's field to a Whitney space and solves the Galerkin system (Study 6b) restores exact power balance but loses the FNO's accuracy (1.4 to 4.9 times its error). | Nothing further planned for H1 |
| H2. A small learned reduced space is accurate off resonance | Partly supported. A single shared W fails, a per-instance oracle W passes at 36 dimensions (median field error 0.044), and the encoder (Gate 2) gets 0.476 to 0.180 median field error at 25 to 400 training instances: PASS through criterion B only (25 to 47% below POD of the same dimension, and below POD-136 from 50 instances), criterion A (2x at 25 instances) not met, 4 to 11 times the oracle, and no size reaches 10% field error. Study 7 then showed the encoder was under-fitted: with 12 bumps (78 dimensions) and 1500 steps it matches the FNO's median field error at refine 4 (0.216 and 0.145 against 0.241 and 0.150 at 100 and 400 instances) and has the lower median energy error; projection pre-training hurts. Mesh transfer, certificate and cost were not rechecked for that model. | Study 4a (consistency and certificate), Study 4b (cost); transfer and certificate for the 78-dimensional encoder |
| H3. Cheap sweeps with omega-independent reduced matrices | The mechanism holds and the cost advantage is measured (scaling study: 3x to 20x per query and 200x to 1200x per extra frequency at 393k unknowns, excluding the cost of obtaining the basis). With the encoder's basis included (Study 4b) the single query is 1.9x to 4.6x faster than a full solve from 24k to 393k unknowns and an extra frequency is 55x to 539x cheaper, but the FNO is far cheaper still (Study 5b) and the transferred encoder is not accurate on finer meshes (Study 5a: median field error 0.64 to 0.90 against 0.18 to 0.24 at refine 4, while the FNO stays at 0.17 to 0.25). Sweep accuracy inside the main band, with one basis predicted at r = 0.4 and reused for all frequencies, was then measured (sweep animation): median field error 0.146, the same as re-predicting the basis at every frequency, over 20 geometries and r from 0.2 to 0.6 at the training mesh. Accuracy across a band with a resonance inside it was only tested in Gate 1b stage (a). A mesh-free encoder with a learnable grid base trained on two meshes (Study 10) cut the transfer degradation (error ratio 1.3 to 1.6 at 98k unknowns against 3.8 to 5.6 for per-node encoders) but lost 33% to 36% of the training-mesh accuracy, still trailing the FNO (0.31 to 0.38 against 0.17 to 0.25), and failed the registered bars. | Training on more meshes (refine 4, 5 and 6) or on the target mesh |
| H4. Residual certificate, conformal calibration and full-wave fallback | Partly tested (verification study): marginal coverage holds for every surrogate, the residual tracks the error for POD but not for coarse spaces, coverage drifts across the band, selection shift is repaired by recalibration, and no full-wave calls are avoided with the current surrogates. For the learned models (Study 4a) the residual does NOT track the error (Spearman 0.26 to 0.34 for the encoder, 0.55 to 0.60 for the FNO, against 0.93 for POD), although marginal coverage is still 0.89 to 0.90; the FNO with the certificate avoids the most calls (38% at tau = 0.2, 400 instances). | Conditional calibration, other indicators; a surrogate accurate to a few per cent |
| H5. Data efficiency and out-of-distribution accuracy against unconstrained operators | Not supported on accuracy. The encoder is 1% to 20% above the FNO's median field error at every training size from 25 to 400 (Gate 2); out-of-distribution accuracy is untested. Study 4a found no compensating advantage: consistency only by construction (shared with POD-Galerkin) and a worse Gauss-law error than the FNO at matched accuracy; the certificate tracks the FNO's error better than the encoder's. Revised by Study 7: a better-trained 78-dimensional encoder is level with the FNO on median field error at refine 4 (and lower on energy error), so the accuracy comparison depends on how well the encoder is trained; the other findings were measured with the Gate 2 encoder. | Out-of-distribution test, if kept in scope |
| H6. Fewer full-wave calls per finished design; manufacturability | A first design loop was built (Study 9, gradient-free, four design parameters, in distribution): screening a pool of 1000 designs with a surrogate and verifying the top candidate by full-wave gets within 10% of the pool optimum with 1 full-wave call (FNO, 12-bump encoder; mean regret 0.064 and 0.013) against 200 for random full-wave search (POD-136 needs 50). Certificate gating added no value (28 to 41 calls for a regret that plain screening reaches with 1 to 5). No wall-clock saving at this mesh. Manufacturability was not built. | Gradient-based design, port or S-parameter objectives, a fabrication filter and a yield measure |
| Resonant accuracy | Out of scope (limitation). | None planned |

Main-family band (from the band pre-check, fixed): omega^2 = r x lambda_1(instance) with r uniform on [0.2, 0.6]; lambda_1 is the first mode of each instance.

Decision rule for H2 and H5, fixed now: if the encoder does not give at least a 2x error reduction against POD or coarse FEM at the smallest training-set size, or the same accuracy at half the dimension, the study continues with the verification layer and the stability diagnosis as its contributions, and no further learned-W variations are tried.


## Abstract

Neural surrogates for Maxwell's equations are fast but usually do not respect the structure of the equations, and they need large simulation datasets. We tested a hybrid in which a network predicts only a small reduced finite-element space and the physics is assembled exactly by a conventional edge-element FEM on a fine mesh. Following Geo-NeW (Neural Whitney Forms), a network predicts a partition of unity W from the instance; W builds Whitney 1-forms, and a small Galerkin system is solved in the span of their lift to the fine mesh. We asked whether this gives fast, Maxwell-consistent, data-efficient and accurate surrogates, and how their answers can be verified.

The test is a pre-registered ladder of studies on one family: 2D time-harmonic Maxwell in a PEC cavity with a dielectric disc (random position, radius and contrast) and a Gaussian current source, at sub-resonant frequencies (resonance was studied, found hard, and declared out of scope). Every protocol and pass criterion was committed before its run, and failures are reported as findings. Compared with POD-Galerkin, an FNO, non-learned Whitney spaces and a per-instance oracle:

- **Capacity:** a per-instance oracle W reaches 4.4% median field error with 36 dimensions (POD of the same size: 27.6%), so the model class can represent the solutions.
- **Learned basis:** an encoder (a small CNN) that predicts W reaches 0.48 to 0.18 median field error with 25 to 400 training instances at 36 dimensions, 25% to 47% below POD of the same size, and was under-fitted: with 12 bumps (78 dimensions) and 1500 steps it matches the FNO at refine 4 (0.216 and 0.145 against 0.241 and 0.150 with 100 and 400 instances) and has the lower stored-energy error. Pre-training with a best-approximation loss hurts. No model reaches the 10% field-error target at 100 instances.
- **Consistency:** the reduced solution satisfies power balance (5e-15 against 2.3 for the FNO) and PEC exactly, but any Galerkin solution does (POD-Galerkin too), and the Gauss-law error at matched accuracy was 23% to 27% worse than the FNO's for the Gate 2 encoder. Adding the FNO's field to a Whitney space restores the equations but loses the accuracy (1.4 to 4.9 times the FNO's error).
- **Cost:** a single query is 1.9 to 4.6 times faster than a full solve from 24k to 393k unknowns and an extra frequency 55 to 539 times cheaper, but an FNO costs 26 ms where the encoder costs 1.4 s at 393k unknowns.
- **Transfer:** an encoder with per-node parameters trained at 1.5k unknowns loses its accuracy on finer meshes (median field error 0.24 to 0.90 by 98k unknowns, while the FNO stays at 0.17 to 0.25); a mesh-free encoder keeps the ratio (1.1 to 1.35) but is less accurate and fits unstably, and a mesh-free encoder with a learnable grid base trained on two meshes (Study 10) limits the error at 98k unknowns to 0.31 to 0.38 (ratio 1.3 to 1.6), against 0.17 to 0.25 for the FNO, at the cost of 33% to 36% of its refine-4 accuracy; it misses the registered bars.
- **Verification:** a residual indicator costs about 0.25% of a full solve and, with split conformal calibration, gives valid average coverage (0.89 to 0.90) for every surrogate; it tracks the error for POD (Spearman 0.93) but not for the learned models (0.26 to 0.60), and no model has a certificate that avoids many full-wave calls at a 5% target.
- **Design loop (first version):** screening 1000 candidate discs with a surrogate and verifying the top one by full-wave gets within 10% of the best design with 1 full-wave call for the FNO and the 12-bump encoder, against 200 calls for random full-wave search (POD-136: 50); gating the candidates with the residual certificate added no value. The task has four parameters and is in distribution, and at this mesh there is no wall-clock saving.
- **Resonance:** near an eigenvalue the field is representable (1.5% to 2.3% error) but the Galerkin solve in the same space has error near 1, because the reduced spectrum is wrong; spectrum-matching, Hodge-split and energy-norm losses did not fix it.

Overall: a learned compatible basis beats a data-driven basis of the same size and, with enough training, is more accurate per parameter than an FNO at the training mesh: with 24,000 parameters it has 0.60 to 0.63 times the error of an FNO of 20,000 parameters and matches an FNO of 1.19 million (Study 8). But in this regime it does not beat the FNO on cost at scale (the 78-dimensional encoder is faster than a full solve only above about 98k unknowns, 2.2 times at 393k), on mesh transfer, or on certificate behaviour. In a first design loop, surrogate screening cuts the full-wave calls needed to find a near-optimal design by a factor of 200 on a four-parameter task. The contributions are the evaluation, the diagnosis of why, the verification layer and this first design-level demonstration.

### Results at a glance

| Study | Question | Outcome |
|---|---|---|
| Gate 0 | Is the construction implemented correctly? | Passes: Gauss law about 1e-13, power balance about 1e-14, tangential trace 0 |
| Gates 1, 1b | Can a small W represent the solutions? | Off resonance yes (oracle 0.044 at 36 dimensions); near resonance the Galerkin solve fails |
| Gates 1c to 1f | Can a spectral, Hodge or energy-norm loss fix resonance? | No; hard stop, resonance declared out of scope |
| Verification study | Is the residual a valid, informative certificate? | Valid on average; tracks POD, not coarse spaces |
| Scaling study | When is a reduced solve cheaper? | 3x to 20x per query at 393k, hundreds of times per extra frequency, excluding the basis cost |
| Baseline data efficiency | How do the FNO and POD behave as data grows? | No baseline reaches 10% field error up to 400 instances |
| Gate 1b stage (b) | Does one shared W suffice? | No; an encoder is needed (oracle passes) |
| Gate 2 | Can an encoder predict W? | Pass through one criterion; 25% to 47% below POD, level with the FNO only at 78 dimensions after Study 7 |
| Studies 4a, 4b | Consistency, certificate and cost of the encoder | Consistency by construction only; certificate does not track; cost 3.2 ms at 1.5k, 4.6x faster than a full solve at 393k |
| Studies 5a, 5b | Transfer to finer meshes; cost of the FNO | Per-node encoder fails to transfer; FNO transfers and is 54 times cheaper |
| Studies 5c, 6a | Is the per-node base the cause? | Smooth transfer: no effect; mesh-free encoder: ratio repaired, accuracy poor and unstable |
| Study 6b | Does adding the FNO's field to a Whitney space work? | No: equations restored, accuracy 1.4 to 4.9 times the FNO's error |
| Study 7 | Is the encoder under-fitted? | Yes: 12 bumps and 1500 steps match the FNO at refine 4; projection pre-training hurts |
| Study 8 | Do the Study 7 gains survive parameter matching, certificate, transfer and cost checks? | Parameter matching: leads (0.60 to 0.63 times a same-size FNO); certificate does not track but accepts 61% at target 0.2; transfer still fails; cost: faster than a full solve only above about 98k unknowns |
| Study 9 | Do surrogates cut full-wave calls in a design loop? | Yes for screening: 1 call (FNO, encoder) against 200 for random search to reach 10% regret, POD-136 50; the certificate gating adds no value; four-parameter, in-distribution task |
| Study 10 | Can a grid-base mesh-free encoder trained on two meshes transfer without losing accuracy? | No (both variants FAIL): training on two meshes helps (error ratio 1.3 to 1.6 at 98k unknowns against 3.8 to 5.6 per-node), but accuracy at the training mesh drops 33% to 36% and the FNO is still better |

## Motivation

- **Cost:** full-wave solves dominate design loops. In 3D, factorisation is the bottleneck, and every design iteration or frequency point repeats it.
- **Data:** training neural operators needs many labelled simulations, which are expensive for the geometries that matter.
- **Consistency:** pure data-driven surrogates give no structural guarantee. Soft physics losses (PINO-style) help on average but not by construction, and FFT-based residuals ring at discontinuous epsilon.
- **Trust:** an optimiser exploits surrogate errors, so a design that looks good on the surrogate may fail full-wave verification. A cheap error certificate is needed, not only average accuracy.
- **Gap:** learned compatible finite-element spaces from Whitney forms exist (Neural Whitney Forms, Geo-NeW [10, 11]) and preserve conservation laws through the exact sequence, with published benchmarks that include electrostatics but, as far as their abstracts show, not time-harmonic Maxwell with edge elements; compatible coarse spaces for Maxwell exist in the multigrid literature [6, 7] and learned multigrid prolongations exist for other problems [8, 9]. See "Literature check and references" at the end: this project is an application and independent evaluation of the idea in a new setting, with a verification layer, not a new method class. The studies below test whether it earns its complexity against the strongest simple alternative, an unconstrained neural operator [13].

## Methodology

**Scope.** 2D time-harmonic Maxwell (TE) in a unit-square PEC cavity with one dielectric disc (centre in [0.3, 0.7]^2, radius in [0.10, 0.25], relative permittivity in [2, 8]) and a Gaussian current source (random position and direction), loss tangent 0.05, edge elements from scikit-fem (lowest-order Nedelec). Frequency: omega^2 = r x lambda_1 of each instance with r uniform on [0.2, 0.6] (sub-resonant). The working mesh is refine 4 (1,504 interior edges); finer meshes (up to 393k unknowns) are used for cost and transfer tests. Impedance boundaries and 3D were not attempted.

**Pipeline as built**

1. **Fine mesh and operators.** The curl-curl matrix K, the epsilon-mass matrix M and the load are assembled exactly; the full-wave reference is a sparse direct solve.
2. **Encoder.** A small CNN reads seven grid channels (epsilon, real and imaginary omega^2, the source components, coordinates) and outputs a correction to shared per-node logits, sampled at the mesh nodes. Variants: a mesh-free base (analytic RBF with trained centres and widths) and a step schedule with a projection loss.
3. **Partition of unity.** A softmax over n bumps (plus a remainder function) that vanish on the boundary: n = 8 (36 dimensions) or 12 (78 dimensions).
4. **Whitney lift.** The coarse 1-forms w_ij = W_i grad(W_j) - W_j grad(W_i) are lifted to fine-mesh edge cochains, M-orthonormalised by a Cholesky whitening with a small jitter.
5. **Reduced solve.** Project K and M onto the lifted basis and solve the small complex Galerkin system; the reduced matrices are real, so each extra frequency costs one small dense solve.
6. **Training.** Supervised with the relative M-norm Galerkin error against the reference solution (the same supervision as the baselines); the optimiser is Adam with cosine annealing. A label-free residual loss was planned (Gate 3) and not run.
7. **Verification layer.** The residual indicator (diagonal-mass norm, O(N)) is the non-conformity score for split conformal prediction on the ratio of error to residual; a surrogate answer is accepted if the certified bound is under the accuracy target and otherwise falls back to a full-wave solve.

**Baselines and controls.** POD-Galerkin (data-driven basis of the same dimension); an FNO on a 64 x 64 grid; non-learned coarse Whitney spaces (88 and 368 dimensions); one shared W with no encoder; and a per-instance oracle W fitted to the reference (an upper bound on what the basis can represent). Variants tried to repair failures: smooth base transfer (A1), a mesh-free encoder (A2), the FNO's field added to a Whitney space (C1), more bumps and projection pre-training (Study 7), and a parameter-matched FNO (Study 8).

**Evaluation discipline.** Each study has a protocol with its pass and failure criteria committed to this file before it is run; results and the reading follow; corrections and deviations are recorded where they happened (for example, studies stopped early at the owner's decision). Cost studies are single-thread, median of seven timings, with the machine load recorded, against SciPy's sparse direct solver. Errors are relative M-norm errors against the full-wave solution on the same mesh.

**Not done.** Resonant accuracy (out of scope); the label-free residual loss; out-of-distribution tests; impedance boundaries and 3D; gradient-based design, port or S-parameter objectives and fabrication constraints (a first, gradient-free screening loop is Study 9; the rest are ideas, in `docs/PROJECT_STATE.md`).

## Evaluation

### Metrics

| Category | Metric |
|---|---|
| Accuracy | Relative field error vs FEM/FDTD; error in scalar quantities (S-parameters, resonance frequency, Q) |
| Consistency | Divergence residual of epsilon*E, always stated with its test space: `gauss_fine` (all fine nodes, the true divergence error) and `gauss_r{L}` (hat functions of a level-L coarse mesh). A reduced space is exact only inside its own test space, so report both; tangential-continuity error at interfaces; complex power-balance violation |
| Data efficiency | Error vs number of training simulations, including the zero-label energy loss |
| Generalisation | Out-of-distribution tests across frequency, geometry class and material contrast |
| Cost | Wall-clock time per solve and per frequency sweep; full-wave calls per finished design |
| Reliability | Conformal coverage achieved vs nominal, including under optimiser-selected candidates; fallback rate |
| Design outcome | Full-wave figure of merit of final candidates; fabrication yield under a tolerance model (if included) |

### Main risks and how we decide

(This table is the original plan. What was actually run, in order, with its protocol and result, is in the sections below; the current status of each hypothesis is in the ledger at the top.)

Each risk is tied to a test. A test has three outcomes: **works**, **needs tuning** (the model class is sufficient but training or features fall short), or **does not work** (the model class itself is insufficient, so tuning will not help and we pivot). The key device is the oracle test in Gate 1, which separates "the idea cannot represent the solution" from "the network is not trained well".

| Gate | Question | Test | Works | Needs tuning | Does not work |
|---|---|---|---|---|---|
| 0. Structure | Is the construction implemented correctly? | Random PoU, no learning. Check discrete Gauss law, curl of grad = 0, power balance. | All at machine precision | n/a | Any violation is a bug. Fix before continuing. |
| 1. Capacity (oracle W) | Does any W exist at this size that solves the problem? | Optimise W directly per instance, with no encoder, on one geometry and frequency. Sweep the number of coarse functions. | Error floor below target at an affordable size | n/a | Floor stays above target as size grows. The risk "small reduced space cannot capture resonant or wideband fields" has materialised. |
| 2. Learnability | Can the encoder predict a good W across a family? | Compare encoder-predicted W to oracle W on held-out designs. | Gap to oracle small | Large gap: change tokens (add epsilon, omega, operator eigenmodes), encoder, loss weights | Gap persists after tuning and the oracle is only good per instance |
| 3. Training signal | Does the residual loss work without labels in the indefinite case? | Compare residual-only, supervised-only and mixed training at fixed data size. | Residual-only close to supervised | Residual-only fails but mixed works: tune weights, add few labels | Neither trains W. Restrict to coercive problems. |
| 4. Baseline | Does the structure help? | Error vs training-set size and OOD error vs unconstrained baseline and potential-head baseline. | Better at small data and OOD | Comparable: revisit features and capacity | Worse at all sizes |
| 5. Reliability | Does the certificate hold under optimisation? | Residual-score conformal coverage on random designs vs on optimiser-selected designs. | Coverage holds on both | Coverage drops on selected designs: recalibrate on selected points | Residual does not track true error |
| 6. Cost | Is it actually faster? | Total time (assembly, encoder, reduced solve, fallbacks) vs fine solve, per sweep and per design. | Clear break-even at realistic mesh sizes | Dominated by assembly or encoder: optimise those | No break-even, since the fine solve is already cheap |

**Proposed thresholds (to agree before running)**

- Gate 0: relative violation below 1e-10.
- Gates 1 to 3: scalar-quantity error (resonance frequency, S-parameters) below 5%, with field error as a secondary figure.
- Gate 4: at least a 2x reduction in error at the smallest training-set size, or a clear OOD gain.
- Gate 5: empirical coverage within 2 points of nominal.

**Gate 1 protocol (fixed before running)**

- Instance: unit square, refine 4 (1568 edges, 1504 interior), dielectric disc at (0.5, 0.5), radius 0.2, epsilon 4; Gaussian current at (0.3, 0.4), sigma 0.08, direction y; loss tan(delta) = 0.05.
- Frequencies from the dense eigenvalues of this cavity (first modes at omega^2 = 7.60 double, 18.8, 28.7): `low` = 0.4 x lambda_1 (quasi-static), `near` = 1.02 x lambda_1 (near resonance), `gap` = midpoint of the first two distinct modes.
- W is a free nodal partition of unity (one logit per node per bump). No encoder. Minimise the M-norm error of the Galerkin solution against the fine solution. This is a capacity test, so the reference is allowed in the loss.
- Sweep n = 4, 6, 8, 12, 16, 24 bumps. Reduced dimension is about n(n+1)/2 (10 to 300).
- Starts: 2 initialisations (RBF-centred bumps, smooth random) x 2 seeds, 500 Adam steps each. Report the best start and the spread.
- Affordable size: reduced dimension at most 10% of interior fine edges (150).
- Pass: scalar-QoI error (stored energy) below 5% at an affordable size. Field error is reported as well, and also the projection (best-approximation) error of the same W, to separate "space cannot represent the field" from "Galerkin solve is unstable".
- Fail (capacity): no affordable size passes and the error has plateaued: best field error at n = 24 is at least 0.8 of that at n = 16. Inconclusive: still decreasing at the largest n, or the optimiser has not converged (loss still falling by more than 5% over the last 20% of steps).
- Context: the same instance solved with `coarse_r1/2/3` (non-learned compatible spaces) at their dimensions.

**Gate 1 results** (protocol above, CPU, float64; rerun on Linux/WSL, which reproduces the verdicts of the first macOS run)

Best of 4 starts. `rel_err` is the M-norm field error, `energy` the stored-energy QoI error, `proj` the best-approximation error of the same W.

| Frequency | n | dim | rel_err | energy | proj | Verdict |
|---|---|---|---|---|---|---|
| low | 4 | 10 | 0.029 | 0.007 | 0.026 | PASS at n = 4 |
| low | 24 | 300 | 0.023 | 0.008 | 0.012 | field error plateaus near 2.5% |
| gap | 8 | 36 | 0.068 | 0.033 | 0.055 | PASS at n = 8 |
| gap | 12 to 24 | 78 to 300 | 0.051 to 0.061 | 0.028 to 0.033 | 0.031 to 0.041 | field error plateaus near 5 to 6% |
| near | 4 to 16 | 10 to 136 | 0.98 | 0.98 | 0.52 to 0.70 | INCONCLUSIVE |
| near | 24 | 259 | 0.58 | 0.29 | 0.44 | flagged converged, but outside the budget (150) |

Same instance with non-learned compatible spaces: `coarse_r3` (368 dims) has field error 0.33 (low), 0.26 (gap), 0.25 (near). This is not a fair comparison, since the oracle W is fitted to this instance's reference solution. It is an upper bound on what learning can do.

Reading:

- Off resonance, capacity exists at small sizes (10 to 36 dims). Gate 1 passes for low and gap. This is one instance with a free per-node W, so it says nothing yet about whether a network can predict such a W (Gate 2).
- The optimiser is fragile: the worst start is often far off (for example worst 0.85 against best 0.023 at low, n = 24, and 0.94 against 0.068 at gap, n = 8). A trained encoder has to find good W reliably, which is a Gate 2 risk.
- Several best runs were flagged unconverged, so their errors are upper bounds on the floor.
- The first (macOS) run and the rerun (Linux/WSL) use the same fixed seeds but differ by a few percent in several entries (for example near, n = 24: 0.67 first run, 0.58 rerun; the convergence flag also changed). Treat the numbers as approximate (up to roughly 15% in the worst entry). The verdicts are the same.
- Near resonance the registered verdict is inconclusive, not a capacity failure.

Exploratory follow-up on the near-resonance case (post hoc, outside the protocol). Reproduce with `uv run python -m experiments.diag_near_resonance`; one instance, best of 4 starts chosen by projection error, CPU. Resonance half-width in omega^2 is 0.05 x omega^2 = 0.39 (the imaginary part), which replaces the 0.19 quoted in an earlier draft of these notes.

| Space | n | dim | proj | Galerkin | min-residual | Galerkin + exact modes | reduced eigenvalues below lambda_1 (lowest) | nearest to omega^2 |
|---|---|---|---|---|---|---|---|---|
| W optimised for proj | 8 | 36 | 0.023 | 1.15 | 1.00 | 0.56 | 2 (5.79) | 7.32 |
| W optimised for proj | 12 | 78 | 0.017 | 1.04 | 1.00 | 0.030 | 3 (1.89) | 8.36 |
| W optimised for proj | 16 | 136 | 0.015 | 1.04 | 1.00 | 0.014 | 3 (1.07) | 8.33 |
| same, then 500 Galerkin steps | 8 to 16 | 36 to 136 | 0.57 to 0.85 | 0.98 to 1.00 | 0.99 to 1.00 | 0.15 to 0.16 | 0 | 11.6 to 11.7 |
| coarse_r1 (non-learned) | | 20 | 0.39 | 0.75 | 1.00 | 0.12 | 2 (7.07) | 7.07 |
| coarse_r2 (non-learned) | | 88 | 0.25 | 0.56 | 0.99 | 0.08 | 2 (7.28) | 7.28 |
| coarse_r3 (non-learned) | | 368 | 0.13 | 0.25 | 0.99 | 0.05 | 2 (7.51) | 7.51 |

True first mode: lambda_1 = 7.604 (double); omega^2 = 7.756; half-width 0.39. "+ exact modes" adds the two fine-mesh eigenvectors at lambda_1 to the space.

- The field is representable. Optimising the projection error gives 0.023, 0.017, 0.015 at n = 8, 12, 16, matching the first exploratory run. The Galerkin solve in those same spaces still has error about 1. Continuing with 500 Galerkin steps does not recover: it destroys representability (proj rises to 0.57 to 0.85) and Galerkin stays near 1.
- A least-squares (minimal-residual, M^-1 norm) solve also gives about 1.0 in every space except when the exact modes are added. The remedy proposed in the plan does not rescue this case.
- The learned spaces have reduced eigenvalues far below lambda_1 (5.8, 1.9, 1.07 at n = 8, 12, 16). The coarse spaces do not: their lowest pair is 7.07, 7.28, 7.51, rising towards 7.604 with refinement. An earlier version of these notes said the coarse spaces have no eigenvalues below lambda_1. That was imprecise: their first pair is slightly below it, since it is a Rayleigh quotient on a subspace that is not exactly curl-orthogonal to the fine gradients. It is a shifted copy of the true mode, not a spurious one.
- Direct test of the spectrum explanation. Adding the two exact resonant eigenvectors to the space brings the Galerkin error from 1.04 to 0.030 (n = 12) and 0.014 (n = 16), which is the projection floor. Adding them at n = 8 only reaches 0.56, and there the reduced spectrum has a mode at 7.32, 0.44 from omega^2, just outside the half-width. So at n = 12 and 16 the field was representable and the Galerkin solve failed only because the reduced operator had no mode near omega^2. This supports the explanation, but it is an intervention with oracle information (the exact modes), on one instance. It shows the resonant subspace is sufficient to fix the failure, not that a trained W will find it.
- Refinement of the explanation. Spurious eigenvalues far from omega^2 appear harmless: at n = 12 and 16 modes at 1.9, 4.5 and 5.5 remain after the fix and the error is at the floor. What matters is the reduced spectrum near omega^2, not the absence of low eigenvalues.
- Only coarse_r3 puts a reduced mode inside the half-width (7.51, distance 0.24), and it has the lowest Galerkin error of the non-augmented spaces (0.25). The ordering across the three coarse spaces follows the distance to omega^2 (0.69, 0.47, 0.24 against Galerkin errors 0.75, 0.56, 0.25), which is consistent with the explanation.
- The minimal-residual error stays at 0.12 to 0.14 even with the exact modes, compared with 0.014 for Galerkin at n = 16. The M^-1 residual norm is not the field-error norm, so it cannot be used as a drop-in replacement for the Galerkin solve in this comparison. Not investigated further.
- Candidate next steps: a spectrum-aware loss that places a reduced eigenvalue near omega^2; a single W trained across a frequency band (which is also what the cheap-sweep claim needs, since W must not depend on omega); continuation in frequency.

**Status after Gate 1: what the implementation shows so far**

Measured:

- The compatible construction is implemented correctly. Gate 0 holds to about 1e-13 for random partitions of unity, with a negative control that fails by 0.1 to 0.7.
- Off resonance, a learned space of 10 to 36 dimensions reaches 3 to 7% field error on one instance, where non-learned compatible spaces need 368 dimensions for 26 to 33%.
- Near resonance, representability and solvability come apart. The field is representable (projection error 0.015 to 0.023), but the Galerkin and least-squares solves in the same spaces give error about 1. Adding the exact resonant eigenvectors restores the projection floor at n = 12 and 16.

Inferred (reasoned from the results, not yet tested unless stated):

1. The consistency guarantee is weaker than the plan wording suggests. From `lift.py`, the lifted space contains the gradients of the n + 1 functions W_i, so the weak Gauss law holds only against those test functions. This is now measured on the oracle W (table below): the Gauss error is 4e-15 to 4e-14 against the W_i, as guaranteed, but 3e-3 to 7e-2 against the fine-mesh nodal functions, and 1e-3 to 4e-2 against the coarse hat sets. It is not exact at any fixed test set other than the learned one. "Maxwell-consistent by construction" should be stated as "exact in the reduced 0-form space".
2. The Gate 1 pass is a weak capacity result. A free per-node W has about 1500 x n parameters and is fitted to one reference solution, so it can nearly memorise one smooth field. It shows there is no representational obstruction for a single instance, and says little about a family. The comparison with `coarse_r3` is also between unequal information.
3. The near-resonance failure is a stability failure, not a capacity failure. Galerkin on a subspace is quasi-optimal only when the reduced indefinite form is stable, and nothing in the training objective enforces that. The gate table has no outcome for this case. Report the ratio of Galerkin error to projection error in the Gate 1 table.
4. A field-error loss saturates near resonance. Every size up to n = 16 sits at error 0.98, which is the plateau where the reduced solution is close to zero, so the gradient carries little information. Optimising the projection error escapes it, and continuing with the Galerkin loss then destroys representability (projection error rises to 0.57 to 0.85). Spurious eigenvalues far from omega^2 appear harmless. The reduced spectrum near omega^2 is what matters.
5. Cheap sweeps and resonances pull against each other. A frequency-independent W must place a reduced eigenvalue within the half-width (0.05 omega^2) of every resonance in the band, so the reduced dimension has to grow with the number of modes in the band. For one resonance this is fine. For wideband problems the natural competitor is a modal or eigenvector-augmented reduced basis, which should be added to the baselines. This is an inference and is untested.
6. The least-squares remedy in the plan is not supported. In the M^-1 residual norm it gives error about 1 wherever Galerkin does, and 0.12 to 0.14 even with the exact modes.
7. The problem family needs a decision. Its frequency range (2 to 12) includes the first resonances (7.6 and 9.9), so near-resonant instances will dominate any family-level error unless the loss is raised, the band narrowed, or resonances treated as the main target.

**Gate 1 consistency check** (`experiments/gate1_consistency.py`, oracle W of the passing cases, best of 4 starts, same instance)

Relative Gauss residual against four test sets, plus the field error. `W` = the learned bumps (the guaranteed set), `r1` to `r3` = hat functions of the coarse meshes, `fine` = all interior fine nodes (the true divergence error).

| Frequency | Space | dim | W | r1 | r2 | r3 | fine | field error |
|---|---|---|---|---|---|---|---|---|
| low | oracle n = 4 | 10 | 4e-15 | 3.3e-3 | 6.3e-3 | 1.1e-2 | 7.2e-2 | 0.029 |
| low | oracle n = 8 | 36 | 9e-15 | 2.4e-3 | 2.4e-3 | 3.8e-3 | 3.9e-2 | 0.026 |
| low | oracle n = 16 | 136 | 4e-14 | 1.6e-3 | 1.7e-3 | 3.3e-3 | 3.2e-2 | 0.024 |
| low | coarse_r3 | 368 | n/a | 8e-15 | 2e-14 | 5e-14 | 0.50 | 0.33 |
| gap | oracle n = 8 | 36 | 5e-15 | 2.0e-2 | 2.8e-2 | 3.7e-2 | 0.16 | 0.068 |
| gap | oracle n = 16 | 136 | 2e-14 | 2.4e-2 | 2.7e-2 | 3.1e-2 | 0.086 | 0.061 |
| gap | coarse_r3 | 368 | n/a | 5e-15 | 7e-15 | 2e-14 | 0.54 | 0.26 |

Reading:

- The guarantee holds exactly where it is supposed to (column W) and nowhere else. A non-learned coarse space is exact on all hat sets up to its own level, which a learned W is not, since its reduced 0-form space is not a nested hat space.
- The fine-mesh divergence error tracks the field error: it is 1.3 to 2.5 times the field error for every oracle row, and 1.5 and 2.1 times for `coarse_r3` (0.50 against 0.33, 0.54 against 0.26). So in these results it is largely approximation error, not a separate property of the construction. The oracle W has lower fine divergence error than `coarse_r3`, which is consistent with its more accurate field, but this does not separate the two.
- Whether compatibility gives lower divergence error than an unconstrained model at equal field error is not shown here. That comparison needs the unconstrained baselines at matched accuracy (Gate 4).
- One instance, with W fitted to the reference solution, so this is an upper bound on what a predicted W would give.

Not yet established: whether an encoder can predict a good W (Gate 2); the time cost of the learned pipeline; any geometry other than a disc in a square; any result across a family of instances.

Suggested order before building a spectrum-aware loss: (a) done, see the consistency check above; (b) Gate 1b stage (a) is done and is stability-limited, see its results below (geometry variation, stage (b), is not yet run); (c) the spectrum-aware loss, now justified by the registered rule.

**Gate 1b protocol (fixed before running)**

Question: can one W, shared across a frequency band that contains a resonance, give an accurate reduced solve at every frequency? This tests the cheap-sweep claim and separates capacity from stability near resonance. Stage (a), frequency only, is registered here. Stage (b), adding disc geometries, is registered separately after (a) is read.

- Instance: the Gate 1 instance (unit square, refine 4, disc at (0.5, 0.5), radius 0.2, epsilon 4, same source, tan(delta) = 0.05), so lambda_1 = 7.604 (double). Band: omega^2 in [6.5, 9.0], which contains only that mode (the next is 18.8).
- Training frequencies: 9 points, 6.5 + 0.3125 k for k = 0 to 8. Held-out frequencies: the 8 midpoints, 6.656 to 8.844, one of which (7.594) is within 0.01 of resonance. Each is multiplied by (1 + 0.05j).
- W: free nodal logits shared by all frequencies, no encoder, n = 4, 8, 12, 16, 24. Affordable means dimension at most 150 (n <= 16). Starts: rbf seed 0 and smooth seed 0, 500 Adam steps, lr 0.05. The best start is chosen by the training objective only, never by held-out error.
- Losses compared, all averaged over training frequencies: `galerkin` (relative M-norm error of the Galerkin solution), `proj` (best-approximation error of the same space), `mixed` (their sum). The reference solutions are allowed in the loss, as in Gate 1.
- Comparators, same frequencies: `coarse_r1/2/3`; and modal bases: the lowest k = 2 and k = 10 exact eigenvectors, each with the gradients of the level-3 coarse hat functions added so that the curl-free part of the solution is representable (`modal2+grad`, `modal10+grad`).
- Reported per space: train and held-out mean and max of the Galerkin field error, the projection error, their ratio at the worst held-out frequency, the stored-energy error, and a convergence flag (the Gate 1 rule).
- Pass: for some loss and an affordable n, the stored-energy error is below 5% at every held-out frequency.
- Stability-limited: the maximum held-out projection error is below 5% at an affordable n, but no affordable n passes. The explanation from the Gate 1 diagnostics (reduced spectrum misses the resonance) is then supported, and the spectrum-aware loss is justified.
- Capacity fail: the maximum held-out projection error stays above 5% at every affordable n and has plateaued (n = 24 at least 0.8 of n = 16). Pivot per the rules of engagement.
- Interpolation fail: the training energy error passes but a held-out frequency does not. This is reported separately, because it points to a frequency-conditioned W rather than a larger space.
- Inconclusive: otherwise, or when the best runs are unconverged.

**Gate 1b results, stage (a)** (`experiments/gate1b.py`, protocol above, CPU; best of 2 starts chosen on the training objective)

Held-out frequencies are the 8 midpoints; "gal" is the Galerkin field error, "proj" the best-approximation error of the same space, "energy" the stored-energy error, each the maximum over held-out frequencies. "train gal" is the mean Galerkin error over the 9 training frequencies.

| Space | Loss | n | dim | gal | proj | energy | train gal |
|---|---|---|---|---|---|---|---|
| oracle W | galerkin | 4 | 10 | 0.99 | 0.96 | 0.98 | 0.96 |
| oracle W | galerkin | 8 | 36 | 0.99 | 0.92 | 0.99 | 0.97 |
| oracle W | galerkin | 12 | 78 | 1.01 | 0.70 | 0.98 | 0.94 |
| oracle W | galerkin | 16 | 136 | 1.06 | 0.69 | 0.98 | 0.94 |
| oracle W | galerkin | 24 | 300 | 0.29 | 0.26 | 0.12 | 0.27 |
| oracle W | proj | 4 | 10 | 1.31 | 0.11 | 0.98 | 1.06 |
| oracle W | proj | 8 | 36 | 1.96 | 0.025 | 4.11 | 1.08 |
| oracle W | proj | 12 | 78 | 1.69 | 0.018 | 3.51 | 0.95 |
| oracle W | proj | 16 | 136 | 1.66 | 0.015 | 3.35 | 0.93 |
| oracle W | proj | 24 | 300 | 1.63 | 0.014 | 3.16 | 0.91 |
| oracle W | mixed | 4 | 10 | 1.00 | 0.10 | 0.98 | 0.97 |
| oracle W | mixed | 8 | 36 | 1.01 | 0.050 | 0.98 | 0.97 |
| oracle W | mixed | 12 | 78 | 1.00 | 0.036 | 0.98 | 0.95 |
| oracle W | mixed | 16 | 136 | 1.17 | 0.032 | 0.96 | 0.93 |
| oracle W | mixed | 24 | 300 | 0.30 | 0.025 | 0.12 | 0.28 |
| coarse_r1 | none | | 20 | 1.39 | 0.46 | 2.11 | 0.85 |
| coarse_r2 | none | | 88 | 0.90 | 0.31 | 1.22 | 0.54 |
| coarse_r3 | none | | 368 | 0.27 | 0.18 | 0.24 | 0.21 |
| modal2 + grad | none | | 115 | 0.16 | 0.16 | 0.026 | 0.11 |
| modal10 + grad | none | | 123 | 0.13 | 0.13 | 0.016 | 0.094 |

The `proj`, n = 4 run is flagged unconverged; all others are converged by the Gate 1 rule. The two modal rows are the two lowest and ten lowest exact eigenvectors plus the gradients of the 113 level-3 coarse hat functions.

Verdict by the registered rule: **STABILITY-LIMITED**. The maximum held-out projection error is below 5% at an affordable size (0.025 at n = 8, 0.015 at n = 16 with the proj loss), but no affordable configuration has held-out energy error below 5%.

Reading:

- The failure is not interpolation. Training Galerkin error is 0.93 to 1.08 at every affordable size, equal to the held-out error. The Galerkin objective could not fit even the training frequencies, which supports inference 4 (a plateau near the zero solution).
- It is not capacity either. One shared W represents the solutions at all 17 frequencies to 2.5% with 36 dimensions (proj loss). So the capacity half of the cheap-sweep claim holds in this band, on this instance, with the reference allowed in the loss.
- The projection-loss spaces overshoot rather than collapse: Galerkin error up to 1.96 and energy error 3 to 4, at 8.2 to 8.5, away from the resonance. This is consistent with a spurious reduced mode inside the band. It is not measured: the reduced spectra of these spaces were not computed in this run. The earlier single-frequency diagnostic found modes missing from the resonance; the band case may add modes in the wrong place.
- At n = 24 (dim 300, over the 150 budget) the Galerkin and mixed losses reach held-out error 0.29 and energy error 12%. Still failing.
- The modal comparators meet the energy criterion with no training (1.6 to 2.6% at dim 115 to 123), but their field error (0.13 to 0.16) is about ten times the learned projection error (0.015). So the learned space is much more expressive and the modal space is stable, and the learned-W claim over modal bases rests on stability being fixed. The modal bases use exact eigenvectors, which cost an eigensolve and must be recomputed for each geometry.
- Not covered: stage (b) (geometry variation), an encoder, the reduced spectra of the Gate 1b spaces, a single-instance caveat throughout.

The registered rule says this outcome justifies the spectrum-aware loss. The reduced spectra of these spaces were then computed, see the next section: both failure kinds occur.

**Gate 1b follow-up: reduced spectra of the band spaces** (post hoc, outside the protocol; `experiments/diag_band_spectrum.py`)

The Gate 1b spaces were retrained with the registered seeds and steps. The held-out errors reproduce the table above exactly, so these are the same spaces. For each space, the reduced eigenvalues in the band [6.5, 9.0] are compared with the true mode at 7.604 (a degenerate pair), with a resonance half-width of 0.38.

| Loss | n | Reduced eigenvalue within half-width of 7.604? (nearest) | Reduced eigenvalues in the band | Not at the true mode | held-out gal | held-out proj |
|---|---|---|---|---|---|---|
| galerkin | 8 | no (52.1) | none | 0 | 0.99 | 0.92 |
| galerkin | 16 | no (5.36) | none | 0 | 1.06 | 0.69 |
| galerkin | 24 | yes (7.630) | 7.63 | 0 | 0.29 | 0.26 |
| proj | 8 | no (8.50) | 8.50, 8.96 | 2 | 1.96 | 0.025 |
| proj | 16 | no (8.34) | 6.74, 8.34 | 2 | 1.66 | 0.015 |
| proj | 24 | no (7.03) | 7.03, 8.31, 8.34 | 3 | 1.63 | 0.014 |
| mixed | 8 | no (22.9) | none | 0 | 1.01 | 0.050 |
| mixed | 16 | no (10.5) | none | 0 | 1.17 | 0.032 |
| mixed | 24 | yes (7.631) | 7.63 | 0 | 0.30 | 0.025 |

Reading:

- The two kinds of failure both occur, and which one depends on the loss. With the Galerkin and mixed losses at n <= 16, the reduced operator has no eigenvalue in the band at all, so the resonance is missing and the solution stays near the zero-solution plateau. With the projection loss, the true mode is missing and two or three eigenvalues sit inside the band at the wrong places (ghost resonances).
- The ghosts line up with the overshoot. The worst held-out frequencies for the projection loss were 8.53 (n = 8), 8.22 (n = 16) and 8.22 (n = 24), and the ghost eigenvalues are at 8.50, 8.34 and 8.31 to 8.34, each within about one half-width.
- No affordable space has a reduced eigenvalue at the true mode. The only spaces that do are the Galerkin and mixed losses at n = 24 (dim 300, over budget), and those are the only ones whose Galerkin error is below 0.5 at every frequency. This agrees with the single-frequency diagnostic, and it is the closest this project has to a controlled link between spectrum and accuracy. It is correlational, since n = 24 differs from the others in several ways.
- Those n = 24 spaces have one eigenvalue near 7.6 where the true mode is a degenerate pair, which was suggested as the reason their error is still 0.29. Gate 1c later refuted this: spaces with two modes at the true eigenvalue have the same error floor.
- A per-frequency summary (median distance to the nearest reduced eigenvalue for failing against passing frequencies) did not separate cleanly, so the evidence is the mode counts and the overshoot locations above, not a per-frequency relation.

Implication for the spectrum-aware loss: it has to do two jobs, place reduced eigenvalues at the true modes (both members of the pair) and keep others out of the band. Matching the lowest k reduced physical eigenvalues to the lowest k true ones, with k large enough to include the first mode beyond the band (18.8), covers both. Open design risks: eigenvalue gradients are unstable for a degenerate pair (use symmetric functions such as the sum over the pair), and the true eigenvalues must be available at training time, which costs one eigensolve per training instance.

**Gate 1c protocol (fixed before running)**

Question: does adding a spectrum-matching term to the loss fix the stability failure of Gate 1b, as the reduced-spectrum diagnostic suggests?

- Setting: identical to Gate 1b (instance, band, training and held-out frequencies, free nodal W with no encoder, sizes n = 4, 8, 12, 16, all within the 150-dimension budget, starts rbf seed 0 and smooth seed 0, 500 Adam steps, lr 0.05, best start chosen on the training objective only).
- Loss: `mixed` (mean Galerkin error plus mean projection error over the training frequencies) plus w times a spectral term. The spectral term is the mean over i = 1 to 4 of (log mu_i - log lambda_i)^2, where mu_i are the four lowest reduced physical eigenvalues (reduced curl-curl pencil, curl-free kernel excluded) and lambda_i = 7.604, 7.604, 18.83, 28.71 are the four lowest true physical eigenvalues of the instance. The fourth is included so that ghost eigenvalues between 7.6 and 18.8 are penalised.
- Weights: w = 0.1, 1 and 10. All three are registered and all are reported; no other weight is tried. The reference is the Gate 1b `mixed` result (w = 0).
- Cost note: the target eigenvalues need one eigensolve per training instance. That is training-time cost and must be counted in Gate 6.
- Spectrum fixed: two reduced eigenvalues within the half-width (0.05 x 7.604 = 0.38) of 7.604, and no other reduced eigenvalue in the band [6.5, 9.0]. Measured on a conditioned basis with the Gate 1b diagnostic code, not on the training-time computation.
- Pass: some affordable configuration has held-out stored-energy error below 5% at every held-out frequency. Whether it matches the modal comparators (1.6 to 2.6%) is reported separately.
- Spectrum fixed but inaccurate: some affordable configuration has the spectrum fixed, but none passes. The spectral explanation is then incomplete (right eigenvalues, wrong eigenvectors or solves). The next step is to examine eigenvector accuracy, not to tune weights.
- Spectrum not reached (the failure criterion, agreed in advance): no affordable configuration fixes the spectrum at any weight. The resonant claim is then unsupported in this formulation (free W, this capacity, this loss). Per the rules of engagement, no further loss tuning; the scope moves to off-resonance or narrowband problems, or to a different formulation (modal enrichment, a larger reduced space).
- Inconclusive: when the deciding configuration is unconverged by the Gate 1 rule.
- Multiplicity: 12 configurations (3 weights x 4 sizes) are examined, so a pass in one of them is weaker evidence than a pass in most. Report the number of passing configurations.

**Gate 1c results** (`experiments/gate1c.py`, protocol above, CPU; best of 2 starts chosen on the training objective)

"Modes at 7.604" counts reduced eigenvalues within the half-width (0.38) of the true mode; "ghosts" counts other reduced eigenvalues in [6.5, 9.0]. Held-out columns are over the 8 held-out frequencies. All dimensions are n(n+1)/2: 10, 36, 78, 136.

| w | n | Modes at 7.604 | Ghosts | Spectrum fixed | gal mean | gal max | proj max | energy max | Converged |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 4 | 0 | 1 | no | 0.99 | 1.07 | 0.44 | 0.99 | no |
| 0.1 | 8 | 1 | 0 | no | 0.42 | 0.49 | 0.16 | 0.27 | no |
| 0.1 | 12 | 1 | 0 | no | 0.29 | 0.30 | 0.034 | 0.14 | yes |
| 0.1 | 16 | 1 | 0 | no | 0.29 | 0.30 | 0.031 | 0.13 | yes |
| 1 | 4 | 1 | 0 | no | 0.87 | 0.91 | 0.42 | 0.86 | yes |
| 1 | 8 | 1 | 0 | no | 0.40 | 0.48 | 0.11 | 0.19 | no |
| 1 | 12 | 2 | 0 | yes | 0.30 | 0.33 | 0.033 | 0.145 | no |
| 1 | 16 | 2 | 0 | yes | 0.29 | 0.30 | 0.027 | 0.139 | no |
| 10 | 4 | 2 | 0 | yes | 0.93 | 0.94 | 0.46 | 0.86 | no |
| 10 | 8 | 2 | 0 | yes | 0.43 | 0.48 | 0.17 | 0.26 | no |
| 10 | 12 | 2 | 0 | yes | 0.29 | 0.31 | 0.035 | 0.133 | no |
| 10 | 16 | 2 | 0 | yes | 0.29 | 0.31 | 0.029 | 0.132 | yes |

Verdict by the registered rule: **SPECTRUM FIXED BUT INACCURATE**. Six of the twelve configurations have the spectrum fixed (w = 1 at n = 12 and 16; w = 10 at every n), and none of the twelve passes the 5% energy criterion. The best is w = 10, n = 16, with held-out energy error 0.132.

Reading:

- The spectral term has a large effect. The Gate 1b `mixed` spaces at the same sizes had Galerkin error about 1.0 and energy error 0.96 to 0.98. With the term, the held-out Galerkin error falls to about 0.29 and the energy error to 0.13 to 0.15 at n = 12 and 16. Misplaced reduced eigenvalues were therefore a main cause of the Gate 1b plateau, and the cause is controllable by the loss.
- It is not sufficient. A floor remains: the mean held-out Galerkin error is 0.29 at n = 12 and 16 for every weight, and the maximum is 0.30 to 0.33. The projection error of the same spaces is 0.03, about ten times lower. This is the same floor as the Gate 1b n = 24 spaces (0.29 to 0.30).
- The floor is spread across the band, not concentrated at the resonance: the maximum is within 5 to 10% of the mean at n = 12 and 16. So the remaining error is not a resonance-localised effect.
- The earlier suggestion that a single reduced mode, against a degenerate pair, explains the n = 24 floor is refuted. At w = 0.1 (n = 12 and 16) there is one mode at the true eigenvalue and the error is 0.29. At w = 10 there are two and the error is also 0.29.
- Neither the weight (0.1 to 10) nor the size (12 versus 16) moves the floor. The modal comparators remain ahead on both measures: held-out energy error 0.016 to 0.026 against 0.13, and field error 0.13 to 0.16 against 0.29.
- Eight of the twelve runs are flagged unconverged by the Gate 1 rule, but the converged ones (w = 0.1 at n = 12 and 16, w = 10 at n = 16) show the same floor, so it is not simply an optimisation artefact.

Per the registered rule the next step is to examine eigenvector accuracy and decompose the error, not to tune weights. The untested candidates for the floor are: the matched reduced eigenvectors are only approximately the true modes; the spectrum above the four matched eigenvalues, or the low-overlap reduced modes, contributes; the solution component outside the resonant pair is solved inaccurately. The diagnostic would report, for the spaces above, the capture error of the two true modes (the M-norm distance of each from the space), the Galerkin error split into its component along the true resonant eigenvectors and the remainder, and the lowest eight reduced physical eigenvalues (the loss only controls four).

Caveats: one instance, the reference is in the loss, and the spectral term needs the true eigenvalues at training time.

**Gate 1c follow-up: error decomposition** (post hoc, outside the protocol; `experiments/diag_error_decomposition.py`)

The two best-documented Gate 1c spaces at n = 16 were retrained with the registered seeds. Both reproduce their Gate 1c held-out errors (w = 10: gal max 0.305, energy 0.135; w = 0.1: 0.298, 0.132). The true solution and the Galerkin error are expanded in the exact fine-mesh eigenbasis and grouped into the curl-free part (the kernel, 481 dimensions), the resonant pair at 7.604, and the other transverse modes. Values are shares of ||e_ref||^2, averaged over the 17 frequencies; the three error shares sum to rel_err^2.

| Quantity | w = 10, n = 16 | w = 0.1, n = 16 |
|---|---|---|
| True solution: kernel / pair / other | 0.077 / 0.919 / 0.004 | same |
| Galerkin error: kernel / pair / other | 0.0726 / 0.0082 / 0.0016 | 0.0720 / 0.0068 / 0.0013 |
| rel_err (mean) | 0.287 | 0.283 |
| At omega^2 = 7.594: true kernel share, error kernel share | 0.017, 0.077 | 0.017, 0.076 |
| Lowest reduced physical eigenvalues | 7.585, 7.648, 18.86, 28.68, 32.8, 33.4, 34.8, 37.0 | 5.98, 7.633, 18.66, 28.49, 28.8, 30.1, 31.1, 32.1 |
| sin of principal angles, true pair in the space | 0.03, 0.43 | 0.03, 0.41 |
| sin of principal angles, reduced resonant eigenvectors against the true pair | 0.29, 0.65 | not defined (one mode) |

Reading:

- About 88 to 90% of the squared error is in the curl-free component, 10% in the resonant pair and 2% elsewhere. The true solution is 92% resonant pair and only 8% curl-free (1.7% at resonance), so the floor is not the resonance response. The Galerkin solution gets the resonant pair right to about 9% in amplitude and gets the curl-free part wrong by roughly its entire size on average (error share 0.073 against a content of 0.077), and by more than its size at resonance.
- Eigenvector accuracy is imperfect: one of the two true degenerate modes lies well inside the space (sin 0.03) and the other does not (sin 0.43). It affects the error little here, probably because this single source mostly excites the well-captured direction. That is not verified, and a different source may expose it.
- The two spaces differ in their spectrum (a mode at 5.98 below the band for w = 0.1, none for w = 10) and have the same floor, so the remaining error is not about the eigenvalues the loss controls.

Intervention (post hoc): add the exact gradients of coarse hat functions to the trained space, enlarging the curl-free part.

| Space | Added gradients | dim | held gal mean / max | energy max | reduced eigenvalues in band |
|---|---|---|---|---|---|
| w = 10, as trained | 0 | 136 | 0.29 / 0.31 | 0.135 | 7.585, 7.648 |
| w = 10 + level-1 hats | 5 | 141 | 0.38 / 0.52 | 0.47 | 7.80, 8.56 |
| w = 10 + level-2 hats | 25 | 161 | 0.57 / 0.93 | 1.26 | 7.98 |
| w = 10 + level-3 hats | 113 | 249 | 0.68 / 1.08 | 1.70 | 8.08 |
| w = 0.1 + level-1 hats | 5 | 141 | 0.31 / 0.38 | 0.30 | 7.72 |
| w = 0.1 + level-2 hats | 25 | 161 | 0.52 / 0.83 | 1.04 | 7.12, 7.93 |
| w = 0.1 + level-3 hats | 113 | 249 | 0.62 / 0.99 | 1.45 | 7.85, 8.03 |

The projection error stays at 0.026 to 0.028 in every row.

- The simple fix fails: giving the curl-free part more room after training makes the Galerkin error worse, not better. The in-band eigenvalues move away from 7.604 (up to 8.08), which is the failure the spectral term was meant to remove.
- The likely reason, not yet tested: the trained resonant vectors carry some curl-free content, and the loss tuned their eigenvalues against the original, small kernel. Enlarging the kernel changes the transverse complement and shifts those eigenvalues. The spectral condition and the size of the curl-free space interact, so they would have to be trained together. The modal comparators avoid this because exact eigenvectors are orthogonal to every gradient.
- So the decomposition shows where the error is, but it does not show why. The explanation "the reduced 0-form space is too small" fits the decomposition and the consistency results, and this intervention neither confirms nor refutes it, because it also moves the spectrum.

Proposed next step, to be registered before running: a Gate 1d that places the larger curl-free space inside the training loop, so that the Galerkin, projection and spectral terms are all computed on the enlarged space. Compare against Gate 1c with the same pass criterion.

**Family and band decision: resonance is included**
*Superseded on 2026-10-04 by the scope decision at the top of this file: resonance is out of scope and the main family is sub-resonant. The measurements below (mode ranges across the family) remain valid.*

Decision: the problem family keeps resonance in its frequency band. Reason: the intended design problems (filters, cavities, antennas) are resonant, and a non-resonant family would weaken the relevance, weaken the comparison with baselines, and make the optimiser-shift test in Gate 5 uninformative.

Measured on 60 random instances of the current family (disc centre 0.3 to 0.7, radius 0.10 to 0.25, epsilon 2 to 8; seed 0; dense eigensolve):

| Mode | min | median | max |
|---|---|---|---|
| lambda_1 | 5.42 | 7.76 | 9.35 |
| lambda_2 | 5.91 | 7.99 | 9.40 |
| lambda_3 | 12.64 | 18.49 | 19.58 |

So the band omega^2 in [2, 12] contains the first resonance doublet of every instance (a degenerate pair when the disc is centred, split otherwise) and no other mode. The resonance location moves by about 4 across the family, so training and evaluation must treat resonance position as an instance property, and spectral targets are per instance (lambda_1 to lambda_4). Band edges and loss tangent (0.05) are fixed here and must not be changed after results are seen.

Consequence for the plan: resonance accuracy is now on the critical path for Gates 2 to 4. Gate 1d below is the last variation to be tried in the current formulation before the scope decision.

**Gate 1d protocol (fixed before running)**

Question: does placing a larger curl-free space inside the training loop remove the Gate 1c error floor?

- Setting: identical to Gate 1c (instance, band, frequencies, free nodal W, starts, 500 Adam steps, lr 0.05, selection on the training objective only), with spectral weight w = 10 only (spectrum was fixed at every size at that weight in Gate 1c).
- Change: before whitening, the fixed exact gradients of the interior hat functions of a coarse mesh (level 1: 5 functions, level 2: 25 functions) are appended to the lifted vectors, so that the Galerkin, projection and spectral terms are all computed on the enlarged space. Level 3 (113 functions) is excluded because it leaves too little of the 150-dimension budget for learned vectors.
- Grid: level 1 and 2, n = 4, 8, 12, 16. Dimension is n(n+1)/2 plus the added functions. Level 2 at n = 16 has dimension 161, over the budget, so it is reported but excluded from the verdict. Seven configurations are in the verdict.
- Reference: Gate 1c, w = 10: held-out mean Galerkin error 0.29 at n = 12 and 16, maximum 0.31, energy error 0.13.
- Pass: some affordable configuration has held-out stored-energy error below 5% at every held-out frequency.
- Improved but not passing: no pass, but some affordable configuration has held-out mean Galerkin error below 0.15 (half the Gate 1c floor).
- No improvement (the failure criterion, agreed in advance): every affordable configuration has held-out mean Galerkin error of 0.15 or more. The floor is then not removed by enlarging the curl-free space in the loop. Per the rules of engagement, no further variation of this formulation; the scope decision is taken (for example modal enrichment, a larger or two-level reduced space, or narrowing the claim).
- Inconclusive: when the deciding configuration is unconverged by the Gate 1 rule.
- Multiplicity: seven configurations are examined. Report how many pass and how many improve.

**Gate 1d results** (`experiments/gate1d.py`, protocol above, CPU; w = 10; best of 2 starts chosen on the training objective)

| Added gradients | n | dim | gal mean | gal max | proj max | energy max | Spectrum fixed | Converged |
|---|---|---|---|---|---|---|---|---|
| 5 (level 1) | 4 | 15 | 0.93 | 0.94 | 0.43 | 0.90 | yes | yes |
| 5 (level 1) | 8 | 41 | 0.35 | 0.40 | 0.082 | 0.15 | yes | no |
| 5 (level 1) | 12 | 83 | 0.295 | 0.32 | 0.033 | 0.138 | yes | yes |
| 5 (level 1) | 16 | 141 | 0.285 | 0.30 | 0.027 | 0.135 | yes | yes |
| 25 (level 2) | 4 | 35 | 0.90 | 0.92 | 0.34 | 0.85 | yes | yes |
| 25 (level 2) | 8 | 61 | 0.35 | 0.39 | 0.084 | 0.17 | yes | no |
| 25 (level 2) | 12 | 103 | 0.295 | 0.33 | 0.029 | 0.19 | yes | yes |
| 25 (level 2) | 16 | 161 | 0.286 | 0.30 | 0.027 | 0.149 | yes | yes |

The level 2, n = 16 row is over the 150-dimension budget and is excluded from the verdict.

Verdict by the registered rule: **NO IMPROVEMENT**. Seven configurations are in the verdict; none passes, and the best held-out mean Galerkin error is 0.285, against the 0.15 criterion. The spectrum is fixed in all eight configurations, including the excluded one.

Reading:

- The floor is unchanged. The held-out mean Galerkin error is 0.285 to 0.295 at n = 12 and 16 for 0, 5 and 25 added gradients (Gate 1c: 0.29). Enlarging the exact curl-free space by a few functions inside the loop does not move it, and unlike the post hoc augmentation it leaves the spectrum fixed.
- Post hoc check (`experiments/diag_longitudinal.py`): the curl-free part of the true solution is the gradient of the solution of a coercive scalar problem, independent of omega. Its share of ||e_ref||^2 is 0.181, 0.017 and 0.157 at omega^2 = 6.5, 7.594 and 9.0, which matches the decomposition above (0.017 and 0.156). Its best approximation by the exact gradients of coarse hat functions is poor and converges slowly: relative error 0.83 with 5 functions, 0.55 with 25, 0.34 with 113.
- This is consistent with the floor. The curl-free part comes from a localised source (Gaussian, sigma 0.08), so its potential is not resolved by a few smooth functions, and the Galerkin solution can only use exact gradients for it. It also explains why 5 against 25 added functions made no difference. It is supporting evidence, not a proof: the learned bumps themselves were not checked against this.
- Reaching a small error by enlarging the reduced curl-free space would need hundreds of functions (113 still leaves 34% error), which is outside the budget and defeats the purpose of a small reduced space.

Per the rules of engagement, no further variation of this formulation is tried. Scope decision, with the options considered:

- Hodge split (recommended for the next registered gate): solve the curl-free part exactly on the fine mesh and learn only the transverse reduced space. The curl-free part is e_L = -(1/omega^2) grad(phi), with phi from one coercive scalar solve per instance (481 unknowns here), independent of omega, so frequency sweeps stay cheap. The transverse part is solved in the learned space after projecting out the curl-free directions, with right-hand side f - M grad(phi). This targets the 88 to 90% of the error that sits in the curl-free part. Caveats: untested; projecting out the curl-free content changes the learned vectors' eigenvalues (as the post hoc augmentation showed), so it has to be trained inside the loop. A scalar solve is also much cheaper than an indefinite Maxwell solve, and more so in 3D, but that is not measured here.
- Modal enrichment: add a few eigenvectors, which the modal comparators suggest is stable but needs an eigensolve per geometry.
- Narrow the claim to off-resonance and narrowband problems, where Gate 1 already passes. This contradicts the decision to include resonance, so it is the fallback.

**Gate 1e protocol (fixed before running)**

Question: does solving the curl-free part exactly on the fine mesh, and learning only the transverse reduced space, remove the Gate 1c/1d error floor?

Formulation (Hodge split). Let Gn be the gradient operator from interior nodes to interior edges, and A0 = Gn^T M Gn (coercive). Per instance, solve A0 phi = Gn^T f once. Then:

- Curl-free part: e_L(omega) = -(1/omega^2) Gn phi, exact, with no dependence on omega beyond the scalar factor.
- Transverse right-hand side: f_T = f - M Gn phi, which annihilates every gradient.
- Learned space: the lifted Whitney vectors V with their curl-free content projected out, V_T = V - Gn A0^-1 Gn^T M V, whitened to Q_T (relative jitter 1e-8, because the projection makes n linear dependencies exact).
- Transverse solve: (Kr - omega^2 Mr) y = Q_T^T f_T with Kr = Q_T^T K Q_T, Mr = Q_T^T M Q_T, and e = e_L + Q_T y. The reduced matrices do not depend on omega.

- Setting: identical to Gate 1c and 1d (instance, band, frequencies, free nodal W, starts, 500 Adam steps, lr 0.05, selection on the training objective only), spectral weight 10, n = 4, 8, 12, 16.
- Loss: mean over training frequencies of the Galerkin error of e against e_ref, plus the transverse best-approximation error (projection error of e_ref - e_L onto span(Q_T), relative to ||e_ref||), plus 10 times the spectral term (mean squared log-ratio between the four lowest reduced eigenvalues of the transverse pencil and 7.604, 7.604, 18.83, 28.71).
- Dimension: the number of independent transverse reduced functions (numerical rank of V_T), at most 150. The exact scalar solve (481 unknowns here) is not a reduced dimension and is counted separately in the cost gate. This is a deliberate accounting choice and is reported with every result.
- Implementation checks before the run (reported, not part of the verdict): the kernel-annihilation residual ||Gn^T M (e_ref - e_L)|| / ||e_ref|| below 1e-10; and a Hodge solve with the ten lowest exact transverse eigenvectors in place of Q_T, which must give a small error, to confirm the plumbing.
- Reference: Gate 1c and 1d, held-out mean Galerkin error 0.285 to 0.295 at n = 12 and 16, energy error 0.13 to 0.15; the modal comparators, energy error 0.016 to 0.026.
- Pass: some affordable configuration has held-out stored-energy error below 5% at every held-out frequency.
- Improved but not passing: no pass, but some affordable configuration has held-out mean Galerkin error below 0.15.
- No improvement (the failure criterion, agreed in advance): every affordable configuration has held-out mean Galerkin error of 0.15 or more. Then the Hodge split does not remove the floor and no further variation of the learned-W formulation is tried. The decision then falls to modal enrichment, or to narrowing the claim to off-resonance, which would reverse the band decision.
- Inconclusive: when the deciding configuration is unconverged by the Gate 1 rule.
- If it passes: resonance stays in the family, and the next gates are Gate 1b stage (b) (geometry variation, where the spectral targets become per-instance) and Gate 2 (encoder).
- Multiplicity: four configurations are examined; report how many pass or improve.

**Gate 1e results** (`experiments/gate1e.py`, protocol above, CPU; w = 10; best of 2 starts chosen on the training objective)

| n | dim | gal mean | gal max | proj max | energy max | Modes at 7.604 | Ghosts in band | Converged |
|---|---|---|---|---|---|---|---|---|
| 4 | 6 | 1.08 | 1.62 | 0.110 | 0.96 | 0 | 0 | no |
| 8 | 28 | 0.91 | 1.56 | 0.014 | 2.84 | 0 | 2 (8.28, 8.29) | yes |
| 12 | 66 | 0.91 | 1.56 | 0.013 | 2.82 | 0 | 2 (8.28, 8.28) | yes |
| 16 | 119 | 0.92 | 1.59 | 0.014 | 2.96 | 0 | 2 (8.29, 8.78) | no |

Verdict by the registered rule: **NO IMPROVEMENT**. No configuration passes, and the best held-out mean Galerkin error is 0.907, against the 0.15 criterion. This is worse than Gates 1c and 1d (0.285 to 0.295), and the spectrum was not fixed in any configuration.

Implementation checks (run before the experiment, all as registered): the kernel-annihilation residual is 9e-14 (required below 1e-10), and a Hodge solve with the ten lowest exact transverse eigenvectors gives held-out error 0.016 and energy error 3e-4, so the exact curl-free part and the transverse right-hand side are correct. Cost, informational for one instance: the scalar curl-free solve takes 1.0 to 1.4 ms (481 unknowns) against 3.9 to 4.5 ms for the fine Maxwell solve (1504 unknowns).

Implementation note: after the curl-free content is projected out, the lifted vectors involving the partition's remainder function are dependent on the others, so only the pairs among the n learned bumps are built. The span matches the all-pairs span to 1e-13 at n = 4 and 8 and to 2e-4 at n = 16, where the numerical rank is threshold sensitive (118 against 119). A ridge of 1e-8 is added to the reduced matrices.

Post hoc diagnostic (outside the protocol; `experiments/diag_hodge.py`):

- Training breakdown at n = 12: both starts end with the lowest reduced pair at 8.278, 8.281 and 8.284, 8.344, about 9% above the true 7.604 and outside the half-width of 0.38. The spectral term contributes 0.008 to the objective, against 0.86 for the Galerkin term. The projection term is 0.012.
- Applying the Hodge split after the fact to the Gate 1c space that had fixed the spectrum (w = 10, n = 16) moves its in-band eigenvalues from 7.585 and 7.652 to a single 8.277, and gives held-out mean error 0.909 and energy error 2.87. That is the same result as the trained Gate 1e spaces.
- The pair sits near 8.28 at n = 8, 12 and 16, so this is not simply an optimiser accident or a capacity limit in n.

Reading (inferred, not tested directly):

- Reduced eigenvalues of a space that is orthogonal to every gradient are upper bounds on the true transverse eigenvalues (min-max principle), so a reduced mode cannot sit below 7.604. The Gate 1c modes at 7.585 and 7.648 were therefore reached with curl-free content mixed into the learned vectors. This fits the measured capture and alignment angles for that space (0.03 and 0.43; 0.29 and 0.65), and it fits the post hoc result above, where removing the curl-free content raised the eigenvalue to 8.28. The "spectrum fixed" status of Gate 1c was partly spurious.
- The transverse space represents the solution well in the M-norm (projection error 0.013) but not in the energy (curl) norm: a fine-scale error of about 1% can carry enough curl energy to raise a Rayleigh quotient by 9%. The projection loss measures only the M-norm and the spectral term, as weighted here, is too weak to compensate. Capturing the resonant modes in the energy norm is what the resonance needs. Whether that is achievable at n <= 16 is untested.
- The registered verdict stands, but this run did not reach the regime the Hodge split was meant to be tested in. It does not show that the split fails to remove the curl-free floor when the spectrum is right; it shows that, with this loss, the learned transverse space does not reach the spectrum. Re-attempting with an energy-norm capture loss would be a second attempt on the same idea, after a registered failure, and would have to be registered as such, with the extra multiplicity acknowledged.
- A conventional comparator emerges: exact curl-free solve plus the ten lowest exact transverse eigenvectors gives 1.6% error and 3e-4 energy error, at the price of one eigensolve per geometry. The learned-W formulation has not matched it.

**Gate 1f protocol (fixed before running). This is a second attempt on the Hodge-split idea, made after Gate 1e met its failure criterion.**

Question: does a loss that measures the capture of the true resonant modes in the energy norm get the Hodge-split learned space to the resonant spectrum, and does the error then fall?

- Setting: identical to Gate 1e (Hodge split, instance, band, frequencies, free nodal W, starts, 500 Adam steps, lr 0.05, selection on the training objective only), sizes n = 4, 8, 12, 16, spectral weight 10.
- New term: for the four lowest true transverse eigenvectors phi_k (7.604, 7.604, 18.83, 28.71; M-orthonormal), the relative energy-norm best-approximation error in span(Q_T), measured in the (K + M) norm: err_k^2 = 1 - (1 + lambda_k) b_k^T (Kr + Mr)^-1 b_k with b_k = Q_T^T M phi_k. The terms are grouped into the resonant pair (their mean, which does not depend on the basis chosen within the degenerate pair), lambda_3 and lambda_4. The loss term is the mean over the three groups of log(err^2 + 1e-6), so that it keeps a useful gradient when the error is small. This replaces the M-norm blindness of the Gate 1e projection term.
- Loss: Gate 1e loss (Galerkin error, transverse M-norm projection error, 10 times the spectral term) plus w_cap times the capture term, with w_cap = 0.1 and 1. Both weights are registered and reported; no other weight is tried.
- Cost note: the capture term needs four exact transverse eigenvectors per training instance. This is training-time cost and must be counted in Gate 6.
- Dimension, spectrum fixed, budget and pass criteria: as in Gate 1e (affordable dimension at most 150; spectrum fixed means two reduced eigenvalues within the half-width 0.38 of 7.604 and none other in [6.5, 9.0]). Also reported: the final capture error for each group.
- Pass: some affordable configuration has held-out stored-energy error below 5% at every held-out frequency.
- Improved but not passing: no pass, but some affordable configuration has held-out mean Galerkin error below 0.15.
- No improvement (the failure criterion, agreed in advance): every affordable configuration has held-out mean Galerkin error of 0.15 or more.
- Hard stop: on no improvement, no further variation of the learned-W formulation is tried. The fallback is taken: the claim narrows to off-resonance and narrowband problems, the exact curl-free solve plus modal truncation is reported as the conventional comparator, and the band decision to include resonance is reversed for the main family (resonance stays as a reported stress test).
- Inconclusive: when the deciding configuration is unconverged by the Gate 1 rule (applied with the magnitude of the loss in the denominator, since the loss can now be negative). One rerun with 1000 steps is allowed in that case, and only in that case.
- Multiplicity: eight configurations are examined here, after four in Gate 1e, so twelve configurations of the Hodge-split idea have been tried in total. Report the count of passes and improvements out of the eight and state the total.

**Gate 1f results** (`experiments/gate1f.py`, protocol above, CPU; spectral weight 10; best of 2 starts chosen on the training objective). Capture is the relative energy-norm error^2 of the true modes in the space.

| w_cap | n | dim | gal mean | gal max | proj max | energy max | Capture pair | Capture lambda_3 | Capture lambda_4 | Ghosts in band | Converged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 4 | 6 | 1.082 | 1.615 | 0.117 | 0.96 | 0.212 | 0.413 | 0.136 | 0 | no |
| 0.1 | 8 | 28 | 0.910 | 1.564 | 0.015 | 2.84 | 0.0746 | 0.116 | 0.042 | 2 (8.281, 8.286) | yes |
| 0.1 | 12 | 66 | 0.908 | 1.561 | 0.013 | 2.83 | 0.0741 | 0.115 | 0.041 | 2 (8.278, 8.279) | yes |
| 0.1 | 16 | 119 | 0.915 | 1.574 | 0.014 | 2.89 | 0.0753 | 0.118 | 0.042 | 2 (8.283, 8.299) | yes |
| 1 | 4 | 6 | 1.080 | 1.605 | 0.123 | 0.97 | 0.226 | 0.529 | 0.055 | 0 | yes |
| 1 | 8 | 28 | 0.923 | 1.586 | 0.016 | 2.95 | 0.0751 | 0.116 | 0.041 | 2 (8.285, 8.292) | yes |
| 1 | 12 | 66 | 0.908 | 1.563 | 0.013 | 2.85 | 0.0739 | 0.115 | 0.041 | 2 (8.276, 8.278) | yes |
| 1 | 16 | 120 | 0.909 | 1.566 | 0.012 | 2.86 | 0.0740 | 0.115 | 0.041 | 2 (8.277, 8.279) | yes |

Sanity checks of the capture term (as registered): error 0 with the ten exact transverse eigenvectors, 1 with a random transverse space.

Verdict by the registered rule: **NO IMPROVEMENT**. None of the eight configurations passes or improves; the best held-out mean Galerkin error is 0.908 (w_cap = 0.1, n = 12). Across Gates 1e and 1f, twelve configurations of the Hodge-split idea were tried: none passed and none improved.

Reading:

- The capture term did not reduce the capture error. The energy-norm error^2 of the resonant pair is 0.074 to 0.075 at n = 8, 12 and 16 for both weights, to two significant figures, and the lowest reduced eigenvalues stay at 8.28, so the result does not depend on size (dim 28 to 120) or weight.
- The capture error accounts for the eigenvalue offset. A Rayleigh-Ritz estimate gives a shift of about 0.074 x (1 + 7.6) = 0.64, against the observed 0.67 (8.28 against 7.604). This supports the Gate 1e diagnosis (the resonant modes are not captured in the energy norm) as the mechanism of the shift. It does not show a remedy: the loss that targets this error could not lower it.
- Open and untested: whether 0.074 is a structural limit of the lifted-Whitney family built from softmax partitions of unity, or a trap that this optimiser reaches from both starts at every size. Distinguishing them would need a further variation (for example a capture-only objective with many starts), which the hard stop excludes.

Decision, as pre-registered: the hard stop is in effect.

- The learned-W formulation is not supported for resonant accuracy after Gates 1b to 1f. No further variation of it is tried.
- The claim narrows to off-resonance and narrowband problems, where Gate 1 already passes.
- The conventional comparator is the exact curl-free solve plus modal truncation (1.6% error and 3e-4 energy error with ten exact transverse modes, at one eigensolve per geometry).
- The band decision to include resonance is reversed for the main family. Resonance stays as a reported stress test (an out-of-distribution frequency range). The new main-family band is still to be defined and registered before any data are generated.

**Band pre-check protocol (fixed before running)**

Purpose: choose the edges of the sub-resonant band objectively, so that resonance does not leak into the main family, using only the oracle on the Gate 1 instance. No family data and no baselines are used, and the band is not revisited after family results are seen.

- Setting: the Gate 1 instance (disc at (0.5, 0.5), radius 0.2, epsilon 4, same source, tan(delta) = 0.05, lambda_1 = 7.604). Frequencies omega^2 = r x lambda_1 x (1 + 0.05j) for r = 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 and 0.9.
- Oracle: free nodal W, n = 8 (36 dimensions), starts rbf seed 0 and smooth seed 0, 500 Adam steps, Galerkin loss as in Gate 1, best start chosen by the training loss.
- A ratio passes when the stored-energy error is below 5%. Reported for every ratio: field error, projection error, energy error.
- Rule: the band is the longest contiguous run of passing ratios among 0.2 to 0.8, at least three ratios long. r = 0.9 is reported as an indication only and is not selectable. If no run of three exists, the choice goes back to the project owner.
- The main family then samples omega^2 = r x lambda_1(instance) with r uniform on the selected band, where lambda_1 is the instance's own first mode.
- Caveat: one instance and one reduced size. The check guards against resonance leakage; it is not a proof that every family member is accurate in the band.

**Band pre-check results** (`experiments/band_precheck.py`, protocol above, CPU; oracle W, n = 8, best of 2 starts)

| r = omega^2 / lambda_1 | omega^2 | field error | projection error | energy error | Converged | Result |
|---|---|---|---|---|---|---|
| 0.2 | 1.52 | 0.019 | 0.017 | 0.001 | no | pass |
| 0.3 | 2.28 | 0.021 | 0.018 | 0.003 | no | pass |
| 0.4 | 3.04 | 0.026 | 0.020 | 0.007 | yes | pass |
| 0.5 | 3.80 | 0.038 | 0.026 | 0.018 | yes | pass |
| 0.6 | 4.56 | 0.062 | 0.042 | 0.042 | yes | pass |
| 0.7 | 5.32 | 0.106 | 0.083 | 0.080 | yes | fail |
| 0.8 | 6.08 | 0.915 | 0.883 | 0.845 | yes | fail |
| 0.9 | 6.84 | 0.946 | 0.705 | 0.989 | yes | fail (indication only) |

Band selected by the registered rule: **r in [0.2, 0.6]**, five passing ratios. The main family samples omega^2 = r x lambda_1(instance) with r uniform on [0.2, 0.6].

Reading:

- The error rises steadily with r (field error 0.019, 0.021, 0.026, 0.038, 0.062, 0.106), which is the influence of the nearby resonance showing before it dominates. The edge at r = 0.6 is marginal (energy error 0.042 against the 5% limit), so other family members may sit closer to the limit than this instance.
- The jump from 0.106 at r = 0.7 to 0.915 at r = 0.8 is abrupt. It may be the oracle optimiser failing (the Gate 1 diagnostics showed the same plateau near the zero solution), not a sharp physical change. This was not investigated, and it does not affect the band choice.
- Two of the five passing runs (r = 0.2 and 0.3) are flagged unconverged by the Gate 1 rule, so their errors are upper bounds.
- Caveats: one instance, one reduced size, two starts. The band is fixed here and is not revisited after family results are seen. Per the scope decision, behaviour above the band is a documented limitation.

**Verification study protocol (fixed before running)**

Question: is the fine-mesh residual, which costs one sparse matrix-vector product, a reliable enough indicator of a surrogate's true error to certify accuracy with calibrated conformal coverage, including when candidates are chosen by their surrogate score, and how many full-wave solves does that avoid? This tests H4. It does not depend on the learned W: the surrogates here are existing non-learned or data-driven baselines, so the certificate is judged independently of whether a learned model is ever accurate enough.

Data and surrogates:

- Family: the main sub-resonant family (omega^2 = r x lambda_1, r uniform on [0.2, 0.6]), refine 4, tan(delta) = 0.05, Gaussian source with random position and direction as in `FamilyConfig`.
- Splits, per repeat: 300 training instances (POD only), 1000 calibration instances, 1000 test instances, from separate seeds (100 + r, 200 + r, 300 + r). Five repeats, r = 0 to 4, each with its own POD training. Report the mean and the standard deviation over repeats.
- Surrogates: `coarse_r2` (dim 88), `coarse_r3` (dim 368), and POD-Galerkin of rank 16, 64 and 128 trained on the 300 training instances. FNO is excluded here (it needs the GPU) and is added in the baseline study.

Quantities per surrogate and instance:

- True error eps = ||e - e_ref||_M / ||e_ref||_M against the fine solution.
- Residual r = f - (K - omega^2 M) e_h on the interior edges, and three indicators: eta_D = ||r||_{D^-1} / ||f||_{D^-1} with D = diag(M) (primary, O(N) and spectrally equivalent to the dual norm); eta_M, the same with the exact M^-1 (reference, to check the equivalence); eta_2, the Euclidean relative residual (secondary).

Conformal certificate (primary indicator eta_D, split conformal on the ratio score):

- Score s = eps / eta on the calibration set. q_hat is the ceil((n + 1)(1 - alpha)) / n empirical quantile. The certified bound is B(x) = q_hat x eta(x), with marginal guarantee P(eps <= B) >= 1 - alpha under exchangeability of calibration and test instances.
- alpha = 0.1 is the graded level; alpha = 0.05 is reported. A naive uncalibrated bound B = eta is reported to show what calibration buys.
- Decision for an accuracy target tau (0.02, 0.05, 0.10, 0.20): accept the surrogate answer if B <= tau, otherwise fall back to a full-wave solve. Reported per tau: acceptance rate (the fraction of full-wave calls avoided), and the false-accept rate (accepted with eps > tau), which must be at most alpha.
- Cost, reported separately and not as a speed claim: median wall-clock of the full solve, the surrogate prediction and the residual, on 100 instances. In this 2D setting the surrogates may not be faster than the full solve; calls avoided is the currency of problems where a full-wave solve is expensive.

Selection test (best-of-N, a proxy for optimiser-induced shift): the figure of merit is FoM(e) = e^H M_R e, the field energy in the target region R = [0.65, 0.85] x [0.15, 0.35] (M_R is the mass matrix weighted by the indicator of R, from the element centroids).

- Selected set: the 10% of test instances with the highest surrogate FoM.
- Reported on the selected set against the full test set: coverage of the certified bound, the false-accept rate, the FoM overestimation (surrogate FoM against the true FoM, the optimiser's curse), and the acceptance rate.
- If coverage on the selected set drops, the recalibrated variant is evaluated: calibrate on the 10% of the calibration set selected by the same rule.
- Verified top-K (K = 10): walk down the test instances ranked by surrogate FoM and collect K verified candidates, where verified means certified at tau = 0.10 or solved by full-wave. Report the full-wave calls used, against K for solving the top K directly, and the mean true FoM of the K collected against the true top K of the pool.
- Conditional coverage in bins of r (0.2 to 0.3, ..., 0.5 to 0.6) is reported, not graded.

Decision rules (per surrogate, on the primary indicator, at alpha = 0.1):

- Tracks: Spearman correlation between log eta_D and log eps on the test set is at least 0.9 (weak between 0.7 and 0.9; does not track below 0.7).
- Valid on random instances: mean empirical coverage over the repeats is within 2 percentage points of 90%.
- Valid under selection: mean coverage on the selected set is within 2 percentage points of 90%.
- Verdicts: CERTIFICATE VALID (tracks, valid on random instances, valid under selection); NEEDS RECALIBRATION (valid on random instances but not under selection); DOES NOT TRACK (Spearman below 0.7, so the residual is not usable as a certificate); otherwise INCONCLUSIVE. The overall statement lists the verdict for each of the five surrogates.
- Savings are reported but not graded: the study establishes whether the certificate is valid, and savings will be revisited with the learned surrogate.

Limits of this study, stated in advance: the surrogates are not the learned model; selection is best-of-N from a pool, not a gradient-based design loop (that belongs to H6); the conformal guarantee is marginal and assumes the test instances come from the same family as the calibration set (distribution shift is not tested); 2D timings say nothing about speed at scale; the target region and the selection fraction are fixed here and are not changed after results are seen.

**Scaling study protocol (fixed before running)**

Question: at what mesh size, and for how many frequencies, is a reduced solve faster than a full-wave solve, and what does the residual certificate cost relative to each? This is the cost half of the project's speed argument. It measures cost only; accuracy of the reduced spaces is established elsewhere.

Setting:

- Meshes: unit square, refine 4, 5, 6, 7, 8 (about 1.6k, 6k, 25k, 100k and 400k edges). The Gate 1 geometry (disc at (0.5, 0.5), radius 0.2, epsilon 4), the Gate 1 source, tan(delta) = 0.05, and omega^2 = 0.4 x lambda_1 (inside the sub-resonant band) with lambda_1 from `fem.first_mode` on that mesh.
- Timing method: one warm-up call and the median of 7 timed calls (`perf_counter`), single thread (`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and the torch thread count all set to 1), no other jobs running. The machine and library versions are recorded with the results.
- Full-wave reference: `fem.solve_fine` (SciPy sparse direct solve of the interior system), reported with and without the operator assembly. Assembly is shared with the reduced methods and is reported separately.
- Reduced methods: random orthonormal dense bases of dimension 16, 64 and 128 (cost proxies for POD or a learned dense basis; accuracy is irrelevant here), and the non-learned compatible `coarse_r2` and `coarse_r3` bases (dimension 88 and 368). A basis is skipped when N x q x 8 bytes exceeds 2 GB, and the skip is reported.
- What is timed for a reduced method: the per-geometry setup (projection of K and M onto the basis, the dense q x q assembly, conversion to torch), the small solve, and the reconstruction of the field (Q y). K is also projected per geometry, because a learned basis changes with the geometry; the cost of a fixed basis (K projected once) is reported as a second variant. The cost of predicting the basis itself (an encoder pass and the Whitney lift) is not included, because no network exists yet.
- Certificate cost: eta_D from one residual evaluation (two sparse matrix-vector products and a diagonal-norm sum).

Reported per mesh and method: assembly, full solve, setup, small solve, single-query total, eta_D time, and the log-log slope of each cost against N.

Derived quantities:

- Single-query win at N: setup + small solve + eta_D < full solve.
- Sweep break-even: with F frequencies on one geometry, the cost is F x full solve against setup + F x (small solve + reconstruction). F* is the smallest F for which the reduced method is cheaper (infinite if the full solve is not slower than the small solve).
- Certified pipeline: expected cost per candidate is (setup + small solve + eta_D) + p x full solve, where p is the fallback fraction. Report p*, the largest fallback fraction at which the pipeline is cheaper than always running the full solve.

Decision rules:

- A 2D speed claim is supportable only if some reduced method of dimension at most 150 wins the single query at some tested mesh, and then only at meshes at least as large as the crossover, and together with an adequate accuracy for that method from other studies. If no method wins at any tested size, the statement is: no single-query advantage in 2D at any tested size, and the speed argument rests on sweeps and on 3D scaling, which is not measured here.
- Sweep results are reported as F* for every mesh and method, without a threshold.

Limits, stated in advance: the full-wave reference is SciPy's default direct solver, and production full-wave solvers are faster per unknown, which would move every crossover to larger N; the dense random bases are a cost proxy, and a learned basis may be sparser or denser; no encoder or Whitney-lift cost is included; single thread and a single machine; 2D only. Extrapolating the measured exponents to 3D is not a measurement and is labelled as such.

**Verification study results** (`experiments/verification.py`, protocol above, CPU; 5 repeats of 300 training, 1000 calibration and 1000 test instances; mean over repeats with the standard deviation in brackets where it matters)

| Surrogate | median error | Spearman eta_D (eta_M, eta_2) | coverage, alpha = 0.1 | naive bound coverage | coverage on selected | recalibrated on selected |
|---|---|---|---|---|---|---|
| coarse_r2 (dim 88) | 0.563 | -0.33 (-0.52, 0.02) | 0.898 (0.017) | 1.000 | 0.804 (0.046) | 0.890 (0.060) |
| coarse_r3 (dim 368) | 0.309 | -0.16 (-0.44, 0.14) | 0.909 (0.010) | 1.000 | 0.832 (0.010) | 0.878 (0.038) |
| pod_16 | 0.763 | 0.92 (0.93, 0.88) | 0.897 (0.006) | 0.712 | 0.792 (0.029) | 0.878 (0.035) |
| pod_64 | 0.397 | 0.92 (0.92, 0.88) | 0.907 (0.010) | 1.000 | 0.964 (0.019) | 0.934 (0.037) |
| pod_128 | 0.283 | 0.93 (0.94, 0.89) | 0.901 (0.008) | 1.000 | 0.938 (0.034) | 0.892 (0.055) |

Verdicts by the registered rules (primary indicator eta_D, alpha = 0.1):

- coarse_r2 and coarse_r3: **DOES NOT TRACK** (Spearman -0.33 and -0.16).
- pod_16: **NEEDS RECALIBRATION** (random coverage 0.897, selected coverage 0.792, a real under-coverage).
- pod_64 and pod_128: **NEEDS RECALIBRATION** by the registered rule, but their selected-set coverage (0.964, 0.938) is above nominal, which is conservative, not an invalid bound. The registered tolerance was symmetric and counted over-coverage as a failure; that was a flaw in the rule as written. The verdict is reported as registered.

Coverage by r bin (alpha = 0.1; reported, not graded):

| Surrogate | r in [0.2, 0.3] | [0.3, 0.4] | [0.4, 0.5] | [0.5, 0.6] |
|---|---|---|---|---|
| coarse_r2 | 0.74 | 0.88 | 0.98 | 0.995 |
| coarse_r3 | 0.79 | 0.89 | 0.97 | 0.99 |
| pod_16 | 1.00 | 0.99 | 0.88 | 0.71 |
| pod_64 | 0.98 | 0.94 | 0.87 | 0.84 |
| pod_128 | 0.99 | 0.93 | 0.87 | 0.82 |

Acceptance and savings (alpha = 0.1): no surrogate is accepted at an accuracy target of 2%, 5% or 10%, except pod_128 at 10% (0.7% accepted, 0.2% false accepts). At a target of 20%, pod_128 accepts 15.6% of candidates (false-accept rate 0.9%) and pod_64 accepts 2.3% (0.3%). In the verified top-10 test every surrogate needs 10 of 10 full-wave calls. So no full-wave calls are avoided at useful accuracy with these surrogates.

Selection test: the surrogate field energy in the target region is overestimated, by a large factor, in the selected set for POD: mean relative bias +665% (pod_16), +1127% (pod_64) and +287% (pod_128), against +329%, +349% and +200% over the whole test set. For the coarse spaces the selected set is underestimated (-32%, -16%). The true figure of merit of the top 10 by surrogate rank, relative to the true top 10: 0.71 (coarse_r2), 0.99 (coarse_r3), 0.54 (pod_16), 0.97 (pod_64), 0.98 (pod_128).

Timing, median per instance (full solve 3.09 ms; eta_D 0.09 ms; the surrogate `predict` of the baselines: 3.28 ms for coarse_r2, 10.1 for coarse_r3, 1.96, 2.41 and 3.67 for POD 16, 64 and 128). These `predict` times include matrix conversion and a re-projection of K on every call, so they overstate what a leaner reduced solve costs; the scaling study times that separately.

Reading:

- The residual is informative for the data-driven surrogates and not for the compatible coarse spaces. For POD, eta_D has Spearman correlation 0.92 to 0.93 with the true error; for the coarse spaces it is negative. The cause was not investigated. A guess, untested: a coarse space cannot resolve the localised source, so its residual does not scale with its error.
- Marginal coverage holds for every surrogate (0.897 to 0.909 against 0.9, and 0.946 to 0.955 against 0.95), including those whose indicator does not track. Validity does not need tracking, but without it the bound is uninformative. The naive uncalibrated bound eta_D covers 100% for four surrogates (conservative) and only 71% for pod_16, so it is neither reliable nor sharp, and calibration is needed.
- Marginal validity hides drift across the band. Coverage varies with r by up to 30 points: the POD bounds under-cover near the upper edge of the band (0.71 to 0.84 at r in [0.5, 0.6], where the resonance is closest) and over-cover at low r, and the coarse bounds do the opposite. This is consistent with the residual-to-error ratio depending on the stability constant, which grows toward resonance. The ratio r itself is not known at test time without an eigensolve, so a per-r calibration is not directly available. Not tested.
- Under selection by surrogate score, coverage drops for the coarse spaces and pod_16 (0.79 to 0.83) and rises for pod_64 and pod_128. Calibrating on the selected portion of the calibration set restores coverage to 0.88 to 0.93, with a wide spread (standard deviation 0.04 to 0.06) because only 100 calibration points are selected.
- The optimiser's curse is large for POD: the candidates a surrogate ranks highest are mostly its spurious overestimates, and none of them is certified. That is the behaviour the certificate is meant to have.
- With these surrogates the certificate is valid but there are almost no savings, because their errors (median 0.28 to 0.76) are far above the 5 to 10% targets. Savings depend on a surrogate that is accurate to a few per cent, which is what the learned model would have to provide (H2) and which has not been shown.

Limits, as registered: the surrogates are not the learned model; selection is best-of-N from a pool, not a gradient-based design loop; the guarantee is marginal and distribution shift is untested; 2D timings do not generalise.

**Scaling study results** (`experiments/scaling.py`, protocol above; single thread, nothing else running, 20-core Intel Core Ultra 7 255HX under WSL, Python 3.12.3, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.1; median of 7 timed calls)

Full-wave reference (SciPy sparse direct solve of the interior system) and the certificate:

| Mesh | unknowns N | operator assembly | full solve | eta_D |
|---|---|---|---|---|
| refine 4 | 1,504 | 0.4 ms | 2.7 ms | 0.03 ms |
| refine 5 | 6,080 | 1.5 ms | 13.4 ms | 0.09 ms |
| refine 6 | 24,448 | 7.7 ms | 86 ms | 0.42 ms |
| refine 7 | 98,048 | 31 ms | 727 ms | 2.2 ms |
| refine 8 | 392,704 | 145 ms | 6,781 ms | 16.7 ms |

Single-query speedup, full solve time divided by (setup + small solve + eta_D), with a per-geometry projection of both K and M (a learned basis changes with the geometry). Values below 1 mean the reduced method loses:

| N | dense q = 16 | dense q = 64 | dense q = 128 | coarse_r2 (q = 88) | coarse_r3 (q = 368) |
|---|---|---|---|---|---|
| 1,504 | 2.7x | 2.1x | 0.82x | 1.7x | 0.14x |
| 6,080 | 5.8x | 2.5x | 0.88x | 2.1x | 0.18x |
| 24,448 | 7.4x | 2.8x | 1.15x | 2.2x | 0.23x |
| 98,048 | 13.4x | 3.7x | 1.6x | 2.9x | 0.46x |
| 392,704 | 20.2x | 6.7x | 3.2x | 5.6x | 1.04x |

Log-log slope of cost against N: full solve 1.42, operator assembly 1.05, eta_D 1.15; reduced setup 1.07 to 1.23; the small solve with field reconstruction 0.6 to 1.0.

Per extra frequency on the same geometry (full solve divided by the small solve with reconstruction), at N = 392,704: 1224x for q = 16, 361x for q = 64, 207x for q = 128, 310x for coarse_r2 and 68x for coarse_r3. At N = 1,504 the same ratios are 90x, 27x, 12x, 25x and 0.9x. The sweep break-even number of frequencies F* is 1 wherever the single query already wins, and at most 2 for q = 128 on the small meshes (9 for coarse_r3 at N = 6,080 and 5 at N = 24,448). If the K projection is done once for a fixed basis, setup falls by 1.3x to 2x and every method of dimension at most 128 wins at every mesh tested.

Certified pipeline: the largest fallback fraction p* at which (surrogate + eta_D + p x full solve) is still cheaper than always running the full solve, at N = 392,704: 0.95 (q = 16), 0.85 (q = 64), 0.69 (q = 128), 0.82 (coarse_r2) and 0.04 (coarse_r3). At N = 1,504: 0.64, 0.51, negative, 0.39 and negative.

Decision rule from the protocol: some reduced method of dimension at most 150 wins the single query at every mesh tested (q = 16, q = 64 and coarse_r2 from N = 1,504; q = 128 from N = 24,448). The cost condition for a 2D speed claim is therefore met. The second condition, adequate accuracy for that method from other studies, is not: the surrogates in the verification study have median errors of 0.28 to 0.76, the random dense bases here have no accuracy at all, and the learned model has not been shown. So the cost half of the speed argument holds with headroom of about 3x to 20x per query and hundreds of times per extra frequency at 393k unknowns, and the accuracy half is unestablished.

Correction to an earlier statement: the verification study's timing line, and my remark that the surrogates were not faster than the full solve in 2D, came from the baselines' `predict`, which converts matrices and re-projects K on every call (3.3 ms for coarse_r2 at N = 1,504). The leaner path timed here costs 1.6 ms for the same basis, against 2.7 ms for the full solve. The earlier remark was an artefact of the baseline implementation and is withdrawn.

Limits, as registered, and one more: the full-wave reference is SciPy's default direct solver, and a production solver (UMFPACK, PARDISO, MUMPS) would be faster per unknown, which would shrink every speedup and move the crossovers to larger N; the dense random bases are a cost proxy; the cost of obtaining the basis (POD snapshots and SVD, or an encoder pass and the Whitney lift for a learned W) is not included, and it is the main missing term; single thread, one machine, 2D only. Extrapolating the measured exponents to 3D is not a measurement. As general background (not measured here), sparse direct solves in 3D are expected to scale worse than in 2D, which would widen the gap.

**Gate 1b stage (b) protocol (fixed before running)**

Question: in the sub-resonant main family, can a single partition of unity shared by all geometries and frequencies (no encoder) give accurate reduced solves, how does it compare with POD at equal dimension, and how much could a geometry-dependent W (an encoder) add?

- Family: the main sub-resonant family (omega^2 = r x lambda_1, r uniform on [0.2, 0.6]), refine 4, tan(delta) = 0.05. Instances differ in disc position, radius and contrast, source position and direction, and frequency.
- Splits per repeat: 200 training and 300 test instances (seeds 400 + r and 500 + r), three repeats.
- Shared W: free nodal logits shared by every instance, sizes n = 4, 8, 12, 16 (dimension 10, 36, 78, 136). Adam, lr 0.05, 500 steps, mini-batches of 16 training instances, loss the mean relative M-norm Galerkin error against the fine solution. Two starts (rbf seed 0 and smooth seed 0); the start is chosen by the mean error over all 200 training instances, never by test error.
- Comparators: POD-Galerkin of the same dimension trained on the same 200 fine solutions (the same supervision); `coarse_r1`, `coarse_r2`, `coarse_r3` (dimension 20, 88, 368); and a per-instance oracle W (an upper bound, with the reference in the loss as in Gate 1), computed in repeat 0 only, for the first 20 test instances, at n = 8 and 16, best of two starts, 500 steps.
- Reported per method and size on the test set: median and mean field error, median stored-energy error, the fraction of instances with energy error below 5%, the fraction with field error below 10%, and the median field error in each r bin. For the oracle comparison, the shared W and POD are also reported on the same 20 instances.
- A method passes at a size when its dimension is at most 150, its median energy error is below 5% and at least 80% of the instances have energy error below 5%.
- Outcomes: GLOBAL W SUFFICES (the shared W passes at some affordable size: no encoder is needed, and Gate 2 reduces to showing it beats POD); ENCODER NEEDED (the shared W fails and the per-instance oracle passes at some size: there is headroom and Gate 2 is justified); NO HEADROOM (neither passes: the learned-W pursuit stops and the fallback in the decision rule applies).
- Structure against POD, reported alongside: the ratio of median field errors, shared W over POD at equal dimension. HELPS if at most 0.5 at two or more sizes, WORSE if above 2 at two or more sizes, COMPARABLE otherwise. This is a preliminary indication without an encoder; the 2x-against-POD rule for H5 is evaluated with the encoder at the next gate.
- Limits, stated in advance: the oracle W is fitted to the reference solution (an upper bound); the shared W is trained with the reference solutions (the same supervision as POD); 200 training instances; a single fine mesh; the oracle subset has 20 instances, so its pass fraction moves in steps of 5% and passing needs at least 16 of 20.

**Baseline data-efficiency study protocol (fixed before running)**

Question: how do the unconstrained neural operator (FNO) and the data-driven POD-Galerkin basis behave as the amount of training data grows, in the sub-resonant main family? This gives the baseline curves that the learned compatible basis will be compared with at Gate 4. It is descriptive: nothing here is graded pass or fail.

- Family: the main sub-resonant family, refine 4. The FNO grid is 64 x 64: the edge-to-grid-to-edge round trip has a median field error of 0.1% at 64, against 8.0% at 32 and 5.5% at 48 (measured on 30 instances), so a 32 grid would put the FNO above the 10% field tolerance by construction.
- Data: per repeat, 400 training instances (seed 400 + r) used as nested subsets of the first n, n = 25, 50, 100, 200, 400, and 300 test instances (seed 500 + r, the same test sets as Gate 1b stage (b)). Three repeats.
- FNO: width 32, 12 Fourier modes, 4 layers, batch 16, Adam with lr 1e-3 and cosine annealing, 200 epochs, float32 on the GPU, no early stopping and no validation set. The hyperparameters are fixed here and not tuned on the test set. Input channels as in `Family.grid_inputs` (epsilon, real and imaginary omega^2, source components, coordinates); targets are the real and imaginary parts of Ex and Ey; the loss is the relative L2 error on the grid.
- POD-Galerkin: requested rank 36 and 136 (the dimensions of the shared-W sizes n = 8 and 16), trained on the same instances. The rank actually obtained is limited by the data (at most twice the number of training instances) and is reported.
- Evaluation: every prediction is mapped to edge coefficients and scored with `metrics.evaluate` against the fine solution, exactly as in Gate 1b stage (b): median and mean field error, median stored-energy error, the fraction of instances with energy error below 5% and with field error below 10%, and the median field error per r bin. For the FNO, the round-trip floor, parameter count, final training loss and fit time are also reported.
- Readouts, not tests: the smallest training size at which each method's mean median field error is below 10% and its mean median energy error is below 5% (or none), and the ratio of the FNO median field error to POD-136 at each size.
- Limits, stated in advance: one FNO architecture and training budget (untuned; a tuned FNO may do better); the epochs are fixed, so larger training sets receive more updates; POD and the FNO use the reference solutions for training; the numbers are for one 2D mesh and family.

**Baseline data-efficiency study results** (`experiments/data_efficiency.py`, protocol above; FNO on the GPU, grid 64, 200 epochs; POD-Galerkin on the CPU; mean over 3 repeats of 300 test instances)

Median field error / median stored-energy error (the pair of numbers per cell), with the fraction of test instances under the 5% energy and 10% field tolerances in brackets:

| Training instances | FNO | POD, requested rank 36 | POD, requested rank 136 |
|---|---|---|---|
| 25 | 0.448 / 0.230 (0.13, 0.00) | 0.634 / 0.156 (0.23, 0.00) | 0.665 / 0.159 (0.21, 0.00), rank 50 |
| 50 | 0.320 / 0.147 (0.19, 0.00) | 0.556 / 0.162 (0.23, 0.00) | 0.378 / 0.069 (0.42, 0.00), rank 100 |
| 100 | 0.241 / 0.095 (0.28, 0.00) | 0.461 / 0.122 (0.26, 0.00) | 0.265 / 0.039 (0.57, 0.02) |
| 200 | 0.188 / 0.066 (0.38, 0.01) | 0.276 / 0.055 (0.46, 0.00) | 0.285 / 0.056 (0.48, 0.01) |
| 400 | 0.150 / 0.049 (0.51, 0.07) | 0.248 / 0.048 (0.51, 0.00) | 0.273 / 0.054 (0.47, 0.02) |

The FNO has 1.19 million parameters and takes 47 s to fit at 400 instances; its grid round-trip floor is at most 0.2%. The rank of POD is limited by the data (at most twice the number of instances), so rank 136 is only reached from 100 instances.

Readouts as registered: no method reaches a mean median field error below 10% at any size. The smallest training size with a mean median energy error below 5% is 400 for the FNO, 400 for POD-36 and 100 for POD-136. The ratio of the FNO median field error to POD-136: 0.67, 0.85, 0.91, 0.66, 0.55 at 25, 50, 100, 200, 400 instances.

Reading:

- With up to 400 training instances no baseline gets the median field error below 10%. The best is the FNO at 0.150, still falling by about 20% per doubling of the data (0.448, 0.320, 0.241, 0.188, 0.150). No extrapolation is claimed.
- The FNO has the lower field error at every size, but POD-136 has the lower energy error at 50 and 100 instances (0.069 and 0.039 against 0.147 and 0.095), and is the only method under 5% median energy error by 100 instances. The FNO reaches 0.049 only at 400.
- POD stops improving: POD-136 is flat from 100 instances (0.265, 0.285, 0.273 median field error), and POD-36 is flat at about 0.25 to 0.28 from 200. This is consistent with a solution manifold that a fixed linear basis of at most 136 dimensions cannot cover, because the family varies the disc, the source position and the source direction together. That explanation is not tested here.
- The errors do not depend on the frequency ratio: at 400 instances the FNO median field error is 0.155, 0.140, 0.148 and 0.158 in the four r bins, and POD-136 0.283, 0.267, 0.272 and 0.281. So the baseline errors in the sub-resonant family are not resonance leakage near the band edge; they come from the geometric and source variability.
- The bar for the learned model, using the registered H5 rule (a 2x error reduction against POD at the smallest training size, or the same accuracy at half the dimension): at 25 training instances POD has median field error 0.63 to 0.67, so a learned model has to reach about 0.32 or lower there. The FNO, at 0.45, does not meet that bar either.
- Limits, as registered: one untuned FNO architecture and training budget (a tuned FNO may do better), fixed epochs, and the reference solutions used for training. The 5% and 10% targets were set before this study; at these data sizes the family is harder than they assumed.

**Gate 1b stage (b) results** (`experiments/gate1b_geometry.py`, protocol above, CPU; 200 training and 300 test instances per repeat, 3 repeats; the per-instance oracle was run in repeat 0 on 20 test instances)

Median field error / median stored-energy error (mean over the repeats), with the fraction of test instances under 5% energy error in brackets:

| Dimension | Shared W | POD (same dimension) | Per-instance oracle W | coarse Whitney |
|---|---|---|---|---|
| 10 (n = 4) | 0.839 / 0.760 (0.00) | 1.008 / 0.481 (0.04) | | |
| 20 (coarse_r1) | | | | 0.835 / 0.641 (0.00) |
| 36 (n = 8) | 0.694 / 0.523 (0.00) | 0.276 / 0.055 (0.46) | 0.044 / 0.024 (0.85) | |
| 78 (n = 12) | 0.602 / 0.389 (0.00) | 0.335 / 0.077 (0.39) | | |
| 88 (coarse_r2) | | | | 0.565 / 0.305 (0.01) |
| 136 (n = 16) | 0.520 / 0.298 (0.00) | 0.285 / 0.056 (0.48) | 0.044 / 0.024 (0.85) | |
| 368 (coarse_r3) | | | | 0.309 / 0.091 (0.13) |

On the 20 oracle instances: the shared W has median field error 0.642 (n = 8) and 0.494 (n = 16) with 0% under 5% energy error; POD has 0.253 and 0.254 with 85% and 40% under 5%. The oracle has field error under 10% on all 20 instances.

Verdict by the registered rules: **ENCODER NEEDED** (the shared W fails at every size; the per-instance oracle passes at n = 8, with median energy error 0.024 and 17 of 20 instances under 5%, against the 16 required). Structure against POD, by the registered rule: **COMPARABLE**, from the ratios of median field error shared W over POD: 0.83, 2.51, 1.80 and 1.83 at n = 4, 8, 12 and 16. The rule needs a ratio above 2 at two or more sizes to call it worse, and only n = 8 exceeds 2; the shared W is nevertheless 1.8 to 2.5 times worse than POD at every size from n = 8.

Reading:

- A single geometry-independent W is far from adequate. Its error improves with size (0.84 to 0.52 median field error) but no test instance reaches 5% energy error at any size in any repeat. The compatible structure alone, without geometry dependence, does not beat a plain data-driven basis.
- There is large headroom for a W that depends on the instance. The per-instance oracle reaches median field error 0.044, about 6 times lower than POD-36 (0.276) and about 16 times lower than the shared W (0.694), at the same dimension. The oracle error is the same at n = 8 and n = 16 (0.044), so for these instances 36 dimensions are enough.
- What the oracle adapts to is the instance as a whole: the disc, the source position and direction, and the frequency are all fitted. An encoder would have to predict a W from those inputs. That it can is plausible (a localised source and the material interfaces suggest where resolution is needed) but untested.
- The oracle is an upper bound and is expensive: each instance needs 500 optimiser steps against the reference solution, far more than a full solve. It shows what is representable, not what an encoder would learn.
- The energy criterion separates the methods poorly on 20 instances (POD also reaches 85% at n = 8); the substantive gap is in the field error.
- Limits, as registered: 20 oracle instances in one repeat, so its pass fraction moves in steps of 5%; the oracle and the shared W use the reference solutions; one mesh. Gate 2 (the encoder) is justified by this verdict, and the bar it has to clear comes from the data-efficiency study above (FNO 0.45 and POD 0.63 to 0.67 median field error at 25 training instances; the registered rule asks for a 2x reduction against POD at the smallest size, or equal accuracy at half the dimension).

**Gate 2 protocol: the encoder (fixed before the test-set run)**

Question: can a network predict the partition of unity W from the instance, so that a 36-dimensional compatible reduced space is much more accurate than POD trained on the same data? This is the decisive test of H2 and the first learned-model entry for H5.

- Model (`baselines/encoder_whitney.py`): n = 8 bumps (dimension 36). The nodal logits are a shared base (initialised from the rbf start, seed 0) plus a CNN correction. The CNN reads the seven grid channels of `Family.grid_inputs` at grid 64 (standardised with the training-set mean and standard deviation), 2x average pooling, four 3 x 3 convolutions with dilation 1, 2, 4, 8 and GELU, then a zero-initialised 1 x 1 convolution to 8 channels, sampled bilinearly at the mesh nodes. The reduced space, whitening and Galerkin solve are those of the shared W. Loss: mean relative M-norm Galerkin error against the fine solution, the supervision POD and the FNO also receive. Adam, batch 8, base learning rate 0.02, CNN learning rate 2e-3, cosine annealing, no weight decay, no early stopping.
- Hyperparameter selection (validation data only, never test data): width in {12, 24} and number of steps in {500, 1000}, four configurations. Each is trained on the first 25 and the first 100 instances of a dataset drawn with seed 901 and scored on 100 instances drawn with seed 900. The configuration with the lowest mean of the two validation median field errors is used for everything below. Nothing else is tuned. (Before this protocol, one smoke run at width 24 with 60, then 1000 steps, was made on 40 validation instances to check that the code trains; its numbers are not used for selection.)
- Test protocol: the data-efficiency setup. Training sizes 25, 50, 100, 200, 400 as nested subsets of the first n instances (seed 400 + r), 300 test instances (seed 500 + r), three repeats, so the test sets and training data are those of the baseline study and its committed numbers serve as the comparators (FNO, POD-36 and POD-136, requested ranks; rank is limited to twice the training size). Scoring through `metrics.evaluate` as before. Also reported: training error, the median time of one `predict` call (including the grid inputs and the lift) against the full solve at this mesh, and the encoder's median field error as a multiple of the per-instance oracle's 0.044.
- Criterion A (the registered H2 and H5 rule, smallest size): at 25 training instances, the mean over repeats of the encoder's median field error is at most half of the smaller of the POD-36 and POD-136 values (0.634 and 0.665 in the baseline study, so at most 0.317), and the encoder is at most half of that repeat's smaller POD value in at least two of the three repeats.
- Criterion B (the same accuracy at under half the dimension): at some size in {50, 100, 200, 400}, where POD-136 has rank at least 100, the encoder's mean median field error is at most that of POD-136. The encoder has dimension 36.
- Verdict: PASS if A or B holds (STRONG PASS if both). FAIL otherwise. On FAIL, the decision rule in the ledger applies: the thesis rests on the verification layer, the stability diagnosis and the baseline study, and no further learned-W variations are tried. Inconclusive only if training diverges or produces non-finite errors in some run.
- Readouts, not graded: comparison with the FNO at each size (the unconstrained operator is the baseline for H5); the fraction of instances under 5% energy and 10% field error; the smallest size at which the median field error is below 10%.
- Multiplicity: two ways to pass (A, B) and four validation configurations; the test sets are used once, for the selected configuration.
- Limits, stated in advance: one architecture and budget; supervised with the reference solutions; dimension 36 only (not n = 16); one mesh; the comparators are the numbers of the earlier study, not rerun, and the FNO there was untuned; with 25 training instances the network is expected to overfit, so criterion A is hard by design.

**Gate 2 results** (`experiments/gate2.py`, protocol above, CPU; selection on validation seeds 900 and 901, then the test run on seeds 400 + r and 500 + r, three repeats of 300 test instances; the comparators are the committed numbers of the baseline study)

Validation selection (mean validation median field error at 25 and 100 training instances): width 12 with 500 steps 0.439, width 12 with 1000 steps 0.419, width 24 with 500 steps 0.389, width 24 with 1000 steps 0.372. Selected: width 24, 1000 steps (21,720 parameters).

Median field error / median stored-energy error of the 36-dimensional encoder, mean over the three repeats, with the baselines' median field error from the earlier study:

| Training instances | Encoder (dimension 36) | Training error | FNO | POD-36 | POD-136 | Encoder as a multiple of the oracle (0.044) |
|---|---|---|---|---|---|---|
| 25 | 0.476 / 0.176 | 0.071 | 0.448 | 0.634 | 0.665 | 10.8 |
| 50 | 0.352 / 0.109 | 0.106 | 0.320 | 0.556 | 0.378 | 8.0 |
| 100 | 0.243 / 0.078 | 0.131 | 0.241 | 0.461 | 0.265 | 5.5 |
| 200 | 0.204 / 0.065 | 0.152 | 0.188 | 0.276 | 0.285 | 4.6 |
| 400 | 0.180 / 0.058 | 0.159 | 0.150 | 0.248 | 0.273 | 4.1 |

Per repeat the median field error varies by at most 0.03 at every size (for example 0.478, 0.484, 0.467 at 25 instances; 0.180, 0.174, 0.185 at 400). The fraction of test instances under 5% energy error is 0.08 to 0.09 at 25 instances and 0.36 to 0.47 at 400; the fraction under 10% field error is 0.00 to 0.02 at every size. No size reaches a median field error below 10%.

Registered criteria:

- Criterion A (at 25 instances at most half of the smaller POD value, so at most 0.317): encoder 0.476, NOT MET (0 of 3 repeats within the bar). The ratio to POD-36 is 0.75 and to POD-136 0.72.
- Criterion B (dimension 36 at most POD-136's error at some size from 50 up): MET at 50, 100, 200 and 400 (ratios to POD-136 of 0.93, 0.92, 0.72, 0.66).
- Verdict: **PASS, through criterion B only** (not a strong pass).

Reading:

- The encoder is a real but modest improvement over the data-driven basis. At equal dimension (36) it has 25 to 47% lower median field error than POD-36 at every size. It beats POD-136, with 3.8 times the dimension, by 7 to 8% at 50 and 100 instances (a small margin, comparable to the spread over repeats) and by 28% and 34% at 200 and 400.
- Criterion B is a weak form of the registered rule, because POD-136 plateaus near 0.27 to 0.29 from 100 instances; part of what is shown is that POD stops improving. The 2x bar at the smallest size (criterion A) is not reached: with 25 instances the encoder is overfitted (training error 0.071, test error 0.476).
- Against the unconstrained operator the encoder shows no accuracy advantage. Its median field error is 1% to 20% above the FNO's at every size (ratios 1.06, 1.10, 1.01, 1.08, 1.20), and its stored-energy error at 400 instances (0.058) is above the FNO's (0.049) and POD-136's (0.054). On the stored energy the picture is mixed, not uniformly in the FNO's favour: the encoder's median energy error is lower at 25, 50 and 100 instances (0.176, 0.109, 0.078 against 0.230, 0.147, 0.095), level at 200 (0.065 against 0.066) and higher at 400, while the FNO has the larger fraction of instances under 5% energy error at every size (0.13, 0.19, 0.28, 0.38, 0.51 against 0.08, 0.11, 0.23, 0.35, 0.41). So the H5 claim, a data-efficiency advantage over the FNO, is not supported by this comparison. Whether the compatible structure buys something else (consistency at matched accuracy, a certificate that behaves) is the subject of Study 4a.
- Most of the oracle's headroom is not captured: the encoder is 4.1 to 10.8 times the per-instance oracle's median field error. At 400 instances the training error (0.159) is close to the test error (0.180), so the encoder is limited by optimisation or capacity, not by generalisation; at 25 instances it is the opposite. Whether a longer or larger model closes more of the gap is untested (each configuration was fixed by the validation selection, and no budget study was run).
- The targets of the original brief (5% energy, 10% field) are not reached by the encoder or by either baseline with up to 400 training instances.
- The encoder `predict` time printed by the script (64.7 ms against a 2.6 ms fine solve) was measured inside a multi-threaded training job while other processes were running on the machine, so it is not used as a result; Study 4b measures it on a quiet machine. The per-fit wall-clock times in the log (77 to 282 s) vary with machine load and carry no information.
- Limits, as registered: one architecture, supervised on reference solutions, dimension 36 only, one mesh, comparators not rerun, and the FNO of the baseline study was untuned.

**Study 4a protocol: consistency and certificate for the learned models (fixed before running; runs after Gate 2)**

Question: does the compatible encoder give anything the unconstrained FNO does not, beyond accuracy (which Gate 2 measures)? Two things are tested: the structural consistency of the answers at matched accuracy (H1 against an unconstrained baseline), and whether the residual certificate is valid for the FNO and for the encoder (H4; the verification study covered only POD and coarse spaces).

- Models, per repeat and training size n in {100, 400} (nested subsets of seed 400 + r, three repeats): the encoder with the configuration selected in Gate 2, the FNO of the baseline study (grid 64, 200 epochs, same settings), and POD-136 (rank limited by the data as before). The encoder and the FNO are retrained, since the earlier fits were not stored.
- Data: 1000 calibration instances (seed 600 + r) and 1000 test instances (seed 700 + r), the main sub-resonant family.
- Consistency metrics, from `metrics.evaluate` on every test instance and every model: `power_balance`, `pec_violation`, `gauss_fine` (the Gauss law against all fine-mesh interior nodal functions), `residual`, and the field error `rel_err`. Medians over the test set, mean over repeats.
- Consistency readouts and verdicts:
  - By construction: the median `power_balance` and `pec_violation` of the FNO are at least 100 times those of the encoder at both sizes. Both are properties of a Galerkin solution in a PEC-compatible space (Gate 0), not empirical findings, and the FNO may pass them trivially for the PEC part; the result is reported either way and labelled as by construction.
  - Substantive: the Gauss law at matched accuracy. On the test instances where both the encoder and the FNO have field error in [0.10, 0.30] (if at least 100 such instances at that size and repeat), the encoder's median `gauss_fine` is at most half of the FNO's at both sizes. If fewer than 100 such instances exist the comparison is reported as insufficient. Verdicts: SUBSTANTIVE ADVANTAGE (holds at both sizes), BY CONSTRUCTION ONLY (the first criterion holds, the second does not), NO ADVANTAGE (neither).
- Certificate: the verification protocol, applied to the three models at each size. Primary indicator eta_D, split conformal on eps / eta with alpha = 0.1, Spearman correlation of log eta_D with log eps, coverage on random test instances and on the selected set (the 10% with the highest surrogate figure of merit in the region R of the verification protocol), recalibration on the selected calibration points when coverage drops, acceptance and false-accept rates at tau in {0.05, 0.10, 0.20}. Decision rules and verdicts exactly as in the verification protocol (CERTIFICATE VALID, NEEDS RECALIBRATION, DOES NOT TRACK, INCONCLUSIVE), one per model and size.
- Cost, reported not graded: median `predict` time per model, and the training time.
- Limits, stated in advance: one encoder configuration and one FNO configuration, both untuned beyond Gate 2; accuracy of all models is poor (median field error 0.15 to 0.25), so acceptance at tau = 0.05 is expected to be near zero and savings are not expected; the matched-accuracy band [0.10, 0.30] is fixed here and not changed after the results; the conformal guarantee is marginal and covers only the same family; no distribution-shift test.

**Study 4a results** (`experiments/study4a.py`, protocol above; encoder width 24 and 1000 steps from the Gate 2 selection, FNO as in the baseline study on the GPU, 8 threads; three repeats of 1000 calibration and 1000 test instances; mean over repeats)

Consistency, median over the test set (relative quantities from `metrics.evaluate`):

| Model | Median field error | Power balance | PEC violation | Gauss law (all fine nodal functions) | Euclidean residual |
|---|---|---|---|---|---|
| Encoder, 100 instances | 0.244 | 4.6e-15 | 0 | 0.337 | 3.30 |
| FNO, 100 instances | 0.235 | 2.30 | 0.039 | 0.288 | 30.5 |
| POD-136, 100 instances | 0.267 | 4.3e-15 | 0 | 0.455 | 1.52 |
| Encoder, 400 instances | 0.177 | 4.8e-15 | 0 | 0.260 | 3.12 |
| FNO, 400 instances | 0.148 | 2.31 | 0.024 | 0.203 | 30.3 |
| POD-136, 400 instances | 0.274 | 4.5e-15 | 0 | 0.439 | 1.23 |

Matched accuracy (instances where the encoder and the FNO both have field error in [0.10, 0.30]; 659 to 669 of 1000 at 100 instances, 860 to 885 at 400): median Gauss-law error of the encoder against the FNO, 0.294 against 0.238 at 100 instances and 0.265 against 0.209 at 400, so the encoder is 1.23 and 1.27 times higher in the same instances, in every repeat.

Consistency verdict by the registered rule: **BY CONSTRUCTION ONLY**. The power balance and PEC criterion holds at both sizes (the FNO's median power-balance error is 2.3, against machine precision for the encoder; its PEC violation is 2 to 4%). The substantive criterion (encoder at most half the FNO's Gauss-law error at matched accuracy) fails at both sizes, in the wrong direction.

Certificate (primary indicator eta_D, alpha = 0.1), mean over repeats:

| Model | Spearman, log eta_D with log eps (per repeat) | Coverage, random | Coverage, selected set (recalibrated) | Accepted / false-accepted at tau = 0.2 | Verdict |
|---|---|---|---|---|---|
| Encoder, 100 | 0.26 (0.20, 0.24, 0.34) | 0.896 | 0.893 (0.920) | 0.000 / 0.000 | DOES NOT TRACK |
| FNO, 100 | 0.55 (0.53, 0.59, 0.54) | 0.893 | 0.977 (0.927) | 0.066 / 0.021 | DOES NOT TRACK |
| POD-136, 100 | 0.93 (0.94, 0.92, 0.92) | 0.900 | 0.937 (0.917) | 0.179 / 0.014 | NEEDS RECALIBRATION |
| Encoder, 400 | 0.34 (0.28, 0.36, 0.39) | 0.893 | 0.913 (0.950) | 0.093 / 0.024 | DOES NOT TRACK |
| FNO, 400 | 0.60 (0.60, 0.64, 0.57) | 0.895 | 0.977 (0.937) | 0.383 / 0.010 | DOES NOT TRACK |
| POD-136, 400 | 0.93 (0.94, 0.93, 0.93) | 0.899 | 0.940 (0.917) | 0.155 / 0.010 | NEEDS RECALIBRATION |

Other indicators: the exact-norm indicator eta_M gives the same picture (Spearman 0.22 and 0.33 for the encoder, 0.59 and 0.62 for the FNO, 0.93 for POD), and the Euclidean residual eta_2 is worse (0.02 and 0.08 for the encoder, 0.38 and 0.48 for the FNO). Nothing is accepted at tau = 0.05 or 0.10 except 1 to 2% for the FNO and POD at 0.10, so essentially no full-wave call is avoided at the target of 5% by any model (at most 0.1%). The verified top-10 needs 10 full-wave calls for every model (no savings). Coverage by r band (0.2 to 0.3, 0.3 to 0.4, 0.4 to 0.5, 0.5 to 0.6): encoder 0.80, 0.87, 0.94, 0.98 at 100 instances and 0.77, 0.88, 0.93, 0.99 at 400; FNO 1.00, 0.99, 0.91, 0.67 and 1.00, 0.99, 0.91, 0.68; POD-136 0.99, 0.94, 0.87, 0.81 and 0.99, 0.93, 0.86, 0.81.

Reading:

- The structural advantage over the FNO exists only by construction, and it is not specific to the learned basis. Any Galerkin solution in a PEC-compatible space satisfies the power balance and has zero PEC violation, and POD-136 (a plain Galerkin basis from snapshots) does so to the same precision. The learned compatible structure adds nothing here beyond what a Galerkin projection already gives.
- The one non-trivial consistency test goes against the encoder. At matched accuracy its Gauss-law error is 23 to 27% higher than the FNO's. Earlier (consistency check) it was found that the divergence error tracks the field error; this comparison shows no compatibility benefit beyond that. The compatible space makes the Gauss law exact only against its own reduced 0-forms, not against all fine-mesh nodal functions, which is what the metric tests.
- The residual certificate does not track the error of the learned models: Spearman 0.26 to 0.34 for the encoder and 0.55 to 0.60 for the FNO, against 0.93 for POD, in every repeat. Coverage is nevertheless valid on average for every model (0.893 to 0.900), because the conformal calibration enforces it; the bound is not informative where it does not track. For the encoder the conditional coverage drifts from 0.77 to 0.80 in the lowest band to 0.99 in the highest.
- The FNO with the certificate avoids the most full-wave calls: at 400 instances and an accuracy target of 0.2, 38% of instances are accepted with a 1% false-accept rate, against 15 to 16% for POD and 9% for the encoder. This is a reported result, not a graded one, and it is at a loose target. It is the opposite of the ranking the compatible-encoder story would predict.
- Cause: not tested. Guess, labelled as such: a learned W concentrates the reduced space where the training solutions have energy, so the dual-norm residual of an instance that falls outside that region is not proportional to its error; POD, whose basis is built from solutions, tracks. Whether a different residual norm or conditional calibration recovers the tracking is open.
- The predict times in the CSV are about 60 ms for both the encoder and the FNO and 4 ms for POD, measured under unknown load, so they are indicative only. The two learned pipelines having the same time suggests a shared component (building the grid inputs, which does a point location for 4096 points), but this is untested; Study 4b profiles it on a quiet machine.
- Limits, as registered: one encoder and one FNO configuration, both with median field error of 0.15 to 0.25; the matched-accuracy band was fixed beforehand; the guarantee is marginal and for the same family; no distribution shift; the acceptance results are at loose targets because no model reaches 10%.

**Study 4b protocol: cost of the encoder (fixed before running; runs after Study 4a, single thread, no other jobs)**

Question: what does one encoder pass cost, where does the time go, and does it make the learned pipeline faster than a full-wave solve at the large meshes of the scaling study? The scaling study excluded this cost.

- Step 1, profile at refine 4 (the Gate 2 model, 300 test instances' worth of calls, median of 30): time of the grid inputs (`Family.grid_inputs`), the CNN forward pass, the sampling at the nodes, the PoU and Whitney lift, the restriction and sparse conversion of K and M, the whitening and reduced solve, and the total.
- Step 2, optimisation without changing the answer: only changes that leave the output identical (relative difference at most 1e-8 against the unoptimised pass on 20 instances), for example precomputed mesh tensors, cached element lookups, no-grad and in-place operations, and the CNN in float32 as now. Report the profile before and after.
- Step 3, scaling: the encoder pass on the scaling-study geometry and source at refine 4 to 7 (refine 8 if memory allows: the lift matrix is E x 36 doubles, about 110 MB), with the Gate 2 CNN weights and the shared base logits transferred to the finer mesh by nearest-node lookup. Accuracy at finer meshes is NOT assessed here (a transferred model has not been validated; that would be a separate gate), so this is a cost measurement only. Per mesh: encoder pass (basis prediction), reduced setup, small solve, eta_D, the full solve (as in the scaling study), the single-query speedup including the encoder pass, and the sweep break-even F*, with the encoder pass counted once as setup.
- Decision rule: the cost claim for the learned pipeline is supportable only at the meshes where the single-query speedup including the encoder pass exceeds 1, and the crossover mesh is reported. If the encoder pass never wins at any tested mesh the statement is: no single-query advantage for the learned pipeline at any tested size, and the cost argument rests on sweeps (F*) alone.
- Clarification added before the run (no threshold or rule changed): wall-clock times in this study are only valid on a quiet machine. The Gate 2 timing of one encoder pass (64.7 ms against 2.6 ms for a full solve) was taken inside a multi-threaded training job while other processes were running, so it is not used as a result. The script records the load average (`/proc/loadavg`, one-minute value) before every timed block, repeats a block once if the load exceeds 1.5 for a single-thread job, and flags rows whose load was still above 1.5. The study is run only after the other training jobs have finished and the owner has closed other heavy processes; load that WSL cannot see (on the Windows side) is a stated limit.
- Limits, stated in advance: SciPy's direct solver is slower than production solvers, so every speedup is optimistic against them; the transferred basis is only a cost proxy; single thread, one machine, 2D only; contention from processes outside WSL cannot be measured.

**Study 4b results** (`experiments/study4b.py`, protocol above; the Gate 2 encoder trained with 100 instances; one torch thread and the load average at most 0.99 for every timed block, nothing flagged; two runs, see below)

Profile at refine 4, median of 30 instances (milliseconds), second run, with the first run in brackets: original `predict` 66.3 (61.8), of which building the grid inputs (`Family.grid_inputs`) 63.2 (57.0). The cached path: inputs 0.13, CNN and sampling at the nodes 1.21, partition of unity and lift 1.10, restriction and conversion of M 0.68, whitening and reduced solve 1.25, total 3.69 (3.46). The optimised path gives the identical output (maximum relative difference 0 over 20 instances).

Cost at larger meshes (the Gate 2 weights and base logits transferred by nearest node; accuracy not assessed), second run; full solve and encoder pass in milliseconds, speedup of the single query including the residual eta_D:

| Mesh | Unknowns | Full solve | Encoder pass (basis part) | Speedup, second run (first run) | Per extra frequency | Full solve per extra frequency |
|---|---|---|---|---|---|---|
| refine 4 | 1,504 | 2.2 | 3.2 (1.7) | 0.69 (0.68) | 0.04 | 55 times |
| refine 5 | 6,080 | 10.6 | 12.1 (5.6) | 0.88 (1.04) | 0.09 | 118 times |
| refine 6 | 24,448 | 92.8 | 47.6 (17.8) | 1.93 (2.06) | 0.39 | 238 times |
| refine 7 | 98,048 | 818 | 264.5 (94.5) | 3.06 (3.04) | 2.93 | 279 times |
| refine 8 | 392,704 | 6,665 | 1,421 (521) | 4.64 (4.72) | 12.4 | 539 times |

The sweep break-even F* is 1 from refine 6 and 2 at refine 4 and 5. The residual eta_D costs 0.03 to 16 ms.

Two runs: the first run timed the per-frequency solve with a complex copy of the whole basis on every call (the 393k by 36 basis is 226 MB), which inflated that column (98.6 ms at 393k, against 12.4 ms in the second run). The script was corrected to the leaner path of the scaling study and rerun with identical settings; the single-query numbers of the two runs agree within a few per cent apart from refine 5, which is break-even (0.88 and 1.04). The second run is the one used.

Decision rule from the protocol: the single query wins at refine 6 and above (24,448 unknowns and more) in the second run; refine 5 is break-even within the run-to-run difference. The cost claim for the learned pipeline is therefore supportable from about 24k unknowns, with a speedup of 1.9 at 24k, 3.1 at 98k and 4.6 at 393k, and it is supportable for extra frequencies from the smallest mesh.

Reading:

- The earlier statement that one encoder `predict` was about 25 times slower than a full solve is withdrawn. 57 to 63 ms of the 62 to 66 ms was the construction of the grid inputs (a point location for 4,096 grid points), which has nothing to do with the learned basis. The FNO and the encoder both showed about 60 ms in Study 4a for the same reason. With the lookup cached the pass is 3.2 ms at refine 4.
- At 393k unknowns the basis prediction (CNN, partition of unity, lift) is 37% of the pass (0.52 of 1.42 s); the rest is the per-geometry reduced setup (projecting K and M onto the 36-dimensional basis, the whitening).
- Per extra frequency the reduced solve is 55 times cheaper than a full solve at 1.5k unknowns and 539 times at 393k.
- Not established: accuracy of the transferred model on the finer meshes (Study 5a); the cost of the FNO next to the encoder (Study 5b). The speedups are against SciPy's default direct solver, which a production solver would shrink; the geometry is one disc and one source; single thread, one machine, 2D only; loads on the Windows side that WSL cannot see are not excluded, although the recorded load average stayed at or below 0.99.

**Study 5a protocol: accuracy of the encoder and the FNO on finer meshes (fixed before running; runs after Study 5b)**

Question: the encoder's cost win (Study 4b) is at meshes of 24k unknowns and more, but the model was trained at 1.5k unknowns and its accuracy on finer meshes is unmeasured. Does the accuracy survive a change of mesh without retraining? A speed claim at scale needs this.

- Models, trained at refine 4 (1,504 unknowns) as in Gate 2 and Study 4a: the encoder (width 24, 1000 steps, 8 bumps) and the FNO (grid 64, 200 epochs, the baseline settings), each with 100 and 400 training instances (nested subsets of seed 400 + r), three repeats. Not retrained on finer meshes.
- Transfer: the encoder's CNN reads the grid inputs of the finer mesh, and the shared base logits are transferred to the finer mesh's nodes by nearest node (as in Study 4b); the FNO reads the same grid inputs and its grid output is mapped to edges with the finer mesh's own edge map. POD is not transferable (its basis is a set of refine-4 edge vectors) and is not included. Non-learned references that are defined on any mesh: `coarse_r2` and `coarse_r3` (dimension 88 and 368).
- Test data: the main sub-resonant family at refine 4, 5, 6 and 7 (1,504, 6,080, 24,448 and 98,048 unknowns). 200 instances per mesh, seed 500, the same draw of geometry, source and r at every mesh (lambda_1 and the reference solution are recomputed on each mesh). Refine 8 is not evaluated, because the full-wave reference costs about 7 s per instance. The error of every model is measured against the full-wave solution on the same mesh.
- Metrics: median field error and median stored-energy error per mesh, model and size (mean over repeats), through `metrics.evaluate`.
- Decision rules, for the encoder at each training size separately:
  - TRANSFER HOLDS at a mesh if the mean median field error at that mesh is at most 1.25 times its refine-4 value.
  - Overall: TRANSFER HOLDS TO REFINE r if it holds at every tested mesh up to r. The speed claim of Study 4b is supported with accuracy at the meshes where this holds and the encoder is faster than the full solve (24k unknowns and above, so refine 6 and 7 need to hold).
  - If the encoder's error exceeds 1.25 times its refine-4 value at refine 6, the statement is: the transferred encoder is not accurate on large meshes, and the speed claim at scale applies only to a model trained there (not tested).
  - The same ratio is reported for the FNO, without a verdict, as the comparison.
- Limits, stated in advance: nearest-node transfer of the base logits is one choice (interpolation is not tried); finer meshes resolve the disc and source better, so the reference solution itself changes with the mesh and the error is measured against the same-mesh solution, not the continuum; 200 instances per mesh; the FNO is untuned; refine 8 is not tested; one family.

**Study 5a results** (`experiments/study5a.py`, protocol above; encoder width 24 and 1000 steps, FNO as in the baseline study, both trained at refine 4 with 100 and 400 instances; three repeats; 200 test instances per mesh with full-wave references on each mesh; no retraining)

Median field error, mean over the three repeats, and the ratio to the refine-4 value in brackets:

| Model | refine 4 (1,504) | refine 5 (6,080) | refine 6 (24,448) | refine 7 (98,048) |
|---|---|---|---|---|
| Encoder, 100 instances | 0.242 | 0.691 (2.86) | 0.822 (3.40) | 0.896 (3.70) |
| Encoder, 400 instances | 0.180 | 0.638 (3.55) | 0.781 (4.34) | 0.870 (4.84) |
| FNO, 100 instances | 0.238 | 0.245 (1.03) | 0.251 (1.05) | 0.250 (1.05) |
| FNO, 400 instances | 0.144 | 0.166 (1.15) | 0.173 (1.20) | 0.170 (1.18) |
| `coarse_r2` (non-learned, 88 dimensions) | 0.565 | 0.581 | 0.586 | 0.588 |
| `coarse_r3` (non-learned, 368 dimensions) | 0.306 | 0.346 | 0.358 | 0.361 |

The three repeats agree closely: for the encoder at 100 instances the median field error at refine 5 is 0.739, 0.663, 0.672, and at refine 7 0.926, 0.881, 0.882; at 400 instances 0.643, 0.614, 0.658 and 0.866, 0.861, 0.884. (The refine-4 value differs slightly from the Gate 2 and Study 4a numbers because of seeds and a different test set.)

Verdict by the registered rule: the encoder does NOT hold at any finer mesh, at either training size (ratios 2.86 to 4.84 against the bar of 1.25), so the transfer holds to refine 4 only, and the accuracy does not hold where the cost win of Study 4b lies (refine 6 and 7). Statement from the protocol: **the transferred encoder is not accurate on large meshes; the speed claim at scale applies only to a model trained there, which is not tested.** The FNO, reported without a verdict, stays within 1.03 to 1.20 times its refine-4 error.

Reading:

- The encoder's accuracy collapses on the first finer mesh (refine 5, a factor of two in resolution) and keeps degrading, reaching a median field error of 0.87 to 0.90 at 98k unknowns, which is almost no accuracy. More training data does not help (400 instances: worse ratios).
- The FNO transfers: 0.17 to 0.25 median field error out to 98k unknowns. Together with Study 5b (6 ms per query at 98k, 26 ms at 393k) the FNO is both accurate and fast at scale, in this family.
- Even the non-learned `coarse_r3` space (0.36 at refine 7) is more accurate than the transferred encoder (0.87 to 0.90). The non-learned spaces change by 3 to 18% across meshes, which is the part of the ratio caused by the reference itself changing with the mesh.
- The cost result of Study 4b (4.6 times faster than a full solve at 393k) was measured for these transferred weights and cannot be attached to an accurate model.
- Cause: not established. The transfer carried the per-node base logits by nearest node (piecewise constant on the coarse cells), and one suspect is that this makes the partition of unity and its gradients unsuitable on a finer mesh; others (the per-node base itself, the lift on a finer mesh, the CNN's grid inputs) are not separated by this study. Study 5c tests the first.
- Limits, as registered: nearest-node transfer is one choice; the error is measured against the same-mesh solution, not the continuum; refine 8 not tested; 200 instances per mesh; the FNO is untuned.

**Study 5b protocol: cost of the FNO next to the encoder (fixed before running; single thread for the CPU parts, quiet machine)**

Question: how does the FNO's cost for one prediction compare with the encoder's and the full solve, at the meshes of the scaling study? Without this the cost advantage of the encoder cannot be put against the simplest alternative.

- Setting: as Study 4b step 3 (the scaling-study geometry and source, omega^2 = 0.4 lambda_1, refine 4 to 8, one warm-up and the median of 7 timed calls, load check before each block with the same limit of 1.5, repeated once and flagged).
- The FNO is the baseline architecture (width 32, 12 modes, 4 layers, grid 64). Timing does not depend on the weights, so a model after one training epoch on a few instances is used; the number of parameters is reported.
- Timed per mesh: (i) the grid inputs, using the cached element lookup as in Study 4b; (ii) the host-to-device copy, the forward pass and the copy back (on the GPU, with `torch.cuda.synchronize`, and also on the CPU with one thread); (iii) the mapping from the grid to edge coefficients (the sparse edge map of the mesh); (iv) the total of i to iii; (v) the residual eta_D of the FNO's answer, so that the figures are comparable with the encoder's single-query total.
- Per extra frequency: the FNO takes omega^2 as an input channel, so each extra frequency is a new forward pass (inputs update, forward, map); the encoder's per-frequency cost is the small reduced solve of Study 4b. Both are reported.
- Readouts, not tests: which of the three (full solve, encoder pass, FNO) is cheapest per single query and per extra frequency at each mesh, and the ratio of the encoder's total to the FNO's. Prediction recorded before running (a guess, not a hypothesis test): the FNO is cheaper than the encoder per single query at the large meshes, because its network cost does not grow with the mesh and only the edge map does.
- Limits, stated in advance: the GPU is the laptop's (RTX 5050) and shares the machine with other work; the FNO's accuracy is not part of this study; SciPy's direct solver is slower than production solvers; single machine, 2D only.

**Study 5b results** (`experiments/study5b.py`, protocol above; the FNO baseline architecture with 1,188,868 parameters, one training epoch, so the numbers are cost only; CPU parts single thread, the forward pass on the RTX 5050; load average at most 0.86, nothing flagged; the encoder numbers are those of the second Study 4b run, a different timing session)

Single query, milliseconds, including the residual eta_D (0.03 to 16 ms, the same for every method):

| Mesh | Unknowns | Full solve | Encoder | FNO (GPU) | FNO parts: inputs, forward, map to edges | Encoder over FNO |
|---|---|---|---|---|---|---|
| refine 4 | 1,504 | 2.9 | 3.2 | 1.7 | 0.05, 1.59, 0.02 | 1.9 |
| refine 5 | 6,080 | 15.2 | 12.1 | 2.4 | 0.04, 2.24, 0.05 | 5.0 |
| refine 6 | 24,448 | 88.5 | 48.0 | 2.2 | 0.05, 1.45, 0.27 | 21.9 |
| refine 7 | 98,048 | 758 | 267 | 6.1 | 0.05, 1.91, 1.56 | 44.1 |
| refine 8 | 392,704 | 6,623 | 1,437 | 26.5 | 0.05, 1.82, 8.27 | 54.2 |

Per extra frequency (milliseconds): encoder 0.04, 0.09, 0.39, 2.93, 12.4 at refine 4 to 8; FNO (one forward pass and map per frequency) 1.66, 2.33, 1.77, 3.52, 10.1. The FNO's forward pass on one CPU thread takes about 5 to 6 ms at every mesh.

Registered readouts: the cheapest per single query is the FNO at every mesh. The prediction recorded in advance (the FNO is cheaper than the encoder per single query at the large meshes) is consistent with the result.

Reading:

- The encoder's cost advantage is over the full solve only, not over the unconstrained alternative. Per single query at 393k unknowns the FNO is 250 times cheaper than a full solve and 54 times cheaper than the encoder pass. The network cost of the FNO is flat (about 1.5 to 2.2 ms on the GPU, 5 to 6 ms on one CPU thread), and only the edge map grows with the mesh (8 ms at 393k).
- At the largest mesh the certificate is the largest part of the FNO pipeline: the residual eta_D costs 16.5 ms against 10 ms for the prediction itself.
- Per extra frequency the encoder is cheaper on the small meshes (0.04 against 1.7 ms at 1.5k unknowns, where the GPU launch overhead sets the FNO's floor) and about equal at 98k and 393k (2.9 against 3.5 ms and 12.4 against 10.1 ms). So the encoder has no per-frequency advantage over the FNO at scale in this measurement either.
- Together with Gate 2 (no accuracy advantage) and Study 4a (no consistency or certificate advantage), this removes the cost argument against the FNO as well. What remains for the encoder is the comparison with the full solve and with POD.
- Limits, as registered, and two more: the FNO's accuracy is not part of this study; the encoder and FNO numbers come from different timing sessions (second Study 4b run and this run) and the full-solve time differs by up to 20% between sessions (for example 2.2 and 2.9 ms at refine 4), so ratios are accurate to about that level; a GPU shared with other work could change the FNO numbers; SciPy's direct solver; one geometry; 2D.

**Study 5c protocol: smooth transfer of the encoder's base logits (variant A1; fixed before running; runs after Study 5a)**

Question: Study 5a found that the encoder trained at refine 4 loses its accuracy on finer meshes. The transfer carried the shared per-node base logits to the finer mesh by nearest node, which makes them piecewise constant on the coarse cells. Is that choice the cause? This is a single change to the transfer, with the same trained models.

- Models: the Gate 2 encoder (width 24, 1000 steps), trained at refine 4 with 100 and 400 instances, three repeats, with the training seeds and model seeds of Study 5a (400 + r and r). They are retrained, since weights are not stored; the encoders of this study are saved to `results/models/` with their configuration.
- Variants on the same trained models: N, nearest node (the transfer of Study 5a); L, linear interpolation of the base logits (P1 interpolation from the refine-4 mesh onto the nodes of the finer mesh). The CNN correction is unchanged.
- Test data: as Study 5a, 200 instances at refine 4, 5, 6 and 7 (seed 500), full-wave references on each mesh, the median field error and median stored-energy error per variant, size and mesh, mean over repeats.
- Sanity check: the N variant should reproduce the Study 5a encoder numbers within the spread between repeats; if not, the difference is reported before anything else is read.
- Decision rules, per training size (as Study 5a): the variant HOLDS at a mesh if its mean median field error is at most 1.25 times its refine-4 value; HOLDS TO REFINE r if it holds at every finer mesh up to r. Outcomes: A1 REPAIRS THE TRANSFER (L holds at refine 6 and 7 for both sizes); PARTIAL (L does not hold but its error at refine 6 is at most half of N's for both sizes); NO EFFECT (otherwise).
- Diagnostic of the mechanism, reported and not graded: for one test instance per mesh (repeat 0, 100 training instances), the mean absolute edge difference of the partition of unity (the mean over interior edges and bumps of |G W|) under N and under L at refine 5, 6 and 7; if the nearest-node transfer causes the failure, N should be much rougher than L.
- Limits, stated in advance: only the base-logit transfer is changed (the CNN, the training and the data are untouched); linear interpolation is one smooth choice among several; the encoders are retrained, so the N numbers differ from Study 5a by the seed-level spread; if L fails too, the cause is not the piecewise-constant transfer, and the remaining candidates (mesh dependence of the per-node base, of the lift or of the CNN's grid inputs) are not distinguished by this study.

**Study 5c results** (`experiments/study5c.py`, protocol above; two of the three registered repeats; 200 test instances per mesh; the same encoders under both transfers)

Deviation from the protocol: the run was stopped by the owner's decision during the third repeat, after the first two repeats were complete, because the machine had become much slower (one-minute load average above 20, the repeat-1 refine-7 step taking 750 s instead of about 330 s) and the outcome was not going to change. The final summary and the roughness diagnostic, which the script writes only at the end, were therefore not produced; the numbers below are from the per-repeat results written after each repeat (`study5c_partial_20261004_052347.csv`).

Median field error, mean over repeats 0 and 1, and the ratio to the refine-4 value in brackets:

| Training instances | Transfer | refine 4 | refine 5 | refine 6 | refine 7 |
|---|---|---|---|---|---|
| 100 | nearest node | 0.244 | 0.701 (2.87) | 0.831 (3.40) | 0.904 (3.70) |
| 100 | linear | 0.244 | 0.591 (2.42) | 0.770 (3.15) | 0.882 (3.61) |
| 400 | nearest node | 0.177 | 0.628 (3.54) | 0.773 (4.35) | 0.863 (4.87) |
| 400 | linear | 0.177 | 0.470 (2.65) | 0.617 (3.48) | 0.728 (4.10) |

Linear over nearest at refine 5, 6, 7: 0.84, 0.93, 0.98 at 100 instances; 0.75, 0.80, 0.84 at 400. Sanity check: the nearest-node numbers agree with the Study 5a encoder (0.691, 0.822, 0.896 at 100 and 0.638, 0.781, 0.870 at 400, three repeats).

Outcome by the registered rules: **NO EFFECT** (the linear variant does not hold at refine 6 or 7, with ratios of 3.1 to 4.1 against the bar of 1.25, and its error at refine 6 is 0.93 and 0.80 of the nearest-node error, not at most half).

Reading:

- Smooth interpolation of the base logits helps somewhat (up to 25% lower error at 400 instances, essentially nothing at 100 instances at the finest meshes) but does not repair the transfer. The piecewise-constant nearest-node transfer is therefore not the main cause of the failure of Study 5a.
- The remaining candidates, not separated here: the per-node base logits as such (any function indexed by nodes of the training mesh), the lift on a finer mesh, and the CNN's grid inputs. Study 6a removes the per-node component altogether.
- Limits: two repeats instead of three; the roughness diagnostic was not obtained.

**Study 6a protocol: a mesh-free encoder (variant A2; fixed before running; runs after Study 5c)**

Question: Study 5a found that the encoder does not transfer to finer meshes, and Study 5c (see its results) tested one cause, the nearest-node transfer of the per-node base logits. Does the encoder transfer when nothing in it is indexed by a mesh node?

- Model: 8 bumps (dimension 36) as in Gate 2, with the logits of bump k at a point x given by an analytic RBF, -|x - c_k|^2 / (2 s_k^2), plus the CNN correction (width 24) sampled at the points. The 8 centres c_k and 8 log-widths log s_k are trained parameters, initialised from the Lloyd rbf start on the refine-4 interior nodes (seed 0, the same start as the Gate 2 base). The model has no per-node parameters, so it can be evaluated on any mesh by passing that mesh's node coordinates. Training as in Gate 2 (1000 steps, batch 8, Adam, learning rate 0.02 for the centres and widths and 2e-3 for the CNN, cosine annealing).
- Data and tests as Study 5a and 5c: refine-4 training with 100 and 400 instances (seeds 400 + r, model seed r), three repeats; 200 test instances at refine 4, 5, 6 and 7 (seed 500) with full-wave references on each mesh.
- Readouts: median field and stored-energy error per size and mesh (mean over repeats), the ratio to the refine-4 value, and the refine-4 parity with the per-node encoder (parity if the refine-4 median field error is at most 1.15 times the per-node encoder's 0.242 at 100 instances and 0.180 at 400, the Study 5a values).
- Outcomes: A2 REPAIRS THE TRANSFER if the median field error at refine 6 and 7 is at most 1.25 times its refine-4 value at both sizes; IMPROVES if not, but the error at refine 6 is at most half of Study 5a's encoder (0.822 at 100 and 0.781 at 400, so at most 0.41 and 0.39); NO EFFECT otherwise. Parity is reported separately: a repaired transfer that costs accuracy at refine 4 is reported as such.
- Limits, stated in advance: an analytic RBF base has less capacity than free per-node logits (the oracle's free logits reach 0.044), so refine-4 accuracy may drop; training is at refine 4 only; if A2 fails, the remaining suspects (the lift on a finer mesh, the CNN's grid inputs, the Cholesky whitening) are not separated by this study.

**Study 6a results** (`experiments/study6a.py`, protocol above; mesh-free encoder, width 24, 1000 steps, 17,384 parameters, trained at refine 4 with 100 and 400 instances; 200 test instances per mesh)

Deviation from the protocol: the run was stopped by the owner's decision during repeat 2, after its refine-6 evaluation and before its refine-7 evaluation, so the final summary was not printed and refine 7 has two repeats instead of three. The numbers below are computed from the per-repeat lines of the log (`results/study6a_run.log`); the per-repeat CSV files hold repeats 0 and 1.

Median field error per repeat (refine 4, 5, 6, 7), then the mean:

| Training instances | Repeat | refine 4 | refine 5 | refine 6 | refine 7 |
|---|---|---|---|---|---|
| 100 | 0 | 0.755 | 0.822 | 0.900 | 1.129 |
| 100 | 1 | 0.724 | 0.775 | 0.828 | 0.875 |
| 100 | 2 | 0.347 | 0.402 | 0.460 | not run |
| 100 | mean (ratio to refine 4) | 0.609 | 0.666 (1.09) | 0.729 (1.20) | 1.002 (1.35, two repeats) |
| 400 | 0 | 0.542 | 0.598 | 0.639 | 0.693 |
| 400 | 1 | 0.712 | 0.791 | 0.817 | 0.923 |
| 400 | 2 | 0.362 | 0.416 | 0.476 | not run |
| 400 | mean (ratio to refine 4) | 0.539 | 0.602 (1.12) | 0.644 (1.19) | 0.808 (1.29, two repeats) |

Training error of the fits (mean of the 100 and 400 instance fits): repeat 0 0.66 and 0.52, repeat 1 0.71 and 0.68, repeat 2 0.27 and 0.35. For comparison the per-node encoder has training error 0.13 and 0.16 and test error 0.24 and 0.18 at refine 4 (Study 5a).

Outcome by the registered rules: **NO EFFECT**, with the refine-4 parity check failing (a refine-4 median field error of 0.61 and 0.54 against 0.242 and 0.180, so LOSES ACCURACY AT REFINE 4). The outcome categories rest on the ratio at refine 6 and 7 and on an absolute improvement over Study 5a's encoder; the first is nearly met (ratios 1.20 and 1.19 at refine 6, 1.35 and 1.29 at refine 7, against the bar of 1.25) and the second is not (0.73 and 0.64 at refine 6 against at most 0.41 and 0.39).

Reading:

- Removing every per-node parameter largely removes the mesh dependence: the error rises by a factor of 1.1 to 1.35 from refine 4 to refine 7, against 2.9 to 4.8 for the per-node encoder. So the per-node base logits were the main cause of the failure in Study 5a, and the nearest-node versus linear choice (Study 5c) was a smaller effect.
- The price is accuracy and stability of training. The Gaussian base has 24 parameters in place of about 4,000, and the fits depend strongly on the seed: refine-4 median errors of 0.34 to 0.36 in one repeat and 0.54 to 0.76 in the other two, with training errors of 0.27 to 0.71. The spread between repeats is larger than any difference between the two training sizes. This points to an optimisation difficulty (the fit lands in different regions from different initial CNN weights) more than to a capacity limit; this is a reading, not a tested explanation.
- Even the best repeat (0.35 at refine 4, about 0.48 at refine 6) is worse than the per-node encoder (0.24 at refine 4) and than the FNO (0.17 to 0.25 on all meshes).
- Limits, as registered, and one more: two or three repeats with a large spread between them, so the means are rough; refine 7 has two repeats.

**Study 6b protocol: the FNO added to a compatible Galerkin space (variant C1; fixed before running; runs after Study 6a)**

Question: the FNO is the most accurate and cheapest surrogate here, and it transfers, but its answers do not satisfy the equations (power balance error 2.3, residual about 30). The non-learned Whitney spaces are mesh-independent and satisfy the Galerkin identities. Does adding the FNO's field to a Whitney space and solving the Galerkin system keep the FNO's accuracy, restore consistency, give a certificate that tracks, and transfer to finer meshes?

- Method (C1): the reduced space is the span of a coarse Whitney basis Q0 (`coarse_r2`, 88 dimensions, or `coarse_r3`, 368) and the real and imaginary parts of the FNO's predicted field, M-orthonormalised with the instance's mass matrix (Cholesky whitening with the jitter of the encoder), then the reduced Galerkin solve. Two variants: `c1_r2` and `c1_r3`. The FNO is the baseline FNO (grid 64, 200 epochs).
- Stage 1, refine 4, as Study 4a: FNO trained with 100 and 400 instances (seeds 400 + r, three repeats), 1000 calibration instances (seed 600 + r) and 1000 test instances (seed 700 + r). Methods: the FNO, `coarse_r2`, `coarse_r3`, `c1_r2`, `c1_r3`. Metrics: median field error, power balance, PEC violation, Gauss-law error and Euclidean residual from `metrics.evaluate`; the certificate with the verification rules (Spearman of log eta_D against log error, coverage on random and selected sets, acceptance and false-accept rates, verdicts).
- Stage 2, transfer: the same FNOs and the C1 spaces rebuilt on each mesh (the coarse spaces are defined on any mesh), tested at refine 4, 5, 6 and 7 with 200 instances (seed 500, as Study 5a); median field error and its ratio to the refine-4 value.
- Criteria, per variant and training size, no pass or fail on any other quantity:
  - Accuracy: RETAINED if the median field error at refine 4 is at most 1.10 times the FNO's, IMPROVED if at most 0.90 times, DEGRADED otherwise.
  - Consistency: RESTORED if the median power-balance error is at most 1e-10 and the median residual is at most one third of the FNO's.
  - Certificate: the verification verdict (CERTIFICATE VALID, NEEDS RECALIBRATION, DOES NOT TRACK, INCONCLUSIVE); tracking requires a Spearman correlation of at least 0.9.
  - Transfer: HOLDS if the median field error at refine 6 and 7 is at most 1.25 times its refine-4 value.
  - Overall: C1 WORKS if, for some variant at both sizes, accuracy is retained or improved, consistency is restored, the certificate tracks, and the transfer holds. PARTIAL if accuracy is retained or improved and consistency is restored but the certificate or the transfer fails. FAILS if accuracy is degraded.
- Stage 3 (cost, single thread, quiet machine, as Studies 4b and 5b): run only if the overall outcome is WORKS or PARTIAL, for the better variant, with its own short protocol written before it runs.
- Limits, stated in advance: the FNO is untuned; a Galerkin solve in a space that contains an arbitrary field is not guaranteed to be quasi-optimal for an indefinite problem, so accuracy could fall below the FNO's; the real and imaginary parts are two separate vectors; the coarse space is not learned (no learned basis is used, so this does not test the encoder idea); one family, 2D; refine 8 is not tested.

**Study 6b results** (`experiments/study6b.py`, protocol above; the FNO of the baseline study with 100 and 400 training instances, added to `coarse_r2` (88 dimensions) and `coarse_r3` (368 dimensions); three repeats; stage 1 with 1000 calibration and 1000 test instances at refine 4, stage 2 with 200 instances per mesh at refine 4 to 7; stage 3, the cost study, not run because the outcome is FAILS)

Stage 1, refine 4, medians over the test set (mean over repeats):

| Model | Median field error | Power balance | PEC violation | Gauss law | Euclidean residual | Spearman (log eta_D, log error) | Certificate verdict | Accepted at tau = 0.2 |
|---|---|---|---|---|---|---|---|---|
| FNO, 100 | 0.235 | 2.30 | 0.039 | 0.288 | 30.5 | 0.55 | DOES NOT TRACK | 0.066 |
| C1 on r2, 100 | 0.685 | 6.4e-15 | 0 | 0.658 | 6.82 | 0.87 | NEEDS RECALIBRATION | 0.000 |
| C1 on r3, 100 | 0.332 | 1.0e-14 | 0 | 0.503 | 2.25 | 0.54 | DOES NOT TRACK | 0.003 |
| FNO, 400 | 0.148 | 2.31 | 0.024 | 0.203 | 30.3 | 0.60 | DOES NOT TRACK | 0.383 |
| C1 on r2, 400 | 0.728 | 7.5e-15 | 0 | 0.676 | 8.56 | 0.92 | NEEDS RECALIBRATION | 0.000 |
| C1 on r3, 400 | 0.340 | 1.0e-14 | 0 | 0.511 | 2.63 | 0.56 | DOES NOT TRACK | 0.001 |
| `coarse_r2` alone | 0.563 | 4.2e-15 | 0 | 0.592 | 1.57 | -0.35 | DOES NOT TRACK | 0.000 |
| `coarse_r3` alone | 0.306 | 8.6e-15 | 0 | 0.477 | 0.97 | -0.19 | DOES NOT TRACK | 0.000 |

Stage 2, median field error by mesh (mean over repeats; the ratio to the refine-4 value is between 0.84 and 1.09 for every C1 variant):

| Model | refine 4 | refine 5 | refine 6 | refine 7 |
|---|---|---|---|---|
| FNO, 100 | 0.238 | 0.245 | 0.251 | 0.250 |
| FNO, 400 | 0.144 | 0.166 | 0.173 | 0.170 |
| C1 on r2, 100 | 0.697 | 0.650 | 0.638 | 0.620 |
| C1 on r2, 400 | 0.744 | 0.628 | 0.646 | 0.625 |
| C1 on r3, 100 | 0.338 | 0.364 | 0.370 | 0.369 |
| C1 on r3, 400 | 0.347 | 0.356 | 0.372 | 0.369 |
| `coarse_r2` | 0.565 | 0.581 | 0.586 | 0.588 |
| `coarse_r3` | 0.306 | 0.346 | 0.358 | 0.361 |

Criteria by variant (as registered): accuracy DEGRADED for all four (C1 is 2.91 and 4.92 times the FNO's error on r2 with 100 and 400 instances, 1.41 and 2.29 times on r3); consistency RESTORED for all four (power balance about 1e-14, residual 2.3 to 8.6 against 30); certificate NEEDS RECALIBRATION for the two r2 variants (Spearman 0.87 and 0.92, random coverage 0.896 and 0.890, selected coverage 0.933) and DOES NOT TRACK for the two r3 variants (0.54 and 0.56); transfer HOLDS for all four. Overall: **FAILS** (accuracy degraded).

Reading:

- Adding the FNO's field to a Whitney space does not carry its accuracy into the Galerkin solution: the result is 1.4 to 4.9 times the FNO's error. It is not merely a collapse onto the coarse space. At refine 4 the enriched space is less accurate than the coarse space it starts from (0.685 and 0.728 against 0.563 on r2; 0.332 and 0.340 against 0.306 on r3), so adding a basis vector that is itself an approximation to the solution made the Galerkin solution worse. The Galerkin solution is not monotone in the space for this indefinite problem.
- Cause: not tested. Consistent with the stability findings of the earlier gates (a space can contain the field to a few per cent while its Galerkin solution is far off), a plausible reading is that the FNO's field, which is not compatible (it has no gradient structure of its own and a PEC violation of 2 to 4%), shifts the reduced spectrum or the stability of the system. This is a hypothesis.
- Consistency is restored, as any Galerkin solve guarantees: power balance at machine precision, PEC exact, the residual 3 to 12 times smaller than the FNO's. The Gauss-law error is not restored (0.50 to 0.68 against the FNO's 0.20 to 0.29).
- The certificate behaves differently from the FNO's: for the r2 variants the residual tracks the error (Spearman 0.87 and 0.92) and the verdict is NEEDS RECALIBRATION, the same category as POD in the verification study, although accuracy is too poor for any instance to be accepted at tau = 0.2.
- The transfer ratios (0.84 to 1.09) say little here: the enriched solution is dominated by the mesh-independent coarse space, whose error is flat across meshes, and it is far less accurate than the FNO on every mesh.
- Limits, as registered, and one more: one FNO, two coarse spaces, the real and imaginary parts added as separate vectors; adding only the FNO's curl part or projecting out its gradient content was not tried.

**Study 7 protocol: is the encoder under-fitted, and do two low-effort changes help? (fixed before running)**

Question: the encoder's training error at 400 instances (0.159) is close to its test error (0.180) and far from the per-instance oracle (0.044), and the Gate 2 validation selection ended at the largest width and step count tried. This suggests under-fitting. Two changes that are cheap to implement are tested: (i) pre-training with a best-approximation (projection) loss before the Galerkin loss, and (ii) more bumps.

- Model: the Gate 2 encoder (per-node base, CNN of width 24, batch 8, learning rates 0.02 and 2e-3, cosine annealing per training stage), refine 4 only (no transfer). Dimension 36 for n = 8 bumps and 78 for n = 12.
- Configurations, all with a total of 1500 optimiser steps:
  - `G8`: n = 8, Galerkin loss for 1500 steps (the control for step count).
  - `P8`: n = 8, projection loss for 500 steps, then the Galerkin loss for 1000 steps.
  - `G12`: n = 12, Galerkin loss for 1500 steps.
  - `P12`: n = 12, projection loss for 500 steps, then the Galerkin loss for 1000 steps.
  - The projection loss is the relative M-norm error of the best approximation of the reference solution in the span of the predicted basis (differentiable, as `oracle.projection_error`); the Galerkin loss is the relative M-norm error of the Galerkin solution, as in Gate 2. A new Adam optimiser and cosine schedule start with each stage.
- Data: as Gate 2 and Study 4a: 100 and 400 training instances (nested subsets of seed 400 + r), 300 test instances (seed 500 + r, the Gate 2 test sets), three repeats, model seed r.
- Reported per configuration and size (mean over repeats): median field and stored-energy error, the final training error (Galerkin loss), and the median best-approximation (projection) error of the predicted basis on the test instances, which separates how good the predicted space is from how well the Galerkin solve uses it. Comparators from committed results: the Gate 2 encoder at 1000 steps (0.243 at 100 and 0.180 at 400 in Study 4a; the Gate 2 table), the FNO (0.241 and 0.150 in the baseline study), POD-136 (0.265 and 0.273).
- Decision rules, each on the mean over repeats of the median field error, per size:
  - More steps help: `G8` is at least 10% lower than the Gate 2 encoder (the 1000-step baseline of the same seeds) at 400 instances.
  - Pre-training helps: `P8` is at least 10% lower than `G8` at both sizes; hurts: at least 10% higher at either size; no effect otherwise. The same rule for `P12` against `G12`.
  - More bumps help: `G12` is at least 10% lower than `G8` at both sizes (the dimension rises from 36 to 78, so this is a cost-accuracy trade, reported as such).
  - Closes the gap to the FNO: some configuration has a mean median field error at most the FNO's at both sizes (reported, not a pass or fail).
  - Under-fitting is supported if the training error falls by at least 10% from the Gate 2 encoder to `G8` at 400 instances.
- Multiplicity: four configurations and four comparisons; the test sets are used once.
- Limits, stated in advance: one architecture and one learning-rate setting; a fixed 1500-step budget (a longer budget is not tested here); the 10% threshold is a stated convention; per-node base, so no transfer claim; with n = 12 the reduced dimension is 78, which is not directly comparable to the 36-dimensional baselines.

**Study 7 results** (`experiments/study7.py`, protocol above; per-node encoder at refine 4, 1500 optimiser steps per configuration, 100 and 400 training instances, three repeats, the Gate 2 test sets of 300 instances; mean over repeats)

| Configuration | Dimension | 100 instances: field / energy / training error / best-approximation error | 400 instances: field / energy / training error / best-approximation error |
|---|---|---|---|
| Gate 2 encoder (n = 8, 1000 steps) | 36 | 0.243 / 0.078 / 0.131 / not measured | 0.180 / 0.058 / 0.159 / not measured |
| `G8` (n = 8, Galerkin loss, 1500 steps) | 36 | 0.234 / 0.074 / 0.112 / 0.191 | 0.164 / 0.052 / 0.140 / 0.137 |
| `P8` (projection 500, then Galerkin 1000) | 36 | 0.247 / 0.081 / 0.129 / 0.196 | 0.182 / 0.061 / 0.158 / 0.147 |
| `G12` (n = 12, Galerkin loss, 1500 steps) | 78 | 0.216 / 0.064 / 0.097 / 0.152 | 0.145 / 0.044 / 0.120 / 0.101 |
| `P12` (projection 500, then Galerkin 1000) | 78 | 0.226 / 0.070 / 0.133 / 0.148 | 0.219 / 0.073 / 0.200 / 0.152 |
| FNO (baseline study) | | 0.241 / 0.095 | 0.150 / 0.049 |

The spread between repeats is at most about 0.015 in the median field error for every configuration.

Registered comparisons:

- More steps (`G8` against the Gate 2 encoder at 400 instances): ratio 0.91, which misses the registered 10% (NO CLEAR HELP); the training error falls by 12% (ratio 0.88), so under-fitting is SUPPORTED by the registered rule, narrowly.
- Pre-training with the projection loss: HURTS at both bump counts (field-error ratios 1.06 and 1.11 at n = 8; 1.05 and 1.51 at n = 12; the n = 12 model at 400 instances ends with a training error of 0.20 against 0.12 for `G12`).
- More bumps (`G12` against `G8`, dimension 36 to 78): ratios 0.92 and 0.89, which miss the registered 10% at 100 instances (NO EFFECT by the rule), although the field error falls by 8 to 11% and the energy error by 14 to 15%.
- Closes the gap to the FNO at both sizes (reported, not graded): `G12` does, with a median field error of 0.216 against 0.241 at 100 instances and 0.145 against 0.150 at 400, and a median energy error of 0.064 against 0.095 and 0.044 against 0.049.

Reading:

- The encoder was under-fitted, not fundamentally weak. Training for 50% longer (`G8`) lowers the error by 4 to 9%, and adding bumps on top (`G12`) by a further 8 to 11%; together, relative to the Gate 2 encoder, that is 11% at 100 instances and 19% at 400. The two effects are individually below the registered thresholds and are not separated further here. The combined model (`G12`: 78 dimensions, 1500 steps) matches the FNO's median field error at refine 4 and has the lower median energy error at both sizes.
- The predicted space is the bottleneck, not the Galerkin solve. The best-approximation error of the predicted space (0.19 and 0.14 for `G8`, 0.15 and 0.10 for `G12`) is within a factor of 1.2 to 1.45 of the Galerkin error: even a perfect solve in the predicted space would lose at most that factor. The per-instance oracle's 0.044 shows the space can be much better, so the remaining gap is in predicting the basis.
- Projection pre-training does not help and can hurt: the best-approximation loss rewards spaces that contain the solution but does not reward spaces that give a stable Galerkin solve, and 1000 Galerkin steps do not recover (the training error of `P12` at 400 instances is 0.20). This explanation is a hypothesis.
- What this does not change: all models here are per-node encoders at refine 4, so the mesh transfer of Study 5a still fails for them; the certificate and consistency comparisons of Study 4a were made with the Gate 2 encoder, not `G12`; the cost of a 78-dimensional space is higher than a 36-dimensional one and was not measured; `G12` was trained for 1500 steps and the FNO for 200 epochs, so the comparison is not at matched compute; and neither reaches the 10% field-error target at 100 instances (0.216 and 0.241), only the 400-instance models come close (0.145 and 0.150).
- Limits, as registered: one architecture and one learning-rate setting, a fixed 1500-step budget, a 10% threshold that is a convention, and a reduced dimension that differs between the n = 8 and n = 12 configurations.

**Study 8 protocol: do the Study 7 gains survive the checks? (fixed before running)**

Question: with 12 bumps (78 dimensions) and 1500 steps the encoder (`G12`) matches the FNO's median field error at refine 4 (Study 7). Four things were not rechecked for that model: whether the match holds against an FNO of the same parameter count, whether the certificate and the consistency comparison of Study 4a change, whether the mesh transfer of Study 5a changes, and what the 78-dimensional model costs.

- Models, trained at refine 4 with 100 and 400 instances (nested subsets of seed 400 + r, three repeats, model seed r): `G12` (n = 12, width 24, Galerkin loss for 1500 steps, as Study 7); the FNO of the baseline study (grid 64, width 32, 12 modes, 4 layers, 200 epochs, 1.19 million parameters); and a small FNO (grid 64, width 8, 6 modes, 4 layers, otherwise the same settings and 200 epochs, about 20,000 parameters, close to the encoder's roughly 24,000). Parameter counts are reported.
- Stage A (refine 4, as Study 4a): 1000 calibration instances (seed 600 + r) and 1000 test instances (seed 700 + r). Per model: median field error, power balance, PEC violation, Gauss-law error and Euclidean residual from `metrics.evaluate`, and the certificate under the verification rules (Spearman of log eta_D against log error, coverage on random and selected sets, acceptance and false-accept rates, verdicts). Matched-accuracy Gauss-law comparison between `G12` and the large FNO on the instances where both have field error in [0.10, 0.30], as in Study 4a.
- Stage B (transfer, as Study 5a): the trained models on the cached test sets at refine 4, 5, 6, 7 (200 instances, seed 500), without retraining; `G12` with the nearest-node transfer of Study 5a. The transfer holds at a mesh if the mean median field error there is at most 1.25 times its refine-4 value.
- Stage C (cost, single thread, quiet machine, the Study 4b harness with 12 bumps and 1500 training steps): profile and single-query cost at refine 4 to 8 for the 78-dimensional encoder, against the full solve and the Study 5b FNO numbers. Readout only; run after stages A and B.
- Decision rules:
  - Parameter-matched comparison, on the stage-A test median field error, per size: `G12` LEADS the small FNO if its mean is at most 0.90 times the small FNO's at both sizes, TRAILS if at least 1.10 times at either size, LEVEL otherwise. The same ratio against the large FNO is reported.
  - Certificate: the verdict (CERTIFICATE VALID, NEEDS RECALIBRATION, DOES NOT TRACK, INCONCLUSIVE) for each model and size, by the verification rules, reported next to the Study 4a verdicts of the Gate 2 encoder (DOES NOT TRACK, Spearman 0.26 and 0.34).
  - Consistency: the Study 4a verdict rule between `G12` and the large FNO (BY CONSTRUCTION ONLY, SUBSTANTIVE ADVANTAGE, NO ADVANTAGE).
  - Transfer: for `G12` and the small FNO, TRANSFER HOLDS TO REFINE r by the rule of Study 5a; the speed claim of Study 4b is supported for `G12` only where accuracy holds at refine 6 and 7.
- Limits, stated in advance: the compute is not matched (1500 encoder steps with batch 8, 200 FNO epochs); the small FNO is one size choice, untuned; `G12` is a per-node model, so the transfer is expected to fail as for the Gate 2 encoder (the question is whether the better fit changes the ratio); one family, 2D; the cost stage is single thread against SciPy's direct solver.

**Study 8 results, stages A and B** (`experiments/study8.py`, protocol above; three repeats; stage A with 1000 calibration and 1000 test instances at refine 4, stage B with 200 instances per mesh at refine 4 to 7; mean over repeats)

Stage A, refine 4, medians over the test set:

| Model (instances) | Parameters | Median field error | Power balance | PEC violation | Gauss law | Euclidean residual | Spearman | Certificate verdict | Accepted / false-accepted at tau = 0.2 |
|---|---|---|---|---|---|---|---|---|---|
| `G12` (100) | 24,000 | 0.213 | 5.4e-15 | 0 | 0.319 | 3.23 | 0.40 | DOES NOT TRACK | 0.005 / 0.003 |
| FNO (100) | 1,188,868 | 0.235 | 2.30 | 0.039 | 0.288 | 30.5 | 0.55 | DOES NOT TRACK | 0.066 / 0.021 |
| Small FNO (100) | 20,452 | 0.355 | 1.06 | 0.063 | 0.346 | 19.4 | 0.28 | DOES NOT TRACK | 0.000 / 0.000 |
| `G12` (400) | 24,000 | 0.141 | 4.9e-15 | 0 | 0.218 | 3.08 | 0.49 | DOES NOT TRACK | 0.609 / 0.027 |
| FNO (400) | 1,188,868 | 0.148 | 2.31 | 0.024 | 0.203 | 30.3 | 0.60 | DOES NOT TRACK | 0.383 / 0.010 |
| Small FNO (400) | 20,452 | 0.224 | 1.34 | 0.040 | 0.282 | 20.8 | 0.31 | DOES NOT TRACK | 0.036 / 0.015 |

Coverage of the certified bound for `G12`: 0.888 and 0.893 on random instances, 0.900 and 0.887 on the selected sets (100 and 400 instances); for the FNO 0.893 and 0.895 random.

Registered outcomes:

- Parameter-matched comparison: **LEADS** (`G12` over the small FNO: 0.60 at 100 and 0.63 at 400, against the bar of at most 0.90 at both sizes). Against the large FNO, reported: 0.91 and 0.95.
- Consistency (`G12` against the large FNO): **BY CONSTRUCTION ONLY**. Power balance and PEC hold to machine precision for `G12` (the FNO: 2.3 and 2 to 4%), but the matched-accuracy Gauss-law error is 1.11 to 1.22 times the FNO's at 100 instances (0.285, 0.290, 0.281 against 0.249, 0.238, 0.253 in the three repeats) and 1.02 to 1.16 times at 400, not at most half.
- Certificate: DOES NOT TRACK for every model and size; the Spearman correlation of `G12` (0.40 and 0.49) is higher than that of the Gate 2 encoder (0.26 and 0.34) and lower than the large FNO's (0.55 and 0.60).
- Transfer (stage B):

| Model | refine 4 | refine 5 | refine 6 | refine 7 | Holds to |
|---|---|---|---|---|---|
| `G12`, 100 | 0.213 | 0.591 (2.78) | 0.720 (3.38) | 0.814 (3.83) | refine 4 |
| `G12`, 400 | 0.143 | 0.569 (3.98) | 0.705 (4.93) | 0.804 (5.62) | refine 4 |
| FNO, 100 | 0.238 | 0.245 (1.03) | 0.251 (1.05) | 0.250 (1.05) | refine 7 |
| FNO, 400 | 0.144 | 0.166 (1.15) | 0.173 (1.20) | 0.170 (1.18) | refine 7 |
| Small FNO, 100 | 0.355 | 0.360 (1.01) | 0.361 (1.02) | 0.361 (1.02) | refine 7 |
| Small FNO, 400 | 0.224 | 0.230 (1.03) | 0.236 (1.06) | 0.236 (1.05) | refine 7 |

Reading:

- At the training mesh the compatible encoder is more accurate per parameter than an FNO: with 24,000 parameters it has 0.60 to 0.63 times the error of an FNO of 20,452 parameters, and matches the FNO of 1.19 million parameters (0.91 and 0.95 times its error, with the lower stored-energy error in Study 7). The small FNO is much worse than the large one (0.355 against 0.238 at 100 instances), so the FNO needs its size here and the Galerkin structure does not. The compute is not matched (1500 encoder steps with batch 8 against 200 FNO epochs), and the encoder's reduced dimension is 78.
- The better-trained encoder has the most useful certificate at a loose target: with 400 instances it accepts 61% of instances at an accuracy target of 0.2 with a 2.7% false-accept rate (the FNO accepts 38% with 1.0%), although the verdict by the Spearman rule is still DOES NOT TRACK; at a target of 0.1 or 0.05 essentially nothing is accepted by any model. The reason is accuracy (its median error is 0.14, so many instances sit below 0.2) more than tracking. The false-accept rates are below the nominal 10%.
- The transfer failure is unchanged: the better fit does not help, the ratios are 2.8 to 5.6, and even the small FNO is flat (1.01 to 1.06). A model that is more accurate per parameter on the training mesh is still not usable on a finer mesh without removing its per-node parameters (Study 6a).
- Limits, as registered: compute not matched; one small FNO size, untuned; per-node model; 2D, one family. Stage C (cost) follows below.

**Study 8 results, stage C: cost of the 12-bump (78-dimensional) encoder** (`experiments/study4b.py --bumps 12 --steps 1500`, the Study 4b harness, one thread, load average at most 1.06, nothing flagged; the model trained with 100 instances; transferred weights, accuracy on finer meshes not assessed)

Profile at refine 4, median of 30 instances (milliseconds): inputs 0.14, CNN and sampling 1.26, partition of unity and lift 1.91, restriction of M 0.73, whitening and reduced solve 2.94, total 6.40 (the 36-dimensional encoder: 3.69); output identical to the original path. Single query including the residual, against the full solve:

| Mesh | Unknowns | Full solve | Encoder pass, 78 dimensions (basis part) | Speedup | 36-dimensional encoder speedup (Study 4b) | Per extra frequency |
|---|---|---|---|---|---|---|
| refine 4 | 1,504 | 2.4 ms | 6.3 ms (3.0) | 0.38 | 0.69 | 0.10 ms |
| refine 5 | 6,080 | 11.7 ms | 24.9 ms (8.6) | 0.47 | 0.88 | 0.27 ms |
| refine 6 | 24,448 | 91.8 ms | 134.1 ms (50.5) | 0.68 | 1.93 | 1.07 ms |
| refine 7 | 98,048 | 804 ms | 683.5 ms (251.9) | 1.17 | 3.06 | 5.64 ms |
| refine 8 | 392,704 | 6,780 ms | 3,002 ms (1,100) | 2.24 | 4.64 | 25.0 ms |

Reading: doubling the reduced dimension from 36 to 78 roughly doubles the encoder pass and halves the speedup. The 78-dimensional encoder is faster than a full solve only from about 98k unknowns (1.17) and 2.2 times faster at 393k, against 3.06 and 4.64 for the 36-dimensional one; an extra frequency is still 271 times cheaper at 393k. So the accuracy gained in Study 7 and 8 is bought with cost: the encoder that matches the large FNO at refine 4 is slower than a full solve on every mesh below 98k unknowns. The FNO remains far cheaper (26.5 ms at 393k, Study 5b). Limits: SciPy's direct solver; one geometry; accuracy of the transferred weights on these meshes is poor (Study 5a and 8).

**Study 9 protocol: a screen-and-certify design loop (H6, level 0; fixed before running)**

Question: does screening candidate designs with a surrogate, and optionally gating them with the residual certificate, find good designs with fewer full-wave calls than searching with full-wave solves alone? This is the project's design-level claim (fewer full-wave calls per finished design at equal figure of merit). It is a demonstration at the working mesh, where a full solve costs only 2.6 ms, so the metric is the number of full-wave calls, not wall-clock time.

- Design task: the main family's cavity at refine 4. A design is a disc (centre x and y in [0.3, 0.7], radius in [0.10, 0.25], relative permittivity in [2, 8]); the source (position in [0.2, 0.8]^2, random direction) and the frequency (omega^2 uniform in [2.0, 3.2], with the usual loss tangent 0.05, so that every design is sub-resonant with r between about 0.2 and 0.6) are fixed within a task. Figure of merit: the field energy e^H M_R e in the region R = [0.65, 0.85] x [0.15, 0.35] (as in the verification study), to be maximised.
- Trials: 5 tasks (seeds 900 to 904 for the source and frequency) x 10 pools (seeds 1000 + task x 10 + k) = 50 trials. A pool is 1000 random designs, uniform over the box. The full-wave figure of merit of every pool design is computed, so the pool optimum is known. Regret of a returned design = 1 - (its full-wave figure of merit) / (the pool optimum).
- Surrogates, trained at refine 4 with 400 instances (seed 400): the FNO of the baseline study, the 12-bump encoder `G12` (1500 Galerkin steps, Study 7) and POD-136. Certificate calibration is global, on 1000 family instances (seed 600) per surrogate: split conformal on the ratio of error to eta_D at alpha = 0.1, with the quantile recomputed on the 10% of calibration instances with the highest surrogate figure of merit (the selection-shift correction of the verification study).
- Policies, per surrogate and pool:
  - A (screen and verify): rank the pool by surrogate figure of merit, verify the top M by full-wave, return the best verified design; M in {1, 2, 5, 10, 20, 50}; calls = M.
  - B (screen, certify, verify): take the top K = 50 by surrogate figure of merit; a candidate is certified if q_hat x eta_D <= tau = 0.2, and uncertified candidates are solved by full-wave; rank by the figure of merit (surrogate for certified, full-wave for the others), verify the top 5 by full-wave; return the best verified design; calls = uncertified candidates + verified certified candidates.
  - C (baseline): full-wave on c randomly chosen pool designs, return the best (its expected regret is estimated from 200 resamples of the pool); evaluated at c in {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000} and at the mean number of calls of B.
- Readouts and decision rules (mean over the 50 trials):
  - Calls to reach regret at most 0.10: for C (the smallest c) and for A (the smallest M), per surrogate. SCREENING WORKS for a surrogate if A needs at most half of C's calls; PARTIAL if between half and the same; NO otherwise.
  - Certificate: B's mean regret against A's at the smallest grid M that is at least B's mean number of calls. CERTIFICATE ADDS VALUE if B's regret is at most 0.8 times A's; NO GAIN otherwise.
  - Optimiser's curse, reported: the mean relative error (surrogate figure of merit over full-wave figure of merit, minus 1) of the surrogate's top-ranked candidate before verification.
  - Cost projection, reported and labelled as a projection and not a measurement: the calls above times the 393k-unknown full-solve time (6.6 s) plus the screening cost of the surrogate over the pool at its measured 393k single-query time (FNO 26.5 ms, `G12` 3.0 s from Studies 5b and 8); the accuracy of the surrogates at that mesh is not established (Study 5a, 8).
- Limits, stated in advance: a four-parameter design space and a random-pool search (not gradient-based); the figure of merit is a field energy, not an S-parameter; the certificate is calibrated on the family, not on the design distribution; at this mesh there is no wall-clock saving; the surrogates are accurate to 14% to 27% in median field error, so large savings are not assured; one calibration level and one tau.

**Study 9 results** (`experiments/study9.py`, protocol above; 5 tasks x 10 pools of 1000 random disc designs = 50 trials; surrogates trained at refine 4 with 400 instances; certificate calibrated on 1000 family instances; mean over the 50 trials)

Mean regret of full-wave random search by number of calls (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000): 0.693, 0.615, 0.498, 0.401, 0.312, 0.204, 0.136, 0.082, 0.028, 0.000. Random search needs **200 full-wave calls** to reach a mean regret of at most 0.10.

Policy A (rank the pool by the surrogate, verify the top M by full-wave), mean regret by M:

| Surrogate | M = 1 | 2 | 5 | 10 | 20 | 50 | Calls to reach regret <= 0.10 | Outcome |
|---|---|---|---|---|---|---|---|---|
| FNO | 0.064 | 0.035 | 0.011 | 0.003 | 0.001 | 0.000 | 1 (against 200) | SCREENING WORKS |
| `G12` (78-dimensional encoder) | 0.013 | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 1 (against 200) | SCREENING WORKS |
| POD-136 | 0.494 | 0.365 | 0.195 | 0.143 | 0.114 | 0.041 | 50 (against 200) | SCREENING WORKS (ratio 0.25) |

Policy B (top 50 by the surrogate, certify with tau = 0.2, uncertified candidates solved by full-wave, verify the top 5):

| Surrogate | Mean full-wave calls (certified candidates in the top 50) | Mean regret | Random search at the same calls | Policy A at the smallest M at least B's calls | Outcome |
|---|---|---|---|---|---|
| FNO | 28.3 (22.4) | 0.006 | 0.285 | 0.000 (M = 50) | NO GAIN |
| `G12` | 40.2 (10.6) | 0.000 | 0.247 | 0.000 (M = 50) | NO GAIN |
| POD-136 | 41.1 (10.8) | 0.041 | 0.229 | 0.041 (M = 50) | NO GAIN |

Optimiser's curse (the surrogate's figure of merit over the full-wave one for its top-ranked candidate, minus 1): FNO +0.10, `G12` -0.07, POD-136 +12.1.

Cost projection at 393k unknowns (a projection, not a measurement, and the accuracy of the surrogates at that mesh is not established): FNO 33 s (screening 1000 designs at 26.5 ms plus one full-wave call) against 1320 s for random full-wave search to reach the same regret; `G12` 3007 s (screening alone is 3000 s at 3.0 s per design) against 1320 s, so the 78-dimensional encoder would not save time at that mesh.

Reading:

- Screening with a surrogate of about 14% to 15% median field error ranks designs well enough that verifying a single top candidate is within 1% to 6% of the pool optimum, 200 times fewer full-wave calls than random search needs to get within 10%. Even POD-136, with a 27% error, needs 4 times fewer calls. This is the project's design-level claim (fewer full-wave calls per finished design at equal figure of merit), demonstrated in a simple setting.
- The certificate gating adds nothing here: policy B spends 28 to 41 calls on average to reach a regret that policy A already reaches with 1 to 5 calls, so it is dominated by plain screening followed by a few verifications. Its role as a trigger for fallbacks costs calls without improving the returned design. The certificate remains useful as a reported guarantee on each accepted answer, not as an efficiency device in this setup.
- The optimiser's curse is large for POD (the figure of merit of its top candidate is overestimated by a factor of 13, consistent with the verification study) and mild for the FNO (+10%) and the encoder (-7%); the verification of the top candidate by full-wave removes it from the returned design in every policy.
- Caveats, in addition to the registered limits: the design space has four parameters and the pool is random, so the optimum is found by ranking, not by local refinement; the surrogates were trained on the same family as the pool (in distribution); the figure of merit is a field energy; a regret of 1% to 6% for the top-1 policy is the resolution of the screening, and a harder landscape (finer differences near the optimum, more parameters) would demand more of the ranking; the mesh is small, so there is no wall-clock saving here, and the projection to 393k is conditional on accuracy that the transfer studies do not support for the encoder (it holds for the FNO only if the FNO's accuracy at that mesh remains as at 98k, which was measured up to 98k unknowns only).

**Study 10 protocol: a grid-base, mesh-free encoder with multi-resolution training (fixed before running)**

Question: the per-node base logits were the main cause of the failed transfer to finer meshes (Studies 5a, 5c, 6a), but the analytic mesh-free base that removed the cause lost accuracy at refine 4 and trained unstably (Study 6a). Can a mesh-free base with enough capacity, and training on more than one mesh, give both the accuracy of the per-node 78-dimensional encoder at refine 4 and transfer to finer meshes? It addresses the scale at which the model is accurate.

- Model (`GridBaseEncoder`): 12 bumps (dimension 78), CNN of width 24 as in Gate 2. The base logits are a learnable tensor of shape (12, 32, 32) on a regular grid over the unit square (12,288 parameters, against about 6,500 for the per-node base), sampled bilinearly at the node coordinates of whatever mesh the model is applied to, plus the CNN correction as before. Nothing is indexed by a mesh node. The base is initialised from the Lloyd rbf start (seed 0) evaluated on the grid points. Training: Galerkin loss, 1500 steps, batch 8, Adam with the Gate 2 learning rates (base 0.02, CNN 2e-3), cosine annealing, as `G12` in Study 7.
- Variants: `GB`, trained at refine 4 only; `GBMR`, trained on refine 4 and refine 5 (the same instances discretised on both meshes, 2n training problems, each batch drawn at random from both). The multi-resolution variant takes about twice the compute per step of `GB`.
- Data: 100 and 400 training instances (seed 400 + r, three repeats, model seed r); tests on the cached 200-instance sets at refine 4, 5, 6 and 7 (seed 500), as Studies 5a and 8; refine 5 is seen by `GBMR` in training with other instances, refine 6 and 7 are unseen by both variants.
- Reference for comparison: the per-node 78-dimensional encoder `G12` of Study 8 (median field error at refine 4: 0.213 at 100 instances and 0.143 at 400; transfer ratios 2.78 to 5.62).
- Decision rules, per variant and training size, on the mean over repeats of the median field error:
  - Parity: the refine-4 error is at most 1.15 times that of `G12` (0.245 at 100 instances, 0.164 at 400).
  - Transfer holds: the error at refine 6 and at refine 7 is at most 1.25 times the variant's own refine-4 value.
  - Outcomes: SOLVES (parity at both sizes and transfer holds at both sizes); TRANSFERS BUT LOSES ACCURACY (transfer holds, parity fails); ACCURATE BUT DOES NOT TRANSFER (parity holds, transfer fails); FAILS (neither).
  - Readouts, not graded: the absolute errors at refine 6 and 7 against the FNO (0.25 and 0.17 at 100 and 400 instances); the spread of the refine-4 error over the three repeats (the mesh-free analytic base ranged from 0.35 to 0.76, so a range above 0.1 is reported as unstable).
- Limits, stated in advance: one grid resolution (32 x 32) and one bump count; the multi-resolution variant is trained on two meshes and tested on two further ones, so it shows how far the benefit extends, not that it holds at 393k unknowns (not tested); a per-instance cost at 78 dimensions of about 2.2 times a full solve at 393k (Study 8) is unchanged by the base; the FNO remains cheaper and transfers; compute is not matched between `GB` and `GBMR`.

**Sweep animation (descriptive, no pass or fail; fixed before rendering)**

Purpose: the animation for the deck, and a first measurement of sweep accuracy, which is untested (Study 8 cost numbers for an extra frequency assume a reused basis). One geometry and source: the first instance of the seed-500 test set at refine 4. Frequencies: 61 values with r = omega^2 / lambda_1 from 0.2 to 0.6 (the main band). Models: the 78-dimensional encoder `G12` and the FNO, trained with 400 instances (seed 400). Shown: the reference field magnitude, the encoder's field when the basis is predicted at each frequency, and the error map; and the relative field error against r for the encoder with the basis re-predicted at every frequency, the encoder with the basis predicted once at the band centre and reused for every frequency (only the small solve repeated), and the FNO. To avoid a single-instance impression, the error curves are also computed for the first 20 test geometries and their median is drawn faintly; the numbers go to a CSV. A static frame of each animation is saved as a fallback.

**Sweep animation results** (`figures/make_animation.py`, the descriptive note above; the 78-dimensional encoder and the FNO trained with 400 instances; 20 test geometries x 61 frequencies, r = 0.2 to 0.6, refine 4; outputs `figures/anim_sweep.gif`, `anim_phase.gif`, their fallback frames and `anim_sweep_errors.csv`)

Median relative field error over the 20 geometries and 61 frequencies: encoder with the basis re-predicted at every frequency 0.146; encoder with the basis predicted once at r = 0.4 and reused for every frequency (only the small Galerkin solve repeated) 0.146; FNO 0.151. For the displayed geometry (the first seed-500 test geometry) the mean over the band is 0.171, 0.170 and 0.165, and the worst frequency of the median curve is 0.175, 0.178 and 0.172. The error of all three rises slowly with r, from about 0.14 at r = 0.2 to about 0.18 at r = 0.6.

Reading:

- Reusing one predicted basis across the whole main band costs no accuracy (0.146 against 0.146). This gives the cost numbers for an extra frequency (55 to 539 times cheaper than a full solve, Studies 4b and 8), which assume a reused basis, an accuracy backing in the sub-resonant band at the training mesh: the cheap-sweep mechanism works as described there, at a field error of 14% to 18%.
- The FNO has no reuse mechanism (a new forward pass per frequency) and is level with the encoder in accuracy across the band.
- Limits: descriptive, one family, refine 4 only, 20 geometries, a basis predicted at the band centre and tested inside [0.2, 0.6] (nothing is claimed near resonance or outside the band); the displayed geometry is a fixed first test instance, not selected for appearance; the error is 14% to 18%, not simulation-level; no pass or fail was set.

**Study 10 results** (`experiments/study10.py`, protocol above; 12 bumps, a learnable 32 x 32 grid base, 29,748 parameters; `GB` trained at refine 4, `GBMR` at refine 4 and 5; 100 and 400 training instances; three repeats; 200 test instances per mesh; mean over repeats)

Median field error, with the ratio to the variant's own refine-4 value in brackets:

| Variant (training instances) | refine 4 | refine 5 | refine 6 | refine 7 | Refine-4 range over repeats |
|---|---|---|---|---|---|
| `GB` (100) | 0.308 | 0.382 (1.24) | 0.427 (1.39) | 0.478 (1.55) | 0.191 (unstable) |
| `GB` (400) | 0.179 | 0.281 (1.57) | 0.329 (1.84) | 0.373 (2.08) | 0.032 |
| `GBMR` (100) | 0.283 | 0.316 (1.12) | 0.345 (1.22) | 0.377 (1.33) | 0.084 |
| `GBMR` (400) | 0.195 | 0.225 (1.15) | 0.264 (1.35) | 0.306 (1.56) | 0.076 |

Reference: the per-node 78-dimensional encoder `G12` has 0.213 (100) and 0.143 (400) at refine 4, so the parity limits are 0.245 and 0.164; the FNO has about 0.25 (100) and 0.17 (400) at refine 6 and 7.

Outcomes by the registered rules: **`GB`: FAILS; `GBMR`: FAILS.** Parity fails for every variant and size (the refine-4 error is 1.25 to 1.45 times that of `G12`), and the transfer ratio exceeds 1.25 at refine 7 for every variant (and at refine 6 for `GB` and for `GBMR` at 400 instances).

Reading:

- Training on two meshes is the effective ingredient, as the interim numbers suggested. `GBMR` degrades far less than `GB` at refine 6 and 7 (ratios 1.22 to 1.56 against 1.39 to 2.08) and has a smaller refine-4 spread (0.08 against 0.19 at 100 instances, where `GB` was unstable). Compared with the earlier encoders at refine 7: `GBMR` has 0.31 to 0.38, against 0.81 to 0.90 for the per-node encoders (ratios 3.8 to 5.6) and 0.8 to 1.0 for the analytic mesh-free encoder.
- It is still not good enough: at 98k unknowns `GBMR` has 1.5 to 1.8 times the FNO's error (0.377 against 0.25 and 0.306 against 0.17), and the price of mesh-freedom is accuracy at the training mesh (0.283 and 0.195 against 0.213 and 0.143 for `G12`, 33% to 36% worse). At 100 instances `GBMR` meets the ratio bar at refine 5 and 6 (1.12 and 1.22) and misses it only at refine 7 (1.33).
- The scale at which this model is usable is therefore wider than before but still bounded: with training on two meshes it stays under about 0.4 median field error out to 98k unknowns (the largest mesh tested; 393k unknowns is untested), still short of the FNO, which is flat.
- Hypotheses, not tested: training on more meshes (refine 4, 5 and 6) should extend the range further; the lost refine-4 accuracy may come from the 1500-step budget being shared across two meshes (`GBMR` takes about twice the compute per step).
- Limits, as registered: one grid resolution and one bump count; two training meshes; compute not matched between `GB` and `GBMR`; refine 8 (393k unknowns) not tested.

**Rules of engagement (the original plan, followed throughout)**

- Run the gates in order, fix thresholds first, then run, and record failures. A result of "does not work at size X" is a valid finding.
- Tune only on validation data, with the selection rule registered beforehand (Gate 2, Study 7).
- Variants beyond the original plan (Studies 5c to 10) were each given a protocol before running. Studies stopped early at the owner's decision (5c and 6a) are marked as deviations where they appear.

## Code

Python (uv, PyTorch, scikit-fem). Tests: `uv run pytest -q` (52 pass, about 5 s; `tests/test_learned_bases.py` covers the step schedule and projection loss, the mesh-free and grid-base encoders, multi-mesh training, the fast inference path, the enriched Galerkin method and the design-loop helper).

- `compat_galerkin/`: `fem.py` (fine-mesh PEC cavity, Nedelec edge elements, curl-curl and epsilon-mass matrices, fine solve, `first_mode`, `region_mass`), `lift.py` (partition of unity, Whitney lift, whitening, reduced solve, all in torch), `oracle.py` (per-instance W optimisation, Gate 1), `problems.py` (the shared family and its sampler; `omega_rel=None` restores the old absolute band, which includes resonance), `metrics.py`, `certify.py` (residual indicators, conformal quantile), `checks.py` (Gate 0), `device.py`.
- `baselines/` (one `Method` interface, see `baselines/README.md`): `fine_fem`, `coarse_whitney`, `pod_galerkin`, `fno`, `shared_whitney`, `encoder_whitney` (a CNN on grid inputs plus per-node base logits; `MeshFreeEncoder` replaces the base by an analytic RBF), `grid_base_encoder` (a learnable grid base and multi-resolution training), `enriched_galerkin` (the FNO's field added to a Whitney space).
- `experiments/`: one script per gate or study (the table in Reproducibility maps each to its section); `diag_*.py` are post hoc diagnostics.
- `figures/`: `make_figures.py` (figures 1 to 11 from the committed CSVs, each as PNG, SVG and a table-view CSV), `make_animation.py` (the sweep and phase animations), `tikz/` (diagrams for the slides).
- `docs/`: `PROJECT_STATE.md` (decisions, corrections, open questions, plan), `SLIDES.md` (slide contents), `verify_numbers.py` (recomputes the numbers quoted in the slides from the result files), `ORIGINAL_NOTES.md` (the first-draft brainstorm, kept verbatim).
- `results/`: gitignored; the CSV and log of each study are force-added.

Convention to remember: scikit-fem's N1 coefficient is the circulation from the higher to the lower vertex (a uniform sign of -1), which is built into the gradient operator G and verified by a test.

## Reproducibility

**Environment.** WSL2 Ubuntu on Windows 11, 20-core CPU, RTX 5050 laptop GPU (8 GB); Python through `uv` (`uv sync`), torch 2.14, scipy 1.18, numpy 2.5. Figures use `uv run --with matplotlib python figures/make_figures.py` (matplotlib is not a project dependency). Seeds are fixed: training 400 + r, test 500 + r, calibration 600 + r, certificate test 700 + r, validation 900 and 901; results differ by up to about 15% between macOS and Linux in the worst Gate 1 entry.

**Running.** From the repo root, one study at a time: `OMP_NUM_THREADS=8 uv run python -u -m experiments.<name>`. Timing studies (`scaling`, `study4b`, `study5b`) need a quiet machine and `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` (the encoder is trained with 8 threads and timed with one). Run only one PyTorch job at a time, smoke tests included. Background jobs under WSL need the shell tool's background option (`nohup ... &` inside a `wsl` call does not survive). Keep the machine awake for long runs (a run was once suspended for hours by sleep). Commit with Windows git (WSL git sees every file as modified because of CRLF line endings); the CRLF warnings are harmless, and a `.gitattributes` could be added if wanted.

| Study | Module (`experiments.`) | Wall time on this machine (approximate) |
|---|---|---|
| Gate 1, 1b, 1c to 1f, band pre-check | `gate1`, `gate1b`, `gate1c`, `gate1d`, `gate1e`, `gate1f`, `band_precheck` | minutes to about an hour each |
| Verification | `verification` | about 30 min |
| Scaling | `scaling` | about 15 min (single thread) |
| Baseline data efficiency | `data_efficiency` | about 25 min (GPU) |
| Gate 1b stage (b) | `gate1b_geometry` | about 2 h |
| Gate 2 | `gate2` | about 55 min |
| Studies 4a, 4b | `study4a`, `study4b` | about 45 min, about 5 min (`--bumps 12 --steps 1500` for the 78-dimensional cost stage) |
| Studies 5a, 5b, 5c | `study5a`, `study5b`, `study5c` | about 40 min, about 5 min, about 30 min (two repeats) |
| Studies 6a, 6b | `study6a`, `study6b` | about 20 min (stopped early), about 50 min |
| Studies 7, 8, 9, 10 | `study7`, `study8`, `study9`, `study10` | about 55 min, about 35 min, about 25 min, about 70 min |
| Sweep animation, figures | `figures/make_animation.py`, `figures/make_figures.py` | about 5 min, about 1 min |

Test datasets with full-wave references are cached in `results/cache/` (the refine-7 set takes about six minutes to build); encoder weights from Studies 5c and 6a are in `results/models/`; the animation arrays in `results/anim_cache.npz` (`--reuse` skips retraining). Every number on the slides can be recomputed with `python3 docs/verify_numbers.py`.

## Literature check and references

Method: a web search of titles and abstracts (publisher, arXiv and journal pages) on 2026-10-04. It is not an exhaustive review, and full texts were not read, so novelty statements are limited to what the abstracts show.

What it found:

- Learned compatible finite-element spaces built from Whitney forms already exist: conditional Neural Whitney Forms [10] (benchmarks: advection-diffusion, shock hydrodynamics, electrostatics, battery thermal runaway) and Geo-NeW [11] (generalisation to unseen geometries with a transformer mesh encoder); a later paper adds Gaussian-process uncertainty quantification with RKHS posterior bounds [12]. None of the abstracts mentions time-harmonic Maxwell with edge elements in a PEC cavity, a residual certificate with conformal calibration, or a design loop. This project is therefore an application and independent evaluation of that idea in a new setting, with a verification layer, not a new method class.
- Classical compatible multigrid for Maxwell [6, 7] and learned multigrid prolongations [8, 9] exist separately; no learned compatible basis used as a reduced Galerkin solver for Maxwell was found in the abstracts reviewed.
- Fourier neural operators are already used as surrogate solvers for electromagnetic scattering and inverse design [15]; the FNO [13] is the natural baseline, consistent with its strength found here.
- The certificate uses split conformal prediction [17, 18]; the optimiser's curse [19] is the effect measured in the design loop (Study 9); adjoint-based inverse design in photonics [20] is the setting for the future-work items; POD and reduced-basis methods [16] are the data-driven baselines' background.

References (checked against the publisher, arXiv or journal pages unless noted):

1. J.-C. Nedelec, Mixed finite elements in R^3, Numerische Mathematik 35 (1980) 315-341.
2. A. Bossavit, Whitney forms: a class of finite elements for three-dimensional computations in electromagnetism, IEE Proceedings A 135(8) (1988) 493-500.
3. H. Whitney, Geometric Integration Theory, Princeton University Press (1957). (Added from memory; not checked in the search.)
4. D. N. Arnold, R. S. Falk, R. Winther, Finite element exterior calculus, homological techniques, and applications, Acta Numerica 15 (2006) 1-155.
5. P. Monk, Finite Element Methods for Maxwell's Equations, Oxford University Press (2003).
6. R. Hiptmair, J. Xu, Nodal auxiliary space preconditioning in H(curl) and H(div) spaces, SIAM Journal on Numerical Analysis 45 (2007) 2483-2509.
7. P. B. Bochev, J. J. Hu, C. M. Siefert, R. S. Tuminaro, An algebraic multigrid approach based on a compatible gauge reformulation of Maxwell's equations, SIAM Journal on Scientific Computing 31 (2008) 557-583.
8. D. Greenfeld, M. Galun, R. Kimmel, I. Yavneh, R. Basri, Learning to optimize multigrid PDE solvers, ICML 2019 (PMLR 97: 4305-4316).
9. I. Luz, M. Galun, H. Maron, R. Basri, I. Yavneh, Learning algebraic multigrid using graph neural networks, ICML 2020 (PMLR 119: 6489-6499).
10. B. Kinch, B. Shaffer, E. Armstrong, M. Meehan, J. Hewson, N. Trask, Structure-preserving digital twins via conditional Neural Whitney Forms, arXiv:2508.06981 (2025).
11. B. D. Shaffer, S. Koohy, B. Kinch, M. A. Hsieh, N. Trask, Structure-preserving learning improves geometry generalization in neural PDEs (Geo-NeW), arXiv:2602.02788 (2026).
12. H. Zhang, A. M. Propp, B. Kinch, H. Owhadi, N. Trask, Structure-preserving neural surrogates with tractable uncertainty quantification, arXiv:2606.11650 (2026).
13. Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya, A. Stuart, A. Anandkumar, Fourier neural operator for parametric partial differential equations, ICLR 2021, arXiv:2010.08895.
14. Z. Li, H. Zheng, N. Kovachki, D. Jin, H. Chen, B. Liu, K. Azizzadenesheli, A. Anandkumar, Physics-informed neural operator for learning partial differential equations, arXiv:2111.03794 (doi:10.1145/3648506).
15. Y. Augenstein, T. Repan, C. Rockstuhl, A neural operator-based surrogate solver for free-form electromagnetic inverse design, arXiv:2302.01934 (2023).
16. J. S. Hesthaven, G. Rozza, B. Stamm, Certified Reduced Basis Methods for Parametrized Partial Differential Equations, SpringerBriefs in Mathematics, Springer (2016).
17. V. Vovk, A. Gammerman, G. Shafer, Algorithmic Learning in a Random World, Springer (2005).
18. A. N. Angelopoulos, S. Bates, A gentle introduction to conformal prediction and distribution-free uncertainty quantification, arXiv:2107.07511 (2021).
19. J. E. Smith, R. L. Winkler, The optimizer's curse: skepticism and postdecision surprise in decision analysis, Management Science 52(3) (2006) 311-322.
20. S. Molesky, Z. Lin, A. Y. Piggott, W. Jin, J. Vuckovic, A. W. Rodriguez, Inverse design in nanophotonics, Nature Photonics 12(11) (2018) 659-670.
21. T. Gustafsson, G. D. McBain, scikit-fem: a Python package for finite element assembly, Journal of Open Source Software 5(52) (2020) 2369.

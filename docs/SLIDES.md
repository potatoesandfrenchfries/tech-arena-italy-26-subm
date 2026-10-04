# Slides: learned compatible Galerkin spaces for fast, Maxwell-consistent surrogates

Contents for the slide deck (lean version: only the quantifiers needed to make each point). Every number was recomputed from the committed result files with `docs/verify_numbers.py`; the README section behind each is named in brackets. Study 10 (a mesh-free encoder trained on two meshes) is included. Figures are in `figures/` (fig6 to fig11, the sweep animation) and `figures/tikz/` (diagrams).

---

## Slide 1. Abstract and motivation

**Title: Can a learned, physics-compatible basis give fast and trustworthy EM surrogates? A pre-registered test.**

**Motivation**
- Design loops need thousands of full-wave solves. A surrogate must be fast, accurate, data-efficient and Maxwell-consistent, and its answers must be checkable.
- Neural operators (FNO) are fast but give no guarantee that the equations hold. Galerkin reduced models are consistent but need a basis built from many full solves.
- Idea (after Neural Whitney Forms and Geo-NeW): learn only a small partition of unity, build Whitney (edge-element) functions from it, keep the physics exact on the fine mesh.

**Abstract**
- Setting: 2D time-harmonic Maxwell, PEC cavity with a dielectric disc, sub-resonant frequencies (resonance is out of scope).
- Method: a CNN encoder (24,000 parameters) predicts the partition of unity; a Whitney lift and a small Galerkin solve (36 or 78 dimensions) give the field; a residual certificate with conformal coverage and a full-wave fallback checks it.
- Findings:
  - Accurate per parameter: median field error 0.21 / 0.14 (100 / 400 training instances), against 0.36 / 0.22 for an FNO of 20,000 parameters; matches an FNO of 1.19 million (0.24 / 0.15). No model reaches 10% at 100 instances.
  - Equations satisfied exactly (power-balance error 5e-15, against 2.3 for the FNO), though POD-Galerkin shares this.
  - Cheap sweeps: one basis reused across the band loses no accuracy (0.146 against 0.146); an extra frequency is 55 to 539 times cheaper than a full solve.
  - Does not transfer to finer meshes (error 0.14 to 0.24 rises to 0.8 to 0.9 at 98k unknowns; the FNO stays flat; training on two meshes cuts this to 0.3 to 0.4) and costs 50 times more than an FNO at scale.
  - The certificate is valid on average (coverage 0.89 to 0.90) but does not track the error of learned models.
  - In a first design loop, surrogate screening cuts full-wave calls 200-fold.

*Speaker notes.* One genuine positive (accuracy per parameter at the training mesh), several bounded negatives (transfer, cost against an FNO, tracking certificate). The contribution is the evaluation, the diagnosis and the verification layer. Scope: 2D, one mesh family, sub-resonant. [README: abstract and results at a glance]

---

## Slide 2. Methodology

- **Problem:** unit-square PEC cavity, dielectric disc (random position, radius, contrast), Gaussian current source, loss tangent 0.05; omega squared = r times the first eigenvalue, r in [0.2, 0.6]. Reference: fine-mesh edge-element solution (1,504 unknowns; finer meshes to 393k for cost and transfer).
- **Model:** 7 grid channels (epsilon, omega squared, source, coordinates) -> CNN -> softmax partition of unity (8 or 12 bumps) -> Whitney lift to the fine mesh -> small Galerkin solve. Physics exact on the fine mesh; only the basis is learned. Loss: Galerkin error against the reference.
- **Comparators:** POD-Galerkin, FNO (1.19M parameters) and a small FNO (20,000 parameters), non-learned Whitney spaces, per-instance oracle (0.044 median field error, the ceiling).
- **Verification:** residual indicator (one sparse product, 0.25% of a solve at 393k), split conformal calibration, accept or fall back to a full-wave solve.
- **Discipline:** protocol and pass criteria committed before each run; failures reported as findings; cost timed single-thread against SciPy's direct solver.

*Diagrams (TikZ, built by `figures/tikz/build.ps1`): `figures/tikz/pipeline.png` (the diagram below) and `figures/tikz/setup.png` (the problem family); `figures/tikz/design_loop.png` goes with slide 4e.*

*Speaker notes.* One diagram: instance -> CNN -> partition of unity -> Whitney lift -> reduced solve -> certificate -> accept or fall back. The oracle is the ceiling; the FNO and POD are the fair baselines. [README: methodology]

---

## Slide 3. Inferences

| Claim | Verdict | Key number |
|---|---|---|
| Accurate per parameter | Yes, at the training mesh | 0.60 to 0.63 times the error of a same-size FNO; level with a 50 times larger one |
| Equations exact | By construction, not specific to the learned basis | Power balance 5e-15 against 2.3 (FNO); POD-Galerkin ties |
| Cheap | Against a full solve, not against an FNO | 4.6x at 393k (36 dim), 2.2x (78 dim); the FNO is 250x |
| Cheap frequency sweeps | Yes, inside the main band | Reused basis: error 0.146 against 0.146 re-predicted |
| Transfers to finer meshes | Not accurately; training on two meshes helps | Per-node: 0.14 to 0.24 rises to 0.8 to 0.9 at 98k unknowns; grid base trained on two meshes: 0.31 to 0.38; FNO 0.17 to 0.25 |
| Certificate | Valid on average, does not track learned models | Coverage 0.89 to 0.90; Spearman 0.34 to 0.49 (POD: 0.93) |
| FNO field added to a Whitney space | Fails | Equations restored, error 1.4 to 4.9 times the FNO's |
| Design loop | Screening works, certificate gating adds nothing | 1 full-wave call against 200 for random search |

**Take-aways**
1. Compatible structure buys parameter efficiency and exact equations at a fixed mesh: 24,000 parameters match an FNO with 1.19 million. It does not buy mesh transfer, cost at scale against an FNO, or a certificate that tracks the error.
2. Training matters: the first encoder was under-fitted; longer training with 12 bumps closed the gap to the FNO (at twice the cost per query).
3. Anything indexed by nodes of the training mesh breaks transfer.
4. Near resonance the failure is the reduced spectrum, not the field (representable to 2%, Galerkin solve error near 1), so resonance was scoped out.

*Speaker notes.* Do not oversell. For "why not just use the FNO?": at scale and across meshes, use it; the compact encoder is the choice when parameter count, exact discrete equations at a fixed mesh and cheap extra frequencies matter. [README: hypothesis ledger]

---

## Slide 4. Inferences and results

### 4a. Accuracy (median field error at the training mesh, 3 repeats; `figures/fig6`, `fig7`)

| Training instances | Encoder 36 dim | Encoder 78 dim | FNO 1.19M | FNO 20k | POD-136 |
|---|---|---|---|---|---|
| 25 | 0.48 | | 0.45 | | 0.67 |
| 100 | 0.24 | 0.21 | 0.24 | 0.36 | 0.27 |
| 400 | 0.18 | 0.14 | 0.15 | 0.22 | 0.27 |

Oracle (per-instance, 36 dim): 0.044. Inference: the encoder beats POD of the same size and, with 12 bumps and longer training, matches the large FNO and beats the same-size one by about 40%. The 10% target is not reached at 100 instances.

### 4b. Cost and transfer (`figures/fig8`, `fig9`)

| Unknowns | Full solve | Encoder 36 dim | Encoder 78 dim | FNO |
|---|---|---|---|---|
| 1,504 | 2.9 ms | 3.2 ms | 6.4 ms | 1.7 ms |
| 24,448 | 89 ms | 48 ms | 135 ms | 2.2 ms |
| 98,048 | 758 ms | 267 ms | 686 ms | 6.1 ms |
| 392,704 | 6.6 s | 1.4 s | 3.0 s | 26.5 ms |

| Median field error, 100 instances | refine 4 (1.5k) | refine 7 (98k) |
|---|---|---|
| Encoder 36 dim, per-node | 0.24 | 0.90 |
| Encoder 78 dim, per-node | 0.21 | 0.81 |
| Mesh-free encoder, 36 dim | 0.61 | 1.0 |
| Grid-base encoder, 78 dim, trained on 2 meshes | 0.28 | 0.38 |
| FNO 1.19M | 0.24 | 0.25 |

Inference: the speed advantage over a full solve is real but the accuracy is gone on finer meshes; training on two meshes recovers much of it (0.38 at 98k unknowns) at a 33% accuracy cost at the training mesh; the FNO is cheaper and keeps its accuracy.

### 4c. Certificate (`figures/fig10`; 400 training instances)

| Model | Spearman (residual vs error) | Accepted at accuracy target 0.2 (false accepts) |
|---|---|---|
| Encoder 78 dim | 0.49 | 61% (2.7%) |
| FNO 1.19M | 0.60 | 38% (1.0%) |
| POD-136 | 0.93 | 16% (1.0%) |

Inference: only POD's residual tracks its error; the better-trained encoder accepts the most at a loose target because it is more accurate. Nothing is accepted at the 5% target.

### 4d. Design loop and frequency sweep (`figures/fig11`, `anim_sweep.gif`)

| Full-wave calls to reach within 10% of the best design | Calls |
|---|---|
| Random full-wave search | 200 |
| FNO or 78-dim encoder screening | 1 |
| POD-136 screening | 50 |

Certificate gating spent 28 to 41 calls for no gain. Task: choose a disc to maximise field energy in a target region (50 trials; four parameters; in distribution). Frequency sweep: a basis predicted once at r = 0.4 and reused over r = 0.2 to 0.6 has the same median error (0.146) as re-predicting at each frequency.

*Speaker notes.* Lead with 4a (the headline), then 4b (the scale story), 4c (what the certificate does and does not give), 4d (design level) and play the sweep animation. State the limits aloud: one family, 2D, SciPy's direct solver, an untuned FNO, compute not matched between the encoder and the FNO. [README: Gate 2, Studies 4a to 9, sweep animation]

---

## Slide 5. Future work

- **Transfer:** training on more meshes (Study 10: two meshes cut the degradation at 98k unknowns from 3.8 to 5.6 times to 1.3 to 1.6 times but failed the bars); then training at the target mesh.
- **Accuracy:** ensembling and label-free per-instance refinement against the residual; label-free training on the residual.
- **Hybrids:** the FNO for accuracy and speed with a compatible step to restore the equations (adding its field to a Galerkin space failed); an FNO with conditionally calibrated certificates as the screening layer.
- **Broader tests:** out-of-distribution accuracy, more complex structures (several inclusions, L-shaped domain), a production direct solver, 3D, resonance by modal enrichment.
- **Design:** gradient-based design with adjoints, certificate-gated trust regions, port and S-parameter objectives, a fabrication filter and a yield measure.

*Speaker notes.* Three questions next: can the mesh-free encoder be made accurate, does a hybrid keep the FNO's accuracy while restoring consistency, does any of it survive out of distribution and on more complex structures. Inverse design with manufacturability and 3D are ideas only. [README: Study 10 protocol, `docs/PROJECT_STATE.md`]

---

## Slide 6. References (selected; the full list with a literature check is at the end of `README.md`)

- Kinch et al. (2025), Structure-preserving digital twins via conditional Neural Whitney Forms, arXiv:2508.06981.
- Shaffer et al. (2026), Structure-preserving learning improves geometry generalization in neural PDEs (Geo-NeW), arXiv:2602.02788.
- Li et al. (2021), Fourier neural operator for parametric PDEs, ICLR; arXiv:2010.08895.
- Nédélec (1980), Mixed finite elements in R^3, Numer. Math. 35; Bossavit (1988), Whitney forms, IEE Proc. A 135; Arnold, Falk, Winther (2006), Finite element exterior calculus, Acta Numerica 15.
- Bochev, Hu, Siefert, Tuminaro (2008), Compatible-gauge algebraic multigrid for Maxwell, SIAM J. Sci. Comput. 31; Hiptmair and Xu (2007), SIAM J. Numer. Anal. 45.
- Angelopoulos and Bates (2021), A gentle introduction to conformal prediction, arXiv:2107.07511.
- Augenstein, Repän, Rockstuhl (2023), Neural operator surrogate solver for electromagnetic inverse design, arXiv:2302.01934.
- Gustafsson and McBain (2020), scikit-fem, J. Open Source Softw. 5(52):2369.

# Original notes (first-draft brainstorm)

Kept verbatim from the end of the first draft of `README.md`. They are superseded by the README; the points that survived are tested in the studies there.

---

# Main Doc

## Brainstorming
Most of these ideas are bad, but who cares atp.

- Low data and training cost. Use PINO-style two-stage training. First learn the operator from coarse data plus PDE residuals at higher resolution. Then fine-tune per design instance using only the PDE loss. This is reported to reduce the data/PDE loss-balancing hyperparameters, at the cost of extra compute from the PDE loss.

- Maxwell consistency by construction. Predict a potential (a Fourier–Helmholtz or vector-potential formulation) instead of raw fields, so the Gauss and Faraday laws hold as hard constraints. Clifford layers are a drop-in alternative for E/B coupling. For varying geometry, use a GINO/Geo-FNO-style encoder.

- Fabrication-aware training. Put a differentiable fabrication model (erosion/dilation, minimum feature size) in the loop. Train the operator on perturbed geometries, so its gradients are reliable near the perturbations that matter. Report robustness with a Hessian-flatness measure, as one diffusion-based design paper does.

- Guaranteed accuracy rather than average accuracy. Wrap the surrogate in split conformal prediction. This gives coverage bands calibrated on held-out data. Fall back to a full-wave solver when estimated error crosses a threshold, as in a hybrid operator-solver scheme that bounds error growth. Verify the final candidates with full-wave simulation.


## Evaluation

- Field error against FDTD/FEM, plus divergence residual and energy or power-balance violation (this measures consistency).
- Out-of-distribution tests across frequency and geometry class.
- Number of full-wave calls per finished design.
Fabrication yield under a tolerance model, and conformal coverage actually achieved.

## Random Abstract

They map onto different parts of the proposal. Geo-FNO mainly helps with speed and flexible geometry input. Geo-NeW mainly helps with Maxwell consistency.

### What each does

- Geo-FNO learns to deform an irregular physical domain into a latent uniform grid, then applies the FFT-based FNO there. Point clouds, meshes and design parameters are all accepted as inputs. It keeps FNO's efficiency while removing the rectangular-grid restriction.

- Geo-NeW (General-Geometry Neural Whitney Forms) learns a differential operator together with compatible reduced finite element spaces on the geometry. According to the authors, it preserves conservation laws exactly through finite element exterior calculus (FEEC) and generalises better to unseen domains. Whitney forms carry an exact-sequence property linking discrete gradient, curl and divergence. Geometry enters through a mesh, via a transformer encoder and as the basis for the learned spaces.

- A related point-cloud version, MEEC-Net, keeps the structural guarantees while removing mesh dependency.

Fit with EM

Geo-NeW is the closer match for consistency.

From my own background knowledge, edge elements (Nédélec) are the standard Maxwell discretisation and are Whitney-type spaces. This is the one idea here designed around the div/curl structure you need.
It speaks to two problems I raised earlier: divergence drift in vector Maxwell surrogates, and singular behaviour near material interfaces in TE polarisation.
The demonstrations I found are on fluid and transport problems, not EM, and the claims are the authors' own.

Geo-FNO fits the design-loop and speed side.

Design parameters can be an input format, and shape optimisation is smooth, which suits gradient-based inverse design.
My reasoning, not from the sources: a smooth deformation cannot change topology (adding a hole, splitting a resonator). It also distorts local wavelength, which matters for oscillatory fields. By coordinate-transformation arguments, the mapped problem is Maxwell with modified, generally anisotropic ε and μ, so the network would have to learn that. Many EM surrogates sidestep geometry by voxelising permittivity on a regular grid, so Geo-FNO's gain is largest for curved or conformal structures and surface-based formulations.

### Possible approach?

- Use Geo-FNO or GINO as the fast stage for exploration in the design loop.

- Use a Geo-NeW-style model as the higher-consistency stage, or as a verifier before full-wave checks.

### Open questions

- Whether FEEC-based learning handles time-harmonic, complex-valued, resonant problems. This is my inference, since I found no such demonstration.
- How radiation boundaries are treated.
- How either approach copes with the high-frequency and spectral-bias issues from earlier.
- Whether the data efficiency carries over from fluids.
Counterpoints
View: FEEC-based learning is the principled route to Maxwell consistency. Counterpoint: it reintroduces meshing, which is part of what neural operators were meant to avoid, and meshing cost may matter in a design loop where geometry changes every iteration.
View: Geo-FNO is the pragmatic choice for geometry. Counterpoint: FNO's low-mode truncation is reported to discard high-frequency content on irregular geometries, and that is where EM resonances live.

![alt text](image.png)

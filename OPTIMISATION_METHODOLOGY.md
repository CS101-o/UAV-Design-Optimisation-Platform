# Optimisation Methodology — Flow5 MDO

_Last updated: 2026-09-17_

---

## 1. Phase 1 — original methodology (LLT + SLSQP)

The original pipeline was a two-stage sequential SLSQP loop with LLT as the forward model.

### Forward model: Lifting Line Theory (LLT)

`analysis.py → compute_trimmed_ld()` implements an analytical lifting-line solver
(Prandtl–Glauert, zero compressibility, elliptic efficiency factor).

- **Wing CDi**: `CL² / (π × AR × e)` where `e ≈ 0.85`
- **CD0**: semi-empirical flat-plate friction + form factor per surface, fixed at `CD0_wing = 0.008`
- **Trim**: closed-form moment balance — given CG and tail volume, tail incidence is solved
  directly in one step, no iteration
- **Speed**: < 1 ms per evaluation; unlimited calls

### Optimiser: SLSQP (Sequential Least Squares Programming)

`optimizer.py → optimize_wing() / optimize_tail()` called `scipy.optimize.minimize(..., method='SLSQP')`.

- **Stage 1 — wing**: AR, taper, twist, sweep, dihedral optimised; tail and fuselage fixed
- **Stage 2 — tail**: SHT_frac, AR_HT, taper_HT, tail_arm, fuse_len, fuse_fin optimised; wing fixed
- **Gradient estimation**: SLSQP queries finite differences — each step needed ~11 function calls
  (one per variable per side)
- **Constraints**: `tail_arm × c_bar ≥ fuse_len` as a hard SLSQP inequality constraint;
  SM, stall, and tail-incidence limits as soft penalty terms added to the objective
- **Typical call count**: ~200 SLSQP iterations × ~11 LLT calls each ≈ 2 000–4 000 LLT calls,
  all completing in well under 1 second total

**Output**: the set of design variables that maximises LLT-predicted trimmed L/D.

---

## 2. Phase 2 — intermediate attempt (Flow5 + Simplex / Nelder-Mead)

Before moving to the full BO surrogate, a Nelder-Mead (Simplex) optimiser was run with
Flow5 TRIUNIFORM as the forward model.

- **Variables**: only 3 (AR, taper, twist) — the rest fixed at LLT-optimal values
- **Method**: `scipy.optimize.minimize(..., method='Nelder-Mead')` — gradient-free, uses only
  function values, so numerical noise from Flow5 does not corrupt the step direction
- **Trim**: Newton iteration on tail incidence inside each evaluation (same as Phase 3)
- **Limitations**:
  - With 3 variables Simplex needs ~60–100 evaluations to converge — feasible, but the result
    is constrained to the subspace the LLT phase already found; it cannot escape that basin
  - 9 tail/fuselage variables remain fixed — cross-coupling between wing and tail is not explored
  - No surrogate — every evaluation calls Flow5; no uncertainty estimate, no acquisition strategy
  - Convergence is slow in high-dimensional spaces: Simplex degenerates as n grows

**Outcome**: Flow5 Simplex closed the gap between LLT and Flow5 predictions for the variables
it optimised, but could not jointly optimise the full 11-variable design space. It confirmed that
Nelder-Mead works with Flow5 noise but cannot replace a proper surrogate-based approach.

---

## 3. What changed and why

### Problem 1 — SLSQP gradients are incompatible with Flow5 (motivated Phase 2)

When we tried to plug Flow5 into the SLSQP loop:

- Flow5 TRIUNIFORM takes ~1 s/call; 2 000 calls = **33 minutes** per optimisation
- Flow5 CSV output has small numerical fluctuations (~0.01 % in CDi) between otherwise
  identical runs due to solver convergence tolerances and file timestamps
- SLSQP finite-difference gradients are dominated by this noise at the design variable step
  size needed for meaningful sensitivity — the gradient estimate is meaningless, and SLSQP
  terminates on a spurious local minimum or hits iteration limits immediately

This motivated moving to a gradient-free method (Phase 2: Simplex). Simplex was gradient-free
and worked correctly with Flow5 noise, but it could only optimise 3 variables and lacked a
surrogate, which motivated Phase 3.

### Problem 2 — LLT and Flow5 do not agree on the neutral point

LLT places the aircraft neutral point at ~0.21 m aft of the wing leading edge
(~1.5 MAC fractions from wing LE) for a typical 2 kg design.
Flow5's Trefftz-plane integration places it at ~0.43 m (~3.0 MAC fractions) for the same geometry.

The ratio is approximately 2:1.

**Consequence**: the CG position set by LLT (NP_LLT − 10 % MAC) is far forward of the true
aerodynamic centre. Flow5 sees this as a static margin of ~160 %, not 10 %. The tail must
generate a large down-force to trim, which is physically wasteful. An attempt to correct this by
using Flow5's own XNP value (parsed from the CSV header) placed the CG too far aft (~3 chord
lengths aft of wing LE), requiring ~14° of tail incidence to trim — impossible within the ±8°
hardware limit. Every evaluation returned a penalty; the optimiser had no signal.

LLT CG was therefore kept as the operating point for trim. The designs are stable and trimmed,
but more stable than strictly intended.

### Problem 3 — Newton trim loop sign error

Both the existing GUI verify code (`drone_mdo_gui.py`) and the first BO prototype had the Newton
step sign wrong:

```python
# Wrong: tail moves in the wrong direction — diverges to ±8° clamp
delta = degrees(-CM / (CLa_tail * SHT_frac)) * 0.6
```

When `CM > 0` (nose-up), a negative delta drives the tail more negative, which *increases* Cm
(empirically: tail −3.8° → Cm = 0.88; tail −8° → Cm = 2.0). The loop always clamped to −8° and
never converged.

At tail = −8° with Cm ≈ 2.0, Flow5's Trefftz-plane wake integration returns `CDi < 0` (a
numerical artefact from wake cancellation between wing and tail at extreme incidence). This
divided CL by a tiny denominator and produced phantom L/D values of 69 and 169 in early tests.

**Fix applied:**

```python
# Correct: positive Cm → positive tail → upward tail lift → nose-down → Cm falls
delta = degrees(+CM / (CLa_tail * VHT)) * 0.8
# where VHT = SHT_frac × tail_arm_chords  (dimensionless tail volume coefficient)
```

Using `SHT_frac` alone (≈ 0.22) instead of `VHT` (≈ 2.3) gave a step 10× too large (numerical
instability). The correct denominator is the full tail volume coefficient.

A CDi < 0 guard was also added (`if CDi_total < 0: return None, None`) to protect the GP
surrogate from any residual Trefftz artefacts.

### Problem 4 — Sequential phase splitting misses cross-coupling

With the SLSQP approach, wing variables are fixed before tail variables are optimised. The true
optimum depends on both simultaneously: a longer tail arm changes the wing aspect ratio that is
worth carrying; a different wing chord changes what SHT_frac means in absolute area. The
sequential split consistently left performance on the table compared to joint optimisation.

---

## 4. Phase 3 — new approach: Gaussian Process Bayesian Optimisation with Flow5

### Forward model: Flow5 TRIUNIFORM (inviscid VLM)

Flow5 runs a full 3-D vortex-lattice solution over the wing + tail panel mesh. CDi is computed
via Trefftz-plane far-field wake integration, which correctly captures span-loading interactions
between wing and tail that LLT's elliptic approximation cannot.

- **CDi**: accurate 3-D far-field integration (~1 % accuracy for AR > 5)
- **Cm**: full pitching moment about the user-specified CG — used directly in the Newton trim loop
- **CD0**: still from LLT friction formulas (Flow5 inviscid only); this is the remaining
  analytical component
- **Speed**: ~1 s/call (TRIUNIFORM inviscid); total budget at 80 calls ≈ 7 minutes

### Optimiser: Gaussian Process BO (`scikit-optimize gp_minimize`)

`flow5_bo_optimizer.py → run_flow5_bo()`.

- **Surrogate**: Gaussian Process with Matérn kernel; models the objective as a smooth function
  and quantifies uncertainty
- **Acquisition**: Expected Improvement (EI) — balances exploration (high uncertainty) with
  exploitation (low predicted objective)
- **Design space**: 11 variables jointly (AR, taper, twist, sweep, dihedral, SHT_frac, AR_HT,
  taper_HT, tail_arm, fuse_len, fuse_fin)
- **Calls**: 80 total; first ~12 are random exploration, remainder are GP-guided
- **Gradient-free**: BO only evaluates the objective at chosen points — no finite differences,
  no gradient noise, no minimum-step-size dilemma
- **Warm-start**: the LLT optimum is passed as an initial point so the GP has a physically
  informed starting observation

### Newton trim loop

Each BO evaluation runs up to 5 Flow5 calls internally:

```
tail_inc_0 = LLT trim angle  (from compute_trimmed_ld — fast, < 1 ms)
for each Newton step:
    write_plane_xml(tail_inc)      → Flow5 geometry
    run_flow5(TRIUNIFORM)          → polar CSV
    extract Cm at CL_cruise
    if |Cm| < 0.005: done
    delta = +Cm / (CLa_tail × VHT) × 0.8   [correct sign]
    tail_inc ← clip(tail_inc + delta, −8°, +8°)
return CDi_Flow5 + CD0_LLT,  tail_inc_final
```

The loop converges in 2–3 steps for most geometries and is physically validated:
base geometry trims to tail = +5.8°, Cm ≈ 0, hybrid L/D ≈ 17.5 (consistent with LLT).

---

## 5. What the new approach brings

| Aspect | Phase 1: LLT + SLSQP | Phase 2: Flow5 + Simplex | Phase 3: Flow5 + GP BO |
|---|---|---|---|
| **Forward model** | Analytical LLT (2-D) | Flow5 TRIUNIFORM (3-D VLM) | Flow5 TRIUNIFORM (3-D VLM) |
| **CDi accuracy** | Elliptic approx (±10 %) | Trefftz far-field (~1 %) | Trefftz far-field (~1 %) |
| **Trim** | Closed-form one-shot | Newton iteration | Newton iteration |
| **Variables** | 5 wing then 6 tail (split) | 3 (wing only, rest fixed) | 11 jointly |
| **Optimiser** | SLSQP (gradient-based) | Nelder-Mead (gradient-free) | GP surrogate + EI (gradient-free) |
| **Gradient required** | Yes — noisy, breaks with Flow5 | No | No |
| **Surrogate** | No | No | Yes — uncertainty + EI |
| **Calls/run** | ~4 000 LLT (< 1 s total) | ~80 Flow5 (1–2 min) | ~80 Flow5 (~7 min with trim) |
| **Phantom L/D** | Impossible (analytical) | Eliminated by CDi guard | Eliminated by CDi guard |
| **Seed sensitivity** | Deterministic | N/A (single run) | Spread 2.68 at 80 calls |
| **Design validity** | Trimmed for LLT, not Flow5 | Trimmed for Flow5 | Trimmed for Flow5 (Cm ≈ 0) |
| **Cross-coupling** | Wing fixes before tail | Wing only (tail ignored) | Full joint optimisation |

### Key gains (Phase 3 over Phase 2)

1. **Physically trimmed designs** — every accepted evaluation has `|Cm| < 0.005` as measured by
   Flow5's own pitching moment, not an analytical approximation.

2. **No phantom L/D values** — the CDi < 0 guard and correct Newton sign together eliminate the
   spurious 69× / 169× outliers that contaminated early test runs.

3. **Accurate induced drag** — Flow5 Trefftz CDi correctly accounts for wing–tail interaction,
   where LLT's elliptic assumption overestimates CDi for designs with large tail volume.

4. **Robust across 20 diverse requirements** — stress test (15 requirement sweep + 5 seed sweep)
   passed 20/20 cases at 30 calls and 20/20 at 80 calls with no phantom values.

5. **Joint optimisation** — the optimizer can trade AR against tail arm, or SHT_frac against
   twist, in a single pass rather than fixing one before tuning the other.

### Known remaining limitations

1. **LLT CG** — CG is still placed by LLT at NP_LLT − 10 % MAC. Flow5's true NP is ~2× further
   aft, so the real static margin is ~160 % rather than 10 %. Designs are over-stable and
   the tail carries more load than an optimally placed CG would require.

2. **CD0 is still analytical** — profile drag uses LLT friction formulas with fixed
   `CD0_wing = 0.008`. A viscous BO with Flow5 QUADS + XFoil would give the full coupled
   viscous-inviscid drag polar but at ~15–25 s/call (20× slower).

3. **Fuselage not in VLM** — Flow5 headless crashes with a body element, so the fuselage is
   modelled analytically (Hoerner CD0) only. It must be added manually in the GUI for viscous
   verification.

4. **Seed spread** — at 80 calls the inter-seed L/D spread is 2.68 (target < 1.0). The GP has
   not fully converged; approximately 150 calls would close this gap.

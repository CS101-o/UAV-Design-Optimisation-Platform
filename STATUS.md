# Flow5 BO Optimizer — Status

_Last updated: 2026-09-17_

---

## What has been achieved

### 1. Flow5 Bayesian Optimisation loop (`flow5_bo_optimizer.py`)

A new file replacing the SLSQP+LLT approach for the inner optimisation loop.

| Item | Detail |
|---|---|
| **Surrogate** | Gaussian Process (scikit-optimize `gp_minimize`) with EI acquisition |
| **Forward model** | Flow5 TRIUNIFORM (inviscid VLM, ~1 s/call) |
| **Trim** | Newton iteration on tail incidence until \|Cm\| < 0.005 |
| **Hybrid L/D** | CDi (Flow5 whole-plane Trefftz) + CD0_profile (LLT friction) |
| **Parallelisation** | Each BO case runs in its own subprocess with isolated XML/output dirs |

### 2. Newton trim step — correct formula

The original code (`drone_mdo_gui.py` and the early BO code) had the Newton step sign wrong.

```python
# Wrong (was):   tail moves in the wrong direction — diverges to ±8° clamp
delta = degrees(-CM / (CLa_tail * SHT_frac)) * 0.6

# Correct (now): positive Cm → more positive tail → nose-down → Cm falls
delta = degrees(+CM / (CLa_tail * VHT)) * 0.8
# where VHT = SHT_frac × tail_arm_chords  (tail volume coefficient)
```

The same fix was applied to `_verify_worker` in `drone_mdo_gui.py`.

### 3. CDi < 0 guard

When the tail clamps at ±8° (untrimmed), Flow5's Trefftz-plane wake integration
can return CDi < 0 — a numerical artefact producing phantom L/D values (169, 69
in early tests). The guard rejects these evaluations and returns penalty=500,
keeping the GP surrogate clean.

### 4. Stress test framework (`stress_test_bo.py`)

- 15 requirement-sweep cases: varied MTOW (1.5–5 kg), cruise speed (14–30 m/s), wing loading (80–160 N/m²)
- 5 seed-sweep cases: same base requirements, random_state 0–4
- 4 parallel workers, CSV output per run

### 5. Stress test results

| Run | n_calls | Fix applied | Pass | Phantom L/D | Seed spread |
|-----|---------|------------|------|-------------|-------------|
| Test 1 | 30 | None | 20/20 | **169, 69** | 151 |
| Test 2 | 30 | Cm > 0.3 guard | 18/20 | No | — (hangs) |
| Test 3 | 30 | CDi < 0 guard | 20/20 | No | 7.4 |
| **Test 4** | **30** | **Correct Newton + CDi guard** | **20/20** | **No** | **2.73** |
| **Test 5** | **80** | **Correct Newton + CDi guard** | **20/20** | **No** | **2.68** |

Test 4 is the baseline with correct trimming. Test 5 (80 calls) found better optima
for some cases (e.g. `large_slow`: 26.4 → 35.9 L/D) but barely changed seed spread.

---

## What has NOT been achieved

### 1. GUI button for BO (`drone_mdo_gui.py`)

The plan called for a "▶ Run BO (Flow5)" button alongside the existing Run button.
The `run_flow5_bo()` entry point exists and works from the command line, but it
is **not yet wired into the GUI**.

### 2. Static margin in Flow5's frame

The CG is auto-placed by LLT at NP_LLT − 10 % MAC. However, Flow5's own NP
(from the polar Cm curve) is systematically ~2× further aft than LLT's NP for
geometries with large tail volume (long arm + large SHT). The aircraft is
correctly trimmed (Cm ≈ 0) but the true SM in Flow5's frame is ~160 %, not 10 %.

**Impact:** trim is valid; L/D is physically correct; but the design is much more
stable than intended and the tail works harder than it needs to.

**Why the XNP correction was removed:** setting CG = XNP_Flow5 − 10 % MAC
placed the CG ~3 chord lengths aft of the wing LE. The required tail angle to
trim this became ~14 °, impossible within the ±8 ° limit. Every evaluation
returned penalty=500 → GP signal starvation.

### 3. Seed spread below 1.0

The stress-test threshold is 1.0 L/D unit. At 80 calls the spread is 2.68,
meaning different random seeds still find slightly different optima. The design
space has multiple local optima of similar quality; GP convergence is good
but not yet at the plateau. Increasing to ~150 calls would likely close
the remaining gap but at ~3× the wall time per case.

### 4. Viscous BO run

The BO uses TRIUNIFORM (inviscid). CD0 is from LLT friction formulas with a
fixed CD0_wing = 0.008. A viscous BO using Flow5 QUADS + XFoil would give a
more accurate drag polar but takes ~15–25 s/evaluation instead of ~1 s —
approximately 20× slower.

### 5. End-to-end GUI verify (option 3)

After the BO finds a best design, the plan was to run the viscous QUADS
verify step in the GUI to confirm L/D at high fidelity. This has not been
done yet.

### 6. Fuselage in headless Flow5

Flow5 crashes headless with a body/fuselage element (`Assertion failed:
m_SignedArea > 0.0`). The fuselage is modelled analytically (Hoerner CD0
formula) but not in the 3D VLM. Must still be added manually in the GUI.

---

## Next steps (suggested order)

1. **Wire BO into GUI** — add "▶ Run BO" button, `_bo_worker` thread, progress callback
2. **End-to-end verify** — take best BO result, run viscous QUADS verify in GUI
3. **SM constraint** — add a Flow5-aware SM guard (reject evaluations where
   Flow5-measured SM < 5 % or > 30 %) so the optimizer targets realistic stability
4. **More BO calls** — go to 150 calls if seed spread < 1.0 is required
5. **Viscous BO** — run QUADS+XFoil in the BO loop for final high-fidelity sweep

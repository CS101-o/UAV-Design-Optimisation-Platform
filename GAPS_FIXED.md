# Gaps Fixed — Flow5 Drone/UAV BO Pipeline

_Last updated: 2026-09-18_

This log records gaps identified against the two PDF parameter references
(`aircraft_wing_aerodynamic_variables.pdf`, `aircraft_tail_aerodynamic_variables.pdf`)
and against the BO objective's own constraint coverage, and how each was closed.
Scope throughout: the drone/UAV Flow5 BO pipeline (`flow5_bo_optimizer.py`,
`drone_mdo_gui.py`, `analysis.py`, `optimizer.py`), with tail layout, airfoil
choice, and high-lift/control-surface *type* kept as fixed configuration
choices — only continuous parameters within that configuration were added.

---

## Gap 0 — BO objective had no stall/trim/stability penalties

**Problem:** `flow5_bo_objective()` in `flow5_bo_optimizer.py` only enforced
two things: a hard tail-arm-vs-fuselage-length constraint, and a guard against
negative Trefftz-plane CDi artefacts. Once Flow5's Newton loop converged
`|Cm| < 0.005`, the design was accepted — even if it operated close to stall,
used nearly all available tail authority, or was dynamically/laterally
unstable. The SLSQP path (`optimizer.py`) already had these checks; the BO
path did not.

**Fix:** Added the same soft penalties to `flow5_bo_objective()`:

```python
penalty = (_stall_penalty(llt['stall_ratio']) +
           _trim_penalty(final_tail_inc) +
           _landing_penalty(llt['V_stall_land'], req.V_stall_land_ms) +
           dynamic_stability.optimizer_penalty(llt['dyn_modes']) +
           lateral_stability.optimizer_penalty(llt['Cn_beta'], llt['Cl_beta'], req))
return -LD + penalty
```

This also surfaced that `dynamic_stability.py` (phugoid/short-period modes)
existed but was never called anywhere in the codebase — dead code. It is now
wired into `analysis.py:compute_trimmed_ld()`, both SLSQP phases, the BO
objective, and the GUI banner/log.

**Important clarification:** this does not change *whether* a design is
trimmed. The Newton trim loop runs to convergence before any penalty is
computed — every accepted design still has `|Cm| < 0.005` from Flow5 itself.
The penalties only change which trimmed design the GP surrogate prefers,
steering it away from ones with no stall margin, no tail authority left, or
poor stability, none of which affects trim.

**Files:** `flow5_bo_optimizer.py`, `analysis.py`, `optimizer.py`,
`lateral_stability.py` (new), `drone_mdo_gui.py`.

---

## Gap 1 — Anhedral not reachable

**Problem:** `WING_VAR_SPECS['dihedral']` bounds were `(0.0, 5.0)` — the
optimizer could never explore negative (anhedral) values, even though the
wing-tail interaction and lateral stability model would be able to judge it.

**Fix:** Bounds widened to `(-3.0, 5.0)`. No physics changes needed — Cl_β
(from `lateral_stability.dihedral_cl_beta`) already responds correctly to
negative dihedral (verified: dihedral=−2° → Cl_β=+0.043/rad, i.e. correctly
flips toward destabilizing as dihedral goes negative).

**File:** `optimizer.py` (`WING_VAR_SPECS`).

---

## Gap 2 — Wash-in not reachable

**Problem:** `WING_VAR_SPECS['twist']` bounds were `(0.5, 4.0)` — washout
only. Wash-in (negative twist, raising tip AoA) was excluded by the bounds
before the stall check ever got a chance to judge it.

**Fix:** Bounds widened to `(-2.0, 4.0)`. Also fixed a latent sign issue this
exposed: the wing-phase objective's washout penalty had a linear term
`0.02 * twist` meant to mildly discourage excessive washout — with twist now
able to go negative, that term would have *rewarded* wash-in instead. Changed
to `0.02 * max(0.0, twist)` so it only ever discourages washout, never
rewards wash-in. Wash-in itself is discouraged by the existing stall-ratio
penalty (`wing_CLmax`'s critical-section method already penalizes designs
that bring tip stall onset earlier), so no new penalty term was needed.

**File:** `optimizer.py` (`WING_VAR_SPECS`, `optimize_wing`'s `penalty_washout`).

---

## Gap 3 & 4 — HT twist and HT dihedral not modeled

**Problem:** The horizontal tail had no twist or dihedral degrees of freedom
at all — `write_plane_xml()` hardcoded `<Twist>0.000</Twist>` and
`<Dihedral>0.000</Dihedral>` on both HT sections.

**Fix:** Added `HT_twist_deg` and `HT_dihedral_deg` as real parameters
through the full chain:
- `analysis.py:compute_trimmed_ld()` — new kwargs, returned in the result dict.
- `optimizer.py` — new `TAIL_VAR_SPECS` entries (`HT_twist` 0–3°,
  `HT_dihedral` ±10°), threaded through both `optimize_wing`'s tail-fixed
  kwargs and `optimize_tail`'s `_params`.
- `flow5_xml.py` — HT tip section's `<Twist>` now `{-HT_twist_deg}`
  (matches the wing's washout convention), both HT sections' `<Dihedral>`
  now `{HT_dihedral_deg}`.
- `flow5_bo_optimizer.py` — new BO variables, `_key_map` entries.
- `drone_mdo_gui.py` — auto-appear in the TAIL variables panel (generic
  section renderer); `tail_init` dict, cur-value labels, and BO-result
  reconstruction updated.

**Caveat, stated plainly:** like wing sweep and HT sweep before it, neither
HT twist nor HT dihedral has any effect in the LLT forward model (LLT treats
the tail as a single lifting surface with one lift-curve slope, not a
spanwise strip theory the way the wing is). Their aerodynamic effect only
appears once Flow5's VLM analyzes the actual 3-D geometry. The SLSQP/LLT
phase will therefore leave them at their initial value (zero gradient
signal); only the BO/Flow5 path can meaningfully move them. This is
documented in `optimizer.py`'s module docstring.

**Dimensionality note:** this raises the joint BO design space from 15 to 17
variables. STATUS.md already flags seed-spread convergence problems at 11
variables and 80 calls; adding more variables with no LLT gradient signal
makes that worse, not better. If BO convergence quality matters more than
completeness for a given run, uncheck `HT_twist`/`HT_dihedral` in the GUI
variables panel (they default to unchecked-safe values of 0°) rather than
running the full set every time.

**Files:** `analysis.py`, `optimizer.py`, `flow5_xml.py`,
`flow5_bo_optimizer.py`, `drone_mdo_gui.py`.

---

## Gap 5 — VT sweep not modeled

**Problem:** The fin/vertical-tail geometry had no sweep — `xoff_VT_tip`
in `flow5_xml.py` was purely the taper-driven quarter-chord offset.

**Fix:** Added `VT_sweep_deg` the same way as HT sweep, with one geometric
correction: the fin is a single non-mirrored surface spanning its full
height `b_VT` (unlike the symmetric wing/HT which mirror about the
centerline and use `b/2`), so the sweep offset uses the *full* `b_VT`:

```python
xoff_VT_tip = (cr_VT - ct_VT) / 4.0 + b_VT * tan(radians(VT_sweep_deg))
```

New `VT_VAR_SPECS` entry (`VT_sweep`, 0–30°, x0=10°, a representative
vertical-tail sweep for rudder effectiveness/ground clearance). Same LLT
caveat as Gap 3/4 — no effect until Flow5 VLM sees the real geometry.

**Files:** `analysis.py`, `optimizer.py`, `flow5_xml.py`,
`flow5_bo_optimizer.py`, `drone_mdo_gui.py`.

---

## Gap 6 — Flaps/ailerons had zero performance requirement to optimize against

**Problem:** `control_surfaces()` computed fixed flap/aileron sizing
fractions (15% S, 25% c̄, etc.) purely for display — nothing in the
objective ever used them, and there was no landing/short-field requirement
in `DroneRequirements` for them to satisfy. The PDF's high-lift-device
section was entirely unaddressed.

**Scope decision:** flap *type* (plain flap) stays a fixed configuration
choice, consistent with earlier scoping. Flap chord/span also stay at their
existing fixed fractions — redesigning flap geometry itself was judged out
of proportion to "give flaps a requirement to satisfy." The one variable
added is **flap deflection**, which is the parameter that actually trades
against a landing-speed requirement.

**Fix:**
1. New requirement field `req.V_stall_land_ms` (default 8.0 m/s) — a
   distinct, flaps-out landing target separate from the existing clean-
   configuration `stall_speed_ms` check. Exposed in the GUI mission panel.
2. New function `analysis.py:landing_stall_check()`:
   ```python
   ΔCLmax = k_flap × (S_flap / S) × sin(δ_flap)     # k_flap = 0.9 (plain flap,
                                                      #  Raymer "Aircraft Design")
   CL_max_land = CL_max_clean_3d + ΔCLmax
   V_stall_land = sqrt(2W / (ρ S CL_max_land))
   ```
   Flaps are assumed fully retracted (zero effect) at the cruise point
   computed everywhere else in `compute_trimmed_ld()` — only this separate
   landing check credits them.
3. New design variable `flap_deflection` (`WING_VAR_SPECS`, 0–40°, x0=15°)
   and new penalty `_landing_penalty()`, wired into both SLSQP phases and
   the BO objective.

**Verified:** flap deflection measurably lowers landing stall speed
(0° → 13.66 m/s, 20° → 13.37 m/s in a smoke test at fixed wing geometry);
the SLSQP wing phase drives `flap_deflection` toward its upper bound (40°)
when the landing-speed penalty is active, exactly as expected — more flap
credit relaxes the landing constraint.

**Aileron sizing was left as-is (not "closed").** There is no roll-rate,
roll-authority, or crosswind-landing requirement anywhere in the mission
specification for an aileron design variable to be judged against.
Fabricating one without the user specifying an actual roll requirement would
be scope creep, not a gap closure — unlike flaps, where a stall-speed-type
requirement already existed for the variable to plug into. If a roll
requirement is added to the mission (e.g. "bank to 30° within 1.5 s"),
aileron area/chord/span should follow the same pattern as flap deflection.

**Files:** `requirements.py`, `analysis.py`, `optimizer.py`,
`flow5_bo_optimizer.py`, `drone_mdo_gui.py`.

---

## Summary of files touched (all 6 gaps + Gap 0)

| File | Change |
|---|---|
| `requirements.py` | + `V_stall_land_ms` |
| `analysis.py` | + `landing_stall_check()`, new `compute_trimmed_ld` kwargs (`HT_twist_deg`, `HT_dihedral_deg`, `VT_sweep_deg`, `flap_deflection_deg`), new returned fields |
| `optimizer.py` | Widened dihedral/twist bounds; + `flap_deflection`, `HT_twist`, `HT_dihedral`, `VT_sweep` specs; + `_landing_penalty`; fixed washout-penalty sign; threaded all new params through both SLSQP phases and `result_to_geom` |
| `flow5_xml.py` | HT `<Twist>`/`<Dihedral>` now driven by geometry dict; VT sweep added to fin tip offset (full `b_VT`, not half) |
| `flow5_bo_optimizer.py` | New BO variables, `_key_map` entries, landing/stall/trim/dynamic/lateral penalties in the objective |
| `drone_mdo_gui.py` | New mission field (landing stall speed); new variables auto-appear via existing generic section renderer; `tail_init`, cur-labels, BO-result reconstruction, and ok/pass-fail displays updated |
| `lateral_stability.py` | New module (Gap 0 dependency): Cn_β weathercock stability, Cl_β dihedral effect |

## Verification

All changes were verified with direct functional smoke tests (not just
compilation): `compute_trimmed_ld()` with anhedral/wash-in/HT-twist/HT-
dihedral/VT-sweep/flap-deflection all supplied; `write_plane_xml()` emitting
the correct `<Twist>`/`<Dihedral>`/sweep-offset values; both SLSQP phases
(`optimize_wing`, `optimize_tail`) running with the new variables active; and
the BO objective executing cleanly across the full 20-variable joint space
with Flow5 mocked out. No Flow5-app-dependent behavior (the Newton trim loop
itself) could be exercised in this environment and should be spot-checked
with a real BO run before relying on it for a final design.

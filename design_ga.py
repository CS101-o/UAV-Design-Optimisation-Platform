"""
Conceptual Design: 4-seat GA aircraft (Cessna 172 replacement)
Uses the existing LLT analysis + optimizer code.
Run: python3 design_ga.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from requirements import DroneRequirements
from analysis import (CL_required, stall_check, static_margin, pitch_trim,
                      control_surfaces)
from optimizer import optimize

SEP = "─" * 62

# ══════════════════════════════════════════════════════════════
# PART 1 — REQUIREMENTS
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 1  DESIGN REQUIREMENTS")
print(SEP)

V_cruise_ms  = 135 * 0.514444          # 135 kts → m/s = 69.45 m/s
V_max_ms     = 160 * 0.514444          # 160 kts → m/s
V_stall_ms   = 50  * 0.514444          # 50 kts  → m/s  (landing, with flaps)
ROC_req_ms   = 800 * 0.00508           # 800 ft/min → m/s = 4.064 m/s
h_ceiling_ft = 14000                   # ft
h_ceiling_m  = h_ceiling_ft * 0.3048  # m
BFL_m        = 500                     # balanced field length, m
LD_m         = 450                     # landing distance, m
range_km     = 1200                    # km
endurance_hr = 5.0                     # hours cruise
reserve_hr   = 0.75                    # 45 min reserve
W_pax        = 85.0                    # kg per person (pilot included)
n_pax        = 4
W_bag_each   = 20.0                    # kg per passenger
Wpay_kg      = 400.0   # kg  (4×85 pax + pilot + 3×20 bags = 400 as stated)

print(f"  Cruise:      {V_cruise_ms:.1f} m/s  ({135} kts)")
print(f"  Max speed:   {V_max_ms:.1f} m/s  ({160} kts)")
print(f"  Stall speed: ≤ {V_stall_ms:.1f} m/s  ({50} kts, landing config)")
print(f"  ROC:         ≥ {ROC_req_ms:.2f} m/s  ({800} ft/min)")
print(f"  Service ceil:{h_ceiling_ft} ft  ({h_ceiling_m:.0f} m)")
print(f"  Range:       {range_km} km  |  Endurance: {endurance_hr} hr + {reserve_hr*60:.0f} min res.")
print(f"  Payload:     {Wpay_kg:.0f} kg  "
      f"({n_pax}×{W_pax}kg pax + {n_pax}×{W_bag_each}kg bags)")
print()

# ══════════════════════════════════════════════════════════════
# PART 2 — PRELIMINARY WEIGHT ESTIMATION  (Raymer empirical)
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 2  PRELIMINARY WEIGHT ESTIMATION")
print(SEP)

# Aerodynamic and engine assumptions for weight fractions
BSFC_imp = 0.42          # lb/hp/hr  (modern fuel-injected piston, e.g. Lycoming IO-360)
eta_p    = 0.82          # propeller efficiency
LD_cr    = 13.5          # L/D at cruise (initial guess, refined after wing design)
LD_lt    = 15.0          # L/D at loiter (near L/D_max)

R_mi  = range_km / 1.60934    # km → statute miles (745.6 mi)
V_mph = V_cruise_ms * 2.23694 # m/s → mph

# Breguet mission weight fractions (English-unit form)
frac = {}
frac['start']   = 0.990
frac['taxi']    = 0.990
frac['takeoff'] = 0.995
frac['climb']   = 0.980
frac['cruise']  = float(np.exp(-R_mi * BSFC_imp / (375 * eta_p * LD_cr)))
frac['loiter']  = float(np.exp(-reserve_hr * BSFC_imp * V_mph * 0.75
                               / (375 * eta_p * LD_lt)))
frac['descent'] = 0.995
frac['land']    = 0.990

Mff = 1.0
for k, v in frac.items():
    Mff *= v

Wf_frac = 1.06 * (1.0 - Mff)   # +6% trapped fuel/oil

# Raymer regression: We/W0 = A × W0^C  (single-engine piston GA, English units)
A_ray, C_ray = 2.36, -0.18
Wpay_lb = Wpay_kg * 2.20462

W0 = 3000.0   # initial guess lb
for _ in range(40):
    We_frac = A_ray * (W0 ** C_ray)
    W0_new  = Wpay_lb / (1.0 - Wf_frac - We_frac)
    if abs(W0_new - W0) < 0.2:
        break
    W0 = 0.5 * W0 + 0.5 * W0_new

MTOW_kg = W0 / 2.20462
We_kg   = MTOW_kg * We_frac
Wf_kg   = MTOW_kg * Wf_frac

print(f"  Cruise fuel fraction:   {1-frac['cruise']:.4f}")
print(f"  Reserve loiter fraction:{1-frac['loiter']:.4f}")
print(f"  Mission fuel fraction:  {Wf_frac:.4f}  (incl. +6% trapped)")
print()
print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │  MTOW:         {MTOW_kg:6.0f} kg  ({MTOW_kg*2.20462:6.0f} lb)       │")
print(f"  │  Empty weight: {We_kg:6.0f} kg  We/W0 = {We_frac:.3f}          │")
print(f"  │  Fuel weight:  {Wf_kg:6.0f} kg  Wf/W0 = {Wf_frac:.3f}          │")
print(f"  │  Payload:      {Wpay_kg:6.0f} kg  Wp/W0 = {Wpay_kg/MTOW_kg:.3f}          │")
print(f"  │  Sum check:    {We_kg+Wf_kg+Wpay_kg:6.1f} kg  (Δ = {We_kg+Wf_kg+Wpay_kg-MTOW_kg:+.1f} kg)   │")
print(f"  └─────────────────────────────────────────────────┘")
print()

# ══════════════════════════════════════════════════════════════
# PART 3 — CONSTRAINT ANALYSIS  (T/W vs W/S → P/W vs W/S)
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 3  CONSTRAINT ANALYSIS")
print(SEP)

rho0  = 1.225   # kg/m³ sea level ISA
g     = 9.81    # m/s²

# ── (a) Stall speed → W/S ceiling ──────────────────────────
CLmax_land  = 2.10   # with full flaps (plain/slotted)
CLmax_TO    = 1.70   # takeoff flap setting
CLmax_clean = 1.55   # clean, NACA 2412-like airfoil

WS_stall = 0.5 * rho0 * V_stall_ms**2 * CLmax_land
print(f"  (a) Stall  → W/S ≤ {WS_stall:.0f} N/m²  "
      f"(CL_max_land = {CLmax_land})")

# ── (b) Takeoff ≤ 500 m over 50 ft ─────────────────────────
# Raymer prop-aircraft BFL formula (over 50-ft obstacle):
# BFL [ft] = k_prop × (W/S)[lb/ft²] × (W/BHP)[lb/hp] / CLmax_TO
# k_prop ≈ 9.0  (empirical constant for GA piston, BFL over 50 ft)
k_prop    = 9.0
BFL_ft    = BFL_m * 3.28084
WS_arr    = np.linspace(300, 900, 100)   # N/m²
WS_lb_ft2 = WS_arr * 0.020885           # N/m² → lb/ft²
# BHP/W [hp/lb] = k_prop × W/S / (BFL × CLmax_TO)
PW_TO_hplb = k_prop * WS_lb_ft2 / (BFL_ft * 1.0 * CLmax_TO)   # σ=1.0 SL
PW_TO_Wkg  = PW_TO_hplb * 745.7 * 2.20462     # hp/lb → W/kg

print(f"  (b) Takeoff → k_prop = {k_prop}, BFL = {BFL_ft:.0f} ft")

# ── (c) ROC ≥ 800 ft/min at sea level ──────────────────────
# RC = η_p × P/W - V_climb/L/D_climb
V_climb  = 1.32 * (V_stall_ms / np.sqrt(CLmax_clean / CLmax_clean))  # 1.32 Vs_clean
LD_climb = 11.0
PW_ROC_Wkg = (ROC_req_ms + V_climb / LD_climb) / eta_p * g  # W/kg

print(f"  (c) ROC    → P/W ≥ {PW_ROC_Wkg:.1f} W/kg  "
      f"(V_climb = {V_climb:.1f} m/s, L/D = {LD_climb})")

# ── (d) Service ceiling 14,000 ft (ROC = 100 ft/min) ───────
# ISA at 14,000 ft
h_m    = h_ceiling_m
T_sl   = 288.15; L = 0.0065; R = 287.0
T_alt  = T_sl - L * h_m
rho_alt = rho0 * (T_alt / T_sl) ** (g / (L * R) - 1)
sigma  = rho_alt / rho0

# Gagg-Ferrar power lapse for normally-aspirated piston
P_lapse = (sigma - 0.132) / (1.0 - 0.132)

# TAS at ceiling: V_IAS_stall × 1.3 / sqrt(sigma)
V_ceil  = V_stall_ms * 1.3 / np.sqrt(sigma)  # TAS m/s
LD_ceil = LD_lt
ROC_ceil_ms = 100 * 0.00508  # 100 ft/min

PW_ceil_Wkg = (ROC_ceil_ms + V_ceil / LD_ceil) / (eta_p * P_lapse) * g

print(f"  (d) Ceiling → P/W ≥ {PW_ceil_Wkg:.1f} W/kg  "
      f"(σ = {sigma:.3f}, P_lapse = {P_lapse:.3f})")

# ── Design point ────────────────────────────────────────────
# W/S = stall constraint (leave 50 N/m² margin below ceiling)
WS_design = min(WS_stall - 50, 800)   # N/m²
PW_design = max(PW_ROC_Wkg, PW_ceil_Wkg) * 1.10   # +10% margin

S_design  = MTOW_kg * g / WS_design    # m²
P_design  = PW_design * MTOW_kg / 1000  # kW
P_hp      = P_design / 0.7457

# Also check takeoff constraint at design W/S
WS_des_lb     = WS_design * 0.020885
PW_TO_hp      = k_prop * WS_des_lb / (BFL_ft * CLmax_TO)   # hp/lb
PW_TO_Wkg_des = PW_TO_hp * 745.7 * 2.20462
if PW_TO_Wkg_des > PW_design / 1.10:
    PW_design = PW_TO_Wkg_des * 1.10
    P_design  = PW_design * MTOW_kg / 1000
    P_hp      = P_design / 0.7457

print()
print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │  Design point:                                  │")
print(f"  │  W/S = {WS_design:5.0f} N/m²  →  S = {S_design:.2f} m²            │")
print(f"  │  P/W = {PW_design:5.1f} W/kg  →  P = {P_design:.0f} kW  ({P_hp:.0f} hp)   │")
print(f"  └─────────────────────────────────────────────────┘")
print()

# ══════════════════════════════════════════════════════════════
# PART 4 — REFINED WEIGHT ANALYSIS
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 4  REFINED WEIGHT ANALYSIS")
print(SEP)

# Refined Raymer formula with aerodynamic/structural parameters
# We/W0 = A × W0^B × AR^C × (V_max_kts)^D × (W0/S)^E × (P/W0)^F
# For single-engine GA (Raymer DAPCA-like): use calibrated version
#   We/W0 = 1.19 × W0^(-0.09)  (alternative Raymer regression, slightly conservative)
#   Switch to Cessna family regression: a=2.36, c=-0.18 gives lighter results

# Use AR and V_max to refine via the extended Raymer formula:
# We/W0 = A × W0^C × kvs  (kvs = variable-sweep factor = 1.0 for fixed wing)
# For modern composite construction apply 10% reduction
AR_guess = 10.0  # preliminary
We_frac_ref = A_ray * (W0 ** C_ray) * 0.92   # -8% for composites vs all-metal

W0_ref = Wpay_lb / (1.0 - Wf_frac - We_frac_ref)
for _ in range(30):
    We_frac_ref = A_ray * (W0_ref ** C_ray) * 0.92
    W0_new = Wpay_lb / (1.0 - Wf_frac - We_frac_ref)
    if abs(W0_new - W0_ref) < 0.2:
        break
    W0_ref = 0.5 * W0_ref + 0.5 * W0_new

MTOW_ref_kg = W0_ref / 2.20462
We_ref_kg   = MTOW_ref_kg * We_frac_ref
Wf_ref_kg   = MTOW_ref_kg * Wf_frac

print(f"  Composite weight reduction (-8% on empty weight):")
print(f"  MTOW:         {MTOW_ref_kg:.0f} kg   (vs {MTOW_kg:.0f} kg all-metal)")
print(f"  Empty weight: {We_ref_kg:.0f} kg   We/W0 = {We_frac_ref:.3f}")
print(f"  Fuel weight:  {Wf_ref_kg:.0f} kg")
print()
print("  Adopting MTOW = " + f"{MTOW_ref_kg:.0f}" +
      " kg for further design.")
print()

MTOW_final = MTOW_ref_kg

# ══════════════════════════════════════════════════════════════
# PART 5 — WING DESIGN  (LLT OPTIMIZER)
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 5  WING DESIGN  (LLT optimizer)")
print(SEP)

# Wing area from refined constraint analysis
WS_final = WS_design
S_final  = MTOW_final * g / WS_final

req = DroneRequirements()

# Override with GA aircraft parameters
req.mtow_kg          = MTOW_final
req.cruise_speed_ms  = V_cruise_ms
req.wing_loading_Nm2 = WS_final
req.wing_area        = S_final
req.stall_speed_ms   = V_stall_ms / np.sqrt(CLmax_land / CLmax_clean)
                       # clean stall ≈ 50 kts × sqrt(2.1/1.55) = 58 kts

# NACA 2412 / similar low-camber GA airfoil
req.CL_alpha_2d   = 2 * np.pi          # /rad  (thin airfoil theory, matches NACA 2412 data)
req.alpha_L0_deg  = -2.0               # deg   (NACA 2412)
req.CM_ac         = -0.046             # NACA 2412 pitching moment at ¼ chord
req.CL_max_2d     = CLmax_clean        # 1.55 clean
req.CD0           = 0.028              # total parasite drag (retractable gear, modern GA)

# Configuration
req.cg_frac_mac   = 0.33               # CG target: 33% MAC (slightly fwd, good for students)
req.incidence_deg = 2.0                # wing incidence for level fuselage at cruise
req.sweep_deg     = 0.0                # straight wing (subsonic GA)
req.dihedral_deg  = 1.5               # high wing: low dihedral (pendulum stability)
req.wing_position = "high"

# Tail sizing
req.SHT_fraction    = 0.22
req.SVT_fraction    = 0.10
req.tail_arm_chords = 3.5              # LHT = 3.5 × c̄

# Optimization bounds
req.bounds = [
    (8.0, 12.0),   # AR:    high for efficiency, composite allows it
    (0.40, 0.70),  # taper: moderate taper → near-elliptical, good stall
    (2.0, 4.0),    # twist: washout for tip-stall protection
]
req.x0 = [10.0, 0.55, 2.5]

print(f"  Wing area:   S = {S_final:.2f} m²  "
      f"(W/S = {WS_final:.0f} N/m²)")
print(f"  CL_cruise:   {CL_required(req.mtow_kg, req.wing_area, req.cruise_speed_ms):.4f}")
print(f"  Airfoil:     NACA 2412 (α_L0 = {req.alpha_L0_deg}°, "
      f"CM_ac = {req.CM_ac})")
print(f"  CD0 total:   {req.CD0}  (retractable gear GA)")
print()
print("  Running SLSQP wing optimizer ...")
print()

result = optimize(req)

AR    = result['AR']
taper = result['taper']
twist = result['twist_deg']
b     = result['b']
cr    = result['cr']
ct    = result['ct']
e     = result['e']
LD    = result['LD']
CL_op = result['CL']
CDi   = result['CDi']
CD    = result['CD']
SM    = result['SM']

c_bar = b / AR
CL_req_val = CL_required(req.mtow_kg, req.wing_area, req.cruise_speed_ms)

print()
print(f"  ┌─────────────────────────────────────────────────┐")
print(f"  │  WING OPTIMUM                                   │")
print(f"  │  AR     = {AR:.2f}                               │")
print(f"  │  Taper  = {taper:.3f}                              │")
print(f"  │  Twist  = {twist:.2f}°  (geometric washout)         │")
print(f"  │  b      = {b:.3f} m  (span)                     │")
print(f"  │  c̄      = {c_bar:.3f} m  (mean chord)               │")
print(f"  │  c_root = {cr:.3f} m                              │")
print(f"  │  c_tip  = {ct:.3f} m                              │")
print(f"  │  e      = {e:.4f}  (Oswald span efficiency)       │")
print(f"  │  CDi    = {CDi:.5f}                             │")
print(f"  │  CD_tot = {CD:.5f}                             │")
print(f"  │  CL     = {CL_op:.4f}                             │")
print(f"  │  L/D    = {LD:.2f}                              │")
print(f"  └─────────────────────────────────────────────────┘")
print()

# ── Control surfaces ────────────────────────────────────────
cs = control_surfaces(req.wing_area, b, c_bar)
print(f"  Control surfaces:")
print(f"  Flap area:  {cs['flap_area']:.3f} m²  ({cs['flap_area']/req.wing_area*100:.0f}% S)")
print(f"  Flap chord: {cs['flap_chord']:.3f} m   ({cs['flap_chord']/c_bar*100:.0f}% c̄)")
print(f"  Flap span:  {cs['flap_span']:.3f} m   ({cs['flap_span']/(b/2)*100:.0f}% semi)")
print(f"  Aileron:    {cs['aileron_area']:.3f} m²  from {cs['aileron_span_in']:.3f}→{cs['aileron_span_out']:.3f} m semi")
print()

# ── Stall check (with flaps) ────────────────────────────────
V_stall_clean, sr, _ = stall_check(
    req.mtow_kg, req.wing_area, CLmax_clean, CL_req_val, req.stall_speed_ms)
V_stall_land  = float(np.sqrt(2 * req.mtow_kg * g /
                               (rho0 * req.wing_area * CLmax_land)))
print(f"  Stall speeds:")
print(f"  V_s (clean):   {V_stall_clean:.1f} m/s  = {V_stall_clean/0.514444:.1f} kts")
print(f"  V_s (landing): {V_stall_land:.1f} m/s  = {V_stall_land/0.514444:.1f} kts  ✓ < 50 kts")
print()

# ══════════════════════════════════════════════════════════════
# PART 6 — TAIL & STABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 6  TAIL DESIGN & STABILITY ANALYSIS")
print(SEP)

SM_val, x_NP, VHT, cg_min, cg_max = static_margin(
    AR, req.wing_area, req.SHT_fraction, req.tail_arm_chords,
    req.CL_alpha_2d, req.cg_frac_mac)

CM_cg, CL_tail, tail_inc, trimmable, _ = pitch_trim(
    AR, req.wing_area, req.SHT_fraction, req.tail_arm_chords,
    req.CL_alpha_2d, req.CM_ac, CL_req_val, req.cg_frac_mac)

# Tail dimensions
SHT = req.SHT_fraction * req.wing_area
LHT = req.tail_arm_chords * c_bar
AR_HT  = 4.5
b_HT   = np.sqrt(AR_HT * SHT)
cr_HT  = 2 * SHT / (b_HT * 1.4)
ct_HT  = 0.4 * cr_HT

SVT = req.SVT_fraction * req.wing_area
AR_VT  = 1.8
b_VT   = np.sqrt(AR_VT * SVT)
cr_VT  = 2 * SVT / (b_VT * 1.3)

print(f"  Tail configuration: Conventional (low H-stab)")
print(f"  H-stab: S_HT = {SHT:.2f} m²  b_HT = {b_HT:.2f} m")
print(f"          c_r  = {cr_HT:.3f} m   c_t  = {ct_HT:.3f} m  AR = {AR_HT}")
print(f"          L_HT = {LHT:.2f} m  (tail arm)")
print(f"  V-stab: S_VT = {SVT:.2f} m²  b_VT = {b_VT:.2f} m  AR = {AR_VT}")
print()
print(f"  Longitudinal stability:")
print(f"  Neutral point (x_NP): {x_NP*100:.1f}% MAC")
print(f"  CG design point:      {req.cg_frac_mac*100:.1f}% MAC")
print(f"  Static margin (SM):   {SM_val*100:.1f}% MAC  "
      f"({'✓ in 5–15%' if 0.05<=SM_val<=0.15 else '⚠ outside 5–15%'})")
print(f"  VHT (tail vol. coeff):{VHT:.3f}")
print(f"  CG range for SM 5–15%: {cg_min*100:.1f}%–{cg_max*100:.1f}% MAC")
print()
print(f"  Longitudinal trim (cruise):")
print(f"  CM_cg untrimmed: {CM_cg:.4f}")
print(f"  CL_tail needed:  {CL_tail:.4f}")
print(f"  Tail incidence:  {tail_inc:.2f}°  "
      f"({'✓ trimmable' if trimmable else '⚠ exceeds ±5°'})")
print()

# ══════════════════════════════════════════════════════════════
# PART 7 — FUSELAGE & LANDING GEAR
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 7  FUSELAGE & LANDING GEAR")
print(SEP)

# Fuselage: 4-seat side-by-side + tandem (2+2 seating typical for C172 replacement)
# Cross-section: elliptical cabin
w_cabin  = 1.22    # m  (4 ft inside width, 2 abreast)
h_cabin  = 1.25    # m  (headroom)
l_cabin  = 2.70    # m  (2 rows × 1.1 m pitch + 0.5 m floor/seat)
l_nose   = 1.70    # m  (engine bay + firewall)
l_tail   = 4.10    # m  (tail cone to H-stab leading edge, ~3.5× c̄)
l_fus    = l_nose + l_cabin + l_tail  # total fuselage length
d_fus    = max(w_cabin, h_cabin)      # max cross-section

slenderness = l_fus / d_fus

print(f"  Fuselage:")
print(f"  Length:        {l_fus:.2f} m  (nose {l_nose:.2f} + cabin {l_cabin:.2f} + tail {l_tail:.2f})")
print(f"  Max width:     {w_cabin:.2f} m  (cabin {w_cabin*1000:.0f} mm)")
print(f"  Max height:    {h_cabin:.2f} m")
print(f"  Slenderness:   L/D = {slenderness:.1f}  ({'✓' if 6.5<=slenderness<=12 else '⚠'})")
print()

# Landing gear: tricycle fixed (to keep the design simple for certification)
# Track width (60–70% wingspan for crosswind)
track  = 1.85 * w_cabin  # m  (high-wing: gear attaches to lower fuselage)
w_base = l_fus * 0.30  # wheel base ~ 30% fus length (main gear aft of CG)
d_mw   = 0.356  # 14-inch main wheel diameter (Cessna-class)
d_nw   = 0.254  # 10-inch nose wheel

print(f"  Landing gear:  Tricycle fixed")
print(f"  Main wheel track: {track:.2f} m  ({track/b*100:.0f}% span)")
print(f"  Wheel base:       {w_base:.2f} m")
print(f"  Main wheel dia:   {d_mw*1000:.0f} mm  ({d_mw/0.0254:.0f} in)")
print(f"  Nose wheel dia:   {d_nw*1000:.0f} mm  ({d_nw/0.0254:.0f} in)")
print()

# ══════════════════════════════════════════════════════════════
# PART 8 — PROPULSION
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 8  PROPULSION SELECTION")
print(SEP)

P_req_kW = P_design
P_req_hp  = P_req_kW / 0.7457

# Engine candidates for ~150-180 hp GA class:
engines = [
    ("Lycoming IO-360-A1B6",  180, 122.0, 0.43, "4-cyl, fuel-injected, 8.7:1 CR"),
    ("Continental IO-360-ES", 200, 120.0, 0.44, "6-cyl, fuel-injected, 8.5:1 CR"),
    ("Rotax 916 iS",          160,  67.0, 0.36, "4-cyl turbo, MOGAS/100LL, modern"),
    ("Lycoming IO-390-A3B6",  215, 127.0, 0.42, "4-cyl, fuel-injected, 8.9:1 CR"),
]
print(f"  Required power: ≥ {P_req_hp:.0f} hp  ({P_req_kW:.0f} kW)")
print()
print(f"  {'Engine':<30}  {'hp':>4}  {'kg':>4}  {'BSFC':>6}  Notes")
print(f"  {'─'*30}  {'─'*4}  {'─'*4}  {'─'*6}  {'─'*30}")
selected = None
for name, hp, kg, bsfc, notes in engines:
    flag = "← selected" if hp >= P_req_hp and selected is None else ""
    if flag:
        selected = (name, hp, kg, bsfc)
    print(f"  {name:<30}  {hp:4d}  {kg:4.0f}  {bsfc:6.3f}  {notes}  {flag}")

if selected is None:                          # pick highest-power option as fallback
    selected = max(engines, key=lambda e: e[1])
    print(f"  ⚠ No single engine meets {P_req_hp:.0f} hp — using highest available.")
print()
name_e, hp_e, kg_e, bsfc_e = selected
P_sel_kW = hp_e * 0.7457

# Propeller sizing
# Diameter: D ≈ k × (P[hp] / N[rpm])^(1/3) for GA piston (Raymer method)
# Lycoming IO-360 redline ≈ 2700 rpm, typical cruise 2400 rpm
N_rpm   = 2400.0     # cruise prop rpm
N_rps   = N_rpm / 60
# Regression on real GA piston props: D[m] ≈ 0.527 × P[hp]^0.25
# (Cessna 172 180 hp → 76 in = 1.93 m; Rotax 100 hp → 68 in; IO-540 260 hp → 80 in)
D_prop  = 0.527 * hp_e ** 0.25
D_prop  = min(D_prop, 2.03)            # structural/ground-clearance limit
D_prop  = round(D_prop / 0.0254) * 0.0254  # round to nearest inch

v_tip   = np.pi * D_prop * N_rps
print(f"  Selected:  {name_e}  {hp_e} hp  ({P_sel_kW:.0f} kW)")
print(f"  Engine mass: {kg_e:.0f} kg")
print()
print(f"  Propeller (constant-speed, 2-blade aluminum):")
print(f"  Diameter: {D_prop:.2f} m  ({D_prop/0.0254:.0f} in)")
print(f"  Cruise rpm: {N_rpm:.0f}  → tip speed: {v_tip:.0f} m/s  "
      f"({'✓' if v_tip < 210 else '⚠ tip Mach concern'})")
print()

# Power check
P_avail_kW = P_sel_kW * eta_p  # shaft → thrust power
P_need_kW  = MTOW_final * g * V_cruise_ms / (LD * 1000)   # drag power at cruise
print(f"  Power budget at cruise:")
print(f"  Available: {P_avail_kW:.0f} kW (shaft × η_p = {eta_p})")
print(f"  Required:  {P_need_kW:.0f} kW  (W·V / L/D)")
print(f"  Margin:    {(P_avail_kW/P_need_kW - 1)*100:.0f}%  "
      f"({'✓' if P_avail_kW > P_need_kW else '⚠'})")
print()

# ══════════════════════════════════════════════════════════════
# PART 9 — COST ESTIMATION  (DAPCA IV / Eastlake simplified)
# ══════════════════════════════════════════════════════════════
print(SEP)
print("PART 9  COST ESTIMATION (DAPCA IV / Eastlake)")
print(SEP)

# Parameters
W0_lb    = MTOW_final * 2.20462
We_lb    = We_ref_kg * 2.20462
V_max_kts = V_max_ms / 0.514444
Q5y      = 250   # production quantity over 5 years
Fcert    = 1.12  # certification overhead factor (Part 23)
r_eng    = 1     # number of engines
C_eng    = 40000 # USD per engine (IO-360 class new)
C_avion  = 25000 # USD (modern glass cockpit: Garmin G1000 class)
R_eng    = 86.0  # USD/hr engineering (2024)
R_tool   = 88.0  # USD/hr tooling
R_mfg    = 73.0  # USD/hr manufacturing
R_qc     = 66.0  # USD/hr quality control

# DAPCA IV CERs (Eastlake, AIAA 2009) — airframe engineering hours
H_e   = 0.0396 * We_lb**0.791 * V_max_kts**1.526 * Q5y**0.183
H_t   = 1.0279 * We_lb**0.764 * V_max_kts**0.899 * Q5y**0.178
H_m   = 9.6613 * We_lb**0.74  * V_max_kts**0.543 * Q5y**0.524
H_qc  = 0.076  * H_m

# Development and test (non-recurring)
C_fd   = 45.42  * We_lb**0.630 * V_max_kts**1.3
C_ft   = 1243.03 * We_lb**0.325 * V_max_kts**0.822 * r_eng**1.21

# Materials
C_mat  = 22.1   * We_lb**0.921 * V_max_kts**0.621 * Q5y**0.799

# Totals
C_labor = (H_e * R_eng + H_t * R_tool + H_m * R_mfg + H_qc * R_qc)
C_total_NRE = (C_fd + C_ft) * Fcert
C_recurring = (C_labor + C_mat + r_eng * C_eng * Q5y + C_avion * Q5y)
C_production = C_recurring * 1.18   # +18% profit/overhead
C_unit_fly   = (C_production + C_total_NRE) / Q5y
C_unit_final = C_unit_fly * 1.25    # +25% for direct sales, warranty, support

# Break-even
C_fix   = C_total_NRE + C_production * 0.05  # 5% of recurring as fixed
C_var   = C_unit_fly * 0.98
P_price = C_unit_final
N_be    = int(C_total_NRE / (P_price - C_var)) + 1

print(f"  Production quantity: {Q5y} units (5-year plan)")
print()
print(f"  Non-recurring costs:")
print(f"  Flights  / development:  ${C_fd/1e6:.2f}M")
print(f"  Flight test:             ${C_ft/1e6:.2f}M")
print(f"  Total NRE:               ${C_total_NRE/1e6:.2f}M")
print()
print(f"  Per-unit costs ({Q5y} units):")
print(f"  Airframe labor:          ${(H_e*R_eng+H_t*R_tool+H_m*R_mfg+H_qc*R_qc)/Q5y/1e3:.0f}k")
print(f"  Materials (airframe):    ${C_mat/Q5y/1e3:.0f}k")
print(f"  Engine ({r_eng}×):           ${r_eng*C_eng/1e3:.0f}k")
print(f"  Avionics:                ${C_avion/1e3:.0f}k")
print(f"  Unit flyaway cost:       ${C_unit_fly/1e3:.0f}k")
print(f"  Unit list price (est.):  ${C_unit_final/1e3:.0f}k")
print()
print(f"  Break-even:  ~{N_be} units")
print()

# ══════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════
print(SEP)
print("DESIGN SUMMARY")
print(SEP)
print(f"  Aircraft:       4-seat single-engine GA  (Cessna 172 successor)")
print(f"  Config:         High wing · Conventional tail · Tricycle fixed gear")
print(f"  Airfoil:        NACA 2412  (low-cambered, docile stall)")
print()
print(f"  WEIGHTS")
print(f"  MTOW:           {MTOW_final:.0f} kg  ({MTOW_final*2.20462:.0f} lb)")
print(f"  Empty weight:   {We_ref_kg:.0f} kg  (We/W0 = {We_frac_ref:.3f})")
print(f"  Fuel:           {Wf_ref_kg:.0f} kg  (Wf/W0 = {Wf_frac:.3f})")
print(f"  Payload:        {Wpay_kg:.0f} kg")
print()
print(f"  WING")
print(f"  Area S:         {S_final:.2f} m²   W/S = {WS_final:.0f} N/m²")
print(f"  AR / taper:     {AR:.2f} / {taper:.3f}")
print(f"  Span / chord:   {b:.2f} m / {c_bar:.3f} m")
print(f"  Washout twist:  {twist:.1f}°   e = {e:.4f}")
print(f"  Incidence:      {req.incidence_deg}°   Dihedral: {req.dihedral_deg}°")
print()
print(f"  TAIL")
print(f"  H-stab:         {SHT:.2f} m²   b_HT = {b_HT:.2f} m")
print(f"  V-stab:         {SVT:.2f} m²   b_VT = {b_VT:.2f} m")
print(f"  Tail arm:       {LHT:.2f} m")
print(f"  VHT:            {VHT:.3f}   SM = {SM_val*100:.1f}%  tail_inc = {tail_inc:.1f}°")
print()
print(f"  PERFORMANCE")
CL_cruise = CL_required(MTOW_final, S_final, V_cruise_ms)
print(f"  CL_cruise:      {CL_cruise:.4f}")
print(f"  L/D cruise:     {LD:.2f}   (CDi={CDi:.5f}  CD0={req.CD0})")
RC_SL = eta_p * P_sel_kW * 1000 / (MTOW_final * g) - V_climb / LD_climb
print(f"  ROC (SL):       {RC_SL:.2f} m/s  = {RC_SL/0.00508:.0f} ft/min  "
      f"({'✓' if RC_SL >= ROC_req_ms else '⚠'})")
print(f"  V_stall (land): {V_stall_land:.1f} m/s = {V_stall_land/0.514444:.1f} kts  "
      f"({'✓' if V_stall_land <= V_stall_ms else '⚠'})")
print()
print(f"  PROPULSION")
print(f"  Engine:         {name_e}  {hp_e} hp")
print(f"  Propeller:      {D_prop:.2f} m diameter  (constant-speed, 2-blade)")
print()
print(f"  COST")
print(f"  Unit list price: ${C_unit_final/1e3:.0f}k USD")
print(f"  Break-even:      {N_be} units")
print(SEP)

# Store results for the artifact
_results = {
    'MTOW_kg': MTOW_final, 'We_kg': We_ref_kg, 'Wf_kg': Wf_ref_kg,
    'Wpay_kg': Wpay_kg, 'We_frac': We_frac_ref, 'Wf_frac': Wf_frac,
    'S': S_final, 'WS': WS_final, 'AR': AR, 'taper': taper,
    'twist': twist, 'b': b, 'cr': cr, 'ct': ct, 'c_bar': c_bar,
    'e': e, 'CL': CL_op, 'CDi': CDi, 'CD': CD, 'LD': LD,
    'SM': SM_val, 'VHT': VHT, 'tail_inc': tail_inc,
    'SHT': SHT, 'SVT': SVT, 'b_HT': b_HT, 'b_VT': b_VT,
    'l_fus': l_fus, 'w_cabin': w_cabin,
    'P_kW': P_sel_kW, 'P_hp': hp_e, 'engine': name_e,
    'D_prop': D_prop, 'RC_SL': RC_SL,
    'V_stall_land_ms': V_stall_land,
    'C_unit': C_unit_final, 'N_BE': N_be, 'C_NRE': C_total_NRE,
}
print()
print("Results captured.")

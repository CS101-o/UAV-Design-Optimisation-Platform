"""
Longitudinal dynamic stability analysis for fixed-wing UAVs.

Computes phugoid and short-period modes from aerodynamic geometry
using analytical approximations (Datcom / Etkin & Reid methods).

Reference: Nelson "Flight Stability and Automatic Control", Ch. 4-5.
"""
import numpy as np

RHO = 1.225   # kg/m³  sea-level ISA
G   = 9.81    # m/s²

# ── Regulation bands (CS-UAV / ASTM F3312-19) ────────────────────────────────
REGS = {
    'phugoid_zeta_min':  0.04,    # lightly damped OK — autopilot corrects
    'sp_zeta_min':       0.35,    # well-damped required
    'sp_zeta_max':       1.30,    # not over-damped (sluggish pitch response)
}


def estimate_iyy(mtow_kg, fuselage_length, tail_arm_m):
    """
    Pitch moment of inertia estimate from mass and key lengths.
    Splits mass into ~70% near CG (battery, wing box, motor)
    and ~30% at the tail arm (empennage mass).
    """
    m_cg   = 0.70 * mtow_kg
    m_tail = 0.30 * mtow_kg
    Iyy = m_cg * (fuselage_length * 0.15)**2 + m_tail * tail_arm_m**2
    return max(Iyy, 0.005)   # floor at 5 g·m²


def longitudinal_modes(CL, CD, LD, SM, CL_alpha_3d,
                        AR, S, c_bar,
                        SHT_frac, tail_arm_chords,
                        mtow_kg, cruise_speed,
                        fuselage_length=None):
    """
    Compute phugoid and short-period natural frequencies and damping ratios.

    Parameters
    ----------
    CL, CD, LD   : cruise lift/drag coefficients and L/D ratio
    SM           : static margin as a fraction of MAC (e.g. 0.10 for 10%)
    CL_alpha_3d  : 3-D lift curve slope [/rad]
    AR           : wing aspect ratio
    S, c_bar     : wing area [m²] and mean chord [m]
    SHT_frac     : horizontal tail area / wing area
    tail_arm_chords : tail moment arm in units of mean chord
    mtow_kg      : max take-off mass [kg]
    cruise_speed : cruise airspeed [m/s]
    fuselage_length : total fuselage length [m]; estimated from geometry if None

    Returns dict with phugoid_zeta, phugoid_omega_n, phugoid_period_s,
    sp_zeta, sp_omega_n, sp_period_s, Iyy, Cm_q, VHT
    """
    V = cruise_speed
    m = mtow_kg
    q = 0.5 * RHO * V**2

    if fuselage_length is None:
        fuselage_length = max(0.5, 5.0 * c_bar)   # ~5× mean chord heuristic

    # ── Tail aerodynamics ─────────────────────────────────────────────────────
    SHT  = SHT_frac * S
    l_HT = tail_arm_chords * c_bar          # physical tail arm [m]

    AR_HT = 4.0
    a_HT  = 2 * np.pi * AR_HT / (2 + AR_HT)   # tail lift-curve slope [/rad]

    VHT        = SHT * l_HT / (S * c_bar)              # tail volume coefficient
    deps_dalpha = 2 * CL_alpha_3d / (np.pi * AR)       # downwash gradient dε/dα

    # ── Non-dimensional pitch derivatives ─────────────────────────────────────
    Cm_alpha_nd = -SM * CL_alpha_3d                     # /rad  (negative = stable)
    Cm_q_nd     = -2 * a_HT * VHT * (l_HT / c_bar)    # /rad  (pitch damping)
    Cm_adot_nd  = -2 * a_HT * VHT * (l_HT / c_bar) * deps_dalpha  # /rad

    # ── Inertia ───────────────────────────────────────────────────────────────
    Iyy = estimate_iyy(m, fuselage_length, l_HT)

    # ── Dimensional stability derivatives ────────────────────────────────────
    # Sign convention: Z positive downward, moment positive nose-up
    # Zw < 0 (lift opposes downward motion),  Mw < 0 (stable),  Mq < 0 (damping)
    Zw     = -q * S * CL_alpha_3d / (m * V)            # [1/s]  negative
    Mw     =  q * S * c_bar  * Cm_alpha_nd / (Iyy * V) # [1/s²] negative
    Mq_dim =  q * S * c_bar**2 * Cm_q_nd / (2 * Iyy * V)  # [1/s]  negative

    # ── Phugoid (Lanchester's approximation) ──────────────────────────────────
    omega_ph  = G * np.sqrt(2.0) / V           # natural frequency [rad/s]
    zeta_ph   = 1.0 / (LD * np.sqrt(2.0))      # damping ratio  ≈ 1 / (√2 · L/D)
    period_ph = 2 * np.pi / omega_ph           # period [s]

    # ── Short period (2-DoF decoupled approximation) ──────────────────────────
    # Characteristic equation: λ² - (Mq + Zw)λ + (Mq·Zw - Mw) = 0
    omega_sp_sq = Mq_dim * Zw - Mw             # always positive for stable aircraft
    omega_sp    = np.sqrt(max(0.0, omega_sp_sq))

    if omega_sp > 1e-6:
        zeta_sp  = -(Mq_dim + Zw) / (2.0 * omega_sp)
        period_sp = 2.0 * np.pi / omega_sp
    else:
        zeta_sp   = 0.0
        period_sp = float('inf')

    return {
        'phugoid_zeta':     float(zeta_ph),
        'phugoid_omega_n':  float(omega_ph),
        'phugoid_period_s': float(period_ph),
        'sp_zeta':          float(zeta_sp),
        'sp_omega_n':       float(omega_sp),
        'sp_period_s':      float(period_sp),
        'Iyy':              float(Iyy),
        'Cm_q':             float(Cm_q_nd),
        'VHT':              float(VHT),
        'fuselage_length':  float(fuselage_length),
    }


def check_regulations(modes, regs=None):
    """Return compliance dict against CS-UAV / ASTM F3312 limits."""
    if regs is None:
        regs = REGS
    ph_ok = modes['phugoid_zeta'] >= regs['phugoid_zeta_min']
    sp_ok = (regs['sp_zeta_min'] <= modes['sp_zeta'] <= regs['sp_zeta_max'])
    return {
        'phugoid_ok':     ph_ok,
        'sp_ok':          sp_ok,
        'all_ok':         ph_ok and sp_ok,
        'phugoid_margin': modes['phugoid_zeta'] - regs['phugoid_zeta_min'],
        'sp_margin_lo':   modes['sp_zeta']      - regs['sp_zeta_min'],
        'sp_margin_hi':   regs['sp_zeta_max']   - modes['sp_zeta'],
    }


def optimizer_penalty(modes, weight_ph=25.0, weight_sp=25.0):
    """Penalty term to add to the optimizer score function."""
    ph_deficit = REGS['phugoid_zeta_min'] - modes['phugoid_zeta']
    sp_lo      = REGS['sp_zeta_min'] - modes['sp_zeta']
    sp_hi      = modes['sp_zeta']    - REGS['sp_zeta_max']
    return (weight_ph * max(0.0, ph_deficit) +
            weight_sp * (max(0.0, sp_lo) + max(0.0, sp_hi)))

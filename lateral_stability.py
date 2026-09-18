"""
Static lateral-directional stability for fixed-wing UAVs.

Computes weathercock stability (Cn_beta, from the vertical tail) and the
dihedral effect (Cl_beta, from wing dihedral + sweep) using the same level
of analytical approximation as dynamic_stability.py's longitudinal modes.

Full dutch-roll / spiral eigenvalue modes are NOT computed here: that would
require roll- and yaw-damping derivatives (Cl_p, Cl_r, Cn_p, Cn_r) that
nothing else in this codebase estimates, and stacking several more unverified
assumptions on top of each other would produce a number that looks precise
but isn't trustworthy. Instead this module checks the two static conditions
that most directly gate mission success — is the aircraft weathercock-stable,
and is the dihedral effect within the band where dutch-roll damping doesn't
degrade — which is the standard conceptual-design-stage check (Roskam
"Airplane Design" Pt. VI-VII; Nelson "Flight Stability and Automatic
Control", Ch. 3).

Reference: Nelson, Ch. 3 (static lateral-directional derivatives);
Roskam, Airplane Design Pt. VI (empennage sizing) and Pt. VII (stability
and control) for the Cl_beta/Cn_beta dutch-roll guideline.
"""
import numpy as np

# Flat empirical fuselage destabilizing correction to Cn_beta, per radian.
# Mirrors the -0.15 MAC flat fuselage correction already applied to the
# longitudinal neutral point in analysis.py: a proper DATCOM K_N sidewash
# chart is out of scope for a conceptual-design tool, so a fixed
# order-of-magnitude decrement is used instead of a fabricated sidewash model.
CN_BETA_FUSE_FUDGE = 0.02   # /rad


def weathercock_cn_beta(SVT, l_VT, S, b, AR_VT):
    """
    Static directional stability derivative Cn_beta [/rad], vertical-tail
    contribution minus a flat fuselage destabilizing correction.

    Cn_beta_VT = a_VT * Vv   (sidewash dσ/dβ not modeled — no sidewash/
    blanketing estimate exists elsewhere in this codebase; treating it as
    zero is a known simplification, not a measured value.)

    Parameters
    ----------
    SVT   : vertical tail area [m²]
    l_VT  : vertical tail moment arm from CG [m]  (≈ LHT in this airframe,
            since the fin is co-located with the horizontal tail)
    S, b  : wing area [m²] and span [m]
    AR_VT : vertical tail aspect ratio
    """
    a_VT = 2 * np.pi * AR_VT / (AR_VT + 2)     # tail lift-curve slope [/rad]
    Vv   = (SVT * l_VT) / (S * b)               # vertical tail volume coeff.
    Cn_beta_VT = a_VT * Vv
    return float(Cn_beta_VT - CN_BETA_FUSE_FUDGE)


def dihedral_cl_beta(dihedral_deg, sweep_deg, CL_cruise, CL_alpha_3d):
    """
    Static roll stability derivative Cl_beta [/rad] (dihedral effect).

    Cl_beta_dihedral ≈ -(CL_alpha/4) * Γ        [Nelson, simplified, straight wing]
    Cl_beta_sweep    ≈ -(CL_cruise/4) * Λ_c/4   [small-angle sweep contribution,
                                                  same fidelity as the dihedral term]

    Negative Cl_beta is stabilizing (roll away from sideslip restores wings level).
    """
    gamma = np.radians(dihedral_deg)
    sweep = np.radians(sweep_deg)
    Cl_beta_dihedral = -(CL_alpha_3d / 4.0) * gamma
    Cl_beta_sweep    = -(CL_cruise    / 4.0) * sweep
    return float(Cl_beta_dihedral + Cl_beta_sweep)


def check_lateral_directional(Cn_beta, Cl_beta, req):
    """Return pass/fail against req.target_cn_beta_min / target_cl_beta_max_ratio."""
    directional_ok = Cn_beta >= req.target_cn_beta_min
    ratio = abs(Cl_beta / Cn_beta) if abs(Cn_beta) > 1e-6 else float('inf')
    ratio_ok = ratio <= req.target_cl_beta_max_ratio
    return {
        'Cn_beta': float(Cn_beta), 'Cl_beta': float(Cl_beta),
        'cl_cn_ratio': float(ratio),
        'directional_ok': directional_ok, 'dihedral_ratio_ok': ratio_ok,
        'lat_dir_ok': directional_ok and ratio_ok,
    }


def optimizer_penalty(Cn_beta, Cl_beta, req, weight_dir=25.0, weight_ratio=10.0):
    """Penalty term to add to the optimizer score function."""
    dir_deficit = req.target_cn_beta_min - Cn_beta
    ratio = abs(Cl_beta / Cn_beta) if abs(Cn_beta) > 1e-6 else 999.0
    ratio_excess = ratio - req.target_cl_beta_max_ratio
    return (weight_dir   * max(0.0, dir_deficit) ** 2 +
            weight_ratio * max(0.0, ratio_excess) ** 2)

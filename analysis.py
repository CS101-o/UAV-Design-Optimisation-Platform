"""
Prandtl-Glauert Lifting Line Theory (LLT) + drag model.
Forward model used by the optimizer — equivalent to one Flow5 VLM run.
"""
import numpy as np
import dynamic_stability
import lateral_stability
try:
    _trapz = np.trapezoid   # NumPy 2.0+
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0


def lifting_line(AR, taper, twist_deg, incidence_deg, alpha_cruise_deg,
                 CL_alpha_2d, alpha_L0_deg, S, N=24):
    """
    Glauert Fourier-series LLT.

    Returns CL, CDi, span efficiency e, lift distribution, and geometry.
    """
    b  = np.sqrt(AR * S)
    cr = 2 * S / (b * (1 + taper))
    ct = taper * cr

    # Cosine spacing avoids endpoint singularities
    theta = np.linspace(np.pi / (2 * N), np.pi - np.pi / (2 * N), N)
    y     = (b / 2) * np.cos(theta)

    # Local chord (linear taper)
    c = cr * (1 - (1 - taper) * np.abs(y) / (b / 2))

    # Local AoA: incidence + cruise AoA + linear washout (0 at root → -twist at tip)
    twist_local = -np.radians(twist_deg) * np.abs(y) / (b / 2)
    alpha_local = np.radians(incidence_deg + alpha_cruise_deg) + twist_local

    # Glauert matrix  Σ An [4b/(c·a0) + n/sinθ] sinθ = α(θ) - α_L0
    A_mat = np.zeros((N, N))
    for i, th in enumerate(theta):
        sth = np.sin(th)
        for n in range(1, N + 1):
            A_mat[i, n - 1] = np.sin(n * th) * (4 * b / (c[i] * CL_alpha_2d) + n / sth)

    An  = np.linalg.solve(A_mat, alpha_local - np.radians(alpha_L0_deg))
    A1  = An[0]
    CL  = np.pi * AR * A1

    # Oswald span efficiency
    delta = sum(n * (An[n - 1] / A1) ** 2 for n in range(2, N + 1)) if abs(A1) > 1e-10 else 0.0
    e     = 1.0 / (1.0 + delta)
    CDi   = CL ** 2 / (np.pi * AR * e)

    # Circulation distribution (normalised to max = 1 for shape comparison)
    gamma = np.array([sum(An[n - 1] * np.sin(n * th) for n in range(1, N + 1))
                      for th in theta])

    gamma_max = np.max(np.abs(gamma))
    if gamma_max > 1e-10:
        gamma_norm = gamma / gamma_max
    else:
        gamma_norm = gamma

    # Perfect ellipse has Γ ∝ sin(θ)
    ellipse_norm   = np.sin(theta)
    ellipticity_err = np.sqrt(np.mean((gamma_norm - ellipse_norm) ** 2))

    return {
        'CL': CL, 'CDi': CDi, 'e': e,
        'ellipticity_err': ellipticity_err,
        'gamma': gamma_norm, 'ellipse': ellipse_norm,
        'theta': theta, 'y': y, 'c': c,
        'b': b, 'cr': cr, 'ct': ct, 'An': An,
    }


def find_cruise_alpha(AR, taper, twist_deg, incidence_deg,
                      CL_required, CL_alpha_2d, alpha_L0_deg, S):
    """Bisection: find AoA that produces exactly CL_required."""
    def CL_at(alpha_deg):
        return lifting_line(AR, taper, twist_deg, incidence_deg, alpha_deg,
                            CL_alpha_2d, alpha_L0_deg, S)['CL']

    lo, hi = -8.0, 18.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if CL_at(mid) < CL_required:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def CL_required(mtow_kg, S, V, rho=1.225):
    return (mtow_kg * 9.81) / (0.5 * rho * V ** 2 * S)


# ---------------------------------------------------------------------------
# Stability & Trim  (Part 6 from design document)
# ---------------------------------------------------------------------------

def wing_3d_lift_slope(AR, CL_alpha_2d):
    """Prandtl correction: 2D → 3D lift curve slope."""
    return CL_alpha_2d * AR / (AR + 2)


def tail_3d_lift_slope(CL_alpha_2d, AR_HT=4.0):
    return CL_alpha_2d * AR_HT / (AR_HT + 2)


def neutral_point(AR, S, SHT_frac, LHT_chords, CL_alpha_2d):
    """
    Neutral point as fraction of MAC (measured from wing leading edge at MAC).

    Uses wing+tail only; fuselage destabilising correction applied separately.
    Formula: x_NP = Σ(x_ac_i × CLα_i × S_i) / Σ(CLα_i × S_i)
    """
    b     = np.sqrt(AR * S)
    c_bar = b / AR
    SHT   = SHT_frac * S
    LHT   = LHT_chords * c_bar

    CLa_wing = wing_3d_lift_slope(AR, CL_alpha_2d)
    CLa_tail = tail_3d_lift_slope(CL_alpha_2d)

    x_ac_wing = 0.25                          # quarter chord (fraction of MAC)
    x_ac_tail = x_ac_wing + LHT / c_bar       # tail AC relative to wing LE at MAC

    x_NP_wt = (x_ac_wing * CLa_wing * S + x_ac_tail * CLa_tail * SHT) / \
              (CLa_wing * S + CLa_tail * SHT)

    # Fuselage destabilises: empirical correction ≈ −15% MAC
    x_NP_aircraft = x_NP_wt - 0.15

    VHT = (LHT * SHT) / (c_bar * S)   # horizontal tail volume coefficient

    return x_NP_aircraft, x_NP_wt, VHT


def static_margin(AR, S, SHT_frac, LHT_chords, CL_alpha_2d, cg_frac_mac):
    """
    Static margin = (x_NP − x_CG) / MAC.  Target 5–15% (doc Part 6).
    Returns SM and the CG range that keeps SM in [0.05, 0.15].
    """
    x_NP, _, VHT = neutral_point(AR, S, SHT_frac, LHT_chords, CL_alpha_2d)
    SM = x_NP - cg_frac_mac
    cg_min = x_NP - 0.15   # CG at SM=15% (most forward)
    cg_max = x_NP - 0.05   # CG at SM=5%  (most aft)
    return SM, x_NP, VHT, cg_min, cg_max


def pitch_trim(AR, S, SHT_frac, LHT_chords, CL_alpha_2d, CM_ac, CL_cruise, cg_frac_mac):
    """
    Longitudinal trim (Part 6 — ΣM_cg = 0).

    Computes untrimmed CM_cg and the tail incidence needed to trim.
    Tail incidence within ±5° → trimmable.
    """
    b     = np.sqrt(AR * S)
    c_bar = b / AR
    SHT   = SHT_frac * S
    LHT   = LHT_chords * c_bar
    VHT   = (LHT * SHT) / (c_bar * S)
    CLa_tail = tail_3d_lift_slope(CL_alpha_2d)

    # CM about CG from wing (before tail)
    # CM_cg = CM_ac + CL × (x_ac − x_cg) / c̄
    x_ac = 0.25
    CM_cg_untrimmed = CM_ac + CL_cruise * (x_ac - cg_frac_mac)

    # CL_tail needed so that CM_cg_untrimmed − VHT × CL_tail = 0
    CL_tail_needed  = CM_cg_untrimmed / VHT
    tail_inc_deg    = float(np.degrees(CL_tail_needed / CLa_tail))
    trimmable       = abs(tail_inc_deg) <= 5.0

    return CM_cg_untrimmed, CL_tail_needed, tail_inc_deg, trimmable, VHT


# E387 2D Cl_max vs Reynolds number (UIUC LSATs wind-tunnel data, smooth model)
_E387_RE    = np.array([ 60e3, 100e3, 150e3, 200e3, 300e3, 460e3])
_E387_CLMAX = np.array([ 0.95,  1.09,  1.15,  1.20,  1.25,  1.30])


def clmax_2d_E387(Re):
    """2D Cl_max of the E387 at a given chord Reynolds number (UIUC data)."""
    return float(np.interp(Re, _E387_RE, _E387_CLMAX))


def wing_CLmax(AR, taper, twist_deg, incidence_deg, S, V_stall_guess,
               CL_alpha_2d=2*np.pi, alpha_L0_deg=-4.0, nu=1.5e-5):
    """
    3D wing CL_max by the critical-section method.

    The LLT lift distribution gives the ratio of local cl to wing CL at each
    span station.  The wing stalls when the most critical station reaches the
    airfoil's 2D Cl_max at its local chord Reynolds number:

        CL_max = min over y of  Clmax_2d(Re(y)) / (cl(y) / CL)

    Stations outboard of 95% semi-span are ignored (tip cl → 0 makes the
    ratio meaningless there and LLT is least accurate at the tip anyway).
    Run at a representative high-lift AoA so the distribution shape includes
    the washout effect near stall.
    """
    r = lifting_line(AR, taper, twist_deg, incidence_deg, 8.0,
                     CL_alpha_2d, alpha_L0_deg, S)
    b, y, c, CL = r['b'], r['y'], r['c'], r['CL']
    if CL <= 1e-6:
        return 1.0

    # Local cl from circulation:  cl(y) = 2Γ(y) / (V c(y)); LLT returns Γ
    # normalised, so recover the ratio via CL = (2/(V S))∫Γdy — work with
    # the normalised shape and rescale so the area-weighted mean equals CL.
    gamma = r['gamma']                       # normalised circulation shape
    cl_shape = gamma / c                     # ∝ local cl
    # scale so that (1/S)∫ cl·c dy = CL  (y spans the full span, descending)
    integral = _trapz(cl_shape * c, -y)
    cl_local = cl_shape * (CL * S / integral)

    inboard = np.abs(y) <= 0.95 * (b / 2)
    Re_local = V_stall_guess * c / nu
    margin = np.array([clmax_2d_E387(Re_local[i]) / cl_local[i]
                       if cl_local[i] > 1e-6 else np.inf
                       for i in range(len(y))])
    return float(CL * np.min(margin[inboard]))


def stall_check(mtow_kg, S, CL_max_2d, CL_cruise, V_stall_req_ms, rho=1.225):
    """
    Stall speed and margin check (Part 1 requirement vs Part 5 geometry).

    stall_ratio = CL_cruise / CL_max  (target < 0.70 — doc Part 5/6)
    V_stall     = sqrt(2W / ρ S CL_max)
    """
    W         = mtow_kg * 9.81
    V_stall   = float(np.sqrt(2 * W / (rho * S * CL_max_2d)))
    stall_ratio = CL_cruise / CL_max_2d
    meets_req   = V_stall <= V_stall_req_ms
    return V_stall, stall_ratio, meets_req


# Empirical plain-flap lift increment factor (Raymer, "Aircraft Design: A
# Conceptual Approach", flap ΔCLmax guidance — plain/simple flaps ≈ 0.9,
# vs. ~1.3 for slotted, ~1.6-1.9 for Fowler). Flap TYPE is a fixed
# configuration choice for this airframe (plain flap); only flap deflection
# is a design variable.
K_FLAP_PLAIN = 0.9


def landing_stall_check(mtow_kg, S, CL_max_clean_3d, flap_area_frac,
                        flap_deflection_deg, V_stall_land_req_ms,
                        rho=1.225, k_flap=K_FLAP_PLAIN):
    """
    Landing (flaps-out) stall speed and CLmax, separate from the clean-
    configuration check in stall_check()/wing_CLmax(). Flaps are assumed
    fully retracted (zero effect) at cruise — only this landing-config
    check credits them.

    ΔCLmax ≈ k_flap × (S_flap / S) × sin(δ_flap)   [Raymer-style flap increment]
    """
    delta_clmax = k_flap * flap_area_frac * np.sin(np.radians(flap_deflection_deg))
    CL_max_land = CL_max_clean_3d + delta_clmax
    W = mtow_kg * 9.81
    V_stall_land = float(np.sqrt(2 * W / (rho * S * CL_max_land)))
    meets_req = V_stall_land <= V_stall_land_req_ms
    return V_stall_land, float(CL_max_land), meets_req


# ---------------------------------------------------------------------------
# Control surface sizing  (Part 5 — typical percentages from design document)
# ---------------------------------------------------------------------------

def control_surfaces(S, b, c_bar):
    """
    Sizes flaps, ailerons from doc typical values.
    Flaps:    10–20% wing area, chord 20–30% c, span 50–60% semi-span
    Ailerons: 8–12% wing area, chord 20–30% c, span 25–50% semi-span
    """
    semi = b / 2
    return {
        'flap_area':        0.15 * S,
        'flap_chord':       0.25 * c_bar,
        'flap_span':        0.55 * semi,
        'aileron_area':     0.10 * S,
        'aileron_chord':    0.25 * c_bar,
        'aileron_span_in':  0.50 * semi,   # inboard start
        'aileron_span_out': 0.90 * semi,   # outboard end (~40% semi-span width)
    }


# ---------------------------------------------------------------------------
# Fuselage and tail drag models  (for two-phase trimmed MDO)
# ---------------------------------------------------------------------------

def fuselage_cd0(length_m, fineness, S_ref, V=20.0, nu=1.5e-5):
    """
    Fuselage parasite drag coefficient referenced to wing area.

    Uses Hoerner's form factor for a body of revolution and turbulent
    flat-plate skin friction (conservative — appropriate for a drone
    fuselage with surface roughness).

    fineness = total_length / max_diameter  (good range: 6–14)
    """
    D    = length_m / fineness
    Re   = V * length_m / nu
    Cf   = 0.074 / Re ** 0.2                                # turbulent flat plate
    FF   = 1.0 + 60.0 / fineness ** 3 + 0.0025 * fineness  # Hoerner body FF
    Swet = np.pi * D * length_m * 0.90                      # ~90% of cylinder
    return Cf * FF * Swet / S_ref


def tail_profile_cd0(SHT, SVT, b_HT, V=20.0, nu=1.5e-5, t_c=0.09):
    """
    Horizontal + vertical tail profile drag referenced to wing area.

    Section: NACA 0009 (t/c = 9%), symmetric.  Both upper and lower
    surfaces are wetted for each panel.

    Returns CD0_tail referenced to S_ref — the caller supplies S_ref by
    dividing the returned value by wing area.
    """
    c_HT = SHT / b_HT if b_HT > 0.01 else 0.12
    Re_t = V * c_HT / nu
    Cf   = 0.074 / Re_t ** 0.2
    FF   = 1.0 + 2.0 * t_c + 60.0 * t_c ** 4   # thin aerofoil form factor
    Swet = 2.0 * (SHT + SVT)                     # top+bottom of HT and VT
    return Cf * FF * Swet                         # absolute — divide by S in caller


def compute_trimmed_ld(AR, taper, twist_deg, SHT_frac, AR_HT, taper_HT,
                        tail_arm_chords, fuse_length, fuse_fineness, req,
                        sweep_deg=0.0, dihedral_deg=0.0,
                        SVT_frac=0.10, AR_VT=1.8, taper_VT=0.50,
                        HT_sweep_deg=0.0, HT_twist_deg=0.0, HT_dihedral_deg=0.0,
                        VT_sweep_deg=0.0, flap_deflection_deg=0.0):
    """
    Compute fully trimmed L/D at cruise for a complete aircraft configuration.

    TRUE TRIM: tail incidence is solved analytically so CM_cg = 0 exactly.
    It is never a design variable and never a penalty — the tail always trims
    the aircraft; we only check whether the required incidence is within the
    physical authority limit (±6°).

    CG AUTO-PLACEMENT: CG is placed at NP − target_SM (default 10% MAC).
    The user never enters a CG position.

    OBJECTIVE: includes wing induced drag, tail induced drag, wing profile
    drag, tail profile drag, and fuselage parasite drag.  Missing any of
    these would give a misleading optimum.

    STABILITY: alongside the longitudinal static margin/trim already solved
    here, this also evaluates longitudinal dynamic modes (phugoid,
    short-period — dynamic_stability.py) and static lateral-directional
    stability (weathercock Cn_beta from the vertical tail, dihedral effect
    Cl_beta from wing dihedral/sweep — lateral_stability.py) so that the
    vertical-tail sizing and dihedral design variables are actually judged
    against a requirement rather than floating free.

    Parameters
    ----------
    AR, taper, twist_deg        : wing planform (Aspect Ratio, taper, washout)
    SHT_frac                    : horizontal tail area / wing area
    AR_HT, taper_HT             : horizontal tail planform
    tail_arm_chords             : LHT = tail_arm_chords × mean chord
    fuse_length, fuse_fineness  : fuselage length [m] and L/d ratio
    req                         : DroneRequirements (mission inputs only)
    sweep_deg                   : quarter-chord wing sweep [deg]  (0 for drone)
    dihedral_deg                : dihedral angle [deg]
    SVT_frac                    : vertical tail area / wing area
    AR_VT, taper_VT             : vertical tail planform
    HT_sweep_deg                : horizontal tail quarter-chord sweep [deg]
    HT_twist_deg                : horizontal tail washout [deg]
    HT_dihedral_deg             : horizontal tail dihedral [deg]
    VT_sweep_deg                : vertical tail (fin) sweep [deg]
    flap_deflection_deg         : plain-flap deflection for the LANDING stall
                                   check only (flaps assumed retracted, zero
                                   effect, at the cruise point computed above)

    Returns
    -------
    dict with all aerodynamic, stability, trim, and geometry quantities.
    """
    S   = req.wing_area
    V   = req.cruise_speed_ms
    nu  = 1.5e-5
    rho = 1.225

    CL_req = CL_required(req.mtow_kg, S, V)

    # ── Wing LLT ─────────────────────────────────────────────────────────
    alpha_c = find_cruise_alpha(AR, taper, twist_deg, 0.0, CL_req,
                                 req.CL_alpha_2d, req.alpha_L0_deg, S)
    r = lifting_line(AR, taper, twist_deg, 0.0, alpha_c,
                     req.CL_alpha_2d, req.alpha_L0_deg, S)
    b      = r['b']
    c_bar  = b / AR
    CDi_w  = r['CDi']

    # ── Tail geometry ─────────────────────────────────────────────────────
    SHT   = SHT_frac * S
    SVT   = SVT_frac * S
    b_HT  = np.sqrt(max(AR_HT * SHT, 0.01))
    cr_HT = 2.0 * SHT / (b_HT * (1.0 + taper_HT))
    ct_HT = taper_HT * cr_HT
    LHT   = tail_arm_chords * c_bar                          # physical arm [m]
    VHT   = (LHT * SHT) / (c_bar * S)                       # tail volume coeff
    CLa_t = 2.0 * np.pi * AR_HT / (AR_HT + 2.0)            # tail lift slope

    # Vertical tail geometry (fin arm = LHT: co-located with the HT, per
    # the fixed-configuration Flow5 plane geometry in flow5_xml.py)
    b_VT  = np.sqrt(max(AR_VT * SVT, 1e-6))
    cr_VT = 2.0 * SVT / (b_VT * (1.0 + taper_VT))
    ct_VT = taper_VT * cr_VT

    # ── Neutral point (wing + tail, fuselage correction) ─────────────────
    CLa_w    = req.CL_alpha_2d * AR / (AR + 2.0)            # 3-D wing slope
    x_ac     = 0.25                                          # wing AC at c/4
    x_ac_t   = x_ac + LHT / c_bar                           # tail AC [MAC frac]
    x_NP_wt  = (x_ac * CLa_w * S + x_ac_t * CLa_t * SHT) / \
               (CLa_w * S + CLa_t * SHT)
    x_NP     = x_NP_wt - 0.15                               # fuselage destabilises

    # ── Auto-place CG at NP − SM_target ──────────────────────────────────
    cg_frac  = x_NP - req.target_sm
    SM       = req.target_sm                                 # exact by construction

    # ── Solve trim exactly: CM_cg = 0 → tail incidence ───────────────────
    # Moment about CG from wing (before tail):
    #   CM_cg = CM_ac + CL_wing × (x_ac − x_cg)
    CM_cg_unt = req.CM_ac + CL_req * (x_ac - cg_frac)

    # Tail must supply:  VHT × CLa_t × i_t = CM_cg_unt
    CL_tail  = CM_cg_unt / VHT                              # tail CL for trim
    tail_inc = float(np.degrees(CL_tail / CLa_t))           # tail incidence [deg]
    trimmable = abs(tail_inc) <= 6.0

    # ── Tail induced drag (referenced to wing area) ───────────────────────
    e_HT    = 0.85                                           # typical tail Oswald
    CDi_t   = (CL_tail ** 2 / (np.pi * AR_HT * e_HT)) * (SHT / S)

    # ── Profile drag breakdown ─────────────────────────────────────────────
    # Wing: E387 at cruise Re (~250k). Laminar bucket gives Cf≈0.005; corrected
    # for typical UAV transition roughness → 0.008 (conservative but stable).
    CD0_w   = 0.008

    CD0_t   = tail_profile_cd0(SHT, SVT, b_HT, V, nu) / S  # referenced to S
    CD0_f   = fuselage_cd0(fuse_length, fuse_fineness, S, V, nu)

    # ── Totals ────────────────────────────────────────────────────────────
    CD_total  = CDi_w + CDi_t + CD0_w + CD0_t + CD0_f
    # Total lift: wing plus tail download/upload contribution
    CL_total  = CL_req + CL_tail * (SHT / S)
    LD_trim   = CL_total / CD_total if CD_total > 1e-8 else 0.0

    # ── Stall (critical section method) ──────────────────────────────────
    CL_max_3d   = wing_CLmax(AR, taper, twist_deg, 0.0, S,
                              req.stall_speed_ms, req.CL_alpha_2d, req.alpha_L0_deg)
    V_stall     = float(np.sqrt(2.0 * req.mtow_kg * 9.81 / (rho * S * CL_max_3d)))
    stall_ratio = CL_req / CL_max_3d
    stall_ok    = V_stall <= req.stall_speed_ms

    # ── Landing stall (flaps out) — separate from the clean check above ──
    cs = control_surfaces(S, b, c_bar)
    flap_area_frac = cs['flap_area'] / S
    V_stall_land, CL_max_land, landing_ok = landing_stall_check(
        req.mtow_kg, S, CL_max_3d, flap_area_frac,
        flap_deflection_deg, req.V_stall_land_ms, rho)

    # ── Wing incidence: fuselage level at cruise, clamped 2–4° ──────────
    incidence_deg = float(np.clip(alpha_c, 2.0, 4.0))

    # ── Longitudinal dynamic stability (phugoid / short-period) ──────────
    dyn_modes = dynamic_stability.longitudinal_modes(
        CL_total, CD_total, LD_trim, SM, CLa_w,
        AR, S, c_bar, SHT_frac, tail_arm_chords,
        req.mtow_kg, V, fuselage_length=fuse_length)
    dyn_regs = dynamic_stability.check_regulations(dyn_modes, {
        'phugoid_zeta_min': req.dyn_ph_zeta_min,
        'sp_zeta_min':      req.dyn_sp_zeta_min,
        'sp_zeta_max':      req.dyn_sp_zeta_max,
    })

    # ── Static lateral-directional stability (weathercock + dihedral) ────
    Cn_beta = lateral_stability.weathercock_cn_beta(SVT, LHT, S, b, AR_VT)
    Cl_beta = lateral_stability.dihedral_cl_beta(dihedral_deg, sweep_deg,
                                                  CL_total, CLa_w)
    lat_dir = lateral_stability.check_lateral_directional(Cn_beta, Cl_beta, req)

    return {
        # ── Trimmed performance ──────────────────────────────────────────
        'LD': LD_trim,
        'CL_total': CL_total, 'CD_total': CD_total,
        'CDi_wing': CDi_w,    'CDi_tail': CDi_t,
        'CD0_wing': CD0_w,    'CD0_tail': CD0_t,  'CD0_fuse': CD0_f,
        # ── Trim ────────────────────────────────────────────────────────
        'CL_tail': CL_tail, 'tail_inc_deg': tail_inc, 'trimmable': trimmable,
        'CM_cg_untrimmed': CM_cg_unt, 'VHT': VHT,
        # ── Stability ───────────────────────────────────────────────────
        'SM': SM, 'x_NP': x_NP, 'cg_frac': cg_frac,
        # ── Wing geometry ────────────────────────────────────────────────
        'e': r['e'], 'ellipticity_err': r['ellipticity_err'],
        'b': b, 'cr': r['cr'], 'ct': r['ct'], 'c_bar': c_bar,
        'AR': AR, 'taper': taper, 'twist_deg': twist_deg,
        'sweep_deg': sweep_deg, 'dihedral_deg': dihedral_deg,
        'alpha_cruise': alpha_c, 'incidence_deg': incidence_deg,
        # ── Tail geometry ────────────────────────────────────────────────
        'b_HT': b_HT, 'cr_HT': cr_HT, 'ct_HT': ct_HT,
        'SHT': SHT, 'SHT_frac': SHT_frac,
        'LHT': LHT, 'AR_HT': AR_HT, 'taper_HT': taper_HT,
        'tail_arm_chords': tail_arm_chords, 'HT_sweep_deg': HT_sweep_deg,
        'HT_twist_deg': HT_twist_deg, 'HT_dihedral_deg': HT_dihedral_deg,
        # ── Vertical tail geometry ──────────────────────────────────────
        'SVT': SVT, 'SVT_frac': SVT_frac, 'AR_VT': AR_VT, 'taper_VT': taper_VT,
        'b_VT': b_VT, 'cr_VT': cr_VT, 'ct_VT': ct_VT, 'VT_sweep_deg': VT_sweep_deg,
        # ── Fuselage ────────────────────────────────────────────────────
        'fuse_length': fuse_length, 'fuse_fineness': fuse_fineness,
        'fuse_diam': fuse_length / fuse_fineness,
        # ── Stall (clean) and landing (flaps out) ────────────────────────
        'V_stall': V_stall, 'stall_ratio': stall_ratio,
        'stall_ok': stall_ok, 'CL_max_3d': CL_max_3d,
        'CL_req': CL_req,
        'flap_deflection_deg': flap_deflection_deg,
        'V_stall_land': V_stall_land, 'CL_max_land': CL_max_land,
        'landing_ok': landing_ok,
        # ── Longitudinal dynamic stability ───────────────────────────────
        'dyn_modes': dyn_modes,
        'phugoid_zeta': dyn_modes['phugoid_zeta'], 'sp_zeta': dyn_modes['sp_zeta'],
        'dyn_ok': dyn_regs['all_ok'],
        # ── Lateral-directional stability ────────────────────────────────
        'Cn_beta': lat_dir['Cn_beta'], 'Cl_beta': lat_dir['Cl_beta'],
        'cl_cn_ratio': lat_dir['cl_cn_ratio'], 'lat_dir_ok': lat_dir['lat_dir_ok'],
    }

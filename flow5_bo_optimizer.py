"""
Bayesian Optimisation with Flow5 TRIUNIFORM (inviscid VLM) as forward model.

Replaces SLSQP+LLT with a Gaussian-Process surrogate (scikit-optimize).
~30 BO evaluations × up to 5 Newton trim steps = ~150 Flow5 calls.
Total wall time: ~6-10 minutes for TRIUNIFORM inviscid.

Per evaluation:
  - LLT called once (<1 ms): geometry dict, initial trim angle, profile drag
  - Flow5 called up to 5 times: Newton iteration on tail incidence until |Cm|<0.005
  - Newton step: Δi_tail = +Cm/(CLa_tail × VHT) × 0.8 (correct sign: positive
    Cm needs positive tail lift to generate nose-down moment)

The CDi < 0 guard rejects Trefftz-plane artefacts that arise at extreme tail
incidence (>±7°), which produce spuriously high L/D (>50).
"""

import os
import math
import time

from flow5_xml import write_plane_xml, write_polar_xml, write_script_xml, XML_DIR
from flow5_run import run_flow5, find_polar_csv, parse_polar_csv, extract_performance_interp
from analysis import compute_trimmed_ld
from optimizer import result_to_geom, _stall_penalty, _trim_penalty, _landing_penalty
import dynamic_stability
import lateral_stability

RHO = 1.225
G   = 9.81

# BO uses the default XML filenames (same as the standard run path).
# Verify uses drone_mdo_best.xml / drone_verify_polar.xml / drone_verify_script.xml,
# so there is no naming conflict.
_BO_SCRIPT = os.path.join(XML_DIR, 'drone_script.xml')



def _flow5_newton_trim(geom, req, llt, initial_tail_inc, n_iter=5, cm_tol=0.005):
    """
    Run Flow5 TRIUNIFORM and iterate tail incidence until |CM| < cm_tol.

    CG is the LLT-derived value (target SM set by analysis.py); the Newton
    loop finds the tail incidence that makes Flow5's Cm equal zero.

    Newton step:
        ∂Cm/∂i_tail = −CLa_tail × VHT  (per radian, negative)
        Δi_tail = +Cm / (CLa_tail × VHT)
        Positive Cm → positive Δi_tail (upward tail lift → nose-down → Cm↓).

    Returns hybrid (LD, final_tail_inc) or (None, None) on Flow5 failure.
    Hybrid L/D = CDi_whole_plane from Flow5 + CD0_profile from LLT friction.
    """
    V      = req.cruise_speed_ms
    S      = req.wing_area
    b      = geom['b']
    c_bar  = b / geom['AR']

    CL_cruise  = (req.mtow_kg * G) / (0.5 * RHO * V**2 * S)
    AR_HT      = geom.get('AR_HT', 4.0)
    CLa_tail   = 2 * math.pi * AR_HT / (AR_HT + 2)   # /rad

    # VHT = SHT_frac × tail_arm_chords (tail volume coefficient).
    # llt['VHT'] is computed by analysis.py from the same geometry.
    VHT = llt.get('VHT', llt.get('SHT_frac', 0.22) * 4.0)

    # Profile drag from LLT friction formulas — independent of CG.
    CD0_profile = llt['CD0_wing'] + llt['CD0_tail'] + llt['CD0_fuse']

    # Polar and script XMLs are fixed for this geometry — write once.
    write_polar_xml(V, S, b, c_bar, method='TRIUNIFORM', viscous=False)
    write_script_xml(-5, 14, 1)

    # ── Newton trim loop ─────────────────────────────────────────────────
    tail_inc  = initial_tail_inc
    last_perf = None

    for _ in range(n_iter):
        geom_iter = {**geom, 'tail_inc_deg': tail_inc}
        write_plane_xml(geom_iter)

        run_flow5(_BO_SCRIPT, timeout=30)
        time.sleep(0.2)

        csv = find_polar_csv()
        if csv is None:
            return None, None
        polar = parse_polar_csv(csv)
        perf  = extract_performance_interp(polar, CL_cruise)
        if perf is None:
            return None, None

        last_perf = perf
        CM = perf['Cm']
        if abs(CM) < cm_tol:
            break

        delta    = math.degrees(CM / (CLa_tail * VHT)) * 0.8
        tail_inc = max(-8.0, min(8.0, tail_inc + delta))

    if last_perf is None:
        return None, None

    CDi_total = last_perf['CDi']
    CL_total  = last_perf['CL']

    # Negative CDi: Trefftz-plane artefact when the tail is near its ±8° clamp
    # and wake cancellation occurs between wing and tail.  Reject so the GP
    # surrogate only sees physically valid observations.
    if CDi_total < 0:
        return None, None

    CD_total  = CDi_total + CD0_profile
    LD_hybrid = CL_total / CD_total if CD_total > 1e-8 else 0.0

    return LD_hybrid, tail_inc


def flow5_bo_objective(x_list, req, var_specs, call_counter, callback=None):
    """
    Objective for gp_minimize: evaluates trimmed L/D via Flow5, returns -LD.

    x_list       : flat list of parameter values in var_specs order.
    call_counter : [int] mutable counter shared with caller.
    callback(n, x_dict, LD, tail_inc) : optional progress hook.
    """
    x_dict = {s['key']: v for s, v in zip(var_specs, x_list)}

    AR        = x_dict.get('AR',        7.0)
    taper     = x_dict.get('taper',    0.40)
    twist_deg = x_dict.get('twist',    1.5)
    sweep_deg = x_dict.get('sweep',    0.0)
    dihedral  = x_dict.get('dihedral', 1.0)
    SHT_frac  = x_dict.get('SHT_frac', 0.22)
    AR_HT     = x_dict.get('AR_HT',   4.0)
    taper_HT  = x_dict.get('taper_HT', 0.50)
    tail_arm  = x_dict.get('tail_arm', 4.0)
    HT_sweep  = x_dict.get('HT_sweep', 0.0)
    HT_twist  = x_dict.get('HT_twist', 0.0)
    HT_dihedral = x_dict.get('HT_dihedral', 0.0)
    fuse_len  = x_dict.get('fuse_len', 0.80)
    fuse_fin  = x_dict.get('fuse_fin', 6.5)
    SVT_frac  = x_dict.get('SVT_frac', 0.10)
    AR_VT     = x_dict.get('AR_VT',    1.8)
    taper_VT  = x_dict.get('taper_VT', 0.50)
    VT_sweep  = x_dict.get('VT_sweep', 0.0)
    flap_deflection = x_dict.get('flap_deflection', 0.0)

    call_counter[0] += 1

    # Physical constraint: tail moment arm must be at least fuselage length.
    # Penalty must be large positive — gp_minimize minimises, valid designs
    # return -LD ≈ -10 to -30, so any penalty > 0 beats any valid design.
    S     = req.wing_area
    c_bar = math.sqrt(AR * S) / AR
    viol  = max(0.0, fuse_len - tail_arm * c_bar)
    if viol > 1e-4:
        return 500.0 + 1000.0 * viol**2

    # LLT for geometry dict, initial trim angle, and profile drag (<1 ms).
    # This is also where stall margin, longitudinal dynamic modes, and
    # lateral-directional stability (weathercock Cn_beta, dihedral Cl_beta)
    # are evaluated — Flow5's Newton trim loop only re-solves Cm, not these.
    try:
        llt = compute_trimmed_ld(
            AR, taper, twist_deg, SHT_frac, AR_HT, taper_HT,
            tail_arm, fuse_len, fuse_fin, req,
            sweep_deg=sweep_deg, dihedral_deg=dihedral,
            flap_deflection_deg=flap_deflection,
            SVT_frac=SVT_frac, AR_VT=AR_VT, taper_VT=taper_VT,
            HT_sweep_deg=HT_sweep, HT_twist_deg=HT_twist,
            HT_dihedral_deg=HT_dihedral, VT_sweep_deg=VT_sweep,
        )
    except Exception:
        return 500.0

    geom = result_to_geom(llt, req.mtow_kg)

    # Flow5 Newton trim loop — returns hybrid L/D
    LD, final_tail_inc = _flow5_newton_trim(geom, req, llt, llt['tail_inc_deg'])
    if LD is None:
        return 500.0

    # Soft penalties: stall margin, tail authority, dynamic and lateral-
    # directional stability. Previously only the hard tail-arm/fuselage and
    # CDi<0 guards were enforced here, so BO could accept a Cm≈0 design that
    # silently violated stall margin or stability requirements.
    penalty = (_stall_penalty(llt['stall_ratio']) +
               _trim_penalty(final_tail_inc) +
               _landing_penalty(llt['V_stall_land'], req.V_stall_land_ms) +
               dynamic_stability.optimizer_penalty(llt['dyn_modes']) +
               lateral_stability.optimizer_penalty(llt['Cn_beta'], llt['Cl_beta'], req))

    if callback:
        callback(call_counter[0], x_dict, LD, final_tail_inc)

    return -LD + penalty


def build_space(var_specs):
    """Convert a list of var spec dicts into a skopt Space."""
    from skopt.space import Real
    return [Real(s['lo'], s['hi'], name=s['key']) for s in var_specs]


def run_flow5_bo(req, var_specs, llt_seed_result=None,
                 n_calls=80, n_initial=12,
                 callback=None, cancel_flag=None, random_state=42):
    """
    Run Bayesian Optimisation with Flow5 TRIUNIFORM as forward model.

    Parameters
    ----------
    req               : DroneRequirements
    var_specs         : list of active spec dicts (WING+TAIL+FUSE, excl. HT_sweep)
    llt_seed_result   : dict from compute_trimmed_ld — used to warm-start the GP
    n_calls           : total BO evaluations (default 80)
    n_initial         : random exploration before GP takes over (default 12)
    callback(n, x_dict, LD, tail_inc) : called after each Flow5 evaluation
    cancel_flag       : [False] — set True from another thread to abort early
    random_state      : BO random seed (vary for seed-independence testing)

    Returns
    -------
    dict: LD, x_dict, n_calls_used, best_llt_LD, cancelled
    """
    from skopt import gp_minimize

    space        = build_space(var_specs)
    call_counter = [0]

    # Map from spec key → result dict key so we can seed GP from LLT result
    _key_map = {
        'AR':       'AR',
        'taper':    'taper',
        'twist':    'twist_deg',
        'sweep':    'sweep_deg',
        'dihedral': 'dihedral_deg',
        'SHT_frac': 'SHT_frac',
        'AR_HT':    'AR_HT',
        'taper_HT': 'taper_HT',
        'tail_arm': 'tail_arm_chords',
        'HT_sweep': 'HT_sweep_deg',
        'HT_twist': 'HT_twist_deg',
        'HT_dihedral': 'HT_dihedral_deg',
        'fuse_len': 'fuse_length',
        'fuse_fin': 'fuse_fineness',
        'SVT_frac': 'SVT_frac',
        'AR_VT':    'AR_VT',
        'taper_VT': 'taper_VT',
        'VT_sweep': 'VT_sweep_deg',
        'flap_deflection': 'flap_deflection_deg',
    }

    x0_seed = None
    if llt_seed_result is not None:
        seed = []
        for s in var_specs:
            res_key = _key_map.get(s['key'])
            val = llt_seed_result.get(res_key, s['x0']) if res_key else s['x0']
            seed.append(float(val))
        x0_seed = [seed]   # gp_minimize expects list of lists

    def _obj(x):
        if cancel_flag is not None and cancel_flag[0]:
            raise KeyboardInterrupt('BO cancelled by user')
        return flow5_bo_objective(x, req, var_specs, call_counter, callback)

    try:
        n_initial_clamped = min(n_initial, n_calls)
        result = gp_minimize(
            _obj, space,
            n_calls=n_calls,
            n_initial_points=n_initial_clamped,
            x0=x0_seed,
            acq_func='EI',
            random_state=random_state,
        )
    except KeyboardInterrupt:
        return {'cancelled': True, 'n_calls_used': call_counter[0]}

    best_x  = result.x
    best_ld = -result.fun
    x_dict  = {s['key']: v for s, v in zip(var_specs, best_x)}

    return {
        'LD':           best_ld,
        'x_dict':       x_dict,
        'n_calls_used': call_counter[0],
        'best_llt_LD':  llt_seed_result.get('LD') if llt_seed_result else None,
        'cancelled':    False,
    }

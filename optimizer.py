"""
Two-phase wing + tail optimizer.

Phase 1 — WING
  Design variables : [AR, taper, twist, sweep, dihedral, flap_deflection]
  Tail + fuselage  : fixed at initial estimates
  Objective        : maximize trimmed L/D, subject to a landing (flaps-out)
                     stall-speed penalty from flap_deflection
  Trim             : tail incidence solved analytically at every evaluation
                     (CM_cg = 0 exactly — never a penalty)

Phase 2 — TAIL + FUSELAGE + VERTICAL TAIL
  Design variables : [SHT_frac, AR_HT, taper_HT, tail_arm, HT_sweep,
                      HT_twist, HT_dihedral, SVT_frac, AR_VT, taper_VT,
                      VT_sweep, fuse_length, fuse_fineness]
  Wing             : fixed at Phase 1 optimum
  Objective        : maximize trimmed L/D (trim still exact)
  Constraint       : |tail_incidence| ≤ 6° (soft penalty if violated)

CG is always auto-placed at NP − target_sm.  It is never a user input.
Wing incidence is derived after optimisation as clip(alpha_cruise, 2–4°).

twist/dihedral bounds allow both signs (wash-in as well as washout, anhedral
as well as dihedral) — the stall check (via wing_CLmax) and the lateral-
directional check (Cl_beta vs Cn_beta, in lateral_stability.py) already
penalise the physically bad corners, so there is no need to pre-exclude
either sign by bounds alone.
"""

import numpy as np
from scipy.optimize import minimize
from analysis import compute_trimmed_ld
import dynamic_stability
import lateral_stability


# ── Default estimates used by Phase 1 while tail is not yet optimised ─────────
TAIL_INIT = {
    'SHT_frac':    0.22,
    'AR_HT':       4.0,
    'taper_HT':    0.50,
    'tail_arm':    4.0,    # × mean chord — must exceed fuse_len/c_bar physically
    'HT_sweep':    0.0,
    'HT_twist':    0.0,
    'HT_dihedral': 0.0,
    'SVT_frac':    0.10,
    'AR_VT':       1.8,
    'taper_VT':    0.50,
    'VT_sweep':    10.0,
}
FUSE_INIT = {
    'length':    0.80,
    'fineness':  6.5,
}

# ── Design variable definitions ───────────────────────────────────────────────
WING_VAR_SPECS = [
    {'key': 'AR',              'label': 'Aspect Ratio (AR)',   'lo': 5.0,  'hi': 12.0, 'x0': 7.0},
    {'key': 'taper',           'label': 'Taper Ratio',         'lo': 0.25, 'hi': 0.55, 'x0': 0.40},
    {'key': 'twist',           'label': 'Twist [deg, +wash-out]', 'lo': -2.0, 'hi': 4.0, 'x0': 1.5},
    {'key': 'sweep',           'label': 'Sweep c/4 [deg]',     'lo': 0.0,  'hi': 8.0,  'x0': 0.0},
    {'key': 'dihedral',        'label': 'Dihedral [deg, ±anhedral]', 'lo': -3.0, 'hi': 5.0, 'x0': 1.0},
    {'key': 'flap_deflection', 'label': 'Flap Deflection [deg]', 'lo': 0.0, 'hi': 40.0, 'x0': 15.0},
]
TAIL_VAR_SPECS = [
    {'key': 'SHT_frac',   'label': 'SHT / S',                  'lo': 0.15, 'hi': 0.40,  'x0': 0.22},
    {'key': 'AR_HT',      'label': 'HT Aspect Ratio',          'lo': 3.0,  'hi': 6.0,   'x0': 4.0},
    {'key': 'taper_HT',   'label': 'HT Taper Ratio',           'lo': 0.35, 'hi': 0.80,  'x0': 0.50},
    {'key': 'tail_arm',   'label': 'Tail Arm [× chord]',       'lo': 3.5,  'hi': 6.0,   'x0': 4.0},
    {'key': 'HT_sweep',   'label': 'HT Sweep [deg]',           'lo': 0.0,  'hi': 25.0,  'x0': 5.0},
    {'key': 'HT_twist',   'label': 'HT Washout [deg]',         'lo': 0.0,  'hi': 3.0,   'x0': 0.0},
    {'key': 'HT_dihedral','label': 'HT Dihedral [deg]',        'lo': -10.0,'hi': 10.0,  'x0': 0.0},
]
FUSE_VAR_SPECS = [
    {'key': 'fuse_len', 'label': 'Fuselage Length [m]',     'lo': 0.50, 'hi': 1.50, 'x0': 0.80},
    {'key': 'fuse_fin', 'label': 'Fineness Ratio (L/d)',    'lo': 5.0,  'hi': 8.0,  'x0': 6.5},
]
VT_VAR_SPECS = [
    {'key': 'SVT_frac', 'label': 'SVT / S',                 'lo': 0.06, 'hi': 0.18, 'x0': 0.10},
    {'key': 'AR_VT',    'label': 'VT Aspect Ratio',         'lo': 1.2,  'hi': 2.5,  'x0': 1.8},
    {'key': 'taper_VT', 'label': 'VT Taper Ratio',          'lo': 0.35, 'hi': 0.70, 'x0': 0.50},
    {'key': 'VT_sweep', 'label': 'VT Sweep [deg]',          'lo': 0.0,  'hi': 30.0, 'x0': 10.0},
]


# ── Penalty helpers ────────────────────────────────────────────────────────────

def _sm_penalty(SM, lo=0.05, hi=0.15, weight=25.0):
    return weight * (max(0.0, lo - SM) + max(0.0, SM - hi)) ** 2


def _stall_penalty(stall_ratio, limit=0.70, weight=50.0):
    return weight * max(0.0, stall_ratio - limit) ** 2


def _trim_penalty(tail_inc_deg, limit=6.0, weight=15.0):
    return weight * max(0.0, abs(tail_inc_deg) - limit) ** 2


def _landing_penalty(V_stall_land, limit, weight=40.0):
    return weight * max(0.0, V_stall_land - limit) ** 2


# ── Step log helper ────────────────────────────────────────────────────────────

def _step_entry(var_names, x_now, x_prev, res, penalty, phase):
    """Build a dict describing what changed this step and the resulting performance."""
    changes, unchanged = [], []
    for name, xi, xp in zip(var_names, x_now, x_prev):
        delta = xi - xp
        if abs(delta) > 1e-6 * max(abs(xi), 1.0):
            changes.append(f'{name}: {xp:.4g}→{xi:.4g}({delta:+.4g})')
        else:
            unchanged.append(f'{name}: {xi:.4g}')
    return {
        'phase':    phase,
        'changes':  changes,
        'unchanged': unchanged,
        'LD':       res['LD'],
        'SM':       res['SM'],
        'tail_inc': res['tail_inc_deg'],
        'trimmable': res['trimmable'],
        'V_stall':  res['V_stall'],
        'stall_ok': res['stall_ok'],
        'penalty':  penalty,
        'geom':     res,   # full geometry for planform redraw
    }


# ── Phase 1: Wing optimisation ────────────────────────────────────────────────

def optimize_wing(req, wing_specs, tail_fixed, fuse_fixed,
                  wing_frozen=None, step_callback=None):
    """
    Optimize active wing variables for maximum trimmed L/D.

    wing_specs  : list of specs for ACTIVE (checked) wing variables only
    wing_frozen : dict {key: value} for INACTIVE (unchecked) wing variables;
                  defaults to each spec's x0 if omitted
    tail_fixed  : dict with SHT_frac, AR_HT, taper_HT, tail_arm
    fuse_fixed  : dict with length, fineness
    """
    # Build a full default dict; active vars will override per step
    defaults = {s['key']: s['x0'] for s in WING_VAR_SPECS}
    frozen   = {**defaults, **(wing_frozen or {})}

    var_names = [s['key'] for s in wing_specs]
    x0        = [s['x0']  for s in wing_specs]
    bounds    = [(s['lo'], s['hi']) for s in wing_specs]

    prev_x   = [np.array(x0)]
    best     = {'LD': -1e9, 'res': None}
    step_log = []

    def _params(x):
        p = dict(frozen)
        for k, v in zip(var_names, x):
            p[k] = v
        return (p['AR'], p['taper'], p['twist'], p['sweep'], p['dihedral'],
                p['flap_deflection'])

    def _tail_kwargs():
        return dict(
            SVT_frac=tail_fixed.get('SVT_frac', 0.10),
            AR_VT=tail_fixed.get('AR_VT', 1.8),
            taper_VT=tail_fixed.get('taper_VT', 0.50),
            HT_sweep_deg=tail_fixed.get('HT_sweep', 0.0),
            HT_twist_deg=tail_fixed.get('HT_twist', 0.0),
            HT_dihedral_deg=tail_fixed.get('HT_dihedral', 0.0),
            VT_sweep_deg=tail_fixed.get('VT_sweep', 0.0),
        )

    def _obj(x):
        AR, taper, twist, sweep, dihedral, flap_deflection = _params(x)
        try:
            res = compute_trimmed_ld(
                AR, taper, twist,
                tail_fixed['SHT_frac'], tail_fixed['AR_HT'],
                tail_fixed['taper_HT'], tail_fixed['tail_arm'],
                fuse_fixed['length'], fuse_fixed['fineness'],
                req, sweep_deg=sweep, dihedral_deg=dihedral,
                flap_deflection_deg=flap_deflection,
                **_tail_kwargs(),
            )
        except Exception:
            return 500.0

        # Penalise excess washout: twist beyond minimum safe level hurts CDi at
        # off-design CL and adds manufacturing complexity.  twist_safe ~ (1-λ)*2°
        # ensures at least enough washout for stall safety; excess is penalised.
        # max(0, twist) keeps this from rewarding negative twist (wash-in) —
        # wash-in is instead discouraged by the stall-ratio penalty below,
        # since it moves tip stall onset earlier (wing_CLmax critical section).
        twist_safe = max(0.0, (1.0 - taper) * 2.0)
        penalty_washout = 0.4 * max(0.0, twist - twist_safe) ** 2 + 0.02 * max(0.0, twist)

        penalty = (_sm_penalty(res['SM']) +
                   _stall_penalty(res['stall_ratio']) +
                   _trim_penalty(res['tail_inc_deg']) +
                   _landing_penalty(res['V_stall_land'], req.V_stall_land_ms) +
                   penalty_washout +
                   dynamic_stability.optimizer_penalty(res['dyn_modes']) +
                   lateral_stability.optimizer_penalty(res['Cn_beta'], res['Cl_beta'], req))

        entry = _step_entry(var_names, x, prev_x[0], res, penalty, 'wing')
        step_log.append(entry)
        prev_x[0] = x.copy()

        if res['LD'] > best['LD'] and penalty < 1.0:
            best['LD']  = res['LD']
            best['res'] = res

        if step_callback:
            step_callback(entry, x.copy(), best.get('res'))

        return -(res['LD']) + penalty

    if not wing_specs:
        # All wing vars frozen — evaluate once and return
        AR, taper, twist, sweep, dihedral, flap_deflection = _params([])
        res = compute_trimmed_ld(
            AR, taper, twist,
            tail_fixed['SHT_frac'], tail_fixed['AR_HT'],
            tail_fixed['taper_HT'], tail_fixed['tail_arm'],
            fuse_fixed['length'], fuse_fixed['fineness'],
            req, sweep_deg=sweep, dihedral_deg=dihedral,
            flap_deflection_deg=flap_deflection,
            **_tail_kwargs(),
        )
        res.update({'converged': True, 'iterations': 0, 'step_log': [], 'phase': 'wing'})
        return res

    sol = minimize(_obj, x0, method='SLSQP', bounds=bounds,
                   options={'ftol': 1e-7, 'maxiter': 300})

    AR, taper, twist, sweep, dihedral, flap_deflection = _params(sol.x)
    final_res = compute_trimmed_ld(
        AR, taper, twist,
        tail_fixed['SHT_frac'], tail_fixed['AR_HT'],
        tail_fixed['taper_HT'], tail_fixed['tail_arm'],
        fuse_fixed['length'], fuse_fixed['fineness'],
        req, sweep_deg=sweep, dihedral_deg=dihedral,
        flap_deflection_deg=flap_deflection,
        **_tail_kwargs(),
    )
    final_res.update({
        'converged': sol.success,
        'iterations': len(step_log),
        'step_log': step_log,
        'phase': 'wing',
    })
    return final_res


# ── Phase 2: Tail + fuselage optimisation ─────────────────────────────────────

def optimize_tail(req, wing_result, tail_specs, fuse_specs,
                  tail_frozen=None, fuse_frozen=None, step_callback=None,
                  vt_specs=None, vt_frozen=None):
    """
    Optimize active tail + fuselage (+ vertical tail) variables for maximum
    trimmed L/D with wing fixed.

    tail_specs   : list of specs for ACTIVE horizontal-tail variables only
    fuse_specs   : list of specs for ACTIVE fuselage variables only
    vt_specs     : list of specs for ACTIVE vertical-tail variables only
    tail_frozen  : dict {key: value} for inactive tail variables
    fuse_frozen  : dict {key: value} for inactive fuselage variables
    vt_frozen    : dict {key: value} for inactive vertical-tail variables
    """
    AR       = wing_result['AR']
    taper    = wing_result['taper']
    twist    = wing_result['twist_deg']
    sweep    = wing_result['sweep_deg']
    dihedral = wing_result['dihedral_deg']
    flap_deflection = wing_result.get('flap_deflection_deg', 0.0)

    vt_specs = vt_specs or []

    tail_defaults = {s['key']: s['x0'] for s in TAIL_VAR_SPECS}
    fuse_defaults = {s['key']: s['x0'] for s in FUSE_VAR_SPECS}
    vt_defaults   = {s['key']: s['x0'] for s in VT_VAR_SPECS}
    tail_frz = {**tail_defaults, **(tail_frozen or {})}
    fuse_frz = {**fuse_defaults, **(fuse_frozen or {})}
    vt_frz   = {**vt_defaults,   **(vt_frozen or {})}

    all_specs  = tail_specs + fuse_specs + vt_specs
    var_names  = [s['key'] for s in all_specs]
    x0         = [s['x0']  for s in all_specs]
    bounds     = [(s['lo'], s['hi']) for s in all_specs]

    n_tail   = len(tail_specs)
    n_fuse   = len(fuse_specs)
    prev_x   = [np.array(x0)] if x0 else [np.array([])]
    best     = {'LD': -1e9, 'res': None}
    step_log = []

    def _params(x):
        tp = dict(tail_frz)
        fp = dict(fuse_frz)
        vp = dict(vt_frz)
        for k, v in zip(var_names[:n_tail], x[:n_tail]):
            tp[k] = v
        for k, v in zip(var_names[n_tail:n_tail + n_fuse], x[n_tail:n_tail + n_fuse]):
            fp[k] = v
        for k, v in zip(var_names[n_tail + n_fuse:], x[n_tail + n_fuse:]):
            vp[k] = v
        return (tp['SHT_frac'], tp['AR_HT'], tp['taper_HT'], tp['tail_arm'],
                tp['HT_sweep'], tp['HT_twist'], tp['HT_dihedral'],
                fp['fuse_len'], fp['fuse_fin'],
                vp['SVT_frac'], vp['AR_VT'], vp['taper_VT'], vp['VT_sweep'])

    def _obj(x):
        (SHT_frac, AR_HT, taper_HT, tail_arm, HT_sweep, HT_twist, HT_dihedral,
         fuse_len, fuse_fin, SVT_frac, AR_VT, taper_VT, VT_sweep) = _params(x)
        try:
            res = compute_trimmed_ld(
                AR, taper, twist,
                SHT_frac, AR_HT, taper_HT, tail_arm,
                fuse_len, fuse_fin,
                req, sweep_deg=sweep, dihedral_deg=dihedral,
                flap_deflection_deg=flap_deflection,
                SVT_frac=SVT_frac, AR_VT=AR_VT, taper_VT=taper_VT,
                HT_sweep_deg=HT_sweep, HT_twist_deg=HT_twist,
                HT_dihedral_deg=HT_dihedral, VT_sweep_deg=VT_sweep,
            )
        except Exception:
            return 500.0

        penalty = (_trim_penalty(res['tail_inc_deg']) +
                   _stall_penalty(res['stall_ratio']) +
                   _landing_penalty(res['V_stall_land'], req.V_stall_land_ms) +
                   dynamic_stability.optimizer_penalty(res['dyn_modes']) +
                   lateral_stability.optimizer_penalty(res['Cn_beta'], res['Cl_beta'], req))

        entry = _step_entry(var_names, x, prev_x[0], res, penalty, 'tail')
        step_log.append(entry)
        prev_x[0] = x.copy()

        if res['LD'] > best['LD'] and penalty < 1.0:
            best['LD']  = res['LD']
            best['res'] = res

        if step_callback:
            step_callback(entry, x.copy(), best.get('res'))

        return -(res['LD']) + penalty

    if not all_specs:
        # All tail/fuse/VT vars frozen — evaluate once and return
        (SHT_frac, AR_HT, taper_HT, tail_arm, HT_sweep, HT_twist, HT_dihedral,
         fuse_len, fuse_fin, SVT_frac, AR_VT, taper_VT, VT_sweep) = _params([])
        res = compute_trimmed_ld(
            AR, taper, twist,
            SHT_frac, AR_HT, taper_HT, tail_arm,
            fuse_len, fuse_fin,
            req, sweep_deg=sweep, dihedral_deg=dihedral,
            flap_deflection_deg=flap_deflection,
            SVT_frac=SVT_frac, AR_VT=AR_VT, taper_VT=taper_VT,
            HT_sweep_deg=HT_sweep, HT_twist_deg=HT_twist,
            HT_dihedral_deg=HT_dihedral, VT_sweep_deg=VT_sweep,
        )
        res.update({'converged': True, 'iterations': 0, 'step_log': [], 'phase': 'tail'})
        return res

    # Physical constraint: tail moment arm ≥ fuselage length
    # tail_arm is in units of mean chord; c_bar = b/AR where b = sqrt(AR*S)
    c_bar_w = np.sqrt(AR * req.wing_area) / AR

    def _arm_ge_fuse(x):
        (SHT_frac, AR_HT, taper_HT, tail_arm, HT_sweep, HT_twist, HT_dihedral,
         fuse_len, fuse_fin, SVT_frac, AR_VT, taper_VT, VT_sweep) = _params(x)
        return tail_arm * c_bar_w - fuse_len   # ≥ 0  (tail arm must exceed fuselage length)

    constraints = [{'type': 'ineq', 'fun': _arm_ge_fuse}]

    sol = minimize(_obj, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints,
                   options={'ftol': 1e-7, 'maxiter': 300})

    (SHT_frac, AR_HT, taper_HT, tail_arm, HT_sweep, HT_twist, HT_dihedral,
     fuse_len, fuse_fin, SVT_frac, AR_VT, taper_VT, VT_sweep) = _params(sol.x)
    final_res = compute_trimmed_ld(
        AR, taper, twist,
        SHT_frac, AR_HT, taper_HT, tail_arm,
        fuse_len, fuse_fin,
        req, sweep_deg=sweep, dihedral_deg=dihedral,
        flap_deflection_deg=flap_deflection,
        SVT_frac=SVT_frac, AR_VT=AR_VT, taper_VT=taper_VT,
        HT_sweep_deg=HT_sweep, HT_twist_deg=HT_twist,
        HT_dihedral_deg=HT_dihedral, VT_sweep_deg=VT_sweep,
    )
    final_res.update({
        'converged': sol.success,
        'iterations': len(step_log),
        'step_log': step_log,
        'phase': 'tail',
    })
    return final_res


# ── Combined: run both phases sequentially ────────────────────────────────────

def optimize_all(req, wing_specs, tail_specs, fuse_specs,
                 tail_fixed=None, fuse_fixed=None, step_callback=None,
                 vt_specs=None, vt_fixed=None):
    """
    Run Phase 1 (wing) then Phase 2 (tail+fuselage+VT) and return both results.

    step_callback(entry, x, best_res) is forwarded to both phases.
    """
    tf = tail_fixed or TAIL_INIT
    ff = fuse_fixed or FUSE_INIT

    wing_res = optimize_wing(req, wing_specs, tf, ff, step_callback=step_callback)
    tail_res = optimize_tail(req, wing_res, tail_specs, fuse_specs,
                             step_callback=step_callback,
                             vt_specs=vt_specs, vt_frozen=vt_fixed)
    return wing_res, tail_res


# ── Geometry dict for Flow5 XML ───────────────────────────────────────────────

def result_to_geom(res, mtow_kg):
    """
    Convert a compute_trimmed_ld result dict into the geom dict that
    flow5_xml.write_plane_xml() requires.
    """
    return {
        'b':            res['b'],
        'cr':           res['cr'],
        'ct':           res['ct'],
        'AR':           res['AR'],
        'taper':        res['taper'],
        'twist_deg':    res['twist_deg'],
        'incidence_deg': res['incidence_deg'],
        'dihedral_deg': res['dihedral_deg'],
        'sweep_deg':    res['sweep_deg'],
        'SHT':          res['SHT'],
        'LHT':          res['LHT'],
        'b_HT':         res['b_HT'],
        'cr_HT':        res['cr_HT'],
        'ct_HT':        res['ct_HT'],
        'HT_sweep_deg':    res.get('HT_sweep_deg', 0.0),
        'HT_twist_deg':    res.get('HT_twist_deg', 0.0),
        'HT_dihedral_deg': res.get('HT_dihedral_deg', 0.0),
        'tail_inc_deg': res['tail_inc_deg'],
        'SVT':          res['SVT'],
        'AR_VT':        res.get('AR_VT', 1.8),
        'taper_VT':     res.get('taper_VT', 0.50),
        'b_VT':         res.get('b_VT'),
        'cr_VT':        res.get('cr_VT'),
        'ct_VT':        res.get('ct_VT'),
        'VT_sweep_deg': res.get('VT_sweep_deg', 0.0),
        'flap_deflection_deg': res.get('flap_deflection_deg', 0.0),
        'mtow_kg':      mtow_kg,
        'cg_frac_mac':  res['cg_frac'],
    }

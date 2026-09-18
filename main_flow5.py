"""
Flow5-driven wing optimizer.
Run:  python3 main_flow5.py

Loop:
  1. Nelder-Mead proposes new [AR, taper, twist]
  2. Python writes plane.xml + polar.xml + script.xml
  3. Flow5 runs headlessly, outputs CSV polar
  4. Python reads CSV, extracts L/D at cruise CL
  5. Optimizer uses L/D as the score → adjusts geometry
  6. Repeat until convergence

Flow5 is the actual VLM solver — not an approximation.
"""
import os
import sys
import numpy as np
from scipy.optimize import minimize

from requirements import DroneRequirements
from analysis   import (CL_required, static_margin, pitch_trim, stall_check,
                        control_surfaces, find_cruise_alpha)
from flow5_xml  import (write_plane_xml, write_polar_xml, write_script_xml,
                         geometry_from_result, XML_DIR, OUTPUT_DIR)
from flow5_run  import run_and_parse

SCRIPT_PATH = os.path.join(XML_DIR, "drone_script.xml")

_iter   = [0]
_best   = [None]


def _build_geom(x, req, tail_inc_deg=None, optimal_incidence_deg=None):
    """Convert optimizer variables to geometry dict."""
    AR, taper, twist = x
    b   = np.sqrt(AR * req.wing_area)
    c_bar = b / AR
    cr  = 2 * req.wing_area / (b * (1 + taper))
    ct  = taper * cr

    SHT   = req.SHT_fraction  * req.wing_area
    LHT   = req.tail_arm_chords * c_bar
    AR_HT = 4.0
    b_HT  = np.sqrt(AR_HT * SHT)
    cr_HT = 2 * SHT / (b_HT * 1.4)
    ct_HT = 0.4 * cr_HT
    SVT   = req.SVT_fraction * req.wing_area

    return {
        'AR': AR, 'taper': taper, 'twist_deg': twist,
        'b': b, 'cr': cr, 'ct': ct,
        'incidence_deg':         req.incidence_deg,
        'optimal_incidence_deg': optimal_incidence_deg or req.incidence_deg,
        'dihedral_deg':          req.dihedral_deg,
        'sweep_deg':             req.sweep_deg,
        'SHT': SHT, 'LHT': LHT, 'b_HT': b_HT, 'cr_HT': cr_HT, 'ct_HT': ct_HT,
        'tail_inc_deg':          tail_inc_deg or 0.0,
        'SVT': SVT,
        'mtow_kg':       req.mtow_kg,
        'cg_frac_mac':   req.cg_frac_mac,
    }


def objective(x, req):
    AR, taper, twist = x
    _iter[0] += 1

    CL_req = CL_required(req.mtow_kg, req.wing_area, req.cruise_speed_ms)

    # LLT pre-estimates: cruise alpha (to centre the Flow5 sweep) and the
    # incidence that levels the fuselage at cruise
    alpha_c = find_cruise_alpha(AR, taper, twist, req.incidence_deg,
                                CL_req, req.CL_alpha_2d, req.alpha_L0_deg,
                                req.wing_area)
    inc_opt = float(np.clip(req.incidence_deg + alpha_c, 2.0, 4.0))

    # Tail incidence estimate from LLT for the run; corrected from Flow5 Cm below
    _, _, tail_inc, _, VHT = pitch_trim(
        AR, req.wing_area, req.SHT_fraction, req.tail_arm_chords,
        req.CL_alpha_2d, req.CM_ac, CL_req, req.cg_frac_mac)

    geom = _build_geom(x, req,
                       tail_inc_deg=tail_inc,
                       optimal_incidence_deg=inc_opt)

    # Write XML files for this geometry — viscous QUADS, narrow sweep
    write_plane_xml(geom)
    write_polar_xml(req.cruise_speed_ms,
                    ref_area=req.wing_area,
                    ref_span=geom['b'],
                    ref_chord=geom['b'] / AR,
                    method='QUADS', viscous=True)
    write_script_xml(aoa_min=int(np.floor(alpha_c - 3)),
                     aoa_max=int(np.ceil(alpha_c + 4)), aoa_step=1)

    # Run Flow5
    perf, err = run_and_parse(SCRIPT_PATH, CL_req, timeout=240)

    if perf is None:
        print(f"  step {_iter[0]:3d}  ⚠ Flow5 failed: {err}")
        return 1000.0  # large penalty — skip this point

    LD = perf['LD']

    # Stability & trim from the Flow5 Cm(CL) curve when available
    SM_f5 = perf.get('SM_flow5')
    if SM_f5 is not None:
        x_np    = req.cg_frac_mac + SM_f5           # NP in MAC fractions
        cg_new  = x_np - 0.10                       # place CG for 10% SM
        Cm_res  = perf['Cm'] + CL_req * (cg_new - req.cg_frac_mac)
        CLa_t   = 2 * np.pi * 4.0 / 6.0             # AR_HT = 4
        tail_inc_req = tail_inc + float(np.degrees(Cm_res / (VHT * CLa_t)))
        penalty_trim = 10.0 * max(0.0, abs(tail_inc_req) - 5.0)
        SM_report = SM_f5
        cg_balanced = cg_new
    else:
        cg_balanced = None
        SM_report, _, _, _, _ = static_margin(AR, req.wing_area, req.SHT_fraction,
                                              req.tail_arm_chords, req.CL_alpha_2d,
                                              req.cg_frac_mac)
        tail_inc_req = tail_inc
        penalty_trim = 10.0 * max(0.0, abs(tail_inc) - 5.0)

    # Penalty: stall margin
    _, stall_ratio, _ = stall_check(req.mtow_kg, req.wing_area, req.CL_max_2d,
                                     CL_req, req.stall_speed_ms)
    penalty_stall = 20.0 * max(0.0, stall_ratio - 0.70)

    score = -LD + penalty_trim + penalty_stall

    print(f"  step {_iter[0]:3d}  AR={AR:.2f}  taper={taper:.3f}  twist={twist:.2f}°"
          f"  →  L/D={LD:.2f}   SM(F5)={SM_report*100:.0f}%  score={score:.3f}")

    if _best[0] is None or LD > _best[0]['LD']:
        _best[0] = {**perf, 'AR': AR, 'taper': taper, 'twist_deg': twist,
                    'geom': geom, 'SM': SM_report, 'tail_inc_deg': tail_inc_req,
                    'cg_balanced': cg_balanced}

    return score


def optimize_with_flow5(req):
    _iter[0] = 0
    _best[0] = None

    print("=" * 62)
    print("  Flow5-driven Wing Optimizer")
    print(f"  Forward model: Flow5 v7.57 QUADS, viscous (XFoil on the fly)")
    print(f"  Method: Nelder-Mead  |  Variables: AR, taper, twist")
    print(f"  Each step runs Flow5 headlessly and reads the CSV polar.")
    print("=" * 62)
    print(f"\n  Fixed: S={req.wing_area:.4f} m²  V={req.cruise_speed_ms} m/s"
          f"  MTOW={req.mtow_kg} kg\n")

    sol = minimize(
        objective, req.x0, args=(req,),
        method='Nelder-Mead',
        bounds=req.bounds,
        options={'xatol': 0.05, 'fatol': 0.1, 'maxiter': 80, 'disp': False},
    )

    best = _best[0]
    if best is None:
        print("\n⚠  Optimizer failed — no successful Flow5 runs.")
        return None

    # Re-write the plane XML with the BEST geometry (the file on disk holds the
    # last-evaluated one), carrying the trimmed tail and the balanced CG.
    best_geom = dict(best['geom'])
    best_geom['tail_inc_deg'] = best['tail_inc_deg']
    if best.get('cg_balanced') is not None:
        best_geom['cg_frac_mac'] = best['cg_balanced']
    write_plane_xml(best_geom)

    print(f"\n{'─'*62}")
    print(f"  CONVERGED after {_iter[0]} Flow5 runs")
    print(f"  Best L/D  : {best['LD']:.2f}  (airframe; add fuselage CD0 for aircraft L/D)")
    print(f"  AR        : {best['AR']:.2f}")
    print(f"  Taper     : {best['taper']:.3f}")
    print(f"  Twist     : {best['twist_deg']:.2f}°")
    print(f"  NP (Flow5): {(req.cg_frac_mac + best['SM'])*100:.1f}% MAC")
    if best.get('cg_balanced') is not None:
        print(f"  CG placed : {best['cg_balanced']*100:.1f}% MAC  (SM = 10%)")
    print(f"  Tail inc  : {best['tail_inc_deg']:.2f}°  (trims at cruise with CG above)")
    print(f"  CL        : {best['CL']:.4f}")
    print(f"  CD        : {best['CD']:.5f}")
    print(f"{'─'*62}\n")

    print("  The optimized plane.xml (best geometry, trimmed, balanced) is at:")
    print(f"  {os.path.join(XML_DIR, 'drone_plane.xml')}")
    print("\n  Open Flow5 → File → Import → load drone_plane.xml")
    print("  to visualise and further analyse the geometry.\n")

    return best


if __name__ == "__main__":
    req = DroneRequirements()
    optimize_with_flow5(req)

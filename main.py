"""
Flow5 Wing Optimizer — CLI entry point.
Run:  python3 main.py

Runs Phase 1 (wing) then Phase 2 (tail + fuselage) and prints a full report.
For the interactive GUI run: python3 drone_mdo_gui.py
"""
from requirements import DroneRequirements
from optimizer import (
    optimize_all,
    WING_VAR_SPECS, TAIL_VAR_SPECS, FUSE_VAR_SPECS,
    TAIL_INIT, FUSE_INIT,
)


def div(title=""):
    w = 62
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*(w - pad - len(title) - 2)}")
    else:
        print("─" * w)


def ok(condition):
    return "✓" if condition else "✗"


def report(wing_res, tail_res, req):
    # Use tail_res as the final combined result (wing is fixed inside it)
    res = tail_res
    S = req.wing_area

    div("PHASE 1 — WING OPTIMISATION")
    print(f"  Status          : {'✓ converged' if wing_res['converged'] else '⚠ did not converge'}"
          f"  ({wing_res['iterations']} steps)")
    print(f"  L/D (wing-only eval)  : {wing_res['LD']:.2f}")
    print(f"  AR              : {wing_res['AR']:.2f}")
    print(f"  Taper           : {wing_res['taper']:.3f}")
    print(f"  Washout         : {wing_res['twist_deg']:.2f}°")
    print(f"  Sweep (c/4)     : {wing_res['sweep_deg']:.1f}°")
    print(f"  Dihedral        : {wing_res['dihedral_deg']:.1f}°")
    print(f"  Tail inc (Ph1)  : {wing_res['tail_inc_deg']:.2f}°  (exact trim at fixed tail estimate)")

    div("PHASE 2 — TAIL + FUSELAGE OPTIMISATION")
    print(f"  Status          : {'✓ converged' if res['converged'] else '⚠ did not converge'}"
          f"  ({res['iterations']} steps)")
    print(f"  L/D (trimmed)   : {res['LD']:.2f}")
    print(f"  SHT / S         : {res['SHT']/S:.3f}")
    print(f"  HT Aspect Ratio : {res['AR_HT']:.2f}")
    print(f"  Tail arm        : {res['LHT']/res['b']*res['AR']:.2f}× mean chord")
    print(f"  Tail inc (trim) : {res['tail_inc_deg']:.2f}°  {'✓' if res['trimmable'] else '⚠ authority exceeded'}")
    print(f"  Fuselage length : {res['fuse_length']:.3f} m")
    print(f"  Fineness L/d    : {res['fuse_fineness']:.1f}")

    div("FINAL CONFIGURATION — WING")
    b    = res['b']
    c_bar = b / res['AR']
    Re   = 1.225 * req.cruise_speed_ms * c_bar / 1.789e-5
    print(f"  Aerofoil        : E387")
    print(f"  Span            : {b:.3f} m")
    print(f"  Root chord      : {res['cr']:.3f} m")
    print(f"  Tip chord       : {res['ct']:.3f} m")
    print(f"  Mean chord      : {c_bar:.3f} m")
    print(f"  Wing area       : {S:.4f} m²")
    print(f"  Aspect ratio    : {res['AR']:.2f}")
    print(f"  Taper ratio     : {res['taper']:.3f}")
    print(f"  Sweep c/4       : {res['sweep_deg']:.1f}°")
    print(f"  Dihedral        : {res['dihedral_deg']:.1f}°")
    print(f"  Geometric twist : {res['twist_deg']:.2f}° (washout)")
    print(f"  Wing incidence  : {res['incidence_deg']:.1f}°  (= cruise AoA, clamped 2–4°)")

    div("FINAL CONFIGURATION — TAIL")
    print(f"  Aerofoil        : NACA 0009 (symmetric)")
    print(f"  H-stab area     : {res['SHT']:.4f} m²  ({res['SHT']/S*100:.0f}% wing area)")
    print(f"  H-stab span     : {res['b_HT']:.3f} m")
    print(f"  H-stab root c   : {res['cr_HT']:.3f} m")
    print(f"  H-stab tip  c   : {res['ct_HT']:.3f} m")
    print(f"  Tail moment arm : {res['LHT']:.3f} m")
    print(f"  V-stab area     : {res['SVT']:.4f} m²")
    print(f"  Tail incidence  : {res['tail_inc_deg']:.2f}°  {ok(res['trimmable'])}")

    div("STABILITY")
    print(f"  Neutral point   : {res['x_NP']*100:.1f}% MAC")
    print(f"  CG (auto)       : {res['cg_frac']*100:.1f}% MAC  (NP − {req.target_sm*100:.0f}% SM)")
    print(f"  Static margin   : {res['SM']*100:.1f}% MAC  {ok(0.05 <= res['SM'] <= 0.15)}")
    print(f"  VHT             : {res['VHT']:.3f}  (typical UAV range 0.35–0.80)")

    div("LONGITUDINAL TRIM  (ΣM_cg = 0)")
    print(f"  CM_ac (E387)    : {req.CM_ac:.3f}")
    print(f"  CM_cg (untrim)  : {res['CM_cg_unt']:.4f}")
    print(f"  Tail CL for trim: {res['CL_tail']:.4f}  ({'downforce' if res['CL_tail'] < 0 else 'upforce'})")
    print(f"  Tail incidence  : {res['tail_inc_deg']:.2f}°  {ok(res['trimmable'])}  (limit ±6°)")

    div("DRAG BREAKDOWN")
    print(f"  CDi wing        : {res['CDi_wing']:.5f}")
    print(f"  CDi tail        : {res['CDi_tail']:.5f}")
    print(f"  CD0 wing        : {res['CD0_wing']:.5f}")
    print(f"  CD0 tail        : {res['CD0_tail']:.5f}")
    print(f"  CD0 fuselage    : {res['CD0_fuse']:.5f}")
    print(f"  CD total        : {res['CD_total']:.5f}")
    print(f"  CL total        : {res['CL_total']:.4f}")
    print(f"  L/D (trimmed)   : {res['LD']:.2f}")

    div("STALL CHECK")
    print(f"  CL_max 2-D      : {req.CL_max_2d:.2f}")
    print(f"  CL at cruise    : {res['CL_req']:.4f}")
    print(f"  V_stall         : {res['V_stall']:.1f} m/s  {ok(res['stall_ok'])}  "
          f"(req {req.stall_speed_ms:.0f} m/s)")
    if not res['stall_ok']:
        cl_need = (req.mtow_kg * 9.81 * 2) / (1.225 * req.stall_speed_ms**2 * S)
        ws_need = 0.5 * 1.225 * req.stall_speed_ms**2 * req.CL_max_2d
        print(f"  ⚠  Need CL_max ≥ {cl_need:.2f} or W/S ≤ {ws_need:.0f} N/m² "
              f"(S ≥ {req.mtow_kg*9.81/ws_need:.3f} m²)")

    div("XFOIL ANALYSIS SETTINGS  (for Flow5)")
    print(f"  Aerofoil        : E387")
    print(f"  Reynolds number : {Re:.0f}")
    print(f"  Mach            : {req.cruise_speed_ms/340:.4f}")
    print(f"  AoA range       : -5° to 20°  step 0.5°")
    print(f"  NCrit           : 9")
    div()


if __name__ == "__main__":
    req = DroneRequirements()

    print("\nRunning two-phase MDO (Phase 1: Wing → Phase 2: Tail+Fuselage)")
    print("Use Ctrl-C to abort, or run drone_mdo_gui.py for the interactive GUI.\n")

    wing_res, tail_res = optimize_all(
        req,
        WING_VAR_SPECS, TAIL_VAR_SPECS, FUSE_VAR_SPECS,
        tail_fixed=TAIL_INIT, fuse_fixed=FUSE_INIT,
    )

    print()
    report(wing_res, tail_res, req)

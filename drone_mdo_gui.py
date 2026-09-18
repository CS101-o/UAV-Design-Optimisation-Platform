"""
drone_mdo_gui.py  —  Two-Phase MDO GUI: Maximize Trimmed L/D

Layout
──────
  Left   : Mission Requirements  (true mission inputs only)
  Middle : Design Variables       (Wing | Tail | Fuselage sections)
  Right  : Planform Canvas        (top-view, updates each iteration)
  Bottom : Control buttons → Best Config Banner → Iteration Log

Optimization
────────────
  Phase 1 (Wing)  : AR, taper, twist, sweep, dihedral, flap deflection
  Phase 2 (Tail)  : SHT/S, AR_HT, taper_HT, tail arm, HT sweep, HT twist,
                    HT dihedral, SVT/S, AR_VT, taper_VT, VT sweep,
                    fuselage length, fuselage fineness
  True trim at every evaluation: tail incidence solved for CM_cg = 0.
  CG auto-placed at NP − target SM (never a user input).
  Every evaluation also checks longitudinal dynamic modes (phugoid,
  short-period) and static lateral-directional stability (weathercock
  Cn_beta from the vertical tail, dihedral effect Cl_beta) against the
  Stability Requirements panel — see dynamic_stability.py / lateral_stability.py.

Verify + Trim button runs Flow5 viscous panel analysis on the best LLT design
for high-fidelity drag validation and trim confirmation.
"""

import os
import sys
import math
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from requirements import DroneRequirements
from optimizer import (WING_VAR_SPECS, TAIL_VAR_SPECS, FUSE_VAR_SPECS, VT_VAR_SPECS,
                       optimize_wing, optimize_tail, result_to_geom)
from flow5_bo_optimizer import run_flow5_bo
from analysis import find_cruise_alpha, compute_trimmed_ld
from flow5_xml import write_plane_xml, XML_DIR
from flow5_run import (run_flow5, find_polar_csv, parse_polar_csv,
                       extract_performance_interp)

RHO  = 1.225
G    = 9.81
NU   = 1.5e-5

CANVAS_W = 420
CANVAS_H = 360


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _build_req_from_gui(mission_vars):
    """Build a DroneRequirements from the GUI mission fields."""
    req = DroneRequirements()
    req.mtow_kg          = float(mission_vars['mtow_kg'].get())
    req.payload_kg       = float(mission_vars['payload_kg'].get())
    req.cruise_speed_ms  = float(mission_vars['cruise_speed_ms'].get())
    req.stall_speed_ms   = float(mission_vars['stall_speed_ms'].get())
    req.wing_loading_Nm2 = float(mission_vars['wing_loading_Nm2'].get())
    req.propulsive_eta   = float(mission_vars['propulsive_eta'].get())
    req.battery_wh       = float(mission_vars['battery_wh'].get())
    req.wing_area        = (req.mtow_kg * G) / req.wing_loading_Nm2
    req.V_stall_land_ms  = float(mission_vars['V_stall_land_ms'].get())
    # ── Stability requirements (previously hidden constants) ────────────
    req.target_sm             = float(mission_vars['target_sm'].get())
    req.dyn_ph_zeta_min        = float(mission_vars['dyn_ph_zeta_min'].get())
    req.dyn_sp_zeta_min        = float(mission_vars['dyn_sp_zeta_min'].get())
    req.dyn_sp_zeta_max        = float(mission_vars['dyn_sp_zeta_max'].get())
    req.target_cn_beta_min     = float(mission_vars['target_cn_beta_min'].get())
    req.target_cl_beta_max_ratio = float(mission_vars['target_cl_beta_max_ratio'].get())
    return req


# ── Planform canvas drawing ───────────────────────────────────────────────────

def draw_planform(canvas, geom, best_geom=None):
    """
    Render a top-down (plan) view of the complete aircraft on a Tkinter Canvas.

    geom / best_geom are dicts from compute_trimmed_ld().
    best_geom is drawn first in green (background), then the current geom
    is drawn on top in blue so both are visible.
    """
    canvas.delete('all')
    cw = CANVAS_W
    ch = CANVAS_H
    cx = cw // 2
    mg = 16        # margin

    def _scale(geom):
        b         = geom.get('b', 1.2)
        LHT       = geom.get('LHT', 0.55)
        fuse_len  = geom.get('fuse_length', 0.8)
        cr_HT     = geom.get('cr_HT', 0.12)
        # Fit half-span in half canvas AND total length in canvas height
        scale_sp  = (cw / 2 - mg) / (b / 2)
        total_lon = fuse_len * 1.1 + LHT + cr_HT * 1.2   # nose to tail tip
        scale_ln  = (ch - 2 * mg) / total_lon
        return min(scale_sp, scale_ln)

    scale = _scale(geom)

    def _draw_aircraft(g, wing_fill, wing_out, tail_fill, fuse_fill):
        b         = g.get('b',    1.2)
        cr        = g.get('cr',   0.22)
        ct        = g.get('ct',   0.11)
        b_HT      = g.get('b_HT', 0.35)
        cr_HT     = g.get('cr_HT', 0.12)
        ct_HT     = g.get('ct_HT', 0.06)
        LHT       = g.get('LHT', 0.55)
        sweep     = g.get('sweep_deg', 0.0)
        fuse_len  = g.get('fuse_length', 0.8)
        fuse_diam = g.get('fuse_diam', 0.08)

        # --- pixel coordinates ---
        fuse_px   = fuse_len  * scale
        fdiam_px  = max(7, fuse_diam * scale)
        wing_y    = mg + fuse_px * 0.30        # wing root LE at 30% fuselage

        # Fuselage (rounded rectangle approximated by oval)
        canvas.create_oval(cx - fdiam_px / 2, mg,
                           cx + fdiam_px / 2, mg + fuse_px,
                           fill=fuse_fill, outline='#888', width=1)

        # Wing (trapezoidal planform, both halves)
        sw_px = (b / 2) * math.tan(math.radians(sweep)) * scale
        tip_x = cx + b / 2 * scale
        tip_y = wing_y + sw_px
        rw = [cx + fdiam_px / 2, wing_y,
              tip_x,             tip_y,
              tip_x,             tip_y + ct * scale,
              cx + fdiam_px / 2, wing_y + cr * scale]
        lw = [cx - fdiam_px / 2, wing_y,
              cx - b / 2 * scale, tip_y,
              cx - b / 2 * scale, tip_y + ct * scale,
              cx - fdiam_px / 2,  wing_y + cr * scale]
        canvas.create_polygon(rw, fill=wing_fill, outline=wing_out, width=1)
        canvas.create_polygon(lw, fill=wing_fill, outline=wing_out, width=1)

        # H-tail (symmetric trapezoid behind wing root)
        tail_y   = wing_y + LHT * scale
        ht_tip_x = cx + b_HT / 2 * scale
        ht_sw_px = (b_HT / 2) * math.tan(math.radians(5.0)) * scale
        rht = [cx + fdiam_px / 3,  tail_y,
               ht_tip_x,           tail_y + ht_sw_px,
               ht_tip_x,           tail_y + ht_sw_px + ct_HT * scale,
               cx + fdiam_px / 3,  tail_y + cr_HT * scale]
        lht = [cx - fdiam_px / 3,  tail_y,
               cx - b_HT / 2 * scale, tail_y + ht_sw_px,
               cx - b_HT / 2 * scale, tail_y + ht_sw_px + ct_HT * scale,
               cx - fdiam_px / 3,     tail_y + cr_HT * scale]
        canvas.create_polygon(rht, fill=tail_fill, outline=wing_out, width=1)
        canvas.create_polygon(lht, fill=tail_fill, outline=wing_out, width=1)

        # V-tail (thin strip on centreline)
        canvas.create_rectangle(cx - 3, tail_y - cr_HT * 0.3 * scale,
                                 cx + 3, tail_y + cr_HT * scale,
                                 fill='#607080', outline='#405060', width=1)

        # CG and NP markers
        cg_frac = g.get('cg_frac', 0.40)
        x_NP    = g.get('x_NP',    0.50)
        cg_y    = wing_y + cg_frac * cr * scale
        np_y    = wing_y + x_NP    * cr * scale
        canvas.create_oval(cx - 5, cg_y - 5, cx + 5, cg_y + 5,
                           fill='#cc2222', outline='#881111', width=1)
        canvas.create_text(cx + 8, cg_y, text='CG', fill='#cc2222',
                           anchor='w', font=('Arial', 7, 'bold'))
        canvas.create_oval(cx - 5, np_y - 5, cx + 5, np_y + 5,
                           fill='', outline='#0055cc', width=2)
        canvas.create_text(cx + 8, np_y, text='NP', fill='#0055cc',
                           anchor='w', font=('Arial', 7))

        # Span dimension arrow
        dim_y = mg + fuse_px + 12
        if dim_y < ch - 4:
            canvas.create_line(cx - b / 2 * scale, dim_y,
                               cx + b / 2 * scale, dim_y,
                               fill='#888', arrow='both', arrowshape=(5, 6, 3))
            canvas.create_text(cx, dim_y + 7, text=f'b = {b:.2f} m',
                               fill='#555', font=('Arial', 7))

    # Draw best in green behind, then current in blue on top
    if best_geom is not None and best_geom is not geom:
        _draw_aircraft(best_geom,
                       wing_fill='#b8e8b8', wing_out='#4a9a4a',
                       tail_fill='#c8f0c8', fuse_fill='#d8f0d8')

    _draw_aircraft(geom,
                   wing_fill='#4a90d9', wing_out='#2c5f9a',
                   tail_fill='#7ab4e8', fuse_fill='#d0d8e8')

    # Performance annotation at top
    LD       = geom.get('LD', 0.0)
    tail_inc = geom.get('tail_inc_deg', 0.0)
    trim_ok  = geom.get('trimmable', False)
    trim_sym = '✓' if trim_ok else '✗'
    canvas.create_text(cx, 8,
                       text=f'L/D = {LD:.2f}   i_tail = {tail_inc:+.1f}°  {trim_sym}',
                       fill='#1a3a6a', font=('Arial', 9, 'bold'))


# ── Worker threads ────────────────────────────────────────────────────────────

class CancelledError(Exception):
    pass


def _mdo_worker(req,
                wing_specs, wing_frozen,
                tail_specs, tail_frozen,
                vt_specs, vt_frozen,
                fuse_specs, fuse_frozen,
                tail_init_fixed, fuse_init_fixed,
                uq, cancel_flag):
    """Run Phase 1 (wing) then Phase 2 (tail+VT+fuselage) sequentially."""
    try:
        def callback(entry, _x, best_res):
            if cancel_flag[0]:
                raise CancelledError()
            uq.put({'type': 'step', 'entry': entry, 'best_res': best_res})

        wing_res = optimize_wing(req, wing_specs, tail_init_fixed, fuse_init_fixed,
                                 wing_frozen=wing_frozen, step_callback=callback)
        uq.put({'type': 'wing_done', 'result': wing_res, 'chain': True})

        tail_res = optimize_tail(req, wing_res, tail_specs, fuse_specs,
                                 tail_frozen=tail_frozen, fuse_frozen=fuse_frozen,
                                 vt_specs=vt_specs, vt_frozen=vt_frozen,
                                 step_callback=callback)
        uq.put({'type': 'tail_done', 'result': tail_res})
    except CancelledError:
        uq.put({'type': 'cancelled'})
    except Exception:
        import traceback
        uq.put({'type': 'error', 'msg': traceback.format_exc()})


def _verify_worker(geom, req, method, uq, cancel_flag):
    """
    Run Flow5 viscous analysis and iterate tail incidence until CM ≈ 0 (trim).

    Each iteration:
      1. Write XML with current tail_inc
      2. Run Flow5 → read CM at cruise CL
      3. Adjust tail_inc by -CM / CLa_tail (Newton step, damped)
      4. Stop when |CM| < 0.005 or after 6 iterations
    """
    try:
        V     = req.cruise_speed_ms
        S     = req.wing_area
        b     = geom['b']
        c_bar = S / b
        CL_cruise = (req.mtow_kg * G) / (0.5 * RHO * V ** 2 * S)

        alpha_est = find_cruise_alpha(
            geom['AR'], geom['taper'], geom['twist_deg'],
            geom['incidence_deg'], CL_cruise,
            2 * math.pi, -4.0, S)

        plane_path  = os.path.join(XML_DIR, 'drone_mdo_best.xml')
        _write_verify_polar_xml(V, S, b, c_bar, method)

        # Approximate tail lift-curve slope for trim correction
        AR_HT   = geom.get('AR_HT', 4.0)
        CLa_tail = 2 * math.pi * AR_HT / (AR_HT + 2)   # /rad

        tail_inc = geom['tail_inc_deg']          # start from LLT solution
        CM_prev  = None
        MAX_ITER = 6
        CM_TOL   = 0.005

        for iteration in range(MAX_ITER):
            if cancel_flag[0]:
                raise CancelledError()

            geom_iter = {**geom, 'tail_inc_deg': tail_inc}
            write_plane_xml(geom_iter, plane_path)

            script_path = _write_verify_script_xml(
                alpha_est - 3, alpha_est + 5, plane_file='drone_mdo_best.xml')

            uq.put({'type': 'status',
                    'msg': f'Verify iter {iteration+1}/{MAX_ITER}: '
                           f'tail_inc={tail_inc:.2f}°  '
                           f'(CM_prev={CM_prev:.4f})…' if CM_prev is not None else
                           f'Verify iter {iteration+1}/{MAX_ITER}: '
                           f'tail_inc={tail_inc:.2f}° (initial)…'})

            run_flow5(script_path, timeout=300)
            time.sleep(0.3)

            csv = find_polar_csv()
            if not csv:
                uq.put({'type': 'verify_done', 'ok': False, 'msg': 'No Flow5 output'})
                return
            polar = parse_polar_csv(csv)
            perf  = extract_performance_interp(polar, CL_cruise)
            if perf is None:
                uq.put({'type': 'verify_done', 'ok': False,
                        'msg': 'CL not in polar range — widen AoA sweep'})
                return

            CM = perf['Cm']
            CM_prev = CM

            uq.put({'type': 'verify_step',
                    'iteration': iteration + 1,
                    'tail_inc': tail_inc,
                    'CM': CM,
                    'LD': perf['LD'],
                    'alpha': perf['alpha']})

            if abs(CM) < CM_TOL:
                break

            # Newton correction: ∂Cm/∂i_tail = −CLa_tail × VHT (per rad)
            # → Δi_tail = +Cm / (CLa_tail × VHT)  (positive Cm → more positive tail)
            SHT_frac = geom.get('SHT', S * 0.22) / S
            c_bar_loc = geom.get('b', 1.2) / geom.get('AR', 7.0)
            LHT      = geom.get('LHT', 0.55)
            VHT_loc  = SHT_frac * LHT / c_bar_loc
            delta_inc_rad = CM / (CLa_tail * max(VHT_loc, 0.5))
            delta_inc_deg = math.degrees(delta_inc_rad) * 0.8   # damped
            tail_inc = max(-8.0, min(8.0, tail_inc + delta_inc_deg))

        trimmed = abs(CM_prev) < CM_TOL if CM_prev is not None else False
        uq.put({'type': 'verify_done', 'ok': True,
                'CL': CL_cruise,
                'CD': perf['CD'], 'CDv': perf['CDv'], 'CDi': perf['CDi'],
                'Cm': perf['Cm'], 'alpha': perf['alpha'],
                'LD_flow5': perf['LD'],
                'tail_inc_final': tail_inc,
                'trimmed': trimmed,
                'iterations': iteration + 1})

    except CancelledError:
        uq.put({'type': 'cancelled'})
    except Exception:
        import traceback
        uq.put({'type': 'error', 'msg': traceback.format_exc()})


def _bo_worker_thread(req, var_specs, llt_seed_result, n_calls, n_initial, uq, cancel_flag):
    """Run Flow5 Bayesian Optimisation in a background thread."""
    try:
        def callback(n, x_dict, LD, tail_inc):
            if cancel_flag[0]:
                raise KeyboardInterrupt('cancelled')
            uq.put({'type': 'bo_step', 'n': n, 'LD': LD,
                    'x_dict': x_dict, 'tail_inc': tail_inc})

        result = run_flow5_bo(
            req, var_specs, llt_seed_result,
            n_calls=n_calls, n_initial=n_initial,
            callback=callback, cancel_flag=cancel_flag,
        )

        if result.get('cancelled'):
            uq.put({'type': 'cancelled'})
        else:
            uq.put({'type': 'bo_done', 'result': result})

    except Exception:
        import traceback
        uq.put({'type': 'error', 'msg': traceback.format_exc()})


# ── Flow5 XML helpers for verification ───────────────────────────────────────

def _write_verify_polar_xml(speed_ms, ref_area, ref_span, ref_chord, method='QUADS'):
    from flow5_xml import XML_DIR, PLANE_NAME
    thin = 'true' if method == 'TRIUNIFORM' else 'false'
    path = os.path.join(XML_DIR, 'drone_verify_polar.xml')
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflPlanePolar version="1.0">
    <Units><length_unit_to_meter>1</length_unit_to_meter>
    <area_unit_to_m2>1</area_unit_to_m2><mass_unit_to_kg>1</mass_unit_to_kg>
    <speed_unit_to_ms>1</speed_unit_to_ms><inertia_unit_to_kgm2>1</inertia_unit_to_kgm2>
    </Units>
    <Polar>
        <Polar_Name>T1-VERIFY</Polar_Name>
        <Plane_Name>{PLANE_NAME}</Plane_Name>
        <Type>FIXEDSPEEDPOLAR</Type>
        <Method>{method}</Method>
        <Thin_Surfaces>{thin}</Thin_Surfaces>
        <Include_Fuse_Moments>false</Include_Fuse_Moments>
        <Ground_Effect>false</Ground_Effect>
        <Wake><FlatPanelWake>true</FlatPanelWake><NX>5</NX>
        <ProgressionFactor>1.100</ProgressionFactor><LengthFactor>30.000</LengthFactor></Wake>
        <Reference_Dimensions>
            <Reference_Dimensions>Custom</Reference_Dimensions>
            <Reference_Area>{ref_area:.6f}</Reference_Area>
            <Reference_Span_Length>{ref_span:.6f}</Reference_Span_Length>
            <Reference_Chord_Length>{ref_chord:.6f}</Reference_Chord_Length>
        </Reference_Dimensions>
        <Fluid><Viscosity>1.5e-05</Viscosity><Density>1.225</Density></Fluid>
        <Viscous_Analysis>
            <Is_Viscous_Analysis>true</Is_Viscous_Analysis>
            <XFoil_OnTheFly>true</XFoil_OnTheFly>
            <From_CL>true</From_CL>
        </Viscous_Analysis>
        <Use_plane_inertia>true</Use_plane_inertia>
        <Fixed_Velocity>{speed_ms:.3f}</Fixed_Velocity>
    </Polar>
</xflPlanePolar>"""
    with open(path, 'w') as f:
        f.write(xml)
    return path


def _write_verify_script_xml(aoa_min, aoa_max, plane_file='drone_mdo_best.xml'):
    from flow5_xml import XML_DIR, OUTPUT_DIR, FOILS_DIR
    path = os.path.join(XML_DIR, 'drone_verify_script.xml')
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE XFL_SCRIPT>
<xflscript version="1.0">
    <Metadata>
        <Make_project_file>false</Make_project_file>
        <Polar_text_output_format>CSV</Polar_text_output_format>
        <Directories>
            <output_dir>{OUTPUT_DIR}</output_dir>
            <plane_definition_xml_dir>{XML_DIR}</plane_definition_xml_dir>
            <plane_analysis_xml_dir>{XML_DIR}</plane_analysis_xml_dir>
            <foil_files_dir>{FOILS_DIR}</foil_files_dir>
            <recursive_scan>true</recursive_scan>
        </Directories>
        <MultiThreading><Allow_Multithreading>true</Allow_Multithreading>
        <Thread_Priority>Normal</Thread_Priority><Max_threads>4</Max_threads></MultiThreading>
        <Double_Precision>true</Double_Precision>
    </Metadata>
    <Plane_Analysis>
        <Plane_Analysis_Output>
            <make_oppoints>false</make_oppoints>
            <make_oppoints_text_file>false</make_oppoints_text_file>
            <compute_derivatives>false</compute_derivatives>
            <make_polars_text_file>true</make_polars_text_file>
            <export_stl_mesh>false</export_stl_mesh>
        </Plane_Analysis_Output>
        <Foil_Dat_Files>
            <Foil_File_Name>e387.dat</Foil_File_Name>
            <Foil_File_Name>naca0009.dat</Foil_File_Name>
        </Foil_Dat_Files>
        <Plane_Definition_Files>
            <Process_All_Files>false</Process_All_Files>
            <Plane_File_Name>{plane_file}</Plane_File_Name>
        </Plane_Definition_Files>
        <Plane_Analysis_Files>
            <Process_All_Files>false</Process_All_Files>
            <Analysis_File_Name>drone_verify_polar.xml</Analysis_File_Name>
        </Plane_Analysis_Files>
        <Plane_Analysis_Data>
            <T12_Range>{math.floor(aoa_min)}, {math.ceil(aoa_max)}, 1</T12_Range>
        </Plane_Analysis_Data>
    </Plane_Analysis>
</xflscript>"""
    with open(path, 'w') as f:
        f.write(xml)
    return path


# ── Main GUI class ────────────────────────────────────────────────────────────

class DroneMDOGui:
    PAD = 6

    def __init__(self, root):
        self.root = root
        root.title('Wing + Tail MDO  —  Maximize Trimmed L/D')
        root.resizable(True, True)
        root.minsize(1080, 720)

        self._uq           = queue.Queue()
        self._cancel_flag  = [False]
        self._thread       = None

        # State
        self._wing_result  = None   # Phase 1 best result
        self._tail_result  = None   # Phase 2 best result
        self._best_geom    = None   # best geometry for canvas (best of both phases)
        self._current_geom = None   # last evaluated geometry
        self._step_count   = 0

        self._build_ui()
        self._poll()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        p = self.PAD
        root = self.root

        # Title
        tk.Label(root, text='Wing + Tail MDO  —  Maximize Trimmed L/D',
                 font=('Helvetica', 13, 'bold'), pady=p).pack(fill=tk.X, padx=p)
        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=p)

        # Main three-column content area
        content = tk.Frame(root)
        content.pack(fill=tk.BOTH, expand=False, padx=p, pady=p)

        self._build_mission_panel(content)
        self._build_variables_panel(content)
        self._build_canvas_panel(content)

        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=p)

        # Control buttons
        self._build_controls(root)

        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=p, pady=(4, 0))

        # Best configuration banner
        self._build_best_banner(root)

        ttk.Separator(root, orient='horizontal').pack(fill=tk.X, padx=p, pady=(4, 0))

        # Iteration log
        self._build_log(root)

    def _build_mission_panel(self, parent):
        p = self.PAD
        lf = tk.LabelFrame(parent, text='Mission Requirements', padx=p, pady=p,
                           font=('Helvetica', 9, 'bold'))
        lf.pack(side=tk.LEFT, fill=tk.Y, padx=(0, p), anchor='n')

        fields = [
            ('MTOW [kg]',              'mtow_kg',          '3.0'),
            ('Payload [kg]',           'payload_kg',       '0.5'),
            ('Cruise speed [m/s]',     'cruise_speed_ms',  '20.0'),
            ('Stall speed [m/s]',      'stall_speed_ms',   '10.0'),
            ('Landing stall speed [m/s]', 'V_stall_land_ms', '8.0'),
            ('Wing loading [N/m²]',    'wing_loading_Nm2', '120.0'),
            ('Battery [Wh]',           'battery_wh',       '100.0'),
            ('Prop. efficiency η',     'propulsive_eta',   '0.75'),
        ]
        # Stability requirements — every one of these gates the optimizer's
        # penalty terms (see analysis.py / optimizer.py / flow5_bo_optimizer.py);
        # exposed here so a mission-driven change doesn't require editing code.
        stability_fields = [
            ('Target static margin',       'target_sm',              '0.10'),
            ('Phugoid ζ min',               'dyn_ph_zeta_min',        '0.04'),
            ('Short-period ζ min',          'dyn_sp_zeta_min',        '0.35'),
            ('Short-period ζ max',          'dyn_sp_zeta_max',        '1.30'),
            ('Weathercock Cn_β min [/rad]', 'target_cn_beta_min',     '0.05'),
            ('Max |Cl_β / Cn_β|',           'target_cl_beta_max_ratio', '1.2'),
        ]
        self._mission_vars = {}
        for row, (label, key, default) in enumerate(fields):
            tk.Label(lf, text=label, anchor='w', font=('Arial', 9)).grid(
                row=row, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=default)
            tk.Entry(lf, textvariable=var, width=8, font=('Arial', 9)).grid(
                row=row, column=1, padx=(8, 0), pady=2)
            self._mission_vars[key] = var

        # Derived wing area display
        tk.Label(lf, text='Wing area [m²]', anchor='w',
                 font=('Arial', 9), fg='#555').grid(
            row=len(fields), column=0, sticky='w', pady=(6, 2))
        self._wing_area_label = tk.Label(lf, text='0.245', anchor='w',
                                          font=('Arial', 9, 'italic'), fg='#555')
        self._wing_area_label.grid(row=len(fields), column=1, padx=(8, 0))

        def _update_area(*_):
            try:
                S = (float(self._mission_vars['mtow_kg'].get()) * G /
                     float(self._mission_vars['wing_loading_Nm2'].get()))
                self._wing_area_label.config(text=f'{S:.4f}')
            except ValueError:
                pass

        stab_row0 = len(fields) + 1
        tk.Label(lf, text='── Stability Requirements ──', font=('Arial', 8, 'bold'),
                 fg='#555').grid(row=stab_row0, column=0, columnspan=2,
                                 sticky='ew', pady=(8, 2))
        for i, (label, key, default) in enumerate(stability_fields):
            row = stab_row0 + 1 + i
            tk.Label(lf, text=label, anchor='w', font=('Arial', 9)).grid(
                row=row, column=0, sticky='w', pady=2)
            var = tk.StringVar(value=default)
            tk.Entry(lf, textvariable=var, width=8, font=('Arial', 9)).grid(
                row=row, column=1, padx=(8, 0), pady=2)
            self._mission_vars[key] = var

        for v in self._mission_vars.values():
            v.trace_add('write', _update_area)
        _update_area()

        # Note about derived quantities
        note = ('CG → auto at NP − target SM\n'
                'Wing incidence → clip(α_cruise, 2–4°)')
        tk.Label(lf, text=note, font=('Arial', 8), fg='#888',
                 justify='left').grid(row=stab_row0 + len(stability_fields) + 1,
                                      column=0, columnspan=2, sticky='w', pady=(8, 0))

    def _build_variables_panel(self, parent):
        p = self.PAD
        outer = tk.LabelFrame(parent, text='Design Variables', padx=p, pady=p,
                              font=('Helvetica', 9, 'bold'))
        outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, p), anchor='n')

        self._active_vars = {}
        self._lo_vars     = {}
        self._hi_vars     = {}
        self._x0_vars     = {}
        self._cur_labels  = {}   # live "current value" display

        header_row = 0
        for col, hdr in enumerate(['Variable', '✓', 'Lower', 'Upper', 'Initial', 'Current']):
            tk.Label(outer, text=hdr, font=('Arial', 9, 'bold'),
                     fg='#333').grid(row=header_row, column=col, padx=4, pady=(0, 4))

        row = header_row + 1

        def _section(title, specs, bg):
            nonlocal row
            tk.Label(outer, text=f'── {title} ──', font=('Arial', 8, 'bold'),
                     fg='#555', bg=bg).grid(
                row=row, column=0, columnspan=6, sticky='ew', pady=(6, 2))
            row += 1
            for s in specs:
                k = s['key']
                tk.Label(outer, text=s['label'], anchor='w',
                         font=('Arial', 9)).grid(row=row, column=0, sticky='w', padx=4, pady=2)

                av = tk.BooleanVar(value=True)
                tk.Checkbutton(outer, variable=av).grid(row=row, column=1, padx=2)
                self._active_vars[k] = av

                lv = tk.StringVar(value=str(s['lo']))
                tk.Entry(outer, textvariable=lv, width=6,
                         font=('Arial', 9)).grid(row=row, column=2, padx=3)
                self._lo_vars[k] = lv

                hv = tk.StringVar(value=str(s['hi']))
                tk.Entry(outer, textvariable=hv, width=6,
                         font=('Arial', 9)).grid(row=row, column=3, padx=3)
                self._hi_vars[k] = hv

                x0v = tk.StringVar(value=str(s['x0']))
                tk.Entry(outer, textvariable=x0v, width=7,
                         font=('Arial', 9)).grid(row=row, column=4, padx=3)
                self._x0_vars[k] = x0v

                cur = tk.Label(outer, text='—', anchor='w', width=8,
                               font=('Courier', 9), fg='#2d7d46')
                cur.grid(row=row, column=5, padx=4)
                self._cur_labels[k] = cur

                row += 1

        _section('WING',          WING_VAR_SPECS, '#f0f4ff')
        _section('TAIL',          TAIL_VAR_SPECS, '#f0fff4')
        _section('VERTICAL TAIL', VT_VAR_SPECS,   '#fff0f8')
        _section('FUSELAGE',      FUSE_VAR_SPECS, '#fff8f0')

    def _build_canvas_panel(self, parent):
        p = self.PAD
        lf = tk.LabelFrame(parent, text='Planform View (top-down)',
                           padx=p, pady=p, font=('Helvetica', 9, 'bold'))
        lf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(lf, width=CANVAS_W, height=CANVAS_H,
                                 bg='#f8f8fc', relief=tk.SUNKEN, bd=1)
        self._canvas.pack()

        legend = tk.Frame(lf)
        legend.pack(fill=tk.X, pady=(4, 0))
        tk.Label(legend, text='■', fg='#4a90d9', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend, text='Current eval', font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 12))
        tk.Label(legend, text='■', fg='#4a9a4a', font=('Arial', 10)).pack(side=tk.LEFT, padx=2)
        tk.Label(legend, text='Best so far', font=('Arial', 8)).pack(side=tk.LEFT)

        self._canvas_label = tk.Label(lf, text='Waiting for first evaluation…',
                                       font=('Arial', 8), fg='#888')
        self._canvas_label.pack()

        # Draw placeholder
        self._canvas.create_text(CANVAS_W // 2, CANVAS_H // 2,
                                  text='Run optimization to see planform',
                                  fill='#aaa', font=('Arial', 11))

    def _build_controls(self, parent):
        p = self.PAD
        ctrl = tk.Frame(parent)
        ctrl.pack(fill=tk.X, padx=p, pady=(p, 0))

        self._run_both_btn = tk.Button(
            ctrl, text='▶  Run Optimisation', command=self._start_both,
            bg='#2d7d46', fg='black', font=('Helvetica', 10, 'bold'), padx=12)
        self._run_both_btn.pack(side=tk.LEFT, padx=(0, 4))

        ttk.Separator(ctrl, orient='vertical').pack(side=tk.LEFT, fill='y',
                                                     padx=8, pady=2)

        self._verify_btn = tk.Button(
            ctrl, text='🔍  Verify + Trim (Flow5)', command=self._start_verify,
            bg='#8b5a2b', fg='black', font=('Helvetica', 10, 'bold'), padx=10)
        self._verify_btn.pack(side=tk.LEFT, padx=4)

        self._bo_btn = tk.Button(
            ctrl, text='⚡  BO Optimise (Flow5)', command=self._start_bo,
            bg='#4a0e8f', fg='black', font=('Helvetica', 10, 'bold'), padx=10)
        self._bo_btn.pack(side=tk.LEFT, padx=4)

        self._cancel_btn = tk.Button(
            ctrl, text='✕  Cancel', command=self._cancel,
            bg='#c0392b', fg='black', font=('Helvetica', 10), padx=8,
            state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=4)

        # Method selector
        ttk.Separator(ctrl, orient='vertical').pack(side=tk.LEFT, fill='y',
                                                     padx=8, pady=2)
        tk.Label(ctrl, text='Flow5 method:', font=('Arial', 9)).pack(side=tk.LEFT)
        self._method_var = tk.StringVar(value='QUADS')
        ttk.Combobox(ctrl, textvariable=self._method_var,
                     values=['QUADS', 'TRIUNIFORM'], width=10,
                     state='readonly').pack(side=tk.LEFT, padx=(4, 0))

        # Progress bar
        pb_frame = tk.Frame(parent)
        pb_frame.pack(fill=tk.X, padx=p, pady=(4, 0))
        self._progress = ttk.Progressbar(pb_frame, mode='indeterminate')
        self._progress.pack(fill=tk.X)

        self._status_label = tk.Label(parent, text='Ready.', anchor='w',
                                       font=('Courier', 9), fg='#555')
        self._status_label.pack(fill=tk.X, padx=p)

    def _build_best_banner(self, parent):
        p = self.PAD
        self._banner = tk.LabelFrame(parent, text='★  Best Configuration Found',
                                      padx=p, pady=p,
                                      font=('Helvetica', 9, 'bold'),
                                      fg='#2d7d46', relief=tk.GROOVE, bd=2)
        self._banner.pack(fill=tk.X, padx=p, pady=(4, 0))

        self._banner_line1 = tk.Label(
            self._banner,
            text='No result yet — run Wing Phase or Both to begin.',
            font=('Courier', 10, 'bold'), fg='#2d7d46', anchor='w')
        self._banner_line1.pack(fill=tk.X)

        self._banner_line2 = tk.Label(
            self._banner, text='', font=('Courier', 9), fg='#333', anchor='w')
        self._banner_line2.pack(fill=tk.X)

        self._banner_line3 = tk.Label(
            self._banner, text='', font=('Courier', 9), fg='#333', anchor='w')
        self._banner_line3.pack(fill=tk.X)

        self._banner_line4 = tk.Label(
            self._banner, text='', font=('Courier', 9), fg='#333', anchor='w')
        self._banner_line4.pack(fill=tk.X)

    def _build_log(self, parent):
        p = self.PAD
        lf = tk.LabelFrame(parent, text='Iteration Log', padx=p, pady=p,
                           font=('Helvetica', 9, 'bold'))
        lf.pack(fill=tk.BOTH, expand=True, padx=p, pady=(0, p))

        self._log = scrolledtext.ScrolledText(
            lf, font=('Courier', 8), state=tk.DISABLED,
            height=10, wrap=tk.NONE, bg='#f8f8f8')
        self._log.pack(fill=tk.BOTH, expand=True)

        # Tag colors
        self._log.tag_config('header', foreground='#1a3a6a', font=('Courier', 8, 'bold'))
        self._log.tag_config('change', foreground='#c05000')
        self._log.tag_config('fixed',  foreground='#666666')
        self._log.tag_config('result_ok',  foreground='#2d7d46', font=('Courier', 8, 'bold'))
        self._log.tag_config('result_bad', foreground='#c0392b', font=('Courier', 8, 'bold'))
        self._log.tag_config('phase',  foreground='#6f42c1', font=('Courier', 8, 'bold'))

    # ── Log writing ───────────────────────────────────────────────────────────

    def _log_write(self, text, tag=None):
        self._log.config(state=tk.NORMAL)
        if tag:
            self._log.insert(tk.END, text, tag)
        else:
            self._log.insert(tk.END, text)
        self._log.see(tk.END)
        self._log.config(state=tk.DISABLED)

    def _log_step(self, entry):
        self._step_count += 1
        phase_label = 'Phase 1 — Wing' if entry['phase'] == 'wing' else 'Phase 2 — Tail'
        geom = entry.get('geom', {})
        ok = (entry['trimmable'] and entry['stall_ok'] and
              geom.get('landing_ok', True) and
              geom.get('dyn_ok', True) and geom.get('lat_dir_ok', True))

        self._log_write(f'\nStep {self._step_count:4d}  [{phase_label}]\n', 'header')

        if entry['changes']:
            self._log_write('  Changed : ', 'fixed')
            self._log_write('  '.join(entry['changes']) + '\n', 'change')
        if entry['unchanged']:
            self._log_write('  Fixed   : ', 'fixed')
            self._log_write('  '.join(entry['unchanged']) + '\n', 'fixed')

        SM_pct  = entry['SM'] * 100
        tail_i  = entry['tail_inc']
        Vs      = entry['V_stall']
        LD      = entry['LD']
        trim_s  = f'i_tail={tail_i:+.2f}° {"✓" if entry["trimmable"] else "✗"}'
        result  = (f'  →  L/D={LD:.3f}   SM={SM_pct:.1f}%   '
                   f'{trim_s}   Vs={Vs:.1f}m/s\n')
        self._log_write(result, 'result_ok' if ok else 'result_bad')

    # ── Banner update ─────────────────────────────────────────────────────────

    def _update_banner(self, res):
        """Refresh the best-configuration banner with the given result dict."""
        if res is None:
            return
        ok = (res['trimmable'] and res['stall_ok'] and
              res.get('landing_ok', True) and
              res.get('dyn_ok', True) and res.get('lat_dir_ok', True))
        ok_sym = '✓' if ok else '✗'
        banner_col = '#2d7d46' if ok else '#c0392b'

        line1 = (f'L/D = {res["LD"]:.3f}   '
                 f'i_tail = {res["tail_inc_deg"]:+.2f}°  (CM ≈ 0.000 ✓)   '
                 f'SM = {res["SM"]*100:.1f}%   '
                 f'Vs = {res["V_stall"]:.1f} m/s   '
                 f'Vs_land = {res.get("V_stall_land", 0.0):.1f} m/s  {ok_sym}')
        line2 = (f'Wing : AR={res["AR"]:.2f}  taper={res["taper"]:.3f}  '
                 f'twist={res["twist_deg"]:.1f}°  sweep={res["sweep_deg"]:.1f}°  '
                 f'b={res["b"]:.3f}m  cr={res["cr"]:.3f}m  ct={res["ct"]:.3f}m')
        line3 = (f'Tail : SHT/S={res["SHT_frac"]:.3f}  AR_HT={res["AR_HT"]:.2f}  '
                 f'tail_arm={res["tail_arm_chords"]:.2f}×c  b_HT={res["b_HT"]:.3f}m  '
                 f'LHT={res["LHT"]:.3f}m  SVT/S={res.get("SVT_frac", 0.0):.3f}  '
                 f'Fuse: L={res["fuse_length"]:.2f}m  f={res["fuse_fineness"]:.1f}')
        line4 = (f'Stability: '
                 f'ζ_ph={res.get("phugoid_zeta", 0):.2f} '
                 f'{"✓" if res.get("dyn_ok", True) else "✗"}   '
                 f'ζ_sp={res.get("sp_zeta", 0):.2f}   '
                 f'Cn_β={res.get("Cn_beta", 0):+.3f}/rad   '
                 f'Cl_β={res.get("Cl_beta", 0):+.3f}/rad   '
                 f'{"✓" if res.get("lat_dir_ok", True) else "✗"}')

        self._banner_line1.config(text=line1, fg=banner_col)
        self._banner_line2.config(text=line2)
        self._banner_line3.config(text=line3)
        self._banner_line4.config(text=line4)

    # ── Current-value column update ───────────────────────────────────────────

    def _update_cur_labels(self, res):
        fmt = {
            'AR':       f'{res["AR"]:.2f}',
            'taper':    f'{res["taper"]:.3f}',
            'twist':    f'{res["twist_deg"]:.2f}°',
            'sweep':    f'{res["sweep_deg"]:.2f}°',
            'dihedral': f'{res["dihedral_deg"]:.2f}°',
            'SHT_frac': f'{res["SHT_frac"]:.3f}',
            'AR_HT':    f'{res["AR_HT"]:.2f}',
            'taper_HT': f'{res["taper_HT"]:.3f}',
            'tail_arm': f'{res["tail_arm_chords"]:.2f}×c',
            'HT_sweep': f'{res.get("HT_sweep_deg", 0.0):.2f}°',
            'HT_twist': f'{res.get("HT_twist_deg", 0.0):.2f}°',
            'HT_dihedral': f'{res.get("HT_dihedral_deg", 0.0):.2f}°',
            'SVT_frac': f'{res.get("SVT_frac", 0.0):.3f}',
            'AR_VT':    f'{res.get("AR_VT", 0.0):.2f}',
            'taper_VT': f'{res.get("taper_VT", 0.0):.3f}',
            'VT_sweep': f'{res.get("VT_sweep_deg", 0.0):.2f}°',
            'flap_deflection': f'{res.get("flap_deflection_deg", 0.0):.1f}°',
            'fuse_len': f'{res["fuse_length"]:.2f} m',
            'fuse_fin': f'{res["fuse_fineness"]:.1f}',
        }
        for k, lbl in self._cur_labels.items():
            if k in fmt:
                lbl.config(text=fmt[k])

    # ── Run / cancel ──────────────────────────────────────────────────────────

    def _busy(self):
        return self._thread is not None and self._thread.is_alive()

    def _set_running(self, running):
        state_off = tk.DISABLED if running else tk.NORMAL
        for btn in (self._run_both_btn, self._verify_btn, self._bo_btn):
            btn.config(state=state_off)
        self._cancel_btn.config(state=tk.NORMAL if running else tk.DISABLED)
        if running:
            self._progress.start(10)
        else:
            self._progress.stop()

    def _get_specs(self):
        """
        Read design variable specs from the GUI.
        Returns (active_specs, frozen_dict) per group.
        active_specs: variables the optimizer can move.
        frozen_dict:  {key: x0} for unchecked variables (held fixed at their initial value).
        """
        def _split(meta):
            active, frozen = [], {}
            for s in meta:
                k = s['key']
                try:
                    x0 = float(self._x0_vars[k].get())
                    lo = float(self._lo_vars[k].get())
                    hi = float(self._hi_vars[k].get())
                except ValueError:
                    messagebox.showerror('Input error', f'Bad value for {k}')
                    return None, None
                if self._active_vars[k].get():
                    active.append({**s, 'lo': lo, 'hi': hi, 'x0': x0})
                else:
                    frozen[k] = x0
            return active, frozen

        wing_a, wing_f = _split(WING_VAR_SPECS)
        tail_a, tail_f = _split(TAIL_VAR_SPECS)
        vt_a,   vt_f   = _split(VT_VAR_SPECS)
        fuse_a, fuse_f = _split(FUSE_VAR_SPECS)
        if wing_a is None or tail_a is None or vt_a is None or fuse_a is None:
            return None
        return wing_a, wing_f, tail_a, tail_f, vt_a, vt_f, fuse_a, fuse_f

    def _get_req(self):
        try:
            return _build_req_from_gui(self._mission_vars)
        except ValueError as e:
            messagebox.showerror('Mission input error', str(e))
            return None

    def _start_both(self):
        if self._busy():
            messagebox.showwarning('Busy', 'Wait for the current run to finish.')
            return
        req = self._get_req()
        if req is None:
            return
        specs = self._get_specs()
        if specs is None:
            return
        wing_a, wing_f, tail_a, tail_f, vt_a, vt_f, fuse_a, fuse_f = specs

        # Phase 1 uses the tail/VT/fuse initial values as fixed estimates
        tail_init = {
            'SHT_frac': float(self._x0_vars['SHT_frac'].get()),
            'AR_HT':    float(self._x0_vars['AR_HT'].get()),
            'taper_HT': float(self._x0_vars['taper_HT'].get()),
            'tail_arm': float(self._x0_vars['tail_arm'].get()),
            'HT_sweep': float(self._x0_vars['HT_sweep'].get()),
            'HT_twist': float(self._x0_vars['HT_twist'].get()),
            'HT_dihedral': float(self._x0_vars['HT_dihedral'].get()),
            'SVT_frac': float(self._x0_vars['SVT_frac'].get()),
            'AR_VT':    float(self._x0_vars['AR_VT'].get()),
            'taper_VT': float(self._x0_vars['taper_VT'].get()),
            'VT_sweep': float(self._x0_vars['VT_sweep'].get()),
        }
        fuse_init = {
            'length':   float(self._x0_vars['fuse_len'].get()),
            'fineness': float(self._x0_vars['fuse_fin'].get()),
        }

        free_wing = [s['key'] for s in wing_a] or ['(all frozen)']
        free_tail = [s['key'] for s in tail_a + vt_a + fuse_a] or ['(all frozen)']

        self._step_count = 0
        self._cancel_flag[0] = False
        self._set_running(True)
        self._log_write(f'\n{"="*70}\n', 'phase')
        self._log_write('  MDO: Phase 1 (Wing) → Phase 2 (Tail + Fuselage)\n', 'phase')
        self._log_write(f'{"="*70}\n', 'phase')
        self._log_write(f'  Wing free:  {free_wing}\n', 'fixed')
        self._log_write(f'  Tail free:  {free_tail}\n\n', 'fixed')
        self._status_label.config(text='Phase 1: optimizing wing…')

        self._thread = threading.Thread(
            target=_mdo_worker,
            args=(req, wing_a, wing_f, tail_a, tail_f, vt_a, vt_f, fuse_a, fuse_f,
                  tail_init, fuse_init, self._uq, self._cancel_flag),
            daemon=True)
        self._thread.start()

    def _start_verify(self):
        if self._busy():
            messagebox.showwarning('Busy', 'Wait for the current run to finish.')
            return
        best = self._tail_result or self._wing_result
        if best is None:
            messagebox.showwarning('No result', 'Run an optimization first.')
            return
        req = self._get_req()
        if req is None:
            return

        geom = result_to_geom(best, req.mtow_kg)
        method = self._method_var.get()

        self._cancel_flag[0] = False
        self._set_running(True)
        self._log_write(f'\n{"="*70}\n', 'phase')
        self._log_write('  FLOW5 VERIFICATION  (viscous panel method)\n', 'phase')
        self._log_write(f'{"="*70}\n\n', 'phase')
        self._status_label.config(text='Verify: launching Flow5…')

        self._thread = threading.Thread(
            target=_verify_worker,
            args=(geom, req, method, self._uq, self._cancel_flag),
            daemon=True)
        self._thread.start()

    def _start_bo(self):
        if self._busy():
            messagebox.showwarning('Busy', 'Wait for the current run to finish.')
            return
        req = self._get_req()
        if req is None:
            return
        specs = self._get_specs()
        if specs is None:
            return
        wing_a, _wing_f, tail_a, _tail_f, vt_a, _vt_f, fuse_a, _fuse_f = specs

        # BO optimises all active variables together (no Phase 1/2 split).
        all_specs = wing_a + tail_a + vt_a + fuse_a
        if not all_specs:
            messagebox.showwarning('No variables', 'Enable at least one design variable.')
            return

        llt_seed = self._tail_result or self._wing_result
        llt_ld   = llt_seed.get('LD', float('nan')) if llt_seed else float('nan')

        self._step_count = 0
        self._cancel_flag[0] = False
        self._set_running(True)

        self._log_write(f'\n{"="*70}\n', 'phase')
        self._log_write('  BAYESIAN OPTIMISATION  (Flow5 TRIUNIFORM inviscid)\n', 'phase')
        self._log_write(f'{"="*70}\n', 'phase')
        self._log_write(
            f'  Active variables ({len(all_specs)}): '
            f'{[s["key"] for s in all_specs]}\n', 'fixed')
        if llt_seed:
            self._log_write(f'  LLT seed L/D: {llt_ld:.3f}\n', 'fixed')
        self._log_write('\n', 'fixed')
        self._status_label.config(text='BO: exploring design space with Flow5…')

        self._thread = threading.Thread(
            target=_bo_worker_thread,
            args=(req, all_specs, llt_seed, 80, 12, self._uq, self._cancel_flag),
            daemon=True)
        self._thread.start()

    def _cancel(self):
        self._cancel_flag[0] = True
        self._status_label.config(text='Cancelling…')

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll(self):
        try:
            while True:
                msg = self._uq.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, msg):
        t = msg['type']

        if t == 'step':
            entry = msg['entry']
            self._log_step(entry)
            # Update current-value column with the latest geometry
            geom = entry.get('geom')
            if geom:
                self._current_geom = geom
                self._update_cur_labels(geom)
            # Update canvas every 5 steps (or when a new best is found)
            best_res = msg.get('best_res')
            if best_res:
                self._best_geom = best_res
                self._update_banner(best_res)
            if self._step_count % 5 == 0 and self._current_geom:
                draw_planform(self._canvas, self._current_geom, self._best_geom)
                self._canvas_label.config(
                    text=f'Step {self._step_count}   L/D = {self._current_geom.get("LD", 0):.3f}')
            self._status_label.config(
                text=(f'Step {self._step_count}  '
                      f'L/D={entry["LD"]:.3f}  '
                      f'SM={entry["SM"]*100:.1f}%  '
                      f'i_tail={entry["tail_inc"]:+.2f}°  '
                      f'{"✓" if entry["trimmable"] else "✗"}'))

        elif t in ('wing_done', 'tail_done'):
            result = msg['result']
            if t == 'wing_done':
                self._wing_result = result
                phase_name = 'Wing Phase (1)'
            else:
                self._tail_result = result
                phase_name = 'Tail Phase (2)'

            # Update best
            best = self._tail_result or self._wing_result
            self._best_geom = best
            self._update_banner(best)
            self._update_cur_labels(best)
            draw_planform(self._canvas, best)
            self._canvas_label.config(
                text=f'{phase_name} complete  L/D = {result["LD"]:.3f}')

            # Print summary to log
            conv = '✓ converged' if result['converged'] else '⚠ not converged'
            self._log_write(f'\n── {phase_name} COMPLETE  ({conv}, '
                            f'{result["iterations"]} steps) ──\n', 'header')
            self._log_write(
                f'  Best L/D = {result["LD"]:.4f}\n'
                f'  Trimmed: i_tail = {result["tail_inc_deg"]:+.2f}°  '
                f'(CM_cg ≈ 0.000 ✓)\n'
                f'  SM = {result["SM"]*100:.1f}%  '
                f'Vs = {result["V_stall"]:.2f} m/s  '
                f'{"✓" if result["stall_ok"] else "✗"}\n'
                f'  Landing: flap_defl={result.get("flap_deflection_deg",0):.1f}°  '
                f'Vs_land={result.get("V_stall_land",0):.2f} m/s  '
                f'{"✓" if result.get("landing_ok", True) else "✗"}\n'
                f'  Dynamic: ζ_ph={result.get("phugoid_zeta",0):.2f}  '
                f'ζ_sp={result.get("sp_zeta",0):.2f}  '
                f'{"✓" if result.get("dyn_ok", True) else "✗"}   '
                f'Lateral: Cn_β={result.get("Cn_beta",0):+.3f}/rad  '
                f'Cl_β={result.get("Cl_beta",0):+.3f}/rad  '
                f'{"✓" if result.get("lat_dir_ok", True) else "✗"}\n'
                f'  CDi_wing={result["CDi_wing"]:.5f}  '
                f'CDi_tail={result["CDi_tail"]:.5f}  '
                f'CD0_wing={result["CD0_wing"]:.5f}  '
                f'CD0_tail={result["CD0_tail"]:.5f}  '
                f'CD0_fuse={result["CD0_fuse"]:.5f}\n', 'result_ok')

            # If chaining to tail (Run Both), status already set
            if not msg.get('chain'):
                self._set_running(False)
                self._status_label.config(
                    text=f'{phase_name} done — L/D = {result["LD"]:.3f}')

        elif t == 'status':
            self._status_label.config(text=msg['msg'][:100])
            self._log_write(msg['msg'] + '\n', 'fixed')

        elif t == 'verify_step':
            n   = msg['iteration']
            ti  = msg['tail_inc']
            CM  = msg['CM']
            LD  = msg['LD']
            tag = 'result_ok' if abs(CM) < 0.005 else 'fixed'
            self._log_write(
                f'  iter {n}: tail_inc={ti:+.2f}°  CM={CM:+.4f}  L/D={LD:.3f}\n', tag)
            self._status_label.config(
                text=f'Verify iter {n}: CM={CM:+.4f}  L/D={LD:.3f}')

        elif t == 'verify_done':
            self._set_running(False)
            if not msg.get('ok'):
                self._log_write(f'\n  Verify FAILED: {msg.get("msg","")}\n', 'result_bad')
                self._status_label.config(text='Verify failed')
                return
            LD_f     = msg['LD_flow5']
            ti_final = msg.get('tail_inc_final', float('nan'))
            trimmed  = msg.get('trimmed', False)
            iters    = msg.get('iterations', '?')
            llt_ld   = (self._tail_result or self._wing_result or {}).get('LD', float('nan'))
            trim_tag = 'result_ok' if trimmed else 'result_bad'
            self._log_write(
                f'\n── FLOW5 TRIM VERIFICATION  ({iters} iteration{"s" if iters != 1 else ""}) ──\n'
                f'  Trim status      : {"✓ TRIMMED  |CM| < 0.005" if trimmed else "⚠ NOT FULLY TRIMMED"}\n'
                f'  Final tail inc   : {ti_final:+.2f}°\n'
                f'  L/D (Flow5)      : {LD_f:.3f}\n'
                f'  CL={msg["CL"]:.4f}  CD={msg["CD"]:.5f}  '
                f'CDv={msg["CDv"]:.5f}  CDi={msg["CDi"]:.5f}\n'
                f'  Cm={msg["Cm"]:+.4f}  α={msg["alpha"]:.2f}°\n'
                f'  LLT predicted    : {llt_ld:.3f}  '
                f'Δ = {LD_f - llt_ld:+.3f}\n',
                trim_tag)
            self._status_label.config(
                text=f'{"Trimmed ✓" if trimmed else "Not trimmed ⚠"} — Flow5 L/D = {LD_f:.3f}')

        elif t == 'bo_step':
            n       = msg['n']
            LD      = msg['LD']
            tail_inc = msg['tail_inc']
            self._step_count += 1
            tag = 'result_ok' if LD > 0 else 'result_bad'
            self._log_write(
                f'  BO eval {n:3d}: L/D={LD:.3f}  i_tail={tail_inc:+.2f}°\n', tag)
            self._status_label.config(
                text=f'BO eval {n}  L/D={LD:.3f}  i_tail={tail_inc:+.2f}°')

        elif t == 'bo_done':
            self._set_running(False)
            result  = msg['result']
            LD      = result['LD']
            x_dict  = result['x_dict']
            n_used  = result['n_calls_used']
            llt_ld  = result.get('best_llt_LD')

            self._log_write(
                f'\n── BO COMPLETE  ({n_used} Flow5 evaluations) ──\n'
                f'  Best L/D (Flow5) : {LD:.3f}\n', 'result_ok')
            if llt_ld is not None:
                self._log_write(
                    f'  LLT predicted    : {llt_ld:.3f}  Δ = {LD - llt_ld:+.3f}\n',
                    'result_ok')
            self._log_write(
                '  Best vars: ' +
                '  '.join(f'{k}={v:.3g}' for k, v in x_dict.items()) + '\n',
                'fixed')

            # Reconstruct full geometry at the BO optimum for banner/canvas
            req = self._get_req()
            if req:
                try:
                    full_res = compute_trimmed_ld(
                        x_dict.get('AR', 7.0),
                        x_dict.get('taper', 0.40),
                        x_dict.get('twist', 1.5),
                        x_dict.get('SHT_frac', 0.22),
                        x_dict.get('AR_HT', 4.0),
                        x_dict.get('taper_HT', 0.50),
                        x_dict.get('tail_arm', 4.0),
                        x_dict.get('fuse_len', 0.80),
                        x_dict.get('fuse_fin', 6.5),
                        req,
                        sweep_deg=x_dict.get('sweep', 0.0),
                        dihedral_deg=x_dict.get('dihedral', 1.0),
                        flap_deflection_deg=x_dict.get('flap_deflection', 0.0),
                        SVT_frac=x_dict.get('SVT_frac', 0.10),
                        AR_VT=x_dict.get('AR_VT', 1.8),
                        taper_VT=x_dict.get('taper_VT', 0.50),
                        HT_sweep_deg=x_dict.get('HT_sweep', 0.0),
                        HT_twist_deg=x_dict.get('HT_twist', 0.0),
                        HT_dihedral_deg=x_dict.get('HT_dihedral', 0.0),
                        VT_sweep_deg=x_dict.get('VT_sweep', 0.0),
                    )
                    full_res['LD'] = LD   # replace LLT L/D with Flow5 L/D
                    self._tail_result = full_res
                    self._best_geom   = full_res
                    self._update_banner(full_res)
                    self._update_cur_labels(full_res)
                    draw_planform(self._canvas, full_res)
                    self._canvas_label.config(
                        text=f'BO complete  L/D = {LD:.3f} (Flow5 TRIUNIFORM)')
                except Exception:
                    pass

            self._status_label.config(
                text=f'BO done — Flow5 L/D = {LD:.3f}')

        elif t == 'cancelled':
            self._set_running(False)
            self._log_write('\n  ✕  Cancelled by user.\n', 'result_bad')
            self._status_label.config(text='Cancelled.')

        elif t == 'error':
            self._set_running(False)
            self._log_write(f'\nERROR:\n{msg["msg"]}\n', 'result_bad')
            self._status_label.config(text='Error — see log')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    DroneMDOGui(root)
    root.mainloop()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
GA4-01 Airfoil Selection + 3D Panel Analysis
============================================
Phase 1  Analytical pre-screen  — 5 NACA candidates ranked by CL_max, CM_ac, L/D
Phase 2  Flow5 VLM2 polars      — all 5 candidates on the GA4-01 wing geometry
Phase 3  Winner selection       — Flow5 L/D + CL_max + analytical CM score
Phase 4  Full-aircraft XML      — wing + H-stab + V-fin + fuselage body for Flow5 GUI
Phase 5  3D panel (QUADS) run   — QUADS polar with fuselage; headless via script

Usage:  python3 /Users/kaanoktem/Flow5/ga4_airfoil_select.py
Output: analysis_xml/ga4_final_plane.xml     (import in Flow5 GUI)
        analysis_xml/ga4_panel_polar.xml     (QUADS 3D polar)
        analysis_xml/ga4_panel_script.xml    (run headlessly)
"""

import os, math, sys, glob, time
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
FOILS_DIR  = os.path.join(WORK_DIR, "foils")
XML_DIR    = os.path.join(WORK_DIR, "analysis_xml")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
FLOW5_APP  = "/Applications/flow5.app/Contents/MacOS/flow5"

for d in (FOILS_DIR, XML_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, WORK_DIR)
from flow5_run import run_flow5, find_polar_csv, parse_polar_csv, extract_performance

# ── GA4-01 design constants (from design_ga.py) ────────────────────────────
MTOW_KG      = 1202.0
S            = 14.74     # m²
AR           = 12.0
TAPER        = 0.700
TWIST_DEG    = 2.0
DIHED_DEG    = 1.5
V_CRUISE     = 69.45     # m/s
CL_CRUISE    = 0.2708
CG_FRAC_MAC  = 0.33
TAIL_INC_DEG = -1.20

b       = math.sqrt(AR * S)               # 13.302 m
c_bar   = b / AR                          # 1.108 m
cr      = 2 * S / (b * (1 + TAPER))       # 1.304 m
ct      = TAPER * cr                      # 0.913 m
LHT     = 3.5 * c_bar                     # 3.878 m   (tail arm)

SHT     = 0.22 * S                        # 3.243 m²
AR_HT   = 4.5
b_HT    = math.sqrt(AR_HT * SHT)          # 3.822 m
cr_HT   = 2 * SHT / (b_HT * 1.40)        # 1.213 m
ct_HT   = 0.40 * cr_HT                   # 0.485 m

SVT     = 0.10 * S                        # 1.474 m²
AR_VT   = 1.8
b_VT    = math.sqrt(AR_VT * SVT)          # 1.629 m
tv      = 0.50
cr_VT   = 2 * SVT / (b_VT * (1 + tv))    # 1.206 m
ct_VT   = tv * cr_VT                     # 0.603 m

x_cg    = CG_FRAC_MAC * c_bar             # 0.366 m aft of wing root LE

# ── Analytical candidate data  (Abbott & von Doenhoff, Re ≈ 3.5 × 10⁶) ───
# All references to Theory of Wing Sections, NACA TN 1428 / Report 537
CANDIDATES = {
    'naca2412': {
        'display':  'NACA 2412',
        'type':     '4digit',
        'm': 0.02, 'p': 0.40, 't': 0.12,
        'CL_max':   1.63,   # Abbott & von Doenhoff Fig 154
        'CM_ac':   -0.047,  # low camber → moderate nose-down
        'alpha_L0': -2.07,  # deg
        'CD_min':   0.0060,
        'CL_opt':   0.10,   # CL at minimum drag
        'notes': 'Cessna 172 standard. Proven. Moderate CL_max, moderate trim penalty.',
    },
    'naca4412': {
        'display':  'NACA 4412',
        'type':     '4digit',
        'm': 0.04, 'p': 0.40, 't': 0.12,
        'CL_max':   1.73,
        'CM_ac':   -0.099,  # high camber → large nose-down moment (worst trim)
        'alpha_L0': -4.05,
        'CD_min':   0.0058,
        'CL_opt':   0.40,
        'notes': 'Good CL_max but CM_ac=-0.099 imposes heavy tail load at cruise.',
    },
    'naca2415': {
        'display':  'NACA 2415',
        'type':     '4digit',
        'm': 0.02, 'p': 0.40, 't': 0.15,
        'CL_max':   1.60,
        'CM_ac':   -0.050,
        'alpha_L0': -2.07,
        'CD_min':   0.0064,
        'CL_opt':   0.10,
        'notes': 'Thicker (15%) for structural depth. Slightly lower CL_max, more drag.',
    },
    'naca23012': {
        'display':  'NACA 23012',
        'type':     '5digit',
        't': 0.12,
        'CL_max':   1.79,   # highest of the five (NACA TN 1428 Fig 171)
        'CM_ac':   -0.013,  # reflexed camber → very low nose-down (best trim)
        'alpha_L0': -1.22,
        'CD_min':   0.0058,
        'CL_opt':   0.30,
        'notes': 'Highest CL_max, lowest |CM_ac|. Best all-round for this design.',
    },
    'naca23015': {
        'display':  'NACA 23015',
        'type':     '5digit',
        't': 0.15,
        'CL_max':   1.72,
        'CM_ac':   -0.007,  # lowest |CM| of all 5 (thicker reflex)
        'alpha_L0': -1.22,
        'CD_min':   0.0063,
        'CL_opt':   0.30,
        'notes': 'Thicker 23012. Best structural depth, lowest CM_ac, slight drag penalty.',
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 0 — NACA profile generators
# ══════════════════════════════════════════════════════════════════════════════

def naca4_profile(m, p, t, n=80):
    """
    NACA 4-digit airfoil coordinates using standard NASA formulas.
    m: max camber fraction  (e.g. 0.02 for '2' in 2412)
    p: max camber position  (e.g. 0.40 for '4' in 2412)
    t: thickness fraction   (e.g. 0.12 for '12' in 2412)
    Returns (xu, zu, xl, zl) upper/lower surface arrays (length n+1).
    """
    beta = np.linspace(0.0, np.pi, n + 1)
    x    = 0.5 * (1.0 - np.cos(beta))          # cosine spacing 0→1

    # Thickness distribution (open trailing edge)
    yt = (t / 0.2) * (0.2969 * np.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    - 0.1015 * x**4)

    # Camber line + slope
    if m == 0.0 or p == 0.0:
        yc     = np.zeros_like(x)
        dycdx  = np.zeros_like(x)
    else:
        yc_f  = (m / p**2)       * (2*p*x - x**2)
        dc_f  = (2*m / p**2)     * (p - x)
        yc_a  = (m / (1-p)**2)   * (1 - 2*p + 2*p*x - x**2)
        dc_a  = (2*m / (1-p)**2) * (p - x)
        yc    = np.where(x < p, yc_f, yc_a)
        dycdx = np.where(x < p, dc_f, dc_a)

    th = np.arctan(dycdx)
    xu = x  - yt * np.sin(th);  zu = yc + yt * np.cos(th)
    xl = x  + yt * np.sin(th);  zl = yc - yt * np.cos(th)
    return xu, zu, xl, zl


def naca5_23xxx_profile(t, n=80):
    """
    NACA 23xxx series (e.g., 23012 for t=0.12, 23015 for t=0.15).
    Reflexed camber line from NACA Report 537 Table I:
      design CL = 0.30 (k1 = 15.957),  max camber at r = 0.2025c
    """
    beta = np.linspace(0.0, np.pi, n + 1)
    x    = 0.5 * (1.0 - np.cos(beta))

    yt = (t / 0.2) * (0.2969 * np.sqrt(x)
                    - 0.1260 * x
                    - 0.3516 * x**2
                    + 0.2843 * x**3
                    - 0.1015 * x**4)

    k1 = 15.957;  r = 0.2025
    yc_f  = (k1/6) * (x**3 - 3*r*x**2 + r**2*(3-r)*x)
    yc_a  = (k1*r**3/6) * (1.0 - x)
    dc_f  = (k1/6) * (3*x**2 - 6*r*x + r**2*(3-r))
    dc_a  = np.full_like(x, -(k1*r**3/6))

    yc    = np.where(x <= r, yc_f, yc_a)
    dycdx = np.where(x <= r, dc_f, dc_a)

    th = np.arctan(dycdx)
    xu = x  - yt * np.sin(th);  zu = yc + yt * np.cos(th)
    xl = x  + yt * np.sin(th);  zl = yc - yt * np.cos(th)
    return xu, zu, xl, zl


def save_dat_file(filename, display_name, xu, zu, xl, zl):
    """Write Selig-format .dat file (TE→LE upper, LE→TE lower)."""
    path = os.path.join(FOILS_DIR, filename)
    with open(path, 'w') as f:
        f.write(f"{display_name}\n")
        for i in range(len(xu) - 1, -1, -1):   # upper: TE → LE
            f.write(f"  {xu[i]:.6f}  {zu[i]:.6f}\n")
        for i in range(1, len(xl)):             # lower: LE → TE
            f.write(f"  {xl[i]:.6f}  {zl[i]:.6f}\n")
    return path


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — Generate .dat files + analytical table
# ══════════════════════════════════════════════════════════════════════════════

def generate_foil_files():
    """Generate .dat files for all candidates + NACA 0009 tail airfoil."""
    foils = [
        ('naca2412.dat',  'NACA 2412',  naca4_profile(0.02, 0.40, 0.12)),
        ('naca4412.dat',  'NACA 4412',  naca4_profile(0.04, 0.40, 0.12)),
        ('naca2415.dat',  'NACA 2415',  naca4_profile(0.02, 0.40, 0.15)),
        ('naca23012.dat', 'NACA 23012', naca5_23xxx_profile(0.12)),
        ('naca23015.dat', 'NACA 23015', naca5_23xxx_profile(0.15)),
    ]
    # NACA 0009 (symmetric tail) — regenerate to ensure consistency
    xu0, zu0, xl0, zl0 = naca4_profile(0.00, 0.00, 0.09)
    save_dat_file('naca0009.dat', 'NACA 0009', xu0, zu0, xl0, zl0)

    for fname, dname, (xu, zu, xl, zl) in foils:
        p = save_dat_file(fname, dname, xu, zu, xl, zl)
        print(f"  [dat] {dname:12s}  →  {os.path.basename(p)}")


def print_analytical_table():
    print()
    print("┌" + "─"*90 + "┐")
    print(f"│  {'ANALYTICAL PRE-SCREEN':^88}│")
    print(f"│  {'Re ≈ 3.5 × 10⁶  (root chord at cruise)  Source: Abbott & von Doenhoff':^88}│")
    print("├" + "─"*18 + "┬" + "─"*8 + "┬" + "─"*8 + "┬" + "─"*8 + "┬" + "─"*8 + "┬" + "─"*36 + "┤")
    hdr = f"│  {'Airfoil':<16}│{'CL_max':^8}│{'CM_ac':^8}│{'α_L0°':^8}│{'CD_min':^8}│  Notes"
    print(hdr)
    print("├" + "─"*18 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*8 + "┼" + "─"*36 + "┤")
    for key, c in CANDIDATES.items():
        print(f"│  {c['display']:<16}│{c['CL_max']:^8.2f}│{c['CM_ac']:^8.3f}"
              f"│{c['alpha_L0']:^8.2f}│{c['CD_min']:^8.4f}│  {c['notes'][:34]}")
    print("└" + "─"*18 + "┴" + "─"*8 + "┴" + "─"*8 + "┴" + "─"*8 + "┴" + "─"*8 + "┴" + "─"*36 + "┘")
    print()
    print("  Analytical ranking (before Flow5):")
    print("    1st  NACA 23012  — highest CL_max, lowest |CM_ac|, good cruise L/D")
    print("    2nd  NACA 23015  — structural advantage, nearly as good as 23012")
    print("    3rd  NACA 2412   — proven baseline, adequate for certification")
    print("    4th  NACA 4412   — CM_ac penalty disqualifies it for clean design")
    print("    5th  NACA 2415   — lowest CL_max of the five")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — Flow5 VLM2 analysis for each candidate
# ══════════════════════════════════════════════════════════════════════════════

PLANE_NAME = "GA4-01"
POLAR_NAME = "GA4-T2-VLM2"


def write_candidate_plane_xml(foil_key):
    """Write xflplane XML for GA4-01 using foil_key as the wing airfoil."""
    c    = CANDIDATES[foil_key]
    dname = c['display']             # "NACA 23012"
    xoff_wing = (cr - ct) / 4.0     # LE offset for 0° QC sweep
    z_tip     = (b/2) * math.tan(math.radians(DIHED_DEG))
    xoff_HT   = (cr_HT - ct_HT) / 4.0
    xoff_VT   = (cr_VT - ct_VT) / 4.0

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflplane version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <area_unit_to_m2>1</area_unit_to_m2>
        <mass_unit_to_kg>1</mass_unit_to_kg>
        <speed_unit_to_ms>1</speed_unit_to_ms>
        <inertia_unit_to_kgm2>1</inertia_unit_to_kgm2>
    </Units>
    <Plane>
        <Name>{PLANE_NAME}</Name>
        <Description>GA4-01 wing={dname}  tail=NACA 0009</Description>

        <Inertia>
            <Point_Mass>
                <Tag>Aircraft</Tag>
                <Mass>{MTOW_KG:.1f}</Mass>
                <coordinates>{x_cg:.4f}, 0, 0</coordinates>
            </Point_Mass>
        </Inertia>

        <!-- MAIN WING -->
        <wing>
            <Name>Main Wing</Name>
            <Type>MAINWING</Type>
            <Position>0, 0, 0</Position>
            <Ry_angle>0.000</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>{DIHED_DEG:.3f}</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>9</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>13</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>{dname}</Left_Side_FoilName>
                    <Right_Side_FoilName>{dname}</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b/2:.6f}</y_position>
                    <Chord>{ct:.6f}</Chord>
                    <xOffset>{xoff_wing:.6f}</xOffset>
                    <Dihedral>{DIHED_DEG:.3f}</Dihedral>
                    <Twist>{-TWIST_DEG:.3f}</Twist>
                    <x_number_of_panels>9</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>{dname}</Left_Side_FoilName>
                    <Right_Side_FoilName>{dname}</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- HORIZONTAL TAIL -->
        <wing>
            <Name>Elevator</Name>
            <Type>ELEVATOR</Type>
            <Position>{LHT:.5f}, 0, -0.350</Position>
            <Ry_angle>{TAIL_INC_DEG:.3f}</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr_HT:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>9</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_HT/2:.6f}</y_position>
                    <Chord>{ct_HT:.6f}</Chord>
                    <xOffset>{xoff_HT:.6f}</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- VERTICAL FIN -->
        <wing>
            <Name>Fin</Name>
            <Type>FIN</Type>
            <Position>{LHT:.5f}, 0, -0.625</Position>
            <Rx_angle>-90.000</Rx_angle>
            <Ry_angle>0.000</Ry_angle>
            <symmetric>false</symmetric>
            <Two_Sided>false</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr_VT:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>7</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_VT:.6f}</y_position>
                    <Chord>{ct_VT:.6f}</Chord>
                    <xOffset>{xoff_VT:.6f}</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

    </Plane>
</xflplane>
"""
    path = os.path.join(XML_DIR, f"ga4_cand_{foil_key}.xml")
    with open(path, 'w') as f:
        f.write(xml)
    return path


def write_comparison_polar_xml():
    """Single VLM2 polar XML reused for all candidates."""
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflPlanePolar version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <area_unit_to_m2>1</area_unit_to_m2>
        <mass_unit_to_kg>1</mass_unit_to_kg>
        <speed_unit_to_ms>1</speed_unit_to_ms>
        <inertia_unit_to_kgm2>1</inertia_unit_to_kgm2>
    </Units>
    <Polar>
        <Polar_Name>{POLAR_NAME}</Polar_Name>
        <Plane_Name>{PLANE_NAME}</Plane_Name>
        <Type>FIXEDSPEEDPOLAR</Type>
        <Method>VLM2</Method>
        <Thin_Surfaces>true</Thin_Surfaces>
        <Include_Fuse_Moments>false</Include_Fuse_Moments>
        <Ground_Effect>false</Ground_Effect>
        <Wake>
            <FlatPanelWake>true</FlatPanelWake>
            <NX>5</NX>
            <ProgressionFactor>1.100</ProgressionFactor>
            <LengthFactor>30.000</LengthFactor>
        </Wake>
        <Reference_Dimensions>
            <Reference_Dimensions>CUSTOM</Reference_Dimensions>
            <Reference_Area>{S:.6f}</Reference_Area>
            <Reference_Span_Length>{b:.6f}</Reference_Span_Length>
            <Reference_Chord_Length>{c_bar:.6f}</Reference_Chord_Length>
        </Reference_Dimensions>
        <Fluid>
            <Viscosity>1.5e-05</Viscosity>
            <Density>1.225</Density>
        </Fluid>
        <Viscous_Analysis>
            <Is_Viscous_Analysis>false</Is_Viscous_Analysis>
        </Viscous_Analysis>
        <Use_plane_inertia>true</Use_plane_inertia>
        <Fixed_Velocity>{V_CRUISE:.3f}</Fixed_Velocity>
    </Polar>
</xflPlanePolar>
"""
    path = os.path.join(XML_DIR, "ga4_polar_vlm2.xml")
    with open(path, 'w') as f:
        f.write(xml)
    return path


def write_candidate_script_xml(foil_key):
    """Script XML that loads the candidate plane + common VLM2 polar."""
    c = CANDIDATES[foil_key]
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
        <MultiThreading>
            <Allow_Multithreading>true</Allow_Multithreading>
            <Thread_Priority>Normal</Thread_Priority>
            <Max_threads>4</Max_threads>
        </MultiThreading>
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
            <Foil_File_Name>{foil_key}.dat</Foil_File_Name>
            <Foil_File_Name>naca0009.dat</Foil_File_Name>
        </Foil_Dat_Files>

        <Plane_Definition_Files>
            <Process_All_Files>false</Process_All_Files>
            <Plane_File_Name>ga4_cand_{foil_key}.xml</Plane_File_Name>
        </Plane_Definition_Files>

        <Plane_Analysis_Files>
            <Process_All_Files>false</Process_All_Files>
            <Analysis_File_Name>ga4_polar_vlm2.xml</Analysis_File_Name>
        </Plane_Analysis_Files>

        <Plane_Analysis_Data>
            <T12_Range>-6, 16, 1</T12_Range>
        </Plane_Analysis_Data>
    </Plane_Analysis>
</xflscript>
"""
    path = os.path.join(XML_DIR, f"ga4_script_{foil_key}.xml")
    with open(path, 'w') as f:
        f.write(xml)
    return path


def run_candidate(foil_key):
    """Run Flow5 VLM2 for one candidate, return performance dict."""
    write_candidate_plane_xml(foil_key)
    script_path = write_candidate_script_xml(foil_key)

    ok, stdout, stderr = run_flow5(script_path, timeout=90)

    time.sleep(0.5)
    csv_path = find_polar_csv()
    if not csv_path:
        return None, f"No CSV output for {foil_key}"

    polar = parse_polar_csv(csv_path)
    perf  = extract_performance(polar, CL_CRUISE)
    if perf is None:
        return None, f"Could not parse polar for {foil_key}"

    # Also extract CL_max: last stable CL in polar (before divergence)
    cls = np.array(polar['CL'])
    if len(cls) == 0:
        return None, "Empty polar"

    # Find CL_max as the highest CL that was reached
    cl_max_f5 = float(np.max(cls))
    perf['CL_max_flow5'] = cl_max_f5
    return perf, None


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — Select winner
# ══════════════════════════════════════════════════════════════════════════════

def select_winner(flow5_results):
    """
    Combined scoring:
      40%  L/D at CL_cruise       (Flow5 VLM2)
      35%  CL_max                 (Flow5 VLM2)
      25%  1/|CM_ac|              (analytical data — VLM CM depends on full model)
    """
    keys   = [k for k, v in flow5_results.items() if v[0] is not None]
    if not keys:
        return list(CANDIDATES.keys())[0]   # fallback: first candidate

    ld_vals    = np.array([flow5_results[k][0]['LD']           for k in keys])
    clmax_vals = np.array([flow5_results[k][0]['CL_max_flow5'] for k in keys])
    cm_vals    = np.array([1.0 / max(abs(CANDIDATES[k]['CM_ac']), 0.001) for k in keys])

    def norm(v):
        rng = v.max() - v.min()
        return (v - v.min()) / rng if rng > 1e-9 else np.ones_like(v)

    scores = 0.40 * norm(ld_vals) + 0.35 * norm(clmax_vals) + 0.25 * norm(cm_vals)
    winner = keys[int(np.argmax(scores))]

    print()
    print("  Flow5 VLM2 comparison:")
    print(f"  {'Airfoil':<14}  {'L/D@cruise':>10}  {'CL_max':>8}  {'CM_ac':>8}  {'Score':>6}")
    print("  " + "─"*52)
    for i, k in enumerate(keys):
        p   = flow5_results[k][0]
        mrk = " ◀ WINNER" if k == winner else ""
        print(f"  {CANDIDATES[k]['display']:<14}  {p['LD']:>10.3f}  "
              f"{p['CL_max_flow5']:>8.3f}  {CANDIDATES[k]['CM_ac']:>8.3f}  "
              f"{scores[i]:>6.3f}{mrk}")
    print()
    return winner


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — Write ga4_final_plane.xml  (full aircraft + fuselage body)
# ══════════════════════════════════════════════════════════════════════════════

def _half_ellipse_pts(ry, rz, n=5):
    """
    Half-ellipse for body cross-section: n points from bottom (y=0, z=-rz) to top (y=0, z=+rz)
    going through the starboard side (y >= 0).
    Flow5/XFLR5 body convention: half-body defined; port side mirrored automatically.
    Bottom→top ordering gives outward-pointing panel normals (+y, ±z for starboard panels).
    Angles go from -90° (bottom) to +90° (top).
    """
    pts = []
    for i in range(n):
        ang = -math.pi/2 + math.pi * i / (n - 1)   # -pi/2 → +pi/2
        pts.append((ry * math.cos(ang), rz * math.sin(ang)))
    return pts


def _build_fuselage_body_xml():
    """
    Build FLATPANELS fuselage body element using confirmed Flow5 binary tag names:
      BodyFrame, BodyHoopRes, BodyAxialRes, Position, Point.
    Half-body convention: points go top (0,rz) → starboard (+y) → bottom (0,-rz).
    Body origin: x=-1.70 (nose forward of wing LE), z=-0.625 (high-wing config).
    """
    # (x_from_nose,  half_y_width,  half_z_height)
    stations = [
        (0.00,  0.008, 0.008),   # nose tip
        (0.25,  0.120, 0.120),   # nose cone forward
        (0.65,  0.350, 0.360),   # windshield base
        (1.10,  0.520, 0.530),   # forward cabin
        (1.70,  0.610, 0.625),   # max section (wing junction)
        (2.40,  0.610, 0.625),   # cabin centre
        (3.20,  0.610, 0.625),   # cabin aft
        (4.40,  0.610, 0.620),   # cabin end / rear bulkhead
        (5.30,  0.520, 0.530),   # tail cone start
        (6.30,  0.380, 0.395),
        (7.20,  0.230, 0.250),
        (8.00,  0.100, 0.115),
        (8.50,  0.035, 0.040),   # tail end
    ]

    frames = []
    for pos, ry, rz in stations:
        pts = _half_ellipse_pts(ry, rz, n=5)   # 5 pts: top→upper-stbd→equator→lower-stbd→bottom
        pts_xml = "\n".join(
            f"                <Point>{y:.5f}, {z:.5f}</Point>"
            for y, z in pts
        )
        frames.append(
            f"            <BodyFrame>\n"
            f"                <Position>{pos:.4f}</Position>\n"
            f"{pts_xml}\n"
            f"            </BodyFrame>"
        )

    return "\n".join(frames)


def write_ga4_final_plane_xml(winner_key, outpath=None, include_body=False):
    """
    Write GA4-01 plane XML: wing + H-stab + V-fin.
    include_body: embed the fuselage <body> element (experimental — may crash
    older Flow5 builds during QUADS meshing; use GUI body editor instead).
    """
    dname = CANDIDATES[winner_key]['display']

    xoff_wing = (cr - ct) / 4.0
    xoff_HT   = (cr_HT - ct_HT) / 4.0
    xoff_VT   = (cr_VT - ct_VT) / 4.0

    fuse_block = ""
    if include_body:
        fuse_frames = _build_fuselage_body_xml()
        fuse_block = f"""
        <!-- ══════════════  FUSELAGE BODY  ══════════════
             8.50 m × 1.22 m elliptical cross-section
             Origin: x=-1.70 (nose), z=-0.625 (high-wing config)
             Half-body convention: 5 pts bottom→top through starboard
             NOTE: body meshing may crash Flow5 QUADS headless mode.
             Add fuselage via Flow5 GUI: Edit → Define Fuselage.
        ════════════════════════════════════════════════ -->
        <body>
            <Name>GA4-01 Fuselage</Name>
            <Description>8.50 m × 1.22 m  FLATPANELS  high-wing GA fuselage</Description>
            <Type>FLATPANELS</Type>
            <BodyAxialRes>17</BodyAxialRes>
            <BodyHoopRes>8</BodyHoopRes>
            <x>-1.7000</x>
            <y>0.0000</y>
            <z>-0.6250</z>
{fuse_frames}
        </body>
"""

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflplane version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <area_unit_to_m2>1</area_unit_to_m2>
        <mass_unit_to_kg>1</mass_unit_to_kg>
        <speed_unit_to_ms>1</speed_unit_to_ms>
        <inertia_unit_to_kgm2>1</inertia_unit_to_kgm2>
    </Units>
    <Plane>
        <Name>GA4-01</Name>
        <Description>
            GA4-01  Four-seat single-engine GA  |  Wing: {dname}  |  Tail: NACA 0009
            MTOW {MTOW_KG:.0f} kg  |  b = {b:.3f} m  |  S = {S:.2f} m²
            AR = {AR:.1f}  taper = {TAPER:.2f}  washout = {TWIST_DEG:.1f}°
        </Description>

        <Inertia>
            <Point_Mass>
                <Tag>Aircraft total  ({MTOW_KG:.0f} kg MTOW)</Tag>
                <Mass>{MTOW_KG:.1f}</Mass>
                <coordinates>{x_cg:.4f}, 0, 0</coordinates>
            </Point_Mass>
        </Inertia>

        <!-- ══════════════  MAIN WING ({dname})  ══════════════ -->
        <wing>
            <Name>Main Wing</Name>
            <Type>MAINWING</Type>
            <Position>0, 0, 0</Position>
            <Ry_angle>0.000</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>{DIHED_DEG:.3f}</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>11</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>15</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>{dname}</Left_Side_FoilName>
                    <Right_Side_FoilName>{dname}</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b/2:.6f}</y_position>
                    <Chord>{ct:.6f}</Chord>
                    <xOffset>{xoff_wing:.6f}</xOffset>
                    <Dihedral>{DIHED_DEG:.3f}</Dihedral>
                    <Twist>{-TWIST_DEG:.3f}</Twist>
                    <x_number_of_panels>11</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>{dname}</Left_Side_FoilName>
                    <Right_Side_FoilName>{dname}</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- ══════════════  HORIZONTAL TAIL (NACA 0009)  ══════════════ -->
        <wing>
            <Name>Elevator</Name>
            <Type>ELEVATOR</Type>
            <Position>{LHT:.5f}, 0, -0.350</Position>
            <Ry_angle>{TAIL_INC_DEG:.3f}</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr_HT:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>9</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_HT/2:.6f}</y_position>
                    <Chord>{ct_HT:.6f}</Chord>
                    <xOffset>{xoff_HT:.6f}</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- ══════════════  VERTICAL FIN (NACA 0009)  ══════════════ -->
        <wing>
            <Name>Fin</Name>
            <Type>FIN</Type>
            <Position>{LHT:.5f}, 0, -0.625</Position>
            <Rx_angle>-90.000</Rx_angle>
            <Ry_angle>0.000</Ry_angle>
            <symmetric>false</symmetric>
            <Two_Sided>false</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000000</y_position>
                    <Chord>{cr_VT:.6f}</Chord>
                    <xOffset>0.000000</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>7</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_VT:.6f}</y_position>
                    <Chord>{ct_VT:.6f}</Chord>
                    <xOffset>{xoff_VT:.6f}</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

{fuse_block}
    </Plane>
</xflplane>
"""
    outpath = outpath or os.path.join(XML_DIR, "ga4_final_plane.xml")
    with open(outpath, 'w') as f:
        f.write(xml)
    return outpath


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — QUADS 3D panel polar + script
# ══════════════════════════════════════════════════════════════════════════════

def write_ga4_quads_polar_xml(outpath=None):
    """
    QUADS method — 3D panel analysis.
    From Flow5 binary: 'Available methods are: LLT, VLM1, VLM2, QUADS, TRIUNIFORM, TRILINEAR'
    QUADS = quad-panel 3D method; Thin_Surfaces=false activates surface thickness.
    """
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE flow5>
<xflPlanePolar version="1.0">
    <Units>
        <length_unit_to_meter>1</length_unit_to_meter>
        <area_unit_to_m2>1</area_unit_to_m2>
        <mass_unit_to_kg>1</mass_unit_to_kg>
        <speed_unit_to_ms>1</speed_unit_to_ms>
        <inertia_unit_to_kgm2>1</inertia_unit_to_kgm2>
    </Units>
    <Polar>
        <Polar_Name>GA4-T2-QUADS-3D</Polar_Name>
        <Plane_Name>GA4-01</Plane_Name>
        <Type>FIXEDSPEEDPOLAR</Type>
        <Method>QUADS</Method>
        <Thin_Surfaces>false</Thin_Surfaces>
        <Include_Fuse_Moments>true</Include_Fuse_Moments>
        <Ground_Effect>false</Ground_Effect>
        <Wake>
            <FlatPanelWake>true</FlatPanelWake>
            <NX>5</NX>
            <ProgressionFactor>1.100</ProgressionFactor>
            <LengthFactor>30.000</LengthFactor>
        </Wake>
        <Reference_Dimensions>
            <Reference_Dimensions>CUSTOM</Reference_Dimensions>
            <Reference_Area>{S:.6f}</Reference_Area>
            <Reference_Span_Length>{b:.6f}</Reference_Span_Length>
            <Reference_Chord_Length>{c_bar:.6f}</Reference_Chord_Length>
        </Reference_Dimensions>
        <Fluid>
            <Viscosity>1.5e-05</Viscosity>
            <Density>1.225</Density>
        </Fluid>
        <Viscous_Analysis>
            <Is_Viscous_Analysis>false</Is_Viscous_Analysis>
        </Viscous_Analysis>
        <Use_plane_inertia>true</Use_plane_inertia>
        <Fixed_Velocity>{V_CRUISE:.3f}</Fixed_Velocity>
    </Polar>
</xflPlanePolar>
"""
    outpath = outpath or os.path.join(XML_DIR, "ga4_panel_polar.xml")
    with open(outpath, 'w') as f:
        f.write(xml)
    return outpath


def write_ga4_panel_script_xml(winner_key, outpath=None):
    """Script to run the 3D QUADS panel analysis headlessly (no fuselage body)."""
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
        <MultiThreading>
            <Allow_Multithreading>true</Allow_Multithreading>
            <Thread_Priority>Normal</Thread_Priority>
            <Max_threads>4</Max_threads>
        </MultiThreading>
        <Double_Precision>true</Double_Precision>
    </Metadata>

    <Plane_Analysis>
        <Plane_Analysis_Output>
            <make_oppoints>false</make_oppoints>
            <make_oppoints_text_file>false</make_oppoints_text_file>
            <compute_derivatives>false</compute_derivatives>
            <make_polars_text_file>true</make_polars_text_file>
            <export_stl_mesh>true</export_stl_mesh>
        </Plane_Analysis_Output>

        <!-- Wing airfoil = {CANDIDATES[winner_key]['display']}, tail = NACA 0009 -->
        <Foil_Dat_Files>
            <Foil_File_Name>{winner_key}.dat</Foil_File_Name>
            <Foil_File_Name>naca0009.dat</Foil_File_Name>
        </Foil_Dat_Files>

        <Plane_Definition_Files>
            <Process_All_Files>false</Process_All_Files>
            <Plane_File_Name>ga4_final_plane.xml</Plane_File_Name>
        </Plane_Definition_Files>

        <Plane_Analysis_Files>
            <Process_All_Files>false</Process_All_Files>
            <Analysis_File_Name>ga4_panel_polar.xml</Analysis_File_Name>
        </Plane_Analysis_Files>

        <Plane_Analysis_Data>
            <T12_Range>-4, 14, 1</T12_Range>
        </Plane_Analysis_Data>
    </Plane_Analysis>
</xflscript>
"""
    outpath = outpath or os.path.join(XML_DIR, "ga4_panel_script.xml")
    with open(outpath, 'w') as f:
        f.write(xml)
    return outpath


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print()
    print("═" * 70)
    print("  GA4-01 AIRFOIL SELECTION  —  analytical + Flow5 VLM2 pipeline")
    print("═" * 70)

    # ── Phase 1: generate .dat files & analytical table ──────────────────
    print("\n[1/5] Generating NACA profile .dat files …")
    generate_foil_files()
    write_comparison_polar_xml()
    print_analytical_table()

    # ── Phase 2: Flow5 VLM2 for all 5 candidates ─────────────────────────
    print("[2/5] Running Flow5 VLM2 for each candidate …")
    flow5_results = {}
    for key in CANDIDATES:
        print(f"       {CANDIDATES[key]['display']:10s} … ", end='', flush=True)
        perf, err = run_candidate(key)
        if err:
            print(f"FAILED ({err})")
            flow5_results[key] = (None, err)
        else:
            print(f"L/D = {perf['LD']:.2f}  CL_max = {perf['CL_max_flow5']:.3f}  "
                  f"CD@cruise = {perf['CD']:.5f}")
            flow5_results[key] = (perf, None)

    # ── Phase 3: select winner ────────────────────────────────────────────
    print("\n[3/5] Selecting winner …")
    winner = select_winner(flow5_results)
    wc     = CANDIDATES[winner]
    print(f"\n  ✓ Winner: {wc['display']}")
    print(f"    Analytical CL_max = {wc['CL_max']:.2f}  CM_ac = {wc['CM_ac']:.3f}")
    print(f"    {wc['notes']}")

    # ── Phase 4: write final plane XML (no body — avoids Flow5 crash) ──────
    print(f"\n[4/5] Writing aircraft plane XML ({wc['display']}) …")
    plane_path = write_ga4_final_plane_xml(winner, include_body=False)
    print(f"       → {plane_path}")
    print(f"       (Fuselage body omitted for headless compatibility — add via GUI)")

    # ── Phase 5: QUADS 3D panel polar + script + run ─────────────────────
    print("\n[5/5] Writing 3D QUADS panel polar + script …")
    polar_path  = write_ga4_quads_polar_xml()
    script_path = write_ga4_panel_script_xml(winner)
    print(f"       polar  → {polar_path}")
    print(f"       script → {script_path}")

    print("\n       Running Flow5 QUADS 3D panel analysis (1053 panels) … ",
          end='', flush=True)
    _, _, _ = run_flow5(script_path, timeout=180)
    time.sleep(1.0)
    csv_path = find_polar_csv()
    if csv_path:
        polar3d = parse_polar_csv(csv_path)
        perf3d  = extract_performance(polar3d, CL_CRUISE)
        if perf3d:
            import numpy as np
            cls3d   = np.array(polar3d['CL'])
            a3d     = np.array(polar3d['alpha'])
            clmax3d = float(cls3d.max())
            amax3d  = float(a3d[cls3d.argmax()])
            print("OK")
            print(f"\n  ┌─ Flow5 QUADS 3D Panel  (wing + H-stab + V-fin, 1053 panels) ────┐")
            print(f"  │  Speed:  {V_CRUISE:.2f} m/s  ({V_CRUISE*1.944:.1f} kt)                            │")
            print(f"  │  At CL_cruise = {CL_CRUISE:.4f}:                                      │")
            print(f"  │    α        = {perf3d['alpha']:+.2f}°                                   │")
            print(f"  │    CL       = {perf3d['CL']:.4f}                                    │")
            print(f"  │    CD_ind   = {perf3d['CDi']:.5f}  (inviscid, no profile drag)    │")
            print(f"  │    L/D_ind  = {perf3d['LD']:.1f}  (induced only)                 │")
            print(f"  │  CL_max     = {clmax3d:.4f}  at α = {amax3d:.1f}°                       │")
            print(f"  └──────────────────────────────────────────────────────────────────┘")
        else:
            print("parsed — no operating point near CL_cruise")
    else:
        print("FAILED — check Flow5 output")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("═" * 70)
    print("  OUTPUT FILES")
    print("═" * 70)
    print(f"  analysis_xml/ga4_final_plane.xml")
    print(f"      Open in Flow5 GUI:  File → Open  (wing + tail + fin)")
    print(f"      Airfoil: {wc['display']}  |  Then add fuselage: Edit → Define Fuselage")
    print(f"  analysis_xml/ga4_panel_polar.xml   — QUADS 3D polar definition")
    print(f"  analysis_xml/ga4_panel_script.xml  — headless re-run script")
    print()
    print("  To re-run headlessly:")
    print(f"    {FLOW5_APP} -s {script_path} -p")
    print()


if __name__ == "__main__":
    main()

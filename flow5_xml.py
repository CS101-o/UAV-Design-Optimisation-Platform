"""
Generates the three XML files Flow5 needs to run in script mode:
  1. drone_plane.xml   — wing + tail geometry  (xflplane format)
  2. drone_polar.xml   — VLM analysis settings  (xflPlanePolar format)
  3. drone_script.xml  — master script that ties everything together
"""
import os
import math
import numpy as np

FLOW5_APP  = "/Applications/flow5.app/Contents/MacOS/flow5"
WORK_DIR   = os.path.dirname(os.path.abspath(__file__))
FOILS_DIR  = os.path.join(WORK_DIR, "foils")
XML_DIR    = os.path.join(WORK_DIR, "analysis_xml")
OUTPUT_DIR = os.path.join(WORK_DIR, "output")

PLANE_NAME = "drone_opt"
POLAR_NAME = "T1-VLM-ThinSurf"


def _x(val, digits=6):
    return f"{val:>{digits+6}.{digits}f}"


def write_plane_xml(geom, path=None):
    """
    Write xflplane XML from a geometry dict.

    geom keys (all in SI, degrees):
        b, cr, ct, AR, taper, twist_deg, incidence_deg, dihedral_deg, sweep_deg
        SHT, LHT, b_HT, cr_HT, ct_HT, tail_inc_deg
        SVT, b_VT, cr_VT, ct_VT
        mtow_kg, cg_frac_mac
    """
    b    = geom['b'];    cr  = geom['cr'];   ct  = geom['ct']
    taper = geom['taper']
    twist = geom['twist_deg']
    inc   = geom['incidence_deg']
    dihed = geom['dihedral_deg']
    # Quarter-chord sweep offset at tip: keeps quarter-chord line straight
    xoff_wing_tip = (cr - ct) / 4.0
    z_wing_tip    = (b / 2.0) * math.tan(math.radians(dihed))

    # Horizontal tail
    LHT   = geom['LHT'];  b_HT  = geom['b_HT']
    cr_HT = geom['cr_HT']; ct_HT = geom['ct_HT']
    t_inc = geom['tail_inc_deg']
    HT_sweep_deg    = geom.get('HT_sweep_deg', 0.0)
    HT_twist_deg    = geom.get('HT_twist_deg', 0.0)
    HT_dihedral_deg = geom.get('HT_dihedral_deg', 0.0)
    xoff_HT_tip = (cr_HT - ct_HT) / 4.0 + (b_HT / 2.0) * math.tan(math.radians(HT_sweep_deg))

    # Vertical fin — geometry computed by analysis.py (SVT_frac/AR_VT/taper_VT
    # design variables) when present; falls back to the historical fixed
    # AR_VT=1.8, taper=0.5 for any caller that hasn't been updated.
    SVT   = geom['SVT']
    AR_VT = geom.get('AR_VT', 1.8)
    b_VT  = geom.get('b_VT') or math.sqrt(AR_VT * SVT)
    cr_VT = geom.get('cr_VT')
    ct_VT = geom.get('ct_VT')
    if cr_VT is None or ct_VT is None:
        tv    = geom.get('taper_VT', 0.5)
        cr_VT = 2 * SVT / (b_VT * (1 + tv))
        ct_VT = tv * cr_VT
    VT_sweep_deg = geom.get('VT_sweep_deg', 0.0)
    # Fin spans the FULL b_VT (not b_VT/2 — unlike the symmetric wing/HT,
    # the fin is a single non-mirrored surface), so the sweep offset uses
    # the full height, not the half-span used for wing/HT.
    xoff_VT_tip = (cr_VT - ct_VT) / 4.0 + b_VT * math.tan(math.radians(VT_sweep_deg))

    # CG position (x from wing LE at root)
    c_bar = b / geom['AR']
    x_cg  = geom['cg_frac_mac'] * c_bar   # x from wing LE at root

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
        <Inertia>
            <Point_Mass>
                <Tag>Total aircraft</Tag>
                <Mass>{geom['mtow_kg']:.4f}</Mass>
                <coordinates>{x_cg:.4f}, 0, 0</coordinates>
            </Point_Mass>
        </Inertia>

        <!-- ═══════════════  MAIN WING  ═══════════════ -->
        <wing>
            <Name>Main Wing</Name>
            <Type>MAINWING</Type>
            <Position>0, 0, 0</Position>
            <Ry_angle>{inc:.3f}</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000</y_position>
                    <Chord>{cr:.5f}</Chord>
                    <xOffset>0.000</xOffset>
                    <Dihedral>{dihed:.3f}</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>11</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>E387</Left_Side_FoilName>
                    <Right_Side_FoilName>E387</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b/2:.5f}</y_position>
                    <Chord>{ct:.5f}</Chord>
                    <xOffset>{xoff_wing_tip:.5f}</xOffset>
                    <Dihedral>{dihed:.3f}</Dihedral>
                    <Twist>{-twist:.3f}</Twist>
                    <x_number_of_panels>7</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>E387</Left_Side_FoilName>
                    <Right_Side_FoilName>E387</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- ═══════════════  HORIZONTAL TAIL  ═══════════════ -->
        <wing>
            <Name>Elevator</Name>
            <Type>ELEVATOR</Type>
            <Position>{LHT:.5f}, 0, 0</Position>
            <Ry_angle>{t_inc:.3f}</Ry_angle>
            <symmetric>true</symmetric>
            <Two_Sided>true</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000</y_position>
                    <Chord>{cr_HT:.5f}</Chord>
                    <xOffset>0.000</xOffset>
                    <Dihedral>{HT_dihedral_deg:.3f}</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>5</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>7</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_HT/2:.5f}</y_position>
                    <Chord>{ct_HT:.5f}</Chord>
                    <xOffset>{xoff_HT_tip:.5f}</xOffset>
                    <Dihedral>{HT_dihedral_deg:.3f}</Dihedral>
                    <Twist>{-HT_twist_deg:.3f}</Twist>
                    <x_number_of_panels>5</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>0</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
            </Sections>
        </wing>

        <!-- ═══════════════  VERTICAL FIN  ═══════════════
             Root lifted +0.03 m clear of the elevator surface: QUADS thick
             meshing cannot handle intersecting bodies — a fin embedded in
             the elevator produces spurious, incidence-dependent pitching
             moments (verified empirically). -->
        <wing>
            <Name>Fin</Name>
            <Type>FIN</Type>
            <Position>{LHT:.5f}, 0, 0.030</Position>
            <Rx_angle>-90.000</Rx_angle>
            <Ry_angle>0.000</Ry_angle>
            <symmetric>false</symmetric>
            <Two_Sided>false</Two_Sided>
            <Sections>
                <Section>
                    <y_position>0.000</y_position>
                    <Chord>{cr_VT:.5f}</Chord>
                    <xOffset>0.000</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>5</x_number_of_panels>
                    <x_panel_distribution>COSINE</x_panel_distribution>
                    <y_number_of_panels>5</y_number_of_panels>
                    <y_panel_distribution>COSINE</y_panel_distribution>
                    <Left_Side_FoilName>NACA 0009</Left_Side_FoilName>
                    <Right_Side_FoilName>NACA 0009</Right_Side_FoilName>
                </Section>
                <Section>
                    <y_position>{b_VT:.5f}</y_position>
                    <Chord>{ct_VT:.5f}</Chord>
                    <xOffset>{xoff_VT_tip:.5f}</xOffset>
                    <Dihedral>0.000</Dihedral>
                    <Twist>0.000</Twist>
                    <x_number_of_panels>5</x_number_of_panels>
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
    path = path or os.path.join(XML_DIR, "drone_plane.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def write_polar_xml(speed_ms, ref_area=None, ref_span=None, ref_chord=None, path=None,
                    method="TRIUNIFORM", viscous=False):
    """Write xflPlanePolar XML for a fixed-speed analysis.

    ref_area, ref_span, ref_chord: explicit reference dimensions (m², m, m).
    If omitted, falls back to defaults derived from requirements.
    method:  TRIUNIFORM (thin surfaces) or QUADS (thick surfaces).
    viscous: True → XFoil-on-the-fly viscous drag at wing sections
             (no 2D polar mesh needed; ~2-3 s per operating point).
    """
    import math
    if ref_area is None:
        from requirements import DroneRequirements
        _req = DroneRequirements()
        ref_area = _req.wing_area
    if ref_span is None:
        ref_span = math.sqrt(7.0 * ref_area)   # AR=7 default
    if ref_chord is None:
        ref_chord = ref_area / ref_span

    thin = 'true' if method == 'TRIUNIFORM' else 'false'
    visc = 'true' if viscous else 'false'

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
        <Method>{method}</Method>
        <Thin_Surfaces>{thin}</Thin_Surfaces>
        <Include_Fuse_Moments>false</Include_Fuse_Moments>
        <Ground_Effect>false</Ground_Effect>
        <Wake>
            <FlatPanelWake>true</FlatPanelWake>
            <NX>5</NX>
            <ProgressionFactor>1.100</ProgressionFactor>
            <LengthFactor>30.000</LengthFactor>
        </Wake>
        <Reference_Dimensions>
            <Reference_Dimensions>Custom</Reference_Dimensions>
            <Reference_Area>{ref_area:.6f}</Reference_Area>
            <Reference_Span_Length>{ref_span:.6f}</Reference_Span_Length>
            <Reference_Chord_Length>{ref_chord:.6f}</Reference_Chord_Length>
        </Reference_Dimensions>
        <Fluid>
            <Viscosity>1.5e-05</Viscosity>
            <Density>1.225</Density>
        </Fluid>
        <Viscous_Analysis>
            <Is_Viscous_Analysis>{visc}</Is_Viscous_Analysis>
            <XFoil_OnTheFly>{visc}</XFoil_OnTheFly>
            <From_CL>true</From_CL>
        </Viscous_Analysis>
        <Use_plane_inertia>true</Use_plane_inertia>
        <Fixed_Velocity>{speed_ms:.3f}</Fixed_Velocity>
    </Polar>
</xflPlanePolar>
"""
    path = path or os.path.join(XML_DIR, "drone_polar.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def write_script_xml(aoa_min=-5, aoa_max=14, aoa_step=1, path=None):
    """Write the master xflscript XML that drives the headless Flow5 run."""
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
            <Foil_File_Name>e387.dat</Foil_File_Name>
            <Foil_File_Name>naca0009.dat</Foil_File_Name>
        </Foil_Dat_Files>

        <Plane_Definition_Files>
            <Process_All_Files>false</Process_All_Files>
            <Plane_File_Name>drone_plane.xml</Plane_File_Name>
        </Plane_Definition_Files>

        <Plane_Analysis_Files>
            <Process_All_Files>false</Process_All_Files>
            <Analysis_File_Name>drone_polar.xml</Analysis_File_Name>
        </Plane_Analysis_Files>

        <Plane_Analysis_Data>
            <T12_Range>{aoa_min}, {aoa_max}, {aoa_step}</T12_Range>
        </Plane_Analysis_Data>
    </Plane_Analysis>
</xflscript>
"""
    path = path or os.path.join(XML_DIR, "drone_script.xml")
    with open(path, "w") as f:
        f.write(xml)
    return path


def geometry_from_result(res, req):
    """Convert optimizer result + requirements into the geom dict for write_plane_xml."""
    AR    = res['AR']
    b     = res['b'];  cr = res['cr'];  ct = res['ct']
    c_bar = b / AR
    SHT   = req.SHT_fraction * req.wing_area
    LHT   = req.tail_arm_chords * c_bar
    AR_HT = 4.0
    b_HT  = math.sqrt(AR_HT * SHT)
    cr_HT = 2 * SHT / (b_HT * 1.4)
    ct_HT = 0.4 * cr_HT
    SVT   = req.SVT_fraction * req.wing_area

    return {
        'b': b, 'cr': cr, 'ct': ct, 'AR': AR, 'taper': res['taper'],
        'twist_deg':     res['twist_deg'],
        'incidence_deg': res['optimal_incidence_deg'],
        'dihedral_deg':  req.dihedral_deg,
        'sweep_deg':     req.sweep_deg,
        'SHT': SHT, 'LHT': LHT, 'b_HT': b_HT, 'cr_HT': cr_HT, 'ct_HT': ct_HT,
        'tail_inc_deg':  res['tail_inc_deg'],
        'SVT': SVT,
        'mtow_kg':       req.mtow_kg,
        'cg_frac_mac':   req.cg_frac_mac,
    }

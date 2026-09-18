import numpy as np


class DroneRequirements:
    """
    Mission and aircraft-level requirements fixed before design begins.

    These come from:
      • Mission specification (payload, speeds, endurance)
      • Constraint analysis output (wing loading, T/W)
      • Airfoil polar data (XFoil / UIUC wind-tunnel)

    Nothing in here is a design decision:
      - CG position    → auto-placed at NP − SM_target during optimisation
      - Wing incidence → derived from cruise AoA after optimisation
      - Tail sizing    → design variables in Phase 2
      - Bounds / x0   → defined in optimizer.py
    """

    def __init__(self):
        # ── Mission specification ──────────────────────────────────────────
        self.mtow_kg           = 3.0      # max take-off mass [kg]
        self.payload_kg        = 0.5      # payload mass [kg]
        self.cruise_speed_ms   = 20.0     # cruise airspeed [m/s]
        self.stall_speed_ms    = 10.0     # clean-configuration stall speed requirement [m/s]
        self.V_stall_land_ms   = 8.0      # landing (flaps-out) stall speed requirement [m/s]
        self.battery_wh        = 100.0    # battery energy [Wh]

        # ── Constraint analysis outputs (Part 3) ───────────────────────────
        self.wing_loading_Nm2  = 120.0    # W/S from constraint diagram [N/m²]
        self.thrust_to_weight  = 0.40     # T/W from constraint diagram
        self.propulsive_eta    = 0.75     # propeller efficiency η

        # ── Derived from mission (not design choices) ──────────────────────
        self.wing_area = (self.mtow_kg * 9.81) / self.wing_loading_Nm2   # [m²]

        # ── Airfoil aerodynamic constants (E387 at Re ≈ 200 k) ───────────
        # Source: XFoil analysis and UIUC LSATs wind-tunnel data
        self.CL_alpha_2d   = 2 * np.pi    # 2-D lift curve slope [/rad]
        self.alpha_L0_deg  = -4.0         # zero-lift angle of attack [deg]
        self.CM_ac         = -0.09        # pitching moment at aerodynamic centre
        self.CL_max_2d     =  1.25        # 2-D max lift coefficient

        # ── Stability target ───────────────────────────────────────────────
        self.target_sm     = 0.10         # static margin target [fraction MAC]

        # ── Dynamic stability limits (CS-UAV / ASTM F3312-19) ─────────────
        self.dyn_ph_zeta_min = 0.04       # phugoid minimum damping ratio
        self.dyn_sp_zeta_min = 0.35       # short-period minimum damping ratio
        self.dyn_sp_zeta_max = 1.30       # short-period maximum damping ratio

        # ── Lateral-directional stability targets ──────────────────────────
        # No single certification standard covers small fixed-wing UAVs;
        # these are representative conceptual-design targets (Roskam
        # "Airplane Design" Pt. VII / Nelson "Flight Stability and Automatic
        # Control" guidance), not a substitute for a full 6-DOF analysis.
        self.target_cn_beta_min      = 0.05   # min weathercock stability [Cn_beta, /rad]
        self.target_cl_beta_max_ratio = 1.2   # max |Cl_beta / Cn_beta| — excess dihedral
                                                # effect relative to Cn_beta degrades
                                                # dutch-roll damping (Roskam Pt.VII)


# Backward-compatibility alias (flow5_xml.py still imports DroneRequirements)
MissionRequirements = DroneRequirements

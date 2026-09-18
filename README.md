# Flow5 UAV Design & Optimisation Suite

A Python toolchain for conceptual aerodynamic design of fixed-wing UAVs, built around [Flow5](https://flow5.com/) as the panel-method solver. Two aircraft are covered: a **3 kg electric drone** (the primary design target) and a **GA4 general-aviation aircraft** (used for airfoil selection validation).

---

## Documentation

Four markdown files describe the project at different levels. Read them in this order when coming to the codebase fresh:

| File | What it covers | When to read it |
|---|---|---|
| **`README.md`** _(this file)_ | Code reference — every module, function signature, and XML schema | When you need to know what a specific file or function does |
| [**`OPTIMISATION_METHODOLOGY.md`**](OPTIMISATION_METHODOLOGY.md) | The three optimisation phases (LLT+SLSQP → Flow5+Simplex → Flow5+GP BO), why each transition was made, and what the current approach brings | Start here to understand *why* the code is structured the way it is |
| [**`STATUS.md`**](STATUS.md) | What has been achieved vs what is still open (GUI integration, SM constraint, viscous BO, seed spread) | Check before starting new work so you know what is and is not done |
| [**`GAPS_FIXED.md`**](GAPS_FIXED.md) | Design-space and constraint gaps that were identified and closed: stall/trim/stability penalties in the BO objective, dihedral/twist bounds, HT twist/dihedral, VT sweep, flap deflection | Read when modifying the objective function, var specs, or `analysis.py` — each gap entry explains what was missing, what was added, and why |

---

## Project layout

```
Flow5/
├── requirements.py          # Drone mission & design requirements
├── analysis.py              # Analytical Lifting-Line Theory (LLT) model
├── optimizer.py             # SLSQP optimizer using LLT (no Flow5 needed)
├── main.py                  # CLI entry point for analytical optimizer
├── app.py                   # Tkinter GUI for analytical optimizer (dark theme)
│
├── flow5_xml.py             # Flow5 XML generators (plane / polar / script)
├── flow5_run.py             # Flow5 headless runner + CSV polar parser
├── main_flow5.py            # Nelder-Mead optimizer using Flow5 VLM
├── drone_mdo_gui.py         # ★ Full MDO GUI — Flow5 QUADS + range maximisation
│
├── ga4_airfoil_select.py    # GA4 5-candidate NACA airfoil selection pipeline
│
├── analysis_xml/            # Generated XML files (plane, polar, script)
├── foils/                   # Airfoil .dat files (Selig format)
└── output/                  # Flow5 CSV polar output (written at run time)
```

---

## Quick start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9+ |
| Flow5 | Any version with `-s` script mode |
| scipy | 1.9+ |
| numpy | 1.21+ |
| matplotlib | 3.5+ (for `app.py`) |

Flow5 must be installed at `/Applications/flow5.app`. If yours is elsewhere, edit `FLOW5_APP` in `flow5_run.py` and `flow5_xml.py`.

```bash
pip install scipy numpy matplotlib
```

### Run the MDO optimiser (recommended entry point)

```bash
cd /Users/kaanoktem/Flow5
python3 drone_mdo_gui.py
```

### Run the analytical GUI

```bash
python3 app.py
```

### Run the analytical CLI optimizer

```bash
python3 main.py
```

### Run the GA4 airfoil selection pipeline

```bash
python3 ga4_airfoil_select.py
```

---

## Files in detail

### `requirements.py` — Drone mission requirements

Defines the `DroneRequirements` class, the single source of truth for drone design targets.

| Parameter | Value | Notes |
|---|---|---|
| MTOW | 3.0 kg | |
| Payload | 0.5 kg | |
| Cruise speed | 20 m/s | |
| Stall speed | 10 m/s | |
| Wing loading | 120 N/m² | → S = 0.245 m² |
| Airfoil | E387 | Re ≈ 256 k |
| CL_max (2D) | 1.25 | XFoil / literature |
| CM_ac | −0.09 | E387 |
| SHT fraction | 0.22 | H-stab area / wing area |
| SVT fraction | 0.10 | V-fin area / wing area |
| Tail arm | 3.0 × chord | |
| CG position | 40% MAC | battery placement assumption |

Optimization bounds stored as `self.bounds`: AR ∈ [5, 10], taper ∈ [0.3, 0.6], twist ∈ [2°, 5°].

---

### `analysis.py` — Analytical aerodynamic model

Implements Prandtl–Glauert **Lifting-Line Theory** (Glauert Fourier-series method). Used as a fast forward model (no Flow5 required). Functions:

| Function | What it does |
|---|---|
| `lifting_line(AR, taper, twist_deg, ...)` | Returns CL, CDi, span efficiency e, and full lift distribution |
| `find_cruise_alpha(...)` | Bisects to find α that produces the required CL |
| `CL_required(mtow, S, V)` | CL = W / (½ρV²S) |
| `stall_check(CL, CL_max)` | Returns stall margin fraction |
| `static_margin(AR, c_bar, SHT, LHT)` | Neutral-point analysis → SM fraction of MAC |
| `pitch_trim(CM_ac, CL, SM)` | Tail incidence required for trim |
| `control_surfaces(...)` | Aileron and elevator sizing |

---

### `optimizer.py` — Analytical SLSQP optimizer

Maximises L/D using `scipy.optimize.minimize(method='SLSQP')` with the `analysis.py` LLT model as the forward evaluator. Numerical gradients via finite differences.

**Design variables:** AR, taper ratio, twist (washout in degrees)  
**Objective:** maximise L/D = CL / (CDi + CD0)  
**Constraint:** stall margin CL_cruise / CL_max ≤ 0.70

Logs every iteration to `_iter_log` for the GUI convergence plot.

---

### `main.py` — Analytical optimizer CLI

Runs `optimizer.optimize()` and prints a full design report:

```
──────────────── Wing Geometry ────────────────
  AR             :  8.24
  Span           :  1.42 m
  Root chord     :  0.238 m
  Taper          :  0.412
  Washout        :  3.1°
  L/D            :  17.3   ✓
  Static margin  :  12.4%  ✓
  Stall margin   :  28.1%  ✓
```

---

### `app.py` — Analytical optimizer GUI (dark theme)

Full Tkinter GUI with matplotlib plots. Runs the analytical optimizer in a background thread.

**Features:**
- Editable mission parameters (MTOW, cruise/stall speed, payload, etc.)
- Convergence plot (L/D vs iteration)
- Lift distribution plot (actual vs elliptic)
- Wing planform preview
- Full results panel with all constraints flagged pass/fail

---

### `flow5_xml.py` — Flow5 XML generation

Generates the three XML files Flow5 needs to run in headless script mode.

#### `write_plane_xml(geom, path=None)`

Writes an `xflplane` XML from a geometry dictionary. All keys in SI units / degrees:

```python
geom = {
    # Wing
    'b': 1.31,        # span [m]
    'cr': 0.267,      # root chord [m]
    'ct': 0.107,      # tip chord [m]
    'AR': 7.0,        'taper': 0.4,  'twist_deg': 2.0,
    'incidence_deg': 3.0,  'dihedral_deg': 1.0,  'sweep_deg': 0.0,
    # Horizontal tail
    'SHT': 0.054,     # area [m²]
    'LHT': 0.562,     # moment arm from wing LE [m]
    'b_HT': 0.435,    'cr_HT': 0.166,  'ct_HT': 0.083,
    'tail_inc_deg': -1.0,
    # Vertical fin (geometry derived internally from SVT)
    'SVT': 0.0245,    # area [m²]
    # Mass
    'mtow_kg': 3.0,   'cg_frac_mac': 0.40,
}
```

Airfoil names are hardcoded: **E387** on wing sections, **NACA 0009** on tail surfaces. The plane internal name is `drone_opt` (constant `PLANE_NAME`).

#### `write_polar_xml(speed_ms, ref_area, ref_span, ref_chord, path=None)`

Writes a `FIXEDSPEEDPOLAR` polar using **TRIUNIFORM** (VLM, thin surfaces). Default cruise speed is read from `DroneRequirements`.

#### `write_script_xml(aoa_min=-5, aoa_max=14, aoa_step=1, path=None)`

Writes the master `xflscript` XML that ties everything together and drives the headless run. AoA sweep range is configurable.

**Directory constants** (all relative to `flow5_xml.py` location):

| Constant | Path |
|---|---|
| `XML_DIR` | `./analysis_xml/` |
| `OUTPUT_DIR` | `./output/` |
| `FOILS_DIR` | `./foils/` |

---

### `flow5_run.py` — Headless Flow5 runner

Launches Flow5 with `-s <script.xml> -p` and collects the CSV polar output.

#### `run_flow5(script_path, timeout=60) → (success, stdout, stderr)`

Clears any existing CSVs in `output/`, then runs Flow5. Returns the subprocess result. Note: Flow5 sometimes exits with code 1 even when the CSV is written — always check for the CSV rather than relying on the return code alone.

#### `find_polar_csv() → path | None`

Returns the most recently modified CSV in `output/` (recursive search).

#### `parse_polar_csv(csv_path) → dict`

Parses the Flow5 whitespace-delimited polar output. Column mapping:

| Index | Column | Key |
|---|---|---|
| 1 | alpha | `alpha` |
| 4 | CL | `CL` |
| 5 | CD | `CD` |
| 7 | CDi | `CDi` |
| 9 | Cm | `Cm` |
| derived | CL/CD | `LD` |

#### `extract_performance_interp(polar_data, CL_target) → dict | None`

Interpolates the polar at exactly `CL_target` and derives longitudinal data
from the Flow5 Cm(CL) curve: `SM_flow5 = −dCm/dCL`, the residual `Cm` at
cruise, and the trim alpha where Cm crosses zero. Tolerates gaps left by
unconverged viscous points; returns None below 3 converged points and flags
`extrapolated=True` when `CL_target` is outside the converged range.

#### `extract_performance(polar_data, CL_target) → dict | None`

Finds the operating point closest to `CL_target`. Returns alpha, CL, CD, CDi, Cm, LD at that point, plus the full arrays.

#### `run_and_parse(script_path, CL_target, timeout=60) → (perf_dict, error_msg)`

Convenience wrapper: run + find + parse + extract in one call.

---

### `main_flow5.py` — Flow5-backed Nelder-Mead optimizer

Replaces the analytical LLT with a real Flow5 VLM evaluation. Three free variables: AR, taper, twist.

**Objective:** `−L/D + penalties`  
**Penalties:** static margin out of [5%, 30%], stall margin violation, trim AoA excess  
**Solver:** `scipy.optimize.minimize(method='Nelder-Mead')`

Run with:
```bash
python3 main_flow5.py
```

---

### `drone_mdo_gui.py` ★ — Full MDO optimiser GUI

The primary optimisation tool. Maximises **electric range** using Flow5 as the aerodynamic evaluator, with a Tkinter GUI for configuration and live progress monitoring.

#### Objective function

```
Range [km] = (η × E_bat [J] × L/D_real) / (MTOW × g × 1000)

where:
  L/D_real  = CL_cruise / (CD_Flow5 + CD0_fuselage)
  CL_cruise = MTOW·g / (½ρV²S)
  E_bat [J] = battery_Wh × 3600
```

Flow5 runs **viscous** (XFoil on the fly at the wing sections), so `CD_Flow5`
already contains the induced *and* profile drag of the wing and tail at each
section's local Reynolds number. `CD0` covers only the fuselage and
interference (default 0.010). Untick "Viscous (XFoil)" for a fast inviscid
exploration pass (then raise CD0 back to ~0.025).

#### Weight model (Auto MTOW)

With "Auto MTOW" ticked, MTOW is rebuilt every evaluation so a bigger wing
costs mass as well as drag:

```
MTOW = payload + battery_Wh/160 + fixed + 0.70·S·√AR + 1.2·(SHT+SVT)
```

Untick it to use the manual MTOW field (legacy behaviour).

#### Free variables (toggled by checkbox)

| Variable | Default | Default bounds | Physical meaning |
|---|---|---|---|
| AR | 7.0 | [5, 10] | Wing aspect ratio |
| taper | 0.40 | [0.3, 0.6] | Tip/root chord ratio |
| twist | 2.0° | [2°, 5°] | Washout (stall safety) |
| S | 0.245 m² | [0.15, 0.50] | Wing area |
| tail_arm | 3.0 × chord | [2.5, 4.5] | H-tail moment arm |
| SHT_frac | 0.22 | [0.15, 0.35] | H-tail area / wing area |

Fixed variables (unchecked) stay at their defaults throughout the run.

#### Constraints

- **Stall speed:** V_stall ≤ target, with CL_max_3d from the **critical-section
  method** — the LLT lift distribution is compared against the E387's 2D
  Clmax at each station's local Reynolds number (UIUC data), so washout and
  taper genuinely change the stall margin.
- **CL_cruise:** must be < 1.0 (keeps XFoil convergence reliable).
- **Trim:** the tail incidence required to trim at cruise (see below) must be
  within ±5°.

Violations add soft quadratic penalties to the objective (units: km of range).

#### Stability & trim (from the Flow5 Cm curve)

Each evaluation extracts the polar at exactly `CL_cruise` by interpolation
(no more nearest-point stair-stepping) and reads the pitching-moment curve:

- **Neutral point:** SM about the run CG is `−dCm/dCL`; the NP follows.
- **CG placement:** the CG is auto-placed at `NP − 10 % MAC` (battery
  position is the designer's free variable), so every candidate is compared
  at the same static margin instead of penalising SM.
- **Trim:** the residual Cm at cruise gives the tail incidence needed to
  trim; it is written into the saved best geometry and penalised beyond ±5°.

The Flow5 alpha sweep is centred on an LLT estimate of the cruise alpha
(±3–4°, 1° step) instead of a blind −4…14° sweep — ~2.5× fewer points per run.

#### Analysis method

Select **QUADS** (3D quad panel method, thick surfaces) or **TRIUNIFORM**
(VLM thin surfaces) from the dropdown. With viscous ON expect ~15–25 s per
evaluation (XFoil runs at every wing section); inviscid QUADS is ~1 s. For
initial exploration untick viscous; run the final optimisation viscous.

#### GUI panels

```
┌────────────────────────────────────────────────────────────┐
│  Mission Requirements  │  Design Variables                  │
│  Payload [kg]          │  Variable    Active  Lo    Hi      │
│  Battery [Wh]          │  AR          ☑       5.0   10.0    │
│  Fixed mass [kg]       │  Taper       ☑       0.3   0.6     │
│  MTOW [kg] (manual)    │  Washout     ☑       2.0   5.0     │
│  Cruise [m/s]          │  Wing area   ☑       0.15  0.50    │
│  Stall  [m/s]          │  Tail arm    ☑       2.5   4.5     │
│  η prop                │  SHT/S       ☑       0.15  0.35    │
│  CD0 fuselage          │                                    │
│  ☑ Auto MTOW           │                                    │
├────────────────────────────────────────────────────────────┤
│  Method: [QUADS▼] ☑Viscous  Max iters: [100] [Run] [Cancel]│
│                                      Best range: 38.4 km   │
│  ████████████████░░░░░░░░░░░ (indeterminate progress bar)  │
│  Eval  23: AR=8.41  S=0.331m²  taper=0.47  twist=3.2°     │
├────────────────────────────────────────────────────────────┤
│  Output (scrollable log)                                   │
│  #23  range=38.40km  L/D=16.2  CL=0.612  SM=11.3%        │
└────────────────────────────────────────────────────────────┘
```

#### Verify + Trim button

Runs the full trim procedure on the optimizer's best design (or on the
defaults if no optimisation has been run yet), all viscous:

1. sweep with neutral tail → measure the neutral point from Flow5's Cm curve
2. place the CG at NP − 10 % MAC, solve the tail incidence analytically
3. re-run and, if the residual moment warrants it, apply one Newton step on
   the *measured* tail effectiveness (the analytic estimate is ~13 % optimistic)

The verified, balanced, trimmed plane is saved to `drone_mdo_best.xml` and the
log reports NP, CG, SM, trim CL vs cruise CL, and the trimmed L/D (which is
the honest number — it includes trim drag). Three Flow5 runs ≈ 1–2 min.

#### Output

On completion, the best design is printed in full and written to:

```
analysis_xml/drone_mdo_best.xml
```

Open this file in Flow5 (File → Import plane) to inspect and post-process the optimised geometry.

---

### `ga4_airfoil_select.py` — GA4 airfoil selection

Automated 5-candidate NACA airfoil comparison for a general-aviation aircraft (MTOW 1202 kg, S = 14.74 m², V_cruise = 69.45 m/s).

#### Pipeline

1. **Generate .dat files** for 5 candidates: NACA 2412, 2415, 4412, 4415, 23012 (Selig format, cosine-spaced 80 points)
2. **VLM2 comparison** — write one `xflplane` + `xflPlanePolar` (TRIUNIFORM, thin surfaces) per candidate, run Flow5 headlessly, parse L/D, CL_max, CM_ac
3. **Score candidates** — weighted score: 40% max L/D + 35% CL_max + 25% 1/|CM_ac|
4. **QUADS final analysis** — re-run the winner with QUADS 3D panel method (`<Method>QUADS</Method>`, `<Thin_Surfaces>false</Thin_Surfaces>`) for a higher-fidelity polar
5. **Generate `ga4_final_plane.xml`** — complete wing + H-stab + V-fin geometry, importable in Flow5 GUI

#### Winner: NACA 4412

| Metric | Value |
|---|---|
| Score | 0.641 |
| CL (cruise) | 0.2499 |
| CDi (Flow5, induced only) | 0.00232 |
| L/D (induced, no CD0) | 107.7 |
| CL_max | 1.697 at α = 14° |

> **Note:** The fuselage body element was excluded from the XML. Flow5 crashes with `Assertion failed: (m_SignedArea>0.0)` on headless body XML. Add the fuselage manually in Flow5 GUI via Edit → Define Fuselage.

---

## Flow5 XML format notes

### Plane XML (`xflplane`)

- Wing sections need `<y_position>`, `<Chord>`, `<xOffset>` (quarter-chord sweep), `<Dihedral>`, `<Twist>`, panel counts, and `<Left_Side_FoilName>` / `<Right_Side_FoilName>`
- Foil names **must exactly match** the first line of the corresponding `.dat` file
- The last section in each wing acts as a boundary marker (`y_number_of_panels=0`)
- Mass inertia is set via `<Point_Mass>` with CG coordinates from wing LE

### Polar XML (`xflPlanePolar`)

- `<Plane_Name>` must match `<Name>` inside the plane XML exactly
- VLM: `<Method>TRIUNIFORM</Method>` + `<Thin_Surfaces>true</Thin_Surfaces>`
- Panel method: `<Method>QUADS</Method>` + `<Thin_Surfaces>false</Thin_Surfaces>`
- `<Reference_Dimensions>Custom</Reference_Dimensions>` with explicit area/span/chord

### Script XML (`xflscript`)

- `<Process_All_Files>false</Process_All_Files>` + explicit `<Plane_File_Name>` prevents Flow5 from loading every XML in the directory
- `<Polar_text_output_format>CSV</Polar_text_output_format>` enables the CSV output that `flow5_run.py` reads
- AoA sweep: `<T12_Range>aoa_min, aoa_max, aoa_step</T12_Range>`

### Foil `.dat` format (Selig)

```
AirfoilName
x_upper_TE  z_upper_TE
...
x_upper_LE  z_upper_LE
x_lower_LE  z_lower_LE
...
x_lower_TE  z_lower_TE
```

---

## CD0 and L/D correction

Flow5 panel methods return **induced drag only** (CDi). Parasite drag must be added manually:

```
CD_total  = CDi_flow5 + CD0
L/D_real  = CL / CD_total
```

| Component | CD0 contribution |
|---|---|
| Wing (profile drag) | ~0.012 |
| Fuselage + boom | ~0.007 |
| Motor nacelle + prop disk | ~0.006 |
| **Total CD0** | **≈ 0.025** |

This value is configurable in `drone_mdo_gui.py` via the "CD0 (parasite)" field.

---

## Coordinate system

Flow5 uses a standard aircraft coordinate system:
- **x** — aft (positive towards tail)
- **y** — starboard (positive to the right)
- **z** — up

Wing LE root is at the origin. CG x-coordinate is measured from the wing LE root.

---

## Known limitations

| Issue | Status |
|---|---|
| Fuselage body XML crashes Flow5 headless | Workaround: add manually via GUI |
| Viscous analysis not enabled in polars | `<Is_Viscous_Analysis>false</Is_Viscous_Analysis>` — CDi is inviscid; CD0 corrects for profile drag |
| Fixed airfoil | E387 on all drone wing sections; changing requires editing `flow5_xml.py` |
| Single operating point | Optimizer evaluates at cruise CL only; off-design not checked in MDO loop |

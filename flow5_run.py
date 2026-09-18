"""
Launches Flow5 in headless script mode and reads the CSV polar output.
"""
import os
import subprocess
import glob
import numpy as np
import time

FLOW5_APP  = "/Applications/flow5.app/Contents/MacOS/flow5"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def run_flow5(script_path, timeout=60):
    """
    Launch Flow5 with -s <script_path> and wait for it to finish.
    Returns (success, stdout, stderr).
    """
    # Clear previous CSV output so we can detect fresh results
    for f in glob.glob(os.path.join(OUTPUT_DIR, "**", "*.csv"), recursive=True):
        os.remove(f)

    cmd = [FLOW5_APP, "-s", script_path, "-p"]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        success = result.returncode == 0
        return success, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Flow5 timed out after {timeout}s"
    except Exception as e:
        return False, "", str(e)


def find_polar_csv():
    """Return the path of the most recently written CSV polar (searches subdirs too)."""
    csvs = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.csv"), recursive=True)
    if not csvs:
        return None
    return max(csvs, key=os.path.getmtime)


def parse_polar_csv(csv_path):
    """
    Parse a Flow5 polar text file (whitespace-delimited, not CSV despite the name).

    Flow5 output column order (fixed):
      0  Ctrl | 1  alpha | 2  beta | 3  phi |
      4  CL   | 5  CD    | 6  CDv  | 7  CDi |
      8  CY   | 9  Cm    | ...     | 16 CL/CD | ...

    Returns a dict with keys: alpha, CL, CD, CDv, CDi, Cm, LD.
    Rows where XFoil failed to converge are simply absent from the file,
    so the alpha list may have gaps — callers must not assume uniform spacing.
    """
    data = {'alpha': [], 'CL': [], 'CD': [], 'CDv': [], 'CDi': [], 'Cm': [], 'LD': []}
    if not csv_path or not os.path.exists(csv_path):
        return data

    with open(csv_path, encoding='utf-8-sig') as f:
        lines = f.readlines()

    # Find the first line that starts the data block (begins with whitespace + digits or '-')
    data_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped[0] == '-'):
            # Verify it has enough whitespace-split tokens to be a data row
            parts = stripped.split()
            if len(parts) >= 17:
                data_start = i
                break

    if data_start is None:
        return data

    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 17:
            continue
        try:
            data['alpha'].append(float(parts[1]))
            data['CL'].append(float(parts[4]))
            data['CD'].append(float(parts[5]))
            data['CDv'].append(float(parts[6]))
            data['CDi'].append(float(parts[7]))
            data['Cm'].append(float(parts[9]))
            ld = float(parts[4]) / float(parts[5]) if float(parts[5]) > 1e-9 else 0.0
            data['LD'].append(ld)
        except (ValueError, IndexError):
            continue

    return data


def extract_performance(polar_data, CL_target):
    """
    From the parsed polar dict, find the operating point closest to CL_target.
    Returns a performance dict or None if data is empty.
    """
    if not polar_data or not polar_data.get('CL'):
        return None

    CL_arr    = np.array(polar_data['CL'])
    CD_arr    = np.array(polar_data['CD'])
    alpha_arr = np.array(polar_data['alpha'])

    if len(CL_arr) == 0:
        return None

    idx = int(np.argmin(np.abs(CL_arr - CL_target)))

    return {
        'alpha':     alpha_arr[idx],
        'CL':        CL_arr[idx],
        'CD':        CD_arr[idx],
        'LD':        polar_data['LD'][idx],
        'CDi':       polar_data['CDi'][idx],
        'Cm':        polar_data['Cm'][idx],
        'CL_arr':    CL_arr,
        'CD_arr':    CD_arr,
        'alpha_arr': alpha_arr,
    }


def extract_performance_interp(polar_data, CL_target):
    """
    Interpolate the polar at exactly CL_target (CL is monotone in alpha over
    the pre-stall sweep, so CL is a valid interpolation abscissa).

    Also derives longitudinal quantities from the Flow5 Cm(CL) curve
    (moments are computed by Flow5 about the plane's CoG):
      SM          = -dCm/dCL           static margin in MAC fractions
      Cm          = Cm at CL_target    residual moment the tail must trim out
      alpha_trim  = alpha where Cm = 0 (None if Cm doesn't cross zero in range)
      CL_trim     = CL at alpha_trim

    Returns None if fewer than 3 converged points.
    'extrapolated' is True when CL_target lies outside the converged CL range
    (numbers are then clamped to the nearest endpoint — treat with suspicion).
    """
    if not polar_data or len(polar_data.get('CL', [])) < 3:
        return None

    order     = np.argsort(polar_data['alpha'])
    alpha_arr = np.array(polar_data['alpha'])[order]
    CL_arr    = np.array(polar_data['CL'])[order]
    CD_arr    = np.array(polar_data['CD'])[order]
    CDv_arr   = np.array(polar_data['CDv'])[order]
    CDi_arr   = np.array(polar_data['CDi'])[order]
    Cm_arr    = np.array(polar_data['Cm'])[order]

    if not np.all(np.diff(CL_arr) > 0):        # non-monotone → post-stall points present
        keep = np.concatenate(([True], np.diff(CL_arr) > 0))
        alpha_arr, CL_arr = alpha_arr[keep], CL_arr[keep]
        CD_arr, CDv_arr, CDi_arr, Cm_arr = CD_arr[keep], CDv_arr[keep], CDi_arr[keep], Cm_arr[keep]
        if len(CL_arr) < 3:
            return None

    extrapolated = not (CL_arr[0] <= CL_target <= CL_arr[-1])

    alpha = float(np.interp(CL_target, CL_arr, alpha_arr))
    CD    = float(np.interp(CL_target, CL_arr, CD_arr))
    CDv   = float(np.interp(CL_target, CL_arr, CDv_arr))
    CDi   = float(np.interp(CL_target, CL_arr, CDi_arr))
    Cm    = float(np.interp(CL_target, CL_arr, Cm_arr))

    # Static margin from the polar: least-squares slope of Cm vs CL
    A  = np.vstack([CL_arr, np.ones_like(CL_arr)]).T
    slope, _ = np.linalg.lstsq(A, Cm_arr, rcond=None)[0]
    SM = float(-slope)

    # Trim point: Cm crosses zero (Cm decreases with alpha for a stable plane)
    alpha_trim = CL_trim = None
    if (Cm_arr.max() >= 0.0 >= Cm_arr.min()):
        cm_order   = np.argsort(Cm_arr)
        alpha_trim = float(np.interp(0.0, Cm_arr[cm_order], alpha_arr[cm_order]))
        CL_trim    = float(np.interp(alpha_trim, alpha_arr, CL_arr))

    return {
        'alpha': alpha, 'CL': CL_target,
        'CD': CD, 'CDv': CDv, 'CDi': CDi, 'Cm': Cm,
        'LD': CL_target / CD if CD > 1e-9 else 0.0,
        'SM_flow5': SM,
        'alpha_trim': alpha_trim, 'CL_trim': CL_trim,
        'extrapolated': extrapolated,
        'alpha_arr': alpha_arr, 'CL_arr': CL_arr, 'CD_arr': CD_arr, 'Cm_arr': Cm_arr,
    }


def run_and_parse(script_path, CL_target, timeout=60):
    """
    Full pipeline: run Flow5, find CSV, parse, return performance dict.
    Returns (perf_dict, error_message).
    perf_dict is None if analysis failed.
    """
    ok, _, stderr = run_flow5(script_path, timeout=timeout)

    if not ok and not glob.glob(os.path.join(OUTPUT_DIR, "**", "*.csv"), recursive=True):
        return None, f"Flow5 failed: {stderr[:200]}"

    # Give Flow5 a moment to finish writing
    time.sleep(0.5)

    csv_path = find_polar_csv()
    if not csv_path:
        return None, "No CSV output found — check Flow5 script/plane XML"

    polar = parse_polar_csv(csv_path)
    perf  = extract_performance_interp(polar, CL_target)
    if perf is None:                       # too few converged points to interpolate
        perf = extract_performance(polar, CL_target)

    if perf is None:
        return None, f"Could not parse polar CSV: {csv_path}"

    return perf, None

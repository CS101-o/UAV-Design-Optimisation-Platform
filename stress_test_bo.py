"""
Parallel stress test for Flow5 Bayesian Optimisation.

Each test case runs in its own subprocess with its own Flow5 workspace
(xml/ and output/ directories), so parallel workers never clobber each other.

Usage:
    python3 stress_test_bo.py                 # 20 cases, 4 workers, 30 calls each
    python3 stress_test_bo.py --workers 2     # fewer parallel workers
    python3 stress_test_bo.py --calls 15      # faster but less optimal
    python3 stress_test_bo.py --req-only      # skip seed sweep (15 cases only)

Output:
    stress_results_<timestamp>.csv   — one row per case
    stress_work/                     — per-case Flow5 workspaces (safe to delete)

Two test suites:
  1. Requirements sweep (15 cases): varied MTOW, cruise speed, wing loading.
     Checks: optimizer finds a valid design for every combination.
  2. Seed sweep (5 cases): same base requirements, random_state 0–4.
     Checks: BO results cluster tightly (GP surrogate is reliable, not random).
"""

import os
import sys
import csv
import math
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR    = Path(__file__).parent
STRESS_WORK = BASE_DIR / 'stress_work'


# ── Test case definitions ─────────────────────────────────────────────────────

# (label, mtow_kg, cruise_ms, wing_loading_Nm2)
REQ_SWEEP = [
    ('base',        3.0, 20.0, 120.0),
    ('light_slow',  1.5, 15.0,  80.0),
    ('light_fast',  1.5, 25.0, 120.0),
    ('heavy_slow',  5.0, 16.0,  90.0),
    ('heavy_fast',  5.0, 28.0, 150.0),
    ('hi_loading',  3.0, 20.0, 160.0),
    ('lo_loading',  3.0, 20.0,  80.0),
    ('fast',        3.0, 30.0, 130.0),
    ('slow',        3.0, 14.0,  90.0),
    ('med_heavy',   4.0, 22.0, 130.0),
    ('med_light',   2.0, 18.0, 100.0),
    ('sprint',      2.5, 28.0, 140.0),
    ('loiter',      2.5, 14.0,  85.0),
    ('large_slow',  5.0, 14.0,  80.0),
    ('small_fast',  1.5, 28.0, 140.0),
]

# (label, mtow_kg, cruise_ms, wing_loading_Nm2, random_state)
SEED_SWEEP = [(f'seed_{i}', 3.0, 20.0, 120.0, i) for i in range(5)]


def _build_cases(n_calls, req_only=False):
    cases = []
    for i, (label, mtow, speed, loading) in enumerate(REQ_SWEEP):
        d = STRESS_WORK / f'{i:02d}_{label}'
        cases.append((label, mtow, speed, loading, 42,
                      str(d / 'xml'), str(d / 'output'), n_calls))
    if not req_only:
        for i, (label, mtow, speed, loading, seed) in enumerate(SEED_SWEEP):
            d = STRESS_WORK / f'{len(REQ_SWEEP) + i:02d}_{label}'
            cases.append((label, mtow, speed, loading, seed,
                          str(d / 'xml'), str(d / 'output'), n_calls))
    return cases


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def run_one_case(packed):
    """
    Runs a single BO case in isolation.

    Module-level globals in flow5_xml / flow5_run / flow5_bo_optimizer are
    patched to point at this case's private workspace BEFORE any call that
    uses them.  With multiprocessing (spawn), each subprocess has its own
    copy of module state so patches don't leak between workers.
    """
    label, mtow, speed, loading, rstate, xml_dir, output_dir, n_calls = packed

    os.makedirs(xml_dir,     exist_ok=True)
    os.makedirs(output_dir,  exist_ok=True)

    # Patch module globals before importing anything that reads them at call time
    import flow5_xml, flow5_run
    flow5_xml.XML_DIR    = xml_dir
    flow5_xml.OUTPUT_DIR = output_dir
    flow5_run.OUTPUT_DIR = output_dir

    # _BO_SCRIPT is computed at import time from XML_DIR, so update it directly
    import flow5_bo_optimizer
    flow5_bo_optimizer._BO_SCRIPT = os.path.join(xml_dir, 'drone_script.xml')

    from requirements import DroneRequirements
    from optimizer import WING_VAR_SPECS, TAIL_VAR_SPECS, FUSE_VAR_SPECS

    t0     = time.time()
    result = None
    error  = ''
    req    = None

    try:
        req = DroneRequirements()
        req.mtow_kg          = mtow
        req.cruise_speed_ms  = speed
        req.wing_loading_Nm2 = loading
        req.stall_speed_ms   = max(8.0, speed * 0.45)
        req.wing_area        = (req.mtow_kg * 9.81) / req.wing_loading_Nm2

        var_specs = [s for s in WING_VAR_SPECS + TAIL_VAR_SPECS + FUSE_VAR_SPECS
                     if s['key'] != 'HT_sweep']

        result = flow5_bo_optimizer.run_flow5_bo(
            req, var_specs,
            llt_seed_result=None,       # no LLT warm-start → fully independent
            n_calls=n_calls,
            n_initial=max(6, n_calls // 4),
            random_state=rstate,
        )

    except Exception:
        error = traceback.format_exc()[-500:]

    elapsed = round(time.time() - t0, 1)

    if result and not result.get('cancelled') and not error:
        xd    = result['x_dict']
        c_bar = math.sqrt(xd['AR'] * req.wing_area) / xd['AR']
        arm_ok = (xd['tail_arm'] * c_bar) >= (xd['fuse_len'] - 1e-3)
        ld_ok  = result['LD'] > 5.0
        return {
            'label':     label,
            'status':    'PASS' if (arm_ok and ld_ok) else 'FAIL',
            'LD':        round(result['LD'], 3),
            'n_calls':   result['n_calls_used'],
            'arm_ok':    arm_ok,
            'ld_ok':     ld_ok,
            'elapsed_s': elapsed,
            'AR':        round(xd.get('AR', 0), 2),
            'taper':     round(xd.get('taper', 0), 3),
            'tail_arm':  round(xd.get('tail_arm', 0), 2),
            'fuse_len':  round(xd.get('fuse_len', 0), 3),
            'SHT_frac':  round(xd.get('SHT_frac', 0), 3),
            'mtow':      mtow,
            'speed':     speed,
            'loading':   loading,
            'rstate':    rstate,
            'error':     '',
        }
    else:
        return {
            'label': label, 'status': 'ERROR', 'LD': float('nan'),
            'n_calls': 0, 'arm_ok': False, 'ld_ok': False,
            'elapsed_s': elapsed, 'AR': 0, 'taper': 0,
            'tail_arm': 0, 'fuse_len': 0, 'SHT_frac': 0,
            'mtow': mtow, 'speed': speed, 'loading': loading,
            'rstate': rstate, 'error': error,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Parallel Flow5 BO stress test')
    ap.add_argument('--workers',  type=int, default=4,
                    help='Parallel worker processes (default 4)')
    ap.add_argument('--calls',    type=int, default=30,
                    help='BO evaluations per case (default 30, ~2 min/case)')
    ap.add_argument('--req-only', action='store_true',
                    help='Run only the 15 requirement-sweep cases')
    args = ap.parse_args()

    STRESS_WORK.mkdir(exist_ok=True)
    cases = _build_cases(args.calls, req_only=args.req_only)
    n = len(cases)

    # Estimate: each call ≈ 4 Flow5 Newton iters × ~1 s + overhead
    est_min = max(1, round(n * args.calls * 5 / args.workers / 60))
    ts      = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_csv = BASE_DIR / f'stress_results_{ts}.csv'

    print(f'\n{"="*62}')
    print(f'  Flow5 BO Stress Test')
    print(f'  Cases: {n}  |  Workers: {args.workers}  |  Calls/case: {args.calls}')
    print(f'  Estimated time: ~{est_min} min')
    print(f'  Results → {out_csv.name}')
    print(f'{"="*62}\n')

    fields = ['label', 'status', 'LD', 'n_calls', 'arm_ok', 'ld_ok',
              'elapsed_s', 'AR', 'taper', 'tail_arm', 'fuse_len', 'SHT_frac',
              'mtow', 'speed', 'loading', 'rstate', 'error']

    passed = failed = errors = 0
    done   = 0
    all_results = []

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one_case, c): c[0] for c in cases}

            for fut in as_completed(futs):
                r = fut.result()
                all_results.append(r)
                writer.writerow({k: r.get(k, '') for k in fields})
                f.flush()
                done += 1

                sym = {'PASS': '✓', 'FAIL': '✗', 'ERROR': '!'}[r['status']]
                ld  = f"{r['LD']:.3f}" if r['LD'] == r['LD'] else '  nan'
                checks = []
                if not r['arm_ok']:  checks.append('ARM_FAIL')
                if not r['ld_ok']:   checks.append('LD_FAIL')
                flag = '  ' + ' '.join(checks) if checks else ''

                print(f"  [{done:2d}/{n}] {sym}  {r['label']:<14}  "
                      f"L/D={ld:<7}  {r['elapsed_s']:.0f}s{flag}")
                if r['error']:
                    print(f"         ! {r['error'][:120].strip()}")

                if r['status'] == 'PASS':   passed += 1
                elif r['status'] == 'FAIL': failed += 1
                else:                       errors += 1

    # Seed-sweep summary: check LD variance across seeds
    if not args.req_only:
        seed_lds = [r['LD'] for r in all_results
                    if r['label'].startswith('seed_') and r['LD'] == r['LD']]
        if len(seed_lds) > 1:
            spread = max(seed_lds) - min(seed_lds)
            print(f'\n  Seed sweep L/D spread: {spread:.3f}  '
                  f'({"✓ consistent" if spread < 1.0 else "⚠ high variance — increase n_calls"})')

    print(f'\n{"─"*40}')
    print(f'  PASS:  {passed}/{n}')
    print(f'  FAIL:  {failed}/{n}')
    print(f'  ERROR: {errors}/{n}')
    print(f'  Full results: {out_csv}\n')


if __name__ == '__main__':
    main()

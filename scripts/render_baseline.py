#!/usr/bin/env python3
"""Render-baseline comparison: catch small brightness/colour regressions that the test suite's tolerances let through.

The test suite catches a rendering error of about 3-4% but not 1% (measured by injecting bugs). That is fine for ordinary work and not for a big
refactor of shared sampling, BSDF or light code. This tool closes the gap by comparing a panel of scenes against renders of the SAME scenes made
BEFORE the change, with the noise measured instead of guessed:

    python scripts/render_baseline.py capture before-refactor      # on the known-good code (about 4-5 minutes)
    ... make the change, rebuild ...
    python scripts/render_baseline.py compare  before-refactor      # exit code 1 if anything moved

What it does
  - Renders each panel scene on the CPU, the GPU recursive backend and the GPU wavefront backend (ray_tracer.exe, linear EXR output) with a fixed
    --seed, three seeds per scene, at 64x64 and a high sample count. A baseline (every pixel, plus a summary) lives under baselines/<name>/ and is
    git-ignored: it is only meaningful for the code it was captured from, so capture a new one rather than committing it.
  - compare renders the same panel with the same seeds. If every pixel matches, nothing changed (bit-identical). Otherwise it measures the change
    pixel by pixel: the same seed gives the same random numbers, so as long as the change does not alter how many random numbers a path uses, old
    and new pixels are strongly correlated and their DIFFERENCE is far quieter than either image. The standard error of the average difference is
    computed from the pixel differences themselves, so it is honest in both cases (when the random streams still line up, and when a change
    reshuffles them and the comparison falls back to ordinary render noise). A shift counts only when it exceeds --tolerance (default 0.3% of the
    channel mean) AND 4 standard errors. A pure refactor can add --expect-identical, which also fails any pair whose pixels differ by more than
    0.01% RMS, because a change that moves no mean (a 2% roughness change, say) still moves pixels. The 4x4 grid of block means is judged the same way (--block-tolerance, default 2% of the image mean) to
    notice a local change that averages out. New non-finite (NaN/Inf) pixels always fail.
  - A shift too small to show in any one scene can still be unmistakable across the panel, so the scenes that differ at all are also pooled per
    backend (--panel-tolerance, default 0.15% of luminance, and at least 4 standard errors, in one direction in 3 of every 4 scenes).

Each backend is compared with its own baseline: this tool says "did my change move the image", not "is the image right" (that is the test suite's
and the pbrt reference scripts' job). A change that is SUPPOSED to move the image (a bug fix) will show up here as moved, which is the point:
read the table and confirm it moved where you meant.

Needs a built x64\\Release\\ray_tracer.exe (use --exe for another path) and a CUDA/OptiX GPU for the rec/wf backends (--backends cpu to skip them).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exr_io  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
BASELINE_DIR = os.path.join(REPO, 'baselines')

# Scene ids (src/TheRestOfYourLife/scene_registry_data.h), chosen to cover the paths most refactors touch.
PANEL = [
    ('A1', 'Cornell box, diffuse + area light'),
    ('A2', 'many spheres, mixed materials'),
    ('A3', 'checker texture'),
    ('B2', 'rough metal'),
    ('B3', 'rough glass'),
    ('B4', 'conductor'),
    ('B5', 'coated diffuse (layered BxDF)'),
    ('B6', 'thin glass'),
    ('B8', 'subsurface wax'),
    ('B15', 'mix material'),
    ('A8', 'cornell smoke (homogeneous medium)'),
    ('E2', 'cloud medium (heterogeneous grid)'),
    ('C2', 'spot light'),
    ('C3', 'distant light'),
    ('C4', 'point light'),
]
BACKENDS = ('cpu', 'rec', 'wf')
BACKEND_NAMES = {'cpu': 'CPU', 'rec': 'GPU recursive', 'wf': 'GPU wavefront'}
GRID = 4
LUM = np.array([0.2126, 0.7152, 0.0722])


def settings_from(args):
    return {'width': args.width, 'cpu_spp': args.cpu_spp, 'gpu_spp': args.gpu_spp, 'depth': args.depth}


def git_info():
    def run(*a):
        try:
            return subprocess.run(['git', *a], cwd=REPO, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ''
    return {'commit': run('rev-parse', '--short', 'HEAD'), 'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
            'dirty': bool(run('status', '--porcelain', '--untracked-files=no'))}


def render(exe, backend, scene, seed, settings, tmpdir):
    """One render -> {'pixels': float32 [h, w, 3] (non-finite already zeroed), 'nonfinite': int, 'hash': str}."""
    out = os.path.join(tmpdir, f'{scene}_{backend}_{seed}.exr')
    if os.path.exists(out):
        os.remove(out)
    env = dict(os.environ)
    args = [exe]
    if backend == 'cpu':
        args.append('--cpu')
        spp = settings['cpu_spp']
    else:
        args.append('--gpu')
        env['RAY_TRACER_WAVEFRONT'] = '1' if backend == 'wf' else '0'
        if backend == 'wf':
            args.append('--wavefront')
        spp = settings['gpu_spp']
    args += ['--seed', str(seed), '--output', out, str(settings['width']), str(spp), str(settings['depth']), scene]
    proc = subprocess.run(args, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(out):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        raise RuntimeError(f'ray_tracer exited {proc.returncode}: ' + ' | '.join(tail))
    _, _, ch = exr_io.read_exr(out)
    rgb = np.stack([ch['R'], ch['G'], ch['B']], axis=-1).astype(np.float32)
    os.remove(out)
    finite = np.isfinite(rgb)
    return {'pixels': np.where(finite, rgb, 0.0).astype(np.float32), 'nonfinite': int((~finite).sum()),
            'hash': hashlib.sha1(rgb.tobytes()).hexdigest()}


def run_panel(args, seeds, label):
    """-> (settings, {scene: {backend: [run, ...]}}, failures); each run is render()'s dict plus 'seed'."""
    settings = settings_from(args)
    scenes = [s for s, _ in PANEL if not args.scenes or s in args.scenes]
    backends = [b for b in BACKENDS if b in args.backends]
    results = {}
    failures = []
    total = len(scenes) * len(backends) * len(seeds)
    done = 0
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix='render_baseline_') as tmp:
        for scene in scenes:
            for backend in backends:
                runs = []
                for seed in seeds:
                    done += 1
                    try:
                        runs.append({'seed': seed, **render(args.exe, backend, scene, seed, settings, tmp)})
                    except Exception as e:  # keep going: report every failure at the end
                        failures.append((scene, backend, seed, str(e)))
                        print(f'  [{label} {done}/{total}] {scene} {BACKEND_NAMES[backend]} seed {seed}: FAILED ({e})', flush=True)
                results.setdefault(scene, {})[backend] = runs
            print(f'  [{label} {done}/{total}] {scene} done ({int(time.time() - t0)} s)', flush=True)
    return settings, results, failures


def cmd_capture(args):
    seeds = list(range(1, args.seeds + 1))
    print(f'Capturing baseline "{args.name}": {len(PANEL)} scenes x {len(args.backends)} backends x {len(seeds)} seeds')
    settings, results, failures = run_panel(args, seeds, 'capture')
    if failures:
        print(f'\n{len(failures)} render(s) failed; the baseline was NOT written. First: {failures[0]}')
        return 2
    path = os.path.join(BASELINE_DIR, args.name)
    os.makedirs(path, exist_ok=True)
    pixels = {}
    summary = {}
    for scene, per in results.items():
        for backend, runs in per.items():
            for r in runs:
                pixels[f'{scene}|{backend}|{r["seed"]}'] = r['pixels']
            summary.setdefault(scene, {})[backend] = [{'seed': r['seed'], 'hash': r['hash'], 'nonfinite': r['nonfinite'],
                                                       'mean': r['pixels'].reshape(-1, 3).mean(axis=0).tolist()} for r in runs]
    np.savez_compressed(os.path.join(path, 'pixels.npz'), **pixels)
    info = git_info()
    with open(os.path.join(path, 'baseline.json'), 'w') as f:
        json.dump({'created': datetime.datetime.now().isoformat(timespec='seconds'), 'git': info, 'settings': settings,
                   'seeds': seeds, 'results': summary}, f, indent=1)
    print(f'\nBaseline written to {os.path.relpath(path, REPO)} (commit {info["commit"]}{", with uncommitted changes" if info["dirty"] else ""}).')
    return 0


def paired_shift(new_stack, old_stack):
    """Average of (new - old) over every pixel and seed, as a fraction of old's mean, with its standard error.

    new_stack/old_stack: float arrays [..., 3] of the same shape. Returns (shift, standard error), both relative, per channel (shape [3]).
    """
    diff = (new_stack.astype(np.float64) - old_stack.astype(np.float64)).reshape(-1, 3)
    mean_old = old_stack.astype(np.float64).reshape(-1, 3).mean(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        shift = diff.mean(axis=0) / mean_old
        se = diff.std(axis=0, ddof=1) / math.sqrt(len(diff)) / mean_old
    return shift, se


def cmd_compare(args):
    path = os.path.join(BASELINE_DIR, args.name)
    if not os.path.exists(os.path.join(path, 'baseline.json')):
        print(f'No baseline "{args.name}" (looked in {path}). Capture one first.')
        return 2
    with open(os.path.join(path, 'baseline.json')) as f:
        base = json.load(f)
    old_pixels = np.load(os.path.join(path, 'pixels.npz'))
    # Same settings and seeds as the baseline, so unchanged code reproduces it bit for bit.
    for key in ('width', 'cpu_spp', 'gpu_spp', 'depth'):
        setattr(args, key, base['settings'][key])
    seeds = base['seeds'][:args.seeds] if args.seeds else base['seeds']
    old_seed_for = {s + args.seed_offset: s for s in seeds}   # --seed-offset: other seeds, paired by position (a check on false alarms)
    seeds = [s + args.seed_offset for s in seeds]
    now = git_info()
    print(f'Comparing against "{args.name}" (captured {base["created"]} at {base["git"]["commit"]}'
          f'{" + uncommitted changes" if base["git"]["dirty"] else ""}); now at {now["commit"]}{" + uncommitted changes" if now["dirty"] else ""}')
    _, results, failures = run_panel(args, seeds, 'compare')

    rows = []   # (severity, text)
    drift = {}  # backend -> [relative luminance shift of each scene that differs at all]
    identical = checked = bad = 0
    for scene, per_backend in results.items():
        for backend, runs in per_backend.items():
            if not runs:
                continue
            tag = f'{scene:4s} {BACKEND_NAMES[backend]:14s}'
            keys = [f'{scene}|{backend}|{old_seed_for[r["seed"]]}' for r in runs]
            if any(k not in old_pixels.files for k in keys):
                rows.append((9, f'{tag} not in baseline'))
                continue
            checked += 1
            old_stack = np.stack([old_pixels[k] for k in keys])
            new_stack = np.stack([r['pixels'] for r in runs])
            if np.array_equal(old_stack, new_stack):
                identical += 1
                continue
            problems = []
            nf_new = sum(r['nonfinite'] for r in runs)
            used = set(old_seed_for.values())
            nf_old = sum(r['nonfinite'] for r in base['results'][scene][backend] if r['seed'] in used)
            if nf_new > nf_old:
                problems.append(f'non-finite pixels {nf_old} -> {nf_new}')
            shift, se = paired_shift(new_stack, old_stack)
            worst = float(np.nanmax(np.abs(shift)))
            # How different are the pixels, whatever the means do? (A change in highlight shape can leave every mean alone.) The wavefront backend
            # adds floats in a nondeterministic order, so two runs of unchanged code can differ in the last bits: ~1e-7 relative, far under the limit.
            rel_rms = float(np.sqrt(np.mean((new_stack.astype(np.float64) - old_stack) ** 2) / max(np.mean(old_stack.astype(np.float64) ** 2), 1e-30)))
            if args.expect_identical and rel_rms > args.identical_rms:
                problems.append(f'pixels changed: RMS difference {rel_rms * 100:.3f}% of the image RMS (this run expects no change)')
            for c, cname in enumerate('RGB'):
                if abs(shift[c]) > args.tolerance / 100 and abs(shift[c]) > 4 * se[c]:
                    problems.append(f'{cname} mean {shift[c] * 100:+.2f}% (noise +-{se[c] * 100:.2f}%)')
            lum_new, lum_old = new_stack @ LUM, old_stack @ LUM
            drift.setdefault(backend, []).append(float((lum_new - lum_old).mean() / lum_old.mean()))
            # local changes: each of the 4x4 blocks, judged against the image's own mean so dark blocks are not over-weighted
            h, w = new_stack.shape[1], new_stack.shape[2]
            img_mean = old_stack.reshape(-1, 3).mean(axis=0)
            local = []
            for by in range(GRID):
                for bx in range(GRID):
                    sl = (slice(None), slice(by * h // GRID, (by + 1) * h // GRID), slice(bx * w // GRID, (bx + 1) * w // GRID))
                    diff = (new_stack[sl].astype(np.float64) - old_stack[sl].astype(np.float64)).reshape(-1, 3)
                    d = diff.mean(axis=0) / img_mean
                    s = diff.std(axis=0, ddof=1) / math.sqrt(len(diff)) / img_mean
                    for c, cname in enumerate('RGB'):
                        if abs(d[c]) > args.block_tolerance / 100 and abs(d[c]) > 4 * s[c]:
                            local.append(f'block ({by},{bx}) {cname} {d[c] * 100:+.1f}% of image mean')
                            break
            problems += local[:2]
            if problems:
                bad += 1
                rows.append((0 if nf_new > nf_old else 1, f'{tag} MOVED  ' + '; '.join(problems[:3]) + (f' (+{len(problems) - 3} more)' if len(problems) > 3 else '')))
            else:
                rows.append((2, f'{tag} ok     pixels differ by {rel_rms * 100:.4f}% RMS, mean shift at most {worst * 100:.3f}%, within noise'))

    # A shift too small to show in any one scene can still be unmistakable across the panel (a 1% loss in light sampling is about -0.5% in a box
    # lit by an area light and -1% under a point light): pool the scenes. Unweighted: scenes are affected by different amounts, so the spread of
    # the per-scene shifts, not the render noise, is the uncertainty of their average.
    panel_bad = 0
    for backend, items in drift.items():
        if len(items) < 3:
            continue
        d = np.array(items)
        D = float(d.mean())
        SE = float(d.std(ddof=1) / math.sqrt(len(d)))
        same_sign = int((np.sign(d) == np.sign(D)).sum())
        print(f'  pooled {BACKEND_NAMES[backend]}: luminance {D * 100:+.3f}% +-{SE * 100:.3f}% over {len(items)} changed scenes, {same_sign} in the same direction')
        if abs(D) > args.panel_tolerance / 100 and abs(D) > 4 * SE and same_sign >= 0.75 * len(items):
            panel_bad += 1
            rows.append((1, f'ALL  {BACKEND_NAMES[backend]:14s} DRIFT  luminance {D * 100:+.2f}% on average over {len(items)} scenes that changed '
                            f'(+-{SE * 100:.2f}%), {same_sign} of them in the same direction'))

    print()
    if identical:
        print(f'{identical}/{checked} scene/backend pairs are bit-identical to the baseline.')
    for _, text in sorted(rows):
        print('  ' + text)
    for scene, backend, seed, err in failures:
        print(f'  {scene:4s} {BACKEND_NAMES[backend]:14s} RENDER FAILED (seed {seed}): {err}')
    if failures:
        return 2
    if bad or panel_bad:
        parts = []
        if bad:
            parts.append(f'{bad} scene/backend pair(s) moved by more than the tolerance')
        if panel_bad:
            parts.append(f'{panel_bad} backend(s) drifted across the panel')
        print('\n' + ' and '.join(parts) + '. If this change is meant to alter the image, check that these are the scenes you expect.')
        return 1
    print('\nNo scene moved by more than the tolerance.')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('capture', 'compare', 'list'):
        p = sub.add_parser(name)
        if name != 'list':
            p.add_argument('name', help='baseline name, e.g. before-refactor')
            p.add_argument('--exe', default=os.path.join(REPO, 'x64', 'Release', 'ray_tracer.exe'))
            p.add_argument('--scenes', type=lambda s: s.split(','), default=None, help='comma-separated scene ids (default: the whole panel)')
            p.add_argument('--backends', type=lambda s: s.split(','), default=list(BACKENDS), help='cpu,rec,wf (default: all three)')
            p.add_argument('--seeds', type=int, default=3 if name == 'capture' else 0,
                           help='seeds per scene (capture default 3; compare defaults to the baseline\'s own)')
        if name == 'capture':
            p.add_argument('--width', type=int, default=64)
            p.add_argument('--cpu-spp', type=int, default=512)
            p.add_argument('--gpu-spp', type=int, default=1024)
            p.add_argument('--depth', type=int, default=8)
        if name == 'compare':
            p.add_argument('--tolerance', type=float, default=0.3, help='per-channel mean, percent (default 0.3)')
            p.add_argument('--panel-tolerance', type=float, default=0.15,
                           help='pooled luminance shift over all changed scenes of one backend, percent (default 0.15)')
            p.add_argument('--block-tolerance', type=float, default=2.0, help='4x4 block means, percent of the image mean (default 2)')
            p.add_argument('--expect-identical', action='store_true',
                           help='for a change that must not alter any image (a pure refactor): also fail when pixels differ by more than --identical-rms')
            p.add_argument('--identical-rms', type=float, default=1e-4,
                           help='with --expect-identical, the largest relative RMS pixel difference still counted as unchanged (default 1e-4, i.e. 0.01%%)')
            p.add_argument('--seed-offset', type=int, default=0,
                           help='render with seeds shifted by this much: a check that unchanged code reports no move even when no pixel matches')
    args = ap.parse_args()
    if args.cmd == 'list':
        if os.path.isdir(BASELINE_DIR):
            for n in sorted(os.listdir(BASELINE_DIR)):
                p = os.path.join(BASELINE_DIR, n, 'baseline.json')
                if os.path.exists(p):
                    with open(p) as f:
                        b = json.load(f)
                    print(f'{n}: captured {b["created"]} at {b["git"]["commit"]} ({b["git"]["branch"]})')
        return 0
    if not os.path.exists(args.exe):
        print(f'ray_tracer.exe not found at {args.exe} - build the solution first (or pass --exe).')
        return 2
    return cmd_capture(args) if args.cmd == 'capture' else cmd_compare(args)


if __name__ == '__main__':
    sys.exit(main())

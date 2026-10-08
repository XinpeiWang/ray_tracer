#!/usr/bin/env python3
"""CPU-versus-GPU parity sweep for the OptiX renderers - the same method the Mac's Metal sweep uses (gpu/metal/metal_cpu_gpu_parity_check.cpp,
docs/METAL_PARITY_STATUS.md), so every GPU backend is judged the same way.

Every scene is rendered at a small size on the CPU and on the GPU backend with the same --seed and compared in linear HDR (.exr):

  * whole-image brightness and the per-channel means within 30%;
  * a 6x6 grid of blocks: the worst block within 50% (a miss up to 1.5x that is only "marginal"); blocks that are almost black on both sides
    (mean below 0.01) are ignored;
  * no NaN/Inf pixels in either image.

A scene that misses and is listed in scripts/backend_parity_known_gaps.txt (one `scene backend reason` per line) is reported as a known gap and
does not fail the run; a scene that FAILS and is not listed does (exit code 1), and a listed scene that now passes is reported so the entry can go.
A scene the GPU refuses to render (a CPU-only feature) is skipped, not failed.

    python scripts/backend_parity.py                       # every scene, GPU recursive then wavefront (several minutes)
    python scripts/backend_parity.py --backend wf          # one backend
    python scripts/backend_parity.py --scenes A1,B3,E1     # some scenes
    python scripts/backend_parity.py --list                # the scenes it would run

Needs a GPU build of ray_tracer.exe (RayTracer_Package next to this repo, or --exe) and an NVIDIA GPU. CI cannot run it (a hosted Windows runner has
no GPU), so it protects a developer machine, like the Metal sweep does on a Mac.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exr_io  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWN_GAPS = os.path.join(REPO, 'scripts', 'backend_parity_known_gaps.txt')
GRID = 6
MEAN_TOL = 0.30
BLOCK_TOL = 0.50
MARGINAL = 1.5
BLOCK_FLOOR = 0.01


def default_exe():
    for p in (os.path.join(REPO, 'RayTracer_Package', 'ray_tracer.exe'), os.path.join(REPO, 'RayTracer_Package', 'ray_tracer'),
              os.path.join(REPO, 'x64', 'Release', 'ray_tracer.exe')):
        if os.path.exists(p):
            return p
    return 'ray_tracer'


def scene_ids():
    """The built-in scenes that need no external files, in registry order, read from the registry source."""
    text = open(os.path.join(REPO, 'src', 'TheRestOfYourLife', 'scene_registry_data.h'), encoding='utf-8').read()
    ids = []
    for m in re.finditer(r'build_curated_pbrt_scene_descriptor\(\s*"([A-Z]\d+)"(.*?)\),\s*\n', text, re.S):
        if 'requires_files=*/true' in m.group(2).replace(' ', ''):
            continue
        ids.append(m.group(1))
    return ids


def known_gaps():
    gaps = {}
    if os.path.exists(KNOWN_GAPS):
        for line in open(KNOWN_GAPS, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 2)
            if len(parts) >= 2:
                gaps[(parts[0], parts[1])] = parts[2] if len(parts) > 2 else ''
    return gaps


def render(exe, backend, scene, args, tmp):
    out = os.path.join(tmp, f'{scene}_{backend}.exr')
    if os.path.exists(out):
        os.remove(out)
    cmd = [exe]
    if backend == 'cpu':
        cmd += ['--cpu']
    else:
        cmd += ['--gpu'] + (['--wavefront'] if backend == 'wf' else [])
    cmd += ['--seed', str(args.seed), '--output', out, str(args.width), str(args.spp), str(args.depth), scene]
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(out):
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-2:]
        raise RuntimeError(' | '.join(tail))
    _, _, ch = exr_io.read_exr(out)
    os.remove(out)
    return np.stack([ch['R'], ch['G'], ch['B']], axis=-1).astype(np.float64)


def ratio(a, b):
    """How far apart two means are, as the larger over the smaller (1.0 = equal); both near zero count as equal."""
    if a < 1e-6 and b < 1e-6:
        return 1.0
    lo, hi = min(a, b), max(a, b)
    return hi / max(lo, 1e-9)


def compare(cpu, gpu):
    """-> (verdict 'pass'|'marginal'|'fail', details string)."""
    bad = int((~np.isfinite(cpu)).sum() + (~np.isfinite(gpu)).sum())
    if bad:
        return 'fail', f'{bad} non-finite pixels'
    lum = np.array([0.2126, 0.7152, 0.0722])
    cl, gl = (cpu @ lum), (gpu @ lum)
    limit = 1.0 + MEAN_TOL
    whole = ratio(cl.mean(), gl.mean())
    channels = [ratio(cpu[..., c].mean(), gpu[..., c].mean()) for c in range(3)]
    h, w = cl.shape
    worst, worst_at = 1.0, (0, 0)
    for by in range(GRID):
        for bx in range(GRID):
            ys, ye, xs, xe = by * h // GRID, (by + 1) * h // GRID, bx * w // GRID, (bx + 1) * w // GRID
            a, b = cl[ys:ye, xs:xe].mean(), gl[ys:ye, xs:xe].mean()
            if a < BLOCK_FLOOR and b < BLOCK_FLOOR:
                continue
            r = ratio(a, b)
            if r > worst:
                worst, worst_at = r, (bx, by)
    detail = f'cpu {cl.mean():.4f} gpu {gl.mean():.4f} (x{whole:.2f}), channels x{max(channels):.2f}, worst block x{worst:.2f} at {worst_at}'
    if whole > limit or max(channels) > limit:
        return 'fail', detail
    block_limit = 1.0 + BLOCK_TOL
    if worst > block_limit * MARGINAL:
        return 'fail', detail
    if worst > block_limit:
        return 'marginal', detail
    return 'pass', detail


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--exe', default=default_exe())
    ap.add_argument('--backend', choices=['rec', 'wf', 'both'], default='both')
    ap.add_argument('--scenes', default='', help='comma-separated scene ids (default: all)')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--spp', type=int, default=64)
    ap.add_argument('--depth', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()

    scenes = [s for s in args.scenes.split(',') if s] or scene_ids()
    if args.list:
        print('\n'.join(scenes))
        return 0
    backends = ['rec', 'wf'] if args.backend == 'both' else [args.backend]
    names = {'rec': 'GPU recursive', 'wf': 'GPU wavefront'}
    gaps = known_gaps()
    counts = {'pass': 0, 'marginal': 0, 'known': 0, 'skipped': 0, 'fail': 0}
    failures, fixed = [], []
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix='backend_parity_') as tmp:
        for i, scene in enumerate(scenes, 1):
            try:
                cpu = render(args.exe, 'cpu', scene, args, tmp)
            except Exception as e:
                print(f'[{i}/{len(scenes)}] {scene}: CPU render failed ({e}) - skipped', flush=True)
                counts['skipped'] += len(backends)
                continue
            for b in backends:
                try:
                    gpu = render(args.exe, b, scene, args, tmp)
                except Exception as e:
                    print(f'[{i}/{len(scenes)}] {scene} {names[b]}: skipped ({e})', flush=True)
                    counts['skipped'] += 1
                    continue
                verdict, detail = compare(cpu, gpu)
                gap = gaps.get((scene, b))
                if verdict == 'fail' and gap is not None:
                    counts['known'] += 1
                    print(f'[{i}/{len(scenes)}] {scene} {names[b]}: KNOWN GAP ({gap}) - {detail}', flush=True)
                elif verdict == 'fail':
                    counts['fail'] += 1
                    failures.append((scene, b, detail))
                    print(f'[{i}/{len(scenes)}] {scene} {names[b]}: FAIL - {detail}', flush=True)
                else:
                    counts[verdict] += 1
                    if gap is not None:
                        fixed.append((scene, b))
                    print(f'[{i}/{len(scenes)}] {scene} {names[b]}: {verdict} - {detail}', flush=True)
    print(f'\n{counts["pass"]} pass, {counts["marginal"]} marginal, {counts["known"]} known gaps, {counts["skipped"]} skipped, {counts["fail"]} FAIL '
          f'({time.time() - t0:.0f} s)')
    for scene, b in fixed:
        print(f'  {scene} {names[b]} is listed as a known gap but passes now - remove it from scripts/backend_parity_known_gaps.txt')
    for scene, b, detail in failures:
        print(f'  FAIL {scene} {names[b]}: {detail}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""Write each bundled scene's size into its own header as "# @rt-size <number>".

The size is the largest side of the scene's bounding box in the scene's own units (src/shared/scene_size.h). The GUI's Live Preview scales its keyboard step by
it, so one key press moves a sensible distance in a 6-unit room and in the 555-unit Cornell box alike. The number lives in the scene file, like the other
"# @rt-" tags (category, description, performance), so it travels with the file.

    python scripts/stamp_scene_sizes.py                  # add the line to every scene that lacks one
    python scripts/stamp_scene_sizes.py --check          # list scenes whose line is missing or no longer matches the geometry (exit 1 if any)
    python scripts/stamp_scene_sizes.py --force          # re-measure and rewrite every line
    python scripts/stamp_scene_sizes.py pbrt_scenes/a.pbrt other/b.pbrt   # only these files

A scene whose geometry cannot be loaded here (a mesh that is not downloaded) is skipped and listed: stamp it where its assets are. A scene with nothing
to measure (only disks and cylinders, or an empty one) gets no line, and the Live Preview falls back to the camera's distance to its target.

Needs a built ray_tracer.exe (x64\\Release; --exe for another).
"""
import argparse
import os
import re
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TAG = re.compile(r'^# @rt-size[ \t]+(\S+)[ \t]*$')


def default_exe():
    for rel in (os.path.join('x64', 'Release', 'ray_tracer.exe'), os.path.join('RayTracer_Package', 'ray_tracer.exe')):
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            return p
    return os.path.join(REPO, 'x64', 'Release', 'ray_tracer.exe')


def tracked_scenes():
    out = subprocess.run(['git', '-C', REPO, 'ls-files', '--', 'pbrt_scenes'], capture_output=True, text=True).stdout
    return [os.path.join(REPO, p) for p in out.splitlines() if p.lower().endswith('.pbrt')]


def split_header(text):
    """(lines, index of the WorldBegin line) or (lines, None) for a file with no WorldBegin (an included library, not a scene)."""
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if line.strip().startswith('WorldBegin'):
            return lines, i
    return lines, None


def existing_size(lines, world):
    for line in lines[:world]:
        m = TAG.match(line.rstrip('\r'))
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def measure(exe, path):
    proc = subprocess.run([exe, '--print-scene-size', path], capture_output=True, text=True, timeout=600, cwd=REPO)
    if proc.returncode != 0:
        return None, (proc.stderr.strip().splitlines() or ['failed'])[-1]
    try:
        return float(proc.stdout.strip().splitlines()[-1]), ''
    except (ValueError, IndexError):
        return None, 'no number printed'


def fmt(size):
    return format(float('%.3g' % size), 'f').rstrip('0').rstrip('.')   # three significant digits, never in exponent form


def stamped(text, size):
    """`text` with its "# @rt-size" line set to `size` (added after the last "# @rt-" tag line of the header, else on top)."""
    eol = '\r\n' if '\r\n' in text else '\n'
    lines = text.replace('\r\n', '\n').split('\n')
    world = next(i for i, l in enumerate(lines) if l.strip().startswith('WorldBegin'))
    new = '# @rt-size ' + fmt(size)
    for i in range(world):
        if TAG.match(lines[i]):
            lines[i] = new
            return eol.join(lines)
    last = -1
    for i in range(world):
        if lines[i].startswith('# @rt-'):
            last = i
    lines.insert(last + 1, new)
    return eol.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='*', help='scene files (default: every tracked .pbrt under pbrt_scenes)')
    ap.add_argument('--exe', default=default_exe())
    ap.add_argument('--check', action='store_true', help='report only; exit 1 if a line is missing or stale')
    ap.add_argument('--force', action='store_true', help='re-measure scenes that already have a line')
    args = ap.parse_args()

    files = [os.path.abspath(f) for f in args.files] or tracked_scenes()
    written = skipped = unchanged = problems = 0
    for path in files:
        rel = os.path.relpath(path, REPO)
        with open(path, 'rb') as f:
            text = f.read().decode('utf-8', errors='surrogateescape')
        lines, world = split_header(text)
        if world is None:
            continue   # an included library, not a scene
        have = existing_size(lines, world)
        if have is not None and not args.force and not args.check:
            unchanged += 1
            continue
        size, why = measure(args.exe, path)
        if size is None:
            print('skip   %-60s (%s)' % (rel, why))
            skipped += 1
            continue
        if size == 0:
            if have is not None:
                print('stale  %-60s has %s, geometry says nothing to measure' % (rel, fmt(have)))
                problems += 1
            else:
                skipped += 1
            continue
        if args.check:
            if have is None:
                print('missing %-59s should be %s' % (rel, fmt(size)))
                problems += 1
            elif abs(have - size) > 0.02 * size:
                print('stale  %-60s has %s, geometry says %s' % (rel, fmt(have), fmt(size)))
                problems += 1
            continue
        if have is not None and abs(have - size) <= 0.005 * size:
            unchanged += 1
            continue
        with open(path, 'wb') as f:
            f.write(stamped(text, size).encode('utf-8', errors='surrogateescape'))
        print('stamp  %-60s %s' % (rel, fmt(size)))
        written += 1
    print('%d written, %d already fine, %d skipped, %d out of date' % (written, unchanged, skipped, problems))
    return 1 if (args.check and problems) else 0


if __name__ == '__main__':
    sys.exit(main())

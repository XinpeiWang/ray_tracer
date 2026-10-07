#!/usr/bin/env python3
"""Size ratchet: stop functions and files from growing without anyone deciding that they should.

Long functions and files are where this project's reviews get hard (a 1,080-line function in the Metal dispatcher was split once and grew back).
This does not demand that every existing offender be fixed; it records today's in scripts/code_size_baseline.list and fails CI when

  * a NEW function longer than FUNCTION_LIMIT lines, or a NEW file longer than FILE_LIMIT lines, appears, or
  * a baselined one grows by more than the tolerance (the larger of 5% and 20 lines).

Shrinking is always fine; after splitting something, run `--update` to lower its baseline (or drop it) and commit the file.

  python3 scripts/check_code_size.py            check (exit 1 on a violation)
  python3 scripts/check_code_size.py --update   rewrite the baseline from the current tree
  python3 scripts/check_code_size.py --list     print the current offenders, longest first

Scope: C++/Objective-C++/Metal sources and headers outside tests/, src/external/, src/data/ (generated tables), generated and build directories. "Function" means a function
or member-function body found by a brace scan (lambdas and bodies nested inside another function are part of their enclosing function), so
the numbers are a good guide, not a compiler's.
"""
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = os.path.join(REPO, "scripts", "code_size_baseline.list")
FUNCTION_LIMIT = 300
FILE_LIMIT = 2000
EXTS = (".cpp", ".h", ".hpp", ".mm", ".metal", ".cu", ".cuh", ".c")
EXCLUDE = ("tests/", "src/external/", "src/data/", "build", "RayTracer_Package", "releases/", "qt_gui/translations/", "pbrt_scenes/")


def source_files():
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split("\n")
    return [f for f in out if f.endswith(EXTS) and not any(f.startswith(e) or ("/" + e) in f for e in EXCLUDE)]


def strip_noise(text):
    """Blank out comments, string and char literals (keeping newlines) so braces inside them do not count."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if two == "//":
            while i < n and text[i] != "\n":
                i += 1
        elif two == "/*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * text.count("\n", i, j))
            i = j
        elif c == '"' or c == "'":
            # raw strings are not used in the scanned code; a ' inside a number (digit separator) is rare enough to ignore
            q, j = c, i + 1
            while j < n and text[j] != q and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            out.append(q + q)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


HEADER_NAME = re.compile(r"([~A-Za-z_][A-Za-z0-9_:<>~]*)\s*\(")


def functions(path):
    """[(name, first_line, line_count)] for function bodies opened directly at file, namespace or type level."""
    text = strip_noise(open(os.path.join(REPO, path), encoding="utf-8", errors="replace").read())
    stack = []            # kinds: 'ns', 'type', 'fn', 'inner'
    found = []
    pending = []          # text of the statement being read, to classify the next '{'
    line = 1
    start = {}
    for ch in text:
        if ch == "\n":
            line += 1
        if ch == "{":
            head = "".join(pending)
            parent = stack[-1] if stack else "top"
            if parent in ("top", "ns", "type"):
                h = head.strip()
                if re.search(r"\b(namespace)\b", h) or re.search(r'extern\s*"', h) or h.endswith('""'):
                    kind = "ns"
                elif re.match(r"^(template\s*<[^>]*>\s*)?(typedef\s+)?(struct|class|union|enum)\b", h) and "(" not in h:
                    kind = "type"
                elif "(" in h and ")" in h and not re.search(r"\b(struct|class|enum)\s+\w+\s*\{?$", h):
                    kind = "fn"
                else:
                    kind = "inner"
            else:
                kind = "inner"
            stack.append(kind)
            if kind == "fn":
                m = HEADER_NAME.findall(head)
                name = m[-1] if m else "?"
                # prefer the last name before the first '(' (the function, not a macro or attribute in its parameters)
                first = HEADER_NAME.search(head)
                if first:
                    name = first.group(1)
                start[len(stack)] = (name, line)
            pending = []
        elif ch == "}":
            if stack:
                kind = stack.pop()
                if kind == "fn":
                    name, first = start.pop(len(stack) + 1)
                    found.append((name, first, line - first + 1))
            pending = []
        elif ch == ";" and (not stack or stack[-1] in ("ns", "type")):
            pending = []
        else:
            pending.append(ch)
    return found


def file_lines(path):
    with open(os.path.join(REPO, path), encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def measure():
    """{key: lines}; key is 'file <path>' or 'func <path>::<name>' (a repeated name keeps its longest body)."""
    sizes = {}
    for path in source_files():
        n = file_lines(path)
        if n > FILE_LIMIT:
            sizes["file " + path] = n
        for name, first, count in functions(path):
            if count > FUNCTION_LIMIT:
                key = "func %s::%s" % (path, name)
                sizes[key] = max(sizes.get(key, 0), count)
    return sizes


def read_baseline():
    base = {}
    if os.path.exists(BASELINE):
        for line in open(BASELINE, encoding="utf-8"):
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                key, _, n = line.rpartition(" ")
                base[key] = int(n)
    return base


def main():
    sizes = measure()
    if "--list" in sys.argv:
        for key, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
            print("%6d  %s" % (n, key))
        return 0
    if "--update" in sys.argv:
        with open(BASELINE, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Written by scripts/check_code_size.py --update. Functions over %d lines and files over %d lines that exist today;\n" % (FUNCTION_LIMIT, FILE_LIMIT))
            f.write("# CI fails when one grows or a new one appears. Lower or delete a line after splitting something. Format: <key> <lines>\n")
            for key, n in sorted(sizes.items()):
                f.write("%s %d\n" % (key, n))
        print("wrote %d entries to %s" % (len(sizes), os.path.relpath(BASELINE, REPO)))
        return 0
    base = read_baseline()
    problems = []
    for key, n in sorted(sizes.items()):
        allowed = base.get(key)
        if allowed is None:
            limit = FILE_LIMIT if key.startswith("file ") else FUNCTION_LIMIT
            problems.append("%s is %d lines (limit %d) and is not in the baseline - split it, or if it truly must be this long run --update and say why in the PR" % (key, n, limit))
        elif n > allowed + max(20, allowed // 20):
            problems.append("%s grew from %d to %d lines (tolerance %d) - split something out, or run --update if the growth is deliberate" % (key, allowed, n, max(20, allowed // 20)))
    stale = [k for k in base if k not in sizes]
    if problems:
        print("Code size ratchet failed:\n  " + "\n  ".join(problems))
        return 1
    print("code size ok: %d known long functions/files, none grew%s" % (len(sizes), " (%d baseline entries are now below the limit - run --update to drop them)" % len(stale) if stale else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

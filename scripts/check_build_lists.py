#!/usr/bin/env python3
"""Check that the two ways of building the test suite list the same test files.

The tests are built twice: by the MSBuild solution (tests/ray_tracer_tests.vcxproj, the full-coverage Windows build that also has the OptiX and
CUDA tests) and by the portable CMake target (tests/CMakeLists.txt, which macOS and the hosted CI runners use). Each lists its source files by
hand, so a new test file added to only one of them is silently not built, not run and not noticed by the other platform. This finds that:

  1. every tests/unit and tests/integration .cpp on disk is in the .vcxproj, and every such .vcxproj entry exists on disk;
  2. every test file CMake lists exists and is also in the .vcxproj;
  3. every test file the CMake target does NOT build is named in tests/portable_build_exclusions.txt (with the reasons, in that file's header),
     so "left out on purpose" is written down and a forgotten file is not.

Usage: python3 scripts/check_build_lists.py          (from anywhere; exit status 1 and a list of problems on a mismatch)
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(REPO, "tests")


def test_files_on_disk():
    found = set()
    for sub in ("unit", "integration"):
        d = os.path.join(TESTS, sub)
        for name in os.listdir(d):
            if name.endswith(".cpp"):
                found.add("%s/%s" % (sub, name))
    return found


def vcxproj_files():
    text = open(os.path.join(TESTS, "ray_tracer_tests.vcxproj"), encoding="utf-8").read()
    files = set()
    for m in re.finditer(r'<ClCompile Include="([^"]+)"', text):
        p = m.group(1).replace("\\", "/")
        if p.startswith(("unit/", "integration/")):
            files.add(p)
    return files


def cmake_files():
    files = set()
    for line in open(os.path.join(TESTS, "CMakeLists.txt"), encoding="utf-8"):
        if line.lstrip().startswith("#"):
            continue
        for m in re.finditer(r"\b((?:unit|integration)/[A-Za-z0-9_]+\.cpp)\b", line):
            files.add(m.group(1))
    return files


def exclusions():
    files = set()
    for line in open(os.path.join(TESTS, "portable_build_exclusions.txt"), encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            files.add(line)
    return files


def main():
    disk, vcx, cmake, excluded = test_files_on_disk(), vcxproj_files(), cmake_files(), exclusions()
    problems = []
    for f in sorted(disk - vcx):
        problems.append("%s exists but is not in tests/ray_tracer_tests.vcxproj (the Windows full build would never compile it)" % f)
    for f in sorted(vcx - disk):
        problems.append("tests/ray_tracer_tests.vcxproj lists %s, which does not exist" % f)
    for f in sorted(cmake - disk):
        problems.append("tests/CMakeLists.txt lists %s, which does not exist" % f)
    for f in sorted((cmake & disk) - vcx):
        problems.append("%s is in tests/CMakeLists.txt but not in tests/ray_tracer_tests.vcxproj" % f)
    for f in sorted(disk - cmake - excluded):
        problems.append("%s is not in tests/CMakeLists.txt (macOS and the hosted CI would never build it) and not in "
                        "tests/portable_build_exclusions.txt - add it to CMake, or to that file with a reason (needs OptiX/CUDA or a real GPU)" % f)
    for f in sorted(excluded & cmake):
        problems.append("%s is listed in tests/portable_build_exclusions.txt but CMake builds it now - remove it from the exclusions" % f)
    for f in sorted(excluded - disk):
        problems.append("tests/portable_build_exclusions.txt lists %s, which does not exist" % f)
    if problems:
        print("Test build lists are out of step:\n  " + "\n  ".join(problems))
        return 1
    print("test build lists agree: %d files on disk, %d in CMake, %d deliberately MSBuild-only" % (len(disk), len(cmake), len(excluded)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

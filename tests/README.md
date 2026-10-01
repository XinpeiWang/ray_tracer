# Ray Tracer Tests

Google Test suite: a large and growing number of tests (4,292 across 570
test suites as of this writing - run `--gtest_list_tests` for the live
count rather than trusting this number, it drifts fast), in `unit/` and
`integration/`.

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) for the full guide, including
the two available build paths (the full-coverage MSVC solution vs. the
portable, SDK-independent CMake target), running/filtering tests, the
quick dev-loop filter, debugging failed tests, and adding new tests.

Quickest path if you already have the MSVC solution built:
```powershell
.\bin\Release\ray_tracer_tests.exe
```

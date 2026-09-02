# Ray Tracer Tests

Google Test suite: **3,830 tests across 524 test suites**, in ~184 files
under `unit/` and `integration/`.

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) for the full guide, including
the two available build paths (the full-coverage MSVC solution vs. the
portable, SDK-independent CMake target), running/filtering tests, the
quick dev-loop filter, debugging failed tests, and adding new tests.

Quickest path if you already have the MSVC solution built:
```powershell
.\bin\Release\ray_tracer_tests.exe
```

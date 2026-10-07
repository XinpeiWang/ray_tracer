# Ray Tracer Tests

Google Test suite: a large and growing number of tests (run `--gtest_list_tests` for the live count), in `unit/` and `integration/`.

See [`TESTING_GUIDE.md`](TESTING_GUIDE.md) for the full guide, including
the two available build paths (the full-coverage MSVC solution vs. the
portable, SDK-independent CMake target), running/filtering tests, the
quick dev-loop filter, debugging failed tests, and adding new tests.

Quickest path if you already have the MSVC solution built:
```powershell
.\bin\Release\ray_tracer_tests.exe
```

## Qt GUI tests (separate from the Google Test suite)

* `qt_gui/tests/downloader_tests.cpp` - a QtTest for the GUI's two "Download missing files" back ends (checksums, resume, upstream-changed
  refusal, the shipped catalogue parsing). It serves its own files from a local socket, so it needs only Qt:
  `cd qt_gui/tests && qmake && make check` (Windows: `nmake check` from a Visual Studio prompt). CI runs it, and builds the whole GUI,
  on macOS and Windows (`qt-gui` job in `.github/workflows/unit-tests.yml`).
* `scripts/gui_selftest.py` - starts the built GUI headless in each `RT_GUI_SELFTEST` mode (`ui`, `options`, `builder`, `diagnostics`, and with
  `--live-preview` the Live Preview checks) and fails on a crash, a hang or a wrong result. It works the same on macOS and Windows.

## Keeping the two test builds in step

The suite is built by `tests/ray_tracer_tests.vcxproj` (MSBuild, everything including the OptiX/CUDA tests) and `tests/CMakeLists.txt` (portable;
macOS and the CI runners). Add a new test file to **both**; a file that needs the OptiX SDK or a real GPU stays out of CMake and goes into
`tests/portable_build_exclusions.txt` instead. `python3 scripts/check_build_lists.py` (also a CI job) fails when the lists disagree.

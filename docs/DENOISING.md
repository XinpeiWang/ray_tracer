# Denoising

A path-traced image is grainy until enough samples are averaged. A denoiser removes most of the grain from a render with few samples, so a clean picture takes a fraction of the time.

| Where | Denoiser | Notes |
|---|---|---|
| Windows, GPU (OptiX), recursive and wavefront | NVIDIA's OptiX AI denoiser | guided by albedo and normal buffers; `--denoise`, `--denoise-blend` |
| **macOS, GPU (Metal)** | **Intel Open Image Denoise (OIDN)** | colour only (Metal writes no albedo/normal buffers); same flags |
| CPU (any platform) | **Intel Open Image Denoise (OIDN)** | the default path tracer only, from the command line (`--denoise`); colour only; needs the OIDN library (see below) - without it the flag warns and the render goes on undenoised |

## On a Mac

Tick **AI denoiser (Open Image Denoise)** in the Render Options tab (the Renderer on the Settings tab must be GPU). The first time, the app asks to download the denoiser: Open Image Denoise is open source (Apache-2.0) but its network weights make the library about 50 MB, so it is not inside the app. The release for your Mac's architecture (arm64 or x86_64) is fetched from OIDN's own GitHub release page, checked against a SHA-256 checksum built into the program, unpacked, and kept in `<your data folder>/user_assets/denoiser/` (with its licence). After that, nothing is downloaded again.

On the command line the same flag works: `ray_tracer --gpu --denoise 800 16 8 cornell-box`. The program looks for the library in, in order: `$RT_OIDN_DIR` (a folder with `libOpenImageDenoise.dylib` in it or in `lib/` below it), `$RAY_TRACER_USER_ASSETS/denoiser/lib` (where the GUI installs it), and Homebrew's `/opt/homebrew/lib` and `/usr/local/lib` (`brew install open-image-denoise`). If it is not found the flag prints a warning and the render goes on without it; the render never fails because of it.

The denoiser works on the linear HDR image before tone mapping, so it also applies to `.exr` output. `--denoise-blend` mixes some of the original grain back (0 = fully denoised, 1 = untouched), as with OptiX. When the denoiser runs, the Metal PNG's small built-in blur (see `RT_METAL_POST_EFFECTS` in METAL_BACKEND.md) is skipped.

What it does not do yet: Live Preview on a Mac is not denoised, and the denoiser sees only the colour image, so very fine texture can be smoothed along with the noise at low sample counts; raise the blend or the sample count if that matters.

## On the CPU, and on Windows

`ray_tracer --cpu --denoise 800 64 8 cornell-box` denoises the CPU render with Open Image Denoise too, on any platform (the default path tracer only; BDPT, MLT, SPPM and the debug integrators warn and ignore the flag). It works on the linear image before it is tone mapped, so `.exr` output is denoised as well. The same `--denoise-blend` applies.

There is no installer button for it on Windows: unpack an Open Image Denoise 2.x release (the Windows one has `bin/OpenImageDenoise.dll` and the DLLs it needs next to it) anywhere and point `RT_OIDN_DIR` at that folder, or copy the release into `<user assets>/denoiser`. The loader looks in the folder itself, `lib/` and `bin/`. On Windows with a GPU the OptiX denoiser is still what `--gpu --denoise` uses.

## For developers

`src/shared/oidn_runtime.h` loads the library with `dlopen` (`LoadLibrary` on Windows) and declares just the dozen C entry points it uses, so nothing links against OIDN and no OIDN headers are needed to build. `denoiseHdr()` copies the image into OIDN buffers (so it works on every OIDN device, including Metal), runs the `RT` filter in HDR mode at high quality and mixes the result by the blend. `qt_gui/denoiser_installer.*` is the downloader (it reuses the asset downloader, with the archives' sizes and checksums pinned in the source). Tests: `tests/unit/oidn_runtime_tests.cpp` (the real denoising test runs when `RT_OIDN_DIR` points at an OIDN release), and the GUI self-test mode `denoiser` (not in the default set: it downloads ~50 MB) installs it through the real installer and denoises through it.

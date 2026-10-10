# Documentation

A guide to this folder and the other documents. The repository's front page is the [README](../README.md).

## Using the renderer

| Document | What it is |
|---|---|
| [INSTALL.md](../INSTALL.md) | Using the portable release package |
| [SCENE_SELECTION.md](SCENE_SELECTION.md) | Scene names and ids, categories, what a scene's info means, header tags, adding a scene |
| [SCENE_BUILDER.md](SCENE_BUILDER.md) | The GUI's Scene Builder: build a scene and save a `.pbrt` |
| [PHOTO_TO_SCENE.md](PHOTO_TO_SCENE.md) | Pictures as textures, one photo to a 3D object, and bringing in a photo scan |
| [VIDEO_GENERATION.md](VIDEO_GENERATION.md) | Camera paths, frames, assembling an MP4 |
| [DENOISING.md](DENOISING.md) | The AI denoisers (OptiX on the GPU, Open Image Denoise on the CPU and Mac), how to turn them on, how well they work |
| [RENDER_PASSES.md](RENDER_PASSES.md) | Albedo, normal, depth, uv and alpha passes for compositing |
| [LOGGING.md](LOGGING.md) | The log file: where it is and what to send with a bug report |
| [LIVE_PREVIEW.md](LIVE_PREVIEW.md) | Live Preview for users: starting it, camera controls, settings, moving objects and saving the arrangement |
| [MAC_LIVE_PREVIEW.md](MAC_LIVE_PREVIEW.md) | The macOS interactive Live Preview, for developers |
| [MAC_VERIFICATION.md](MAC_VERIFICATION.md) | What was checked on a real Mac after shared changes made on Windows |
| [ERROR_CODE_REFERENCE.md](ERROR_CODE_REFERENCE.md) | Every error code, what it means, what to try |
| [../pbrt_scenes/README.md](../pbrt_scenes/README.md) | The scene folder: names, header tags, the pbrt subset read |

## What the renderer supports

| Document | What it is |
|---|---|
| [GPU_SCENE_COMMON.md](GPU_SCENE_COMMON.md) | Proposal: one scene-to-GPU conversion shared by the OptiX and Metal backends |
| [BACKEND_SUPPORT.md](BACKEND_SUPPORT.md) | One table: what the CPU, OptiX (recursive, wavefront) and Metal renderers each support |
| [PBRT_SUPPORT.md](PBRT_SUPPORT.md) | What happens to each pbrt-v4 directive when a scene file is loaded, per backend (Full / Approx / Fallback / Unsupported) |
| [FEATURE_INVENTORY.md](FEATURE_INVENTORY.md) | Feature by feature and backend by backend, and the gaps from pbrt-v4 |
| [MULTIPLE_IMPORTANCE_SAMPLING.md](MULTIPLE_IMPORTANCE_SAMPLING.md) | How MIS is used |
| [CLOSED_FORMS_FOUND_THE_BUGS.md](CLOSED_FORMS_FOUND_THE_BUGS.md) | Why the tests check closed-form answers, and what that found |
| [METAL_BACKEND.md](METAL_BACKEND.md), [METAL_PARITY_STATUS.md](METAL_PARITY_STATUS.md) | The Metal backend and how closely it matches the CPU renderer ([history/](history/) has the long diary) |

## Building and contributing

| Document | What it is |
|---|---|
| [BUILD.md](../BUILD.md) | Building on Windows and macOS, the fast incremental loop, troubleshooting |
| [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) | Where things are in the repository |
| [CONTRIBUTING.md](../CONTRIBUTING.md), [CODING_STANDARDS.md](../CODING_STANDARDS.md) | How to contribute, and the code style |
| [tests/TESTING_GUIDE.md](../tests/TESTING_GUIDE.md) | Running and adding tests |
| [scripts/README.md](../scripts/README.md), [releases/README.md](../releases/README.md) | The scripts, and making a release package |
| [gpu/optix/README.md](../gpu/optix/README.md) | The OptiX backends |
| [qt_gui/QT_GUI_DOCUMENTATION.md](../qt_gui/QT_GUI_DOCUMENTATION.md) | The Qt GUI's code, themes and installation |
| [CHANGELOG.md](../CHANGELOG.md), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) | What changed, and the licences of what is included |

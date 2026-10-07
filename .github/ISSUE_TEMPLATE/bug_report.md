---
name: Bug report
about: A render looks wrong, a backend disagrees with another, a crash, or a build failure
title: ""
labels: bug
---

**What is wrong**
One or two sentences. If it is a wrong picture, say what you expected and why (a closed form, another
backend, pbrt-v4, a real photo).

**How to reproduce**
- Scene id or `.pbrt` file (attach it if it is yours):
- Full command line (backend flags such as `--cpu`, `--gpu`, `--gpu --wavefront`, `--bdpt`, `--sppm`, size, samples, depth):
- Does another backend render it correctly? (CPU / OptiX recursive / OptiX wavefront / Metal)

**Your setup**
- OS and version:
- GPU and driver (or "CPU only"):
- Built from source (commit) or release (version):

**Output**
Paste the end of the console log, or the error message. Attach the image if you can (a small crop is fine).

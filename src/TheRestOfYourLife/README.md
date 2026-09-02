# src/TheRestOfYourLife/

This is the **primary CPU path tracer implementation** for this project -
materials, lights, cameras, the pbrt-v4 scene builder, and the BDPT/MLT/SPPM
integrators all live here. Despite the name, it is not scoped to the "Ray
Tracing: The Rest Of Your Life" book anymore; that book is where this
directory started, and the name stuck as the codebase grew well beyond it.
See the repo root [`README.md`](../../README.md)'s Project Structure section
and [`docs/FEATURE_INVENTORY.md`](../../docs/FEATURE_INVENTORY.md) for what
actually lives here today.

A rename was considered and deliberately not done: this is the
most-#include'd directory in the whole repository, so a rename is a large,
purely cosmetic, repo-wide mechanical change for a naming-clarity win this
file already delivers more cheaply.

Its GPU counterpart is [`gpu/optix/`](../../gpu/optix/) (see that
directory's own README) and the shared, backend-agnostic pieces both sides
depend on live in [`src/shared/`](../shared/).

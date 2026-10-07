# Closed forms found the bugs: what testing a renderer against its own backends missed

*A write-up of the correctness work on this renderer. Every number below is a measurement from this repository's own tests and commit messages.*

This project has three ways to render the same scene (a CPU path tracer, two OptiX GPU backends, and a Metal one on macOS) plus four more integrators (BDPT, MLT, SPPM and a few debug ones). The obvious test is to render a scene with all of them and check that they agree. It is a good test and it found a lot. It also has a blind spot, and most of what follows is about that blind spot.

## Agreement tells you something is inconsistent

Comparing backends found the loud bugs. A glass sphere that was 2.5% dark under BDPT only. A GPU backend that read 17% low on a floor next to a small light. A path tracer 3.4 times too bright next to a cone-shaped light, where BDPT disagreed with it.

But a disagreement only says *one of these is wrong*, not which, and it says nothing when they are all wrong in the same way. For that you need a number that does not come from any of your backends: a **closed form**.

## The cheap closed forms

You do not need a reference renderer. A few scenes have exact answers:

* **A furnace.** A diffuse sphere with albedo 0.5 under a uniform white sky radiates exactly 0.5 everywhere, at depth 1. Nothing else in the scene can contribute.
* **A closed shell.** A closed shell of a diffuse-transmitting material (reflect 0.2, transmit 0.6) in a white room reads 0.2, 0.56, 0.632, 0.6464 ... 0.65 at depths 1, 2, 3, 4 and in the limit.
* **Beer-Lambert.** A pure absorber with sigma = (0.1, 0.4, 0.9) over a chord of 2 transmits exp(-sigma * 2) = (0.82, 0.45, 0.17) of a uniform sky, per channel.
* **A point light over a plate.** Radiance is rho * L * (r/d)^2 * cos for a small sphere light at distance d, or rho * I * cos / (pi d^2) for a point light.

Each costs a dozen lines of scene file and a test that reads the image mean.

## What they caught that comparison could not

| Closed form | What every backend read | Why backends agreed |
|---|---|---|
| Diffuse sphere under a white sky, depth 1 (0.5) | the CPU path tracer 0.09 | all three path tracers stopped one segment early, so none could see it |
| White-furnace sphere of a measured BRDF (albedo 0.1 / 0.2 / 0.4) | 7.8 / 8.1 / 3.5 on every backend | the material returned the bare f instead of f * cos / pdf, on all of them |
| Diffuse-transmission shell (0.6464 at depth 4) | CPU 0.39, recursive GPU 0.25, `--simplepath` 1.40 | three different defects, so three different wrong answers |
| Chromatic absorber (0.82 / 0.45 / 0.17) | grey 0.47 | every medium model flattened the extinction to one luminance value |
| Diffuse-transmission plate under a point light (1.59 behind, 0.95 in front) | both GPU backends black; CPU right | the GPU never took a light sample at that material, so it could not find a point light |

The first two are the kind of bug that makes CPU-versus-GPU testing feel safe while it is not.

## Closed forms also say which side is wrong

A closed form is just as useful when the backends do disagree. Next to a cone-shaped light the path tracer read 1.4549 and BDPT read 0.4255. A comparison alone would not say which. The probe point has an exact answer, 0.4255: BDPT was right, and the *older* path-tracer code was wrong (its light-sampling density counted only the first of the two places a ray can cross the cone, so the light was 3.4 times too bright). Without the closed form, a reasonable person fixes BDPT to match.

The same trick found a float32 bug. A small sphere light 480 units away read 65% (recursive GPU) and 73% (wavefront GPU) of its closed form. The sphere intersection subtracted two numbers near 230,000, so the hit point was about 0.005 units off, the shadow ray stopped 0.002 short of it, hit the light's own surface and was counted as blocked. The farther the light, the more samples were lost. pbrt-v4 avoids this with an improved-precision discriminant; porting it brought both backends to the closed form within 0.1%.

## Other things that paid off

* **Restrict the estimator to find the strategy.** BDPT combines many ways of connecting a path with MIS weights that must sum to one. Running subsets of those strategies alone, and in pairs, showed that the weights summed to one for strategies with at most two light-path vertices and not for the rest. That pointed straight at a reverse density that was never set at the first vertex.
* **Read what the reference actually does.** More than once the right answer came from reading pbrt-v4's source for the case, instead of inferring it from a difference between two of our own backends.
* **Make a warning say what is true now.** When coloured fog was fixed in BDPT, the warning that said "fog is wrong here" became a note about cost ("rendered as three passes"), and a test pins the note along with the numbers.
* **Fix the test infrastructure when it lies.** One CUDA launch failure poisoned a whole test process, so every later GPU test *skipped* and reported green-looking numbers. The skip count is part of the result.

## The shape of the suite

Over 5,000 tests run in under ten minutes on a desktop GPU, split into a fast tier and a slow one. The CPU-only part (every integrator, against closed forms) is set up to run on a GitHub-hosted runner with no GPU too; that job was added recently and has not yet been seen to pass there. Where a backend still differs from another, [docs/PBRT_SUPPORT.md](PBRT_SUPPORT.md) says so, with the numbers.

If you build a renderer, build the furnace first.

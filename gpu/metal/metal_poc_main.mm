// The standalone metal_poc executable's own main() - split out of
// metal_poc.mm (phase 3 of real GPU integration, see docs/
// METAL_GPU_FEASIBILITY.md's own section on this) so that file's code
// (MetalPocApp, metal_render_main()) can ALSO be linked into ray_tracer
// itself, which already has its own main() (launcher/main.cpp) - two
// definitions of main() in the same executable won't link. This file is
// linked ONLY into the standalone metal_poc target, never into
// ray_tracer, so it's the one place that distinction actually matters.
int metal_poc_cli_main(int argc, const char** argv);

int main(int argc, const char** argv) {
    return metal_poc_cli_main(argc, argv);
}

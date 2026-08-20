#pragma once
// launcher/diagnostics.h -- system-compatibility report for --diagnose.
//
// Gathers OS/CPU/RAM, GPU/CUDA/OptiX (via gpu/optix/optix_interface.h's
// OptixDiagnostics), disk space/output-directory writability, and scene
// asset availability into one human-readable report, printed to stdout and
// optionally written to --output's path (see LaunchArgs::custom_output_path
// - reused rather than adding a second output flag).

#include "launcher_args.h"

// Always returns SUCCESS (src/TheRestOfYourLife/error_codes.h) - a "GPU
// unavailable" or "asset missing" finding is valid report content, not a
// launcher error.
int run_diagnostics(const LaunchArgs& args);

#include "diagnostics.h"

#ifdef RT_HAVE_OPTIX
#include "../gpu/optix/optix_interface.h"
#else
#include "optix_stub.h"
#endif
#include "../cpu_renderer/cpu_interface.h"
#include "../src/TheRestOfYourLife/error_codes.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

std::string format_bytes(unsigned long long bytes) {
	double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
	std::ostringstream oss;
	oss.precision(2);
	oss << std::fixed << gb << " GB";
	return oss.str();
}

// e.g. 12040 -> "12.4" (CUDA version encoding: major*1000 + minor*10).
std::string format_cuda_version(int version) {
	if (version <= 0) return "n/a";
	std::ostringstream oss;
	oss << (version / 1000) << "." << ((version % 1000) / 10);
	return oss.str();
}

void append_os_cpu_ram(std::ostringstream& report) {
#ifdef _WIN32
	report << "OS: Windows\n";
#elif defined(__APPLE__)
	report << "OS: macOS\n";
#elif defined(__linux__)
	report << "OS: Linux\n";
#else
	report << "OS: Unknown\n";
#endif

	unsigned cores = std::thread::hardware_concurrency();
	report << "CPU: " << (cores > 0 ? std::to_string(cores) : std::string("unknown"))
		   << " logical cores\n";

#ifdef _WIN32
	MEMORYSTATUSEX statex{};
	statex.dwLength = sizeof(statex);
	if (GlobalMemoryStatusEx(&statex)) {
		// One value per line (not "X free / Y total" on one line) so every
		// line in the report is a single key: value pair - both for CLI
		// readability and so the GUI's line-based colorizer (see
		// mainwindow_slots.cpp's styleDiagnosticsLine) has exactly one fact
		// to classify per line instead of two crammed together.
		report << "RAM Free: " << format_bytes(statex.ullAvailPhys) << "\n";
		report << "RAM Total: " << format_bytes(statex.ullTotalPhys) << "\n";
	} else {
		report << "RAM: not available\n";
	}
#else
	// Best-effort scope (matches root CMakeLists.txt's own macOS/Linux
	// posture) - no portable RAM query exists pre-C++26.
	report << "RAM: not available on this platform\n";
#endif
}

void append_gpu(std::ostringstream& report) {
	OptixDiagnostics diag{};
	optix_get_diagnostics(&diag);
	if (diag.available) {
		report << "GPU: " << diag.device_name << "\n";
		report << "Driver Version: " << format_cuda_version(diag.cuda_driver_version) << "\n";
		report << "CUDA Runtime: " << format_cuda_version(diag.cuda_runtime_version) << "\n";
		report << "VRAM Free: " << format_bytes(diag.vram_free_bytes) << "\n";
		report << "VRAM Total: " << format_bytes(diag.vram_total_bytes) << "\n";
		report << "OptiX: available (ABI v" << diag.optix_abi_version << ")\n";
	} else {
		report << "GPU: not detected / not usable\n";
		if (diag.cuda_driver_version > 0 || diag.cuda_runtime_version > 0) {
			report << "Driver Version: " << format_cuda_version(diag.cuda_driver_version) << "\n";
			report << "CUDA Runtime: " << format_cuda_version(diag.cuda_runtime_version) << "\n";
		}
		report << "OptiX: not available (" << diag.failure_reason << ")\n";
	}
}

// Checks disk free space and write access for the directory a render would
// actually write into - args.custom_output_path's parent dir if given,
// otherwise "output" (approximating main.cpp's own default of
// <exe_dir>/output/, since this runs before argv[0] resolution and the
// common case - both direct CLI use and GUI's QProcess launch, which sets
// its working directory to the exe's own directory - has cwd == exe dir).
void append_disk(std::ostringstream& report, const std::string& custom_output_path) {
	namespace fs = std::filesystem;
	try {
		fs::path dir;
		if (!custom_output_path.empty()) {
			fs::path p(custom_output_path);
			dir = p.has_parent_path() && !p.parent_path().empty() ? p.parent_path() : fs::path(".");
		} else {
			dir = "output";
		}

		std::error_code ec;
		fs::path space_target = fs::exists(dir, ec) ? dir : fs::current_path();
		auto space_info = fs::space(space_target, ec);
		if (!ec) {
			report << "Disk Free: " << format_bytes(space_info.free) << "\n";
		} else {
			report << "Disk Free: not available (" << ec.message() << ")\n";
		}

		fs::create_directories(dir, ec);
		fs::path probe = dir / ".rt_diag_probe.tmp";
		bool writable = false;
		{
			std::ofstream f(probe);
			writable = f.good();
		}
		if (writable) {
			fs::remove(probe, ec);
		}
		report << "Output Directory: " << dir.string() << " ("
			   << (writable ? "writable" : "NOT writable") << ")\n";
	} catch (const std::exception& e) {
		report << "Disk/output-directory check failed: " << e.what() << "\n";
	}
}

// Real per-file existence checks for pbrt-backed scenes (which have a
// queryable path via cpu_scene_pbrt_path_by_id()); everything else only
// gets a coarse "does it need files at all" tally, since no structured
// filename list exists for the hand-built C++ scenes (see this feature's
// plan for why - the actual filenames only ever appear as free text inside
// each scene's description or as literals inside its build_*() function).
void append_scene_assets(std::ostringstream& report) {
	namespace fs = std::filesystem;
	int total = cpu_scene_count();
	int requires_files_count = 0;
	int pbrt_checked = 0, pbrt_present = 0;
	std::vector<std::string> missing_pbrt;

	for (int i = 0; i < total; ++i) {
		const char* id = cpu_scene_id(i);
		if (!id || !*id) continue;

		if (cpu_scene_requires_files_by_id(id)) {
			++requires_files_count;
		}

		const char* pbrt_path = cpu_scene_pbrt_path_by_id(id);
		if (pbrt_path && *pbrt_path) {
			++pbrt_checked;
			std::error_code ec;
			if (fs::exists(pbrt_path, ec)) {
				++pbrt_present;
			} else {
				missing_pbrt.emplace_back(std::string(id) + " (" + pbrt_path + ")");
			}
		}
	}

	std::error_code ec;
	bool models_dir_present = fs::exists("models", ec) || fs::exists("../models", ec);

	// One key: value pair per line - see append_os_cpu_ram()'s own comment
	// for why (this used to be two facts crammed onto one "Scene assets:"
	// line, plus a third crammed onto the "pbrt-backed scene files" line).
	report << "Models Directory: " << (models_dir_present ? "present" : "MISSING") << "\n";
	report << "Scenes Requiring External Files: " << requires_files_count << " of " << total
		   << " (see each scene's description for exact filenames)\n";
	report << "Pbrt Scene Files Present: " << pbrt_present << " of " << pbrt_checked << "\n";
	if (!missing_pbrt.empty()) {
		report << "Pbrt Scene Files Missing: ";
		for (size_t i = 0; i < missing_pbrt.size(); ++i) {
			if (i) report << ", ";
			report << missing_pbrt[i];
		}
		report << "\n";
	}
}

} // namespace

int run_diagnostics(const LaunchArgs& args) {
	std::ostringstream report;
	report << "=== Ray Tracer System Diagnostics ===\n";
	append_os_cpu_ram(report);
	append_gpu(report);
	append_disk(report, args.custom_output_path);
	append_scene_assets(report);
	report << "======================================\n";

	std::string text = report.str();
	std::cout << text;

	if (!args.custom_output_path.empty()) {
		std::ofstream f(args.custom_output_path);
		if (f) {
			f << text;
		} else {
			std::cerr << "Warning: could not write diagnostics report to "
					  << args.custom_output_path << "\n";
		}
	}

	return SUCCESS;
}

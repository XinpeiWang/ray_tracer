/**
 * @file backend_consistency_tests.cpp
 * @brief Source-level drift guards for two places where the same fact is
 * maintained by hand in two files, on two platforms, with nothing else
 * tying them together.
 *
 * These are text-parsing tests on purpose: the Metal backend
 * (gpu/metal/*.mm) and the macOS build (root CMakeLists.txt) can't be
 * compiled or run on the Windows machines/CI runners this suite mostly
 * runs on, but the *facts* they duplicate can still be checked - and a
 * drift in either is exactly the kind of mistake that only surfaces
 * later, on the other platform.
 *
 * 1. Metal's "which scenes are supported" list vs. its dispatcher.
 *    cpu_scene_metal_hand_authored_supported() (cpu_renderer/
 *    cpu_interface.cpp) is what the GUI/CLI consult to decide whether to
 *    offer a scene on Metal; MetalPocApp::buildHandAuthoredScene()
 *    (gpu/metal/metal_poc.mm) is what actually builds it. A scene in the
 *    list with no builder is offered and then fails at render time; a
 *    builder with no list entry is dead code the GUI never offers.
 *
 * 2. The CPU-side source files each build system compiles. The MSBuild
 *    projects (Windows) and root CMakeLists.txt (macOS/Linux) each list the
 *    same source files by hand. PR #1 was exactly this drift: the CMake
 *    cpu_renderer target was missing files the MSBuild one had, which only
 *    showed up as a link error the first time anyone built on a Mac.
 *    Compared as a UNION across the CPU-side targets, not per target,
 *    because the two systems legitimately place some files in different
 *    targets (e.g. src/data/rgb_spectrum_table_data.cpp) - what must never
 *    differ is "a file one build system compiles and the other never does".
 */

#include <gtest/gtest.h>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// tests/unit/<this file> -> repo root, resolved from the compile-time path so
// it works regardless of the working directory the test binary runs from.
fs::path repoRoot() {
    return fs::path(__FILE__).parent_path().parent_path().parent_path();
}

std::string readFile(const fs::path &rel) {
    const fs::path p = repoRoot() / rel;
    std::ifstream in(p, std::ios::binary);
    if (!in) return {};
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    // Drop a UTF-8 BOM (tests/CMakeLists.txt and some .vcxproj files have one).
    if (s.size() >= 3 && static_cast<unsigned char>(s[0]) == 0xEF &&
        static_cast<unsigned char>(s[1]) == 0xBB && static_cast<unsigned char>(s[2]) == 0xBF) {
        s.erase(0, 3);
    }
    return s;
}

// Removes `// ...` to end of line, so an ID or path that only appears inside
// a comment (this codebase comments heavily) is never mistaken for real code.
std::string stripSlashComments(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '/' && i + 1 < s.size() && s[i + 1] == '/') {
            while (i < s.size() && s[i] != '\n') ++i;
            if (i < s.size()) out += '\n';
        } else {
            out += s[i];
        }
    }
    return out;
}

// Removes `# ...` to end of line (CMake comments).
std::string stripHashComments(const std::string &s) {
    std::string out;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line)) {
        const size_t h = line.find('#');
        out += (h == std::string::npos ? line : line.substr(0, h));
        out += '\n';
    }
    return out;
}

// Index of the '}' / ')' matching the opener at `open`, or npos.
size_t matchingClose(const std::string &s, size_t open, char openCh, char closeCh) {
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == openCh) ++depth;
        else if (s[i] == closeCh && --depth == 0) return i;
    }
    return std::string::npos;
}

std::set<std::string> allMatches(const std::string &text, const std::regex &re, size_t group = 1) {
    std::set<std::string> out;
    for (std::sregex_iterator it(text.begin(), text.end(), re), end; it != end; ++it) {
        out.insert((*it)[group].str());
    }
    return out;
}

std::string joined(const std::set<std::string> &s) {
    std::string out;
    for (const std::string &x : s) out += (out.empty() ? "" : ", ") + x;
    return out.empty() ? "(none)" : out;
}

std::set<std::string> minus(const std::set<std::string> &a, const std::set<std::string> &b) {
    std::set<std::string> out;
    for (const std::string &x : a) if (!b.count(x)) out.insert(x);
    return out;
}

// ---- 1. Metal supported-scene list vs. dispatcher -------------------------

std::set<std::string> metalListedScenes() {
    const std::string src = stripSlashComments(readFile("cpu_renderer/cpu_interface.cpp"));
    const size_t fn = src.find("cpu_scene_metal_hand_authored_supported(const char* scene_id)");
    if (fn == std::string::npos) return {};
    const size_t set = src.find("kSupported", fn);
    if (set == std::string::npos) return {};
    const size_t open = src.find('{', set);
    if (open == std::string::npos) return {};
    const size_t close = matchingClose(src, open, '{', '}');
    if (close == std::string::npos) return {};
    return allMatches(src.substr(open, close - open), std::regex("\"([A-Z][0-9]+)\""));
}

std::set<std::string> metalDispatchedScenes() {
    const std::string src = stripSlashComments(readFile("gpu/metal/metal_poc.mm"));
    const size_t fn = src.find("bool MetalPocApp::buildHandAuthoredScene(");
    if (fn == std::string::npos) return {};
    const size_t open = src.find('{', fn);
    if (open == std::string::npos) return {};
    const size_t close = matchingClose(src, open, '{', '}');
    if (close == std::string::npos) return {};
    return allMatches(src.substr(open, close - open),
                      std::regex("scene_id\\s*==\\s*\"([A-Z][0-9]+)\""));
}

// ---- 2. Build-system source lists -----------------------------------------

bool isCompiledSource(const std::string &p) {
    const auto ends = [&](const char *ext) {
        const std::string e(ext);
        return p.size() >= e.size() && p.compare(p.size() - e.size(), e.size(), e) == 0;
    };
    return ends(".cpp") || ends(".c");
}

std::set<std::string> msbuildSources(const std::string &project) {
    const std::string xml = readFile(fs::path(project) / (project + ".vcxproj"));
    std::set<std::string> out;
    for (const std::string &inc : allMatches(xml, std::regex("<ClCompile\\s+Include=\"([^\"]+)\""))) {
        std::string p = inc;
        for (char &c : p) if (c == '\\') c = '/';
        // Relative to the project's own directory.
        const std::string norm = (fs::path(project) / p).lexically_normal().generic_string();
        if (isCompiledSource(norm)) out.insert(norm);
    }
    return out;
}

std::set<std::string> cmakeSources(const std::string &target) {
    const std::string cm = stripHashComments(readFile("CMakeLists.txt"));
    // add_library(<target> ...) or add_executable(<target> ...), whole-word.
    const std::regex decl("add_(?:library|executable)\\s*\\(\\s*" + target + "(?![A-Za-z0-9_])");
    std::smatch m;
    if (!std::regex_search(cm, m, decl)) return {};
    const size_t open = cm.find('(', static_cast<size_t>(m.position(0)));
    const size_t close = matchingClose(cm, open, '(', ')');
    if (close == std::string::npos) return {};
    std::set<std::string> out;
    for (std::string tok : allMatches(cm.substr(open, close - open),
                                      std::regex("([A-Za-z0-9_${}/\\.\\-]+\\.(?:cpp|c))(?![A-Za-z0-9_])"))) {
        const std::string prefix = "${CMAKE_SOURCE_DIR}/";
        if (tok.compare(0, prefix.size(), prefix) == 0) tok.erase(0, prefix.size());
        if (tok.find("${") != std::string::npos) continue;  // unresolved variable: can't compare
        out.insert(fs::path(tok).lexically_normal().generic_string());
    }
    return out;
}

}  // namespace

TEST(BackendConsistency, SourceTreeIsReachableFromTestFile) {
    // Everything below reads the repo's own source files via __FILE__. If this
    // ever fails, the checks below would pass vacuously - fail loudly instead.
    ASSERT_TRUE(fs::exists(repoRoot() / "CMakeLists.txt"))
        << "expected the repo root at " << repoRoot().string();
    ASSERT_TRUE(fs::exists(repoRoot() / "gpu" / "metal" / "metal_poc.mm"));
}

TEST(BackendConsistency, MetalSupportedSceneListMatchesDispatcher) {
    const std::set<std::string> listed = metalListedScenes();
    const std::set<std::string> dispatched = metalDispatchedScenes();

    // Parser sanity: the real lists have ~80 entries each. A near-empty result
    // means the file's shape changed under this parser, not that they agree.
    ASSERT_GT(listed.size(), 40u) << "could not parse cpu_scene_metal_hand_authored_supported()'s list";
    ASSERT_GT(dispatched.size(), 40u) << "could not parse MetalPocApp::buildHandAuthoredScene()";

    EXPECT_TRUE(minus(listed, dispatched).empty())
        << "Listed as Metal-supported in cpu_renderer/cpu_interface.cpp but MetalPocApp::"
           "buildHandAuthoredScene() (gpu/metal/metal_poc.mm) has no builder for them - the "
           "GUI/CLI would offer these scenes on Metal and then fail at render time: "
        << joined(minus(listed, dispatched));
    EXPECT_TRUE(minus(dispatched, listed).empty())
        << "Have a Metal builder in buildHandAuthoredScene() but are missing from "
           "cpu_scene_metal_hand_authored_supported() - dead code, the GUI/CLI will never "
           "offer them on Metal: "
        << joined(minus(dispatched, listed));
}

TEST(BackendConsistency, CpuSideSourceFilesMatchAcrossBuildSystems) {
    // CPU-side = the targets every platform builds. The GPU backends (optix_renderer,
    // realtime_renderer, metal_*) exist in only one build system by design.
    std::set<std::string> msbuild, cmake;
    for (const char *p : {"cpu_renderer", "launcher", "scene_metadata"}) {
        const std::set<std::string> s = msbuildSources(p);
        ASSERT_FALSE(s.empty()) << "could not parse " << p << "/" << p << ".vcxproj";
        msbuild.insert(s.begin(), s.end());
    }
    // The launcher is the `ray_tracer` executable in the CMake build.
    for (const char *t : {"cpu_renderer", "ray_tracer", "scene_metadata"}) {
        const std::set<std::string> s = cmakeSources(t);
        ASSERT_FALSE(s.empty()) << "could not parse CMake target " << t;
        cmake.insert(s.begin(), s.end());
    }

    EXPECT_TRUE(minus(msbuild, cmake).empty())
        << "Compiled by the Windows MSBuild projects but by NO CPU-side target in the root "
           "CMakeLists.txt - the macOS/Linux build will fail to link (this is the PR #1 bug "
           "class): "
        << joined(minus(msbuild, cmake));
    EXPECT_TRUE(minus(cmake, msbuild).empty())
        << "Compiled by the root CMakeLists.txt but by NO CPU-side MSBuild project - the "
           "Windows build is missing it: "
        << joined(minus(cmake, msbuild));
}

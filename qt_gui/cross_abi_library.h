#ifndef CROSS_ABI_LIBRARY_H
#define CROSS_ABI_LIBRARY_H

#include <QString>
#include <QDir>

#ifdef Q_OS_WIN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// ============================================================================
// cross_abi_library
// ============================================================================
// Shared LoadLibrary/GetProcAddress (Windows) / dlopen/dlsym (else) helpers
// for loading an MSVC-built DLL from this MinGW-built Qt binary - x64
// Windows has one calling convention regardless of compiler, so a MinGW
// process CAN call an extern "C" export dynamically even though it can't
// link the MSVC .lib directly (see qt_gui/scene_metadata_client.cpp's own
// header comment for the fuller rationale).
//
// Used by scene_metadata_client.cpp (scene_metadata.dll) and
// realtime_preview_session.cpp (realtime_renderer.dll) - each still keeps
// its own DllHandle-shaped struct, typed function-pointer list, and
// std::call_once-guarded singleton (their symbol lists and error-handling
// policy on partial resolution genuinely differ); this header only holds
// the three load/lookup/close primitives both of them need, so a future
// fix to those (e.g. a load-failure diagnostic, a new platform) lands once
// instead of needing to be hand-copied a third time.
// ============================================================================
namespace cross_abi_library {

// HMODULE (Windows) and dlopen()'s return type are both opaque handles that
// fit in a void* - returned as void* so callers need no platform-conditional
// field for it, only these three functions do.
#ifdef Q_OS_WIN
inline void* loadLibrary(const QString& dir, const QString& filename) {
	const QString path = QDir(dir).filePath(filename);
	return static_cast<void*>(LoadLibraryW(reinterpret_cast<const wchar_t*>(path.utf16())));
}
inline void* lookupSymbol(void* module, const char* name) {
	return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(module), name));
}
inline void closeLibrary(void* module) {
	FreeLibrary(static_cast<HMODULE>(module));
}
#else
inline void* loadLibrary(const QString& dir, const QString& filename) {
	const QString path = QDir(dir).filePath(filename);
	return dlopen(path.toUtf8().constData(), RTLD_NOW | RTLD_LOCAL);
}
inline void* lookupSymbol(void* module, const char* name) {
	return dlsym(module, name);
}
inline void closeLibrary(void* module) {
	dlclose(module);
}
#endif

} // namespace cross_abi_library

#endif // CROSS_ABI_LIBRARY_H

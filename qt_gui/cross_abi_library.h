#ifndef CROSS_ABI_LIBRARY_H
#define CROSS_ABI_LIBRARY_H

#include <QString>
#include <QDir>

#ifdef Q_OS_WIN
// Without this, windows.h's own min/max macros text-substitute any later
// bare max(/min( in a translation unit that includes this header - e.g.
// realtime_preview_session.cpp's std::min(...) calls, which broke under
// MSVC (error C2589, "std::" followed by the macro-expanded token) but
// happened to compile under MinGW's qmake spec (which defines NOMINMAX by
// default). Same guard already used for the same reason in every CUDA .cu
// file in this project - see e.g. wavefront_device_helpers.h's own.
#ifndef NOMINMAX
#define NOMINMAX
#endif
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
// Set by loadLibrary() only on FAILURE (captured immediately, before any
// other call could reset the underlying dlerror()/GetLastError() state) -
// see lastLoadError()'s own comment below for why this exists at all.
inline QString& loadErrorStorage() {
	static QString err;
	return err;
}

// The real reason the MOST RECENT loadLibrary() call in THIS process
// failed (across every caller - scene_metadata_client.cpp and
// realtime_preview_session.cpp share this one slot, so a caller must read
// it immediately after its own failed loadLibrary() call, before any other
// loadLibrary() call from anywhere else could overwrite it). Added because
// every caller used to just check "is the handle null" and print a
// GUESSED reason ("make sure the file is present") with no actual
// diagnosis - a real gap found when a user's own dylib failed to load for
// an entirely different reason (a macOS Gatekeeper/quarantine block on an
// ad-hoc-signed, downloaded dylib) that message actively misled them
// about. Empty string if the last loadLibrary() call succeeded or none has
// been made yet.
inline QString lastLoadError() { return loadErrorStorage(); }

#ifdef Q_OS_WIN
inline void* loadLibrary(const QString& dir, const QString& filename) {
	const QString path = QDir(dir).filePath(filename);
	void* handle = static_cast<void*>(LoadLibraryW(reinterpret_cast<const wchar_t*>(path.utf16())));
	if (!handle) {
		const DWORD code = GetLastError();
		LPWSTR msgBuf = nullptr;
		FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
			nullptr, code, 0, reinterpret_cast<LPWSTR>(&msgBuf), 0, nullptr);
		loadErrorStorage() = msgBuf ? QString::fromWCharArray(msgBuf).trimmed()
		                            : QStringLiteral("Windows error code %1").arg(code);
		if (msgBuf) LocalFree(msgBuf);
	}
	return handle;
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
	void* handle = dlopen(path.toUtf8().constData(), RTLD_NOW | RTLD_LOCAL);
	if (!handle) {
		const char* err = dlerror();
		loadErrorStorage() = err ? QString::fromUtf8(err)
		                         : QStringLiteral("dlopen() failed with no dlerror() message");
	}
	return handle;
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

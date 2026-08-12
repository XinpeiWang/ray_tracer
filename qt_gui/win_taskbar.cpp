#include "win_taskbar.h"

// QtGlobal must come first - it is what defines Q_OS_WIN.
#include <QtGlobal>

#if defined(Q_OS_WIN)

#include <QWidget>

// <shobjidl.h> pulls in the ITaskbarList3 definition. objbase.h gives us
// CoCreateInstance/CoInitializeEx.
#include <objbase.h>
#include <shobjidl.h>

namespace win_taskbar {
namespace {

ITaskbarList3 *g_taskbar = nullptr;
bool g_initAttempted = false;

// ITaskbarList3::SetProgressValue takes a completed/total pair rather than a
// percentage; a 10000-step denominator gives two decimal places of resolution,
// matching what Blender's GHOST layer uses.
constexpr ULONGLONG kProgressScale = 10000;

TBPFLAG toFlag(State state) {
	switch (state) {
	case State::NoProgress:    return TBPF_NOPROGRESS;
	case State::Indeterminate: return TBPF_INDETERMINATE;
	case State::Normal:        return TBPF_NORMAL;
	case State::Error:         return TBPF_ERROR;
	case State::Paused:        return TBPF_PAUSED;
	}
	return TBPF_NOPROGRESS;
}

HWND handleOf(QWidget *window) {
	// winId() is the public route to the native handle. Qt Creator reaches for
	// a private platform-interface header instead, which we deliberately avoid.
	return window ? reinterpret_cast<HWND>(window->winId()) : nullptr;
}

} // namespace

void init() {
	if (g_initAttempted) return;
	g_initAttempted = true;

	// The window must already exist for the taskbar button to be registered.
	// Strictly, Windows documents waiting for the "TaskbarButtonCreated"
	// message before calling HrInit; in practice creating the interface once
	// the window is shown works, and is what Qt Creator does.
	//
	// The result is deliberately discarded: Qt has normally already
	// initialised COM on the GUI thread, which returns S_FALSE or
	// RPC_E_CHANGED_MODE rather than a usable failure. Either way the
	// CoCreateInstance below is the real test of whether this can work, so
	// there is nothing useful to branch on here.
	(void)CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);

	if (FAILED(CoCreateInstance(CLSID_TaskbarList, nullptr, CLSCTX_INPROC_SERVER,
								IID_PPV_ARGS(&g_taskbar)))) {
		g_taskbar = nullptr;
		return;
	}
	if (FAILED(g_taskbar->HrInit())) {
		g_taskbar->Release();
		g_taskbar = nullptr;
	}
}

void setProgress(QWidget *window, double fraction) {
	if (!g_taskbar) return;
	HWND hwnd = handleOf(window);
	if (!hwnd) return;
	fraction = qBound(0.0, fraction, 1.0);
	// Note SetProgressValue implicitly switches the button to TBPF_NORMAL and
	// clears TBPF_INDETERMINATE, so there's no need to set the state as well.
	g_taskbar->SetProgressValue(hwnd, static_cast<ULONGLONG>(fraction * kProgressScale),
								kProgressScale);
}

void setState(QWidget *window, State state) {
	if (!g_taskbar) return;
	HWND hwnd = handleOf(window);
	if (!hwnd) return;
	g_taskbar->SetProgressState(hwnd, toFlag(state));
}

void shutdown() {
	if (g_taskbar) {
		g_taskbar->Release();
		g_taskbar = nullptr;
	}
}

} // namespace win_taskbar

#else  // !Q_OS_WIN

namespace win_taskbar {
void init() {}
void setProgress(QWidget *, double) {}
void setState(QWidget *, State) {}
void shutdown() {}
} // namespace win_taskbar

#endif

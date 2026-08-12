#ifndef WIN_TASKBAR_H
#define WIN_TASKBAR_H

class QWidget;

// ============================================================================
// Windows taskbar progress
// ============================================================================
// Qt 6 has NO API for this. QWinTaskbarButton/QWinTaskbarProgress lived in
// QtWinExtras, which was removed in Qt 6 "due to warranting a cross-platform
// solution" (doc.qt.io/qt-6/extras-changes-qt6.html); the replacement bugs
// QTBUG-94008 / QTBUG-94009 are both still open. So the only route is calling
// ITaskbarList3 through COM directly - which is exactly what Qt Creator does
// (coreplugin/progressmanager/progressmanager_win.cpp) and what Blender does
// in GHOST_WindowWin32.cc.
//
// Every function here is a no-op on non-Windows builds, so callers need no
// #ifdef of their own.
// ============================================================================
namespace win_taskbar {

enum class State {
	NoProgress,     // clears the bar - MUST be set when a job ends, or it sticks
	Indeterminate,  // marquee, for phases with no measurable progress
	Normal,
	Error,          // red
	Paused,         // yellow
};

// Safe to call more than once; does nothing if the platform integration is
// unavailable. Call after the window has been shown.
void init();

void setProgress(QWidget *window, double fraction);  // fraction in [0, 1]
void setState(QWidget *window, State state);

void shutdown();

} // namespace win_taskbar

#endif // WIN_TASKBAR_H

#ifndef CRASH_RECOVERY_H
#define CRASH_RECOVERY_H

// After a session that did not exit cleanly (app_log.h notices: a crash, a kill, a power cut), the next start asks what to do before anything that could be the
// cause is loaded. The default is to carry on as normal - the unsaved Scene Builder scene is restored, which is usually exactly what a user wants after a
// crash. "Start fresh" is for a program that keeps crashing at start: it puts the window layout back to the default and sets the unsaved scene aside (renamed,
// never deleted), so a bad scene or a bad saved position cannot crash it again. "Show Log Folder" opens the log, which ends with what the program was doing.

#include <QString>

namespace crash_recovery {

// Shows the question if the previous session ended unexpectedly. Call after QApplication and the translators exist, before MainWindow is created.
// Does nothing under the self-test, or when RAY_TRACER_NO_CRASH_PROMPT is set.
void offerAfterUncleanExit();

// What "Start fresh" does (also called by the self-test): forgets the saved window and pane layout, sets the unsaved scene aside.
// Returns the path the unsaved scene was moved to ("" if there was none).
QString startFresh();

}  // namespace crash_recovery

#endif  // CRASH_RECOVERY_H

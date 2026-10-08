#ifndef APP_LOG_H
#define APP_LOG_H

// The GUI's log file: everything the program does that could matter when something goes wrong, kept on disk (the Log Output tab is only the current session's
// render output and is gone when the program closes - or crashes). One line per entry, timestamped, in
//   macOS:    ~/Library/Logs/Ray Tracer/ray_tracer_gui.log
//   Windows:  %APPDATA%/Ray Tracer Project/Ray Tracer/logs/ray_tracer_gui.log   (Qt's AppDataLocation)
// RAY_TRACER_LOG_DIR names another folder. The file rotates at 5 MB (three old ones are kept) and never leaves the computer; "Help > Show Log Folder" opens it.
//
// What goes in: a header per session (version, OS, CPU, Qt, paths), every user operation (clicks, menu choices, selections, typed values, shortcuts, dialogs - see
// ui_logger.h) and the Scene Builder's edits, the Log Output pane's own lines (render commands and output), Qt's warnings, whatever the renderer libraries print on
// stderr while they run inside the program, and a note when the previous session did not exit cleanly.
//
// Thread-safe, never throws, and a failure to write is silent (a log must not be the reason the program stops).

#include <QString>
#include <QStringList>

#include "../src/shared/log_format.h"

namespace AppLog {

// Call once, right after QApplication is created. Opens the file, writes the session header, installs the Qt message handler and (when stderr is not a
// terminal) sends the process's stderr into the file too.
void init();

void write(log_format::Level level, const QString &category, const QString &message);
inline void debug(const QString &category, const QString &message) { write(log_format::Level::Debug, category, message); }
inline void info(const QString &category, const QString &message) { write(log_format::Level::Info, category, message); }
inline void warn(const QString &category, const QString &message) { write(log_format::Level::Warn, category, message); }
inline void error(const QString &category, const QString &message) { write(log_format::Level::Error, category, message); }

QString filePath();     // the current log file ("" if the folder could not be made)
QString folderPath();
QStringList tail(int lines);   // the last lines of the file, for a report
bool previousSessionEndedUnexpectedly();

// Writes the "session ended" line and removes the running marker; connected to aboutToQuit by init().
void shutdown();

}  // namespace AppLog

#endif  // APP_LOG_H

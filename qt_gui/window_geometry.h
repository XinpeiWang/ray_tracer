#ifndef WINDOW_GEOMETRY_H
#define WINDOW_GEOMETRY_H

// Remembers where the main window was and how the Scene Builder's panes were divided, between runs (QSettings, settings_keys.h's location). A saved position is
// used only if its title bar is still reachable on a connected screen (src/shared/window_placement.h): after a monitor is unplugged the window comes back on
// the remaining one, centred, at the default size.

#include <QByteArray>

class QWidget;
class QSplitter;

namespace window_geometry {

// Applies the saved geometry (size, position, maximized) to `window`. Returns false - and leaves the window as it is - when nothing is saved or the saved
// place can no longer be reached; the caller then uses its default.
bool restore(QWidget *window);
void save(const QWidget *window);
void forget();
void forgetSplitters();   // the Scene Builder's pane sizes ("builder/...")   // "Reset Window Layout": the next start uses the defaults too

// A splitter's pane sizes under `key` (e.g. "builder/mainSplit"). restore() ignores a saved state that does not fit the splitter (wrong pane count).
void restoreSplitter(QSplitter *splitter, const char *key);
void saveSplitter(const QSplitter *splitter, const char *key);

}  // namespace window_geometry

#endif  // WINDOW_GEOMETRY_H

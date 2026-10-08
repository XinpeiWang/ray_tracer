# The program's log file

When something goes wrong, the most useful thing to send with a bug report is the log file. The GUI writes one, on disk, for every session; the **Log Output** tab is only the current render's output and is gone when the program closes.

**Where it is:** *Help > Show Log Folder* opens it. The file is `ray_tracer_gui.log` in

| System | Folder |
|---|---|
| macOS | `~/Library/Logs/Ray Tracer/` |
| Windows | `%APPDATA%\Ray Tracer Project\Ray Tracer\logs\` |

`RAY_TRACER_LOG_DIR` names another folder. The file rotates at 5 MB (`ray_tracer_gui.log.1` to `.3` are the older ones). It stays on your computer; nothing is sent anywhere. The Diagnostics report ends with a *Log file* section naming it.

## What is in it

One line per entry, so `grep` and `tail` work: `2026-10-07 22:08:58.028 [INFO ] builder: edit: +1 object (Cylinder)`.

| Category | What |
|---|---|
| `session` | A header per run: version, macOS/Windows version, CPU architecture and count, Qt version, program path, working folder, the per-user folders, the settings file, the language. And, if the previous run did not exit cleanly (crash, kill, power loss), a warning saying so: the lines just above this run's header are its last actions. |
| `ui` | Every user operation: button clicks (with the new state of a checkbox), combo box choices, typed values (when editing finishes and only if they changed), sliders, tab changes, list and table clicks, menu choices, keyboard shortcuts, and every dialog that opens (a message box with its text, which is usually the error you saw). Each names the control as you see it and the tab it is on. Password fields are never logged. |
| `builder` | The Scene Builder's work as readable differences: `edit: object 'Ball': position -1.3,1,0.4 -> 1.25,1,-1`, `edit: +5 objects (Table top, ...)`, `undo:`, `redo:`, new, open, save, add to scene list (with the path written). A drag or typing is one entry, written when it pauses (and always before a render starts, and when the program quits). A deleted or added item is named (`-1 object (Sphere)`). Also: `select: object 'Ball' (list)` / `(layout view)` / `(3D view)`, `3D view: tool Rotate (E key)`, `scene problem - note: ...` (each problem once, when it appears), `3D view: cannot show the mesh <file>: <why>`, `restored the unsaved scene from ...` (or why it could not be), and a failing backup write. |
| `builder-render` | The command line of each Scene Builder preview or final render, the scene notes, the result (exit code, time, picture size), and on a failure the last lines of the renderer's output. |
| `pane` | Every line of the Log Output tab: the render command, the renderer's output (one in 25 of the per-scanline progress lines), downloads, thumbnails, Live Preview settings. |
| `startup` | Where the start-up time goes: one line per stage (QApplication, translators, building each tab, theme and font, showing the window), indented by nesting, with how long it took and when it ended, then `window up N ms after the program started; slowest stages: ...` (a warning past 3 s). A slow start shows up in any bug report's log. |
| `queue` | The Progress tab's render queue: each job queued, started, finished (done / failed / cancelled, with the time and reason), moved, removed, cleared. |
| `window` | The saved window layout restored, refused (its screen is gone) or reset. |
| `diagnostics` | The Diagnostics report, line by line. |
| `photo`, `photo-install` | Object from a photo (the photo, the result or error) and the photo helper installer's output. |
| `qt` and other Qt categories | Qt's own warnings. |
| (no category) | Anything the renderer libraries print on stderr while they run inside the program (Live Preview, the Metal and OptiX backends), when the program was not started from a terminal. |

## After a crash

If the previous run did not exit cleanly, the next start asks before it loads anything: **Start normally** (the default; the unsaved Scene Builder scene is restored), **Start fresh** (resets the window layout and sets the unsaved scene aside - renamed to `scene_builder_autosave.set-aside-<time>.pbrt`, never deleted - for a program that keeps stopping at start), or **Show Log Folder**. `RAY_TRACER_NO_CRASH_PROMPT=1` skips the question.

## For developers

* `qt_gui/app_log.h` is the API (`AppLog::info("category", "text")`); `qt_gui/ui_logger.h` hooks every control as it is first shown, so a new control is logged without any code. `src/shared/log_format.h` (one-line entries, rotation) and `src/shared/scene_doc_diff.h` (the builder's change descriptions) are Qt-free and unit-tested.
* The GUI self-test (`scripts/gui_selftest.py`, builder mode) checks that the file exists and has the header, a tab change, a click, edits, undo and redo, save and open, a selection in the list, a middle item named when deleted, the 3D tool keys, a scene problem, an unreadable mesh, and the last edit before the widget is destroyed.
* Where you add a user-visible operation that does not go through a control (a drop, a file chosen in a native dialog), log it explicitly.

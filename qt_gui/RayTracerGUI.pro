# svg is required for QIcon to load the SVG icons in resources.qrc. Without
# the module (and without imageformats/qsvg.dll alongside the exe) QIcon
# silently renders nothing at all rather than reporting an error.
# multimedia/multimediawidgets back the Preview tab's embedded QVideoWidget/
# QMediaPlayer playback for video-mode renders (see assembleVideoAutomatically()
# in mainwindow_slots.cpp) - needs Qt6Multimedia.dll/Qt6MultimediaWidgets.dll
# plus the ffmpeg backend plugin/runtime DLLs deployed alongside the exe.
QT += core gui widgets svg multimedia multimediawidgets

CONFIG += c++17

RESOURCES += resources.qrc

# Translations: source text is English, written directly into every tr()
# call - see language_switch.cpp's own comment for why switching is
# restart-to-apply rather than live. lrelease/embed_translations are qmake's
# built-in features: lrelease auto-compiles each .ts to a .qm at build time,
# and embed_translations packages the .qm files as Qt resources under an
# "i18n" prefix (:/i18n/raytracer_<code>.qm), NOT ":/translations/..." even
# though the source .ts files live in translations/ - confirmed against the
# generated release/qmake_qmake_qm_files.qrc, and load-bearing: main.cpp's
# QTranslator::load() call has to use the same ":/i18n/..." path. No manual
# resources.qrc entry needed. Regenerate the .ts files after adding/changing
# tr() calls with:
#   lupdate RayTracerGUI.pro
TRANSLATIONS += translations/raytracer_zh_CN.ts \
                translations/raytracer_es.ts \
                translations/raytracer_ja.ts \
                translations/raytracer_fr.ts
CONFIG += lrelease embed_translations

TARGET = RayTracerGUI
TEMPLATE = app

# Output directory
DESTDIR = $$PWD/../RayTracer_Package

# Source files
SOURCES += \
	main.cpp \
	mainwindow.cpp \
	mainwindow_tabs.cpp \
	mainwindow_tabs_render.cpp \
	mainwindow_tabs_output.cpp \
	mainwindow_style.cpp \
	mainwindow_slots.cpp \
	mainwindow_actions.cpp \
	icon_tint.cpp \
	palette_data.cpp \
	palette_file.cpp \
	theme_load.cpp \
	scene_metadata_client.cpp \
	theme.cpp \
	theme_switch.cpp \
	language_switch.cpp \
	font_switch.cpp \
	recent_renders.cpp \
	win_taskbar.cpp

HEADERS += \
	mainwindow.h \
	mainwindow_widgets.h \
	mainwindow_jobtypes.h \
	camera_math.h \
	icon_tint.h \
	palette_data.h \
	palette_file.h \
	render_output_parser.h \
	scene_metadata_client.h \
	scene_technique_notes.h \
	theme.h \
	win_taskbar.h

# scene_metadata.dll/.dylib/.so (loaded dynamically at runtime - see
# scene_metadata_client.cpp's LoadLibrary/dlopen branches) is not linked
# here; it just needs to end up alongside the GUI binary, which the
# Windows post-build step (RayTracer_Package) or the equivalent macOS
# packaging step handles.

# Platform-specific settings
win32 {
	# Windows specific flags - removed /std:c++17 as it's MSVC-specific
	# MinGW uses -std=c++17 automatically from CONFIG += c++17

	# Add application icon
	RC_ICONS = app_icon.ico
	RC_FILE = app_icon.rc

	# win_taskbar.cpp drives the taskbar progress button through ITaskbarList3.
	# Qt 6 has no API for this - QtWinExtras (QWinTaskbarProgress) was removed
	# and its replacement bugs are still open - so it goes through COM directly,
	# which needs ole32 for CoCreateInstance/CoInitializeEx.
	LIBS += -lole32

	# Only the Windows build links against a scene_metadata.dll built by the
	# CUDA/OptiX-enabled MSBuild launcher (ray_tracer.exe, RT_HAVE_OPTIX - see
	# launcher.vcxproj). RT_GUI_HAVE_GPU tells mainwindow.cpp's setup code the
	# "Use GPU" toggle is meaningful here; it's absent on every other platform
	# scope below, where the CLI this GUI launches is always the CPU-only
	# CMake build (root CMakeLists.txt's ray_tracer target, no RT_HAVE_OPTIX).
	DEFINES += RT_GUI_HAVE_GPU
}

macx {
	# No .icns is generated yet - app_icon.png (already in the repo) is the
	# source to run through iconutil/sips if/when a polished bundle icon is
	# wanted. Cosmetic only; omitting ICON just falls back to Qt's default.
	# ICON = app_icon.icns
}

# Default rules for deployment
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

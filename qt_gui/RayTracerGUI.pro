# svg is required for QIcon to load the SVG icons in resources.qrc. Without
# the module (and without imageformats/qsvg.dll alongside the exe) QIcon
# silently renders nothing at all rather than reporting an error.
QT += core gui widgets svg

CONFIG += c++17

RESOURCES += resources.qrc

TARGET = RayTracerGUI
TEMPLATE = app

# Output directory
DESTDIR = $$PWD/../RayTracer_Package

# Source files
SOURCES += \
	main.cpp \
	mainwindow.cpp \
	mainwindow_tabs.cpp \
	mainwindow_style.cpp \
	mainwindow_slots.cpp \
	mainwindow_actions.cpp \
	scene_metadata_client.cpp \
	win_taskbar.cpp

HEADERS += \
	mainwindow.h \
	scene_metadata_client.h \
	win_taskbar.h

# scene_metadata.dll (MSVC-built, loaded dynamically at runtime via
# LoadLibrary/GetProcAddress from kernel32 - see scene_metadata_client.cpp,
# already linked implicitly by MinGW) is not linked here; it just needs to
# end up alongside RayTracerGUI.exe in RayTracer_Package, which its own
# post-build step already handles.

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
}

# Default rules for deployment
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

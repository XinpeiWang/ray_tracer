QT += core gui widgets

CONFIG += c++17

TARGET = RayTracerGUI
TEMPLATE = app

# Output directory
DESTDIR = $$PWD/../RayTracer_Package

# Source files
SOURCES += \
	main.cpp \
	mainwindow.cpp

HEADERS += \
	mainwindow.h

# Platform-specific settings
win32 {
	# Windows specific flags - removed /std:c++17 as it's MSVC-specific
	# MinGW uses -std=c++17 automatically from CONFIG += c++17
}

# Default rules for deployment
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# QtTest for asset_downloader.cpp and scene_packs.cpp (see downloader_tests.cpp). Standalone: it needs only Qt, not the renderer.
#   cd qt_gui/tests && qmake && make && ./downloader_tests       (Windows: nmake / jom, downloader_tests.exe)
QT = core network testlib
CONFIG += console testcase c++17
CONFIG -= app_bundle
TEMPLATE = app
TARGET = downloader_tests

SOURCES += \
	downloader_tests.cpp \
	../asset_downloader.cpp \
	../scene_packs.cpp \
	../../src/external/miniz.c

HEADERS += \
	../asset_downloader.h \
	../scene_packs.h

INCLUDEPATH += ../../src/external

# Only the two generated data files the tests check (not the GUI's whole resources.qrc).
RESOURCES += downloader_tests.qrc

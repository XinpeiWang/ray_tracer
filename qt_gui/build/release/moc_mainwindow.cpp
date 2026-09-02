/****************************************************************************
** Meta object code from reading C++ file 'mainwindow.h'
**
** Created by: The Qt Meta Object Compiler version 69 (Qt 6.11.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../mainwindow.h"
#include <QtGui/qtextcursor.h>
#include <QtCore/qmetatype.h>

#include <QtCore/qtmochelpers.h>

#include <memory>


#include <QtCore/qxptype_traits.h>
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'mainwindow.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 69
#error "This file was generated using the moc from 6.11.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

#ifndef Q_CONSTINIT
#define Q_CONSTINIT
#endif

QT_WARNING_PUSH
QT_WARNING_DISABLE_DEPRECATED
QT_WARNING_DISABLE_GCC("-Wuseless-cast")
namespace {
struct qt_meta_tag_ZN10MainWindowE_t {};
} // unnamed namespace

template <> constexpr inline auto MainWindow::qt_create_metaobjectdata<qt_meta_tag_ZN10MainWindowE_t>()
{
    namespace QMC = QtMocConstants;
    QtMocHelpers::StringRefStorage qt_stringData {
        "MainWindow",
        "onRenderClicked",
        "",
        "onStopClicked",
        "onQualityPresetChanged",
        "index",
        "onCameraPresetChanged",
        "onVideoPresetChanged",
        "onCameraDistanceChanged",
        "distance",
        "onSceneChanged",
        "onModeChanged",
        "onIntegratorChanged",
        "onProgressUpdate",
        "percentage",
        "onRenderComplete",
        "success",
        "message",
        "totalTime",
        "outputPath",
        "onLogMessage",
        "onElapsedTick",
        "onRemoveSelectedQueueItem",
        "onClearQueue",
        "onRunDiagnosticsClicked",
        "onDiagnosticsReportReady",
        "report",
        "onDiagnosticsFailed",
        "onGenerateThumbnailsClicked",
        "onThumbnailReady",
        "sceneId",
        "onThumbnailsAllDone",
        "copyLogToClipboard",
        "saveLogToFile",
        "clearLog",
        "showAboutDialog",
        "copyDiagToClipboard",
        "saveDiagReportToFile"
    };

    QtMocHelpers::UintData qt_methods {
        // Slot 'onRenderClicked'
        QtMocHelpers::SlotData<void()>(1, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onStopClicked'
        QtMocHelpers::SlotData<void()>(3, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onQualityPresetChanged'
        QtMocHelpers::SlotData<void(int)>(4, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onCameraPresetChanged'
        QtMocHelpers::SlotData<void(int)>(6, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onVideoPresetChanged'
        QtMocHelpers::SlotData<void(int)>(7, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onCameraDistanceChanged'
        QtMocHelpers::SlotData<void(double)>(8, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Double, 9 },
        }}),
        // Slot 'onSceneChanged'
        QtMocHelpers::SlotData<void(int)>(10, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onModeChanged'
        QtMocHelpers::SlotData<void(int)>(11, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onIntegratorChanged'
        QtMocHelpers::SlotData<void(int)>(12, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 5 },
        }}),
        // Slot 'onProgressUpdate'
        QtMocHelpers::SlotData<void(int)>(13, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Int, 14 },
        }}),
        // Slot 'onRenderComplete'
        QtMocHelpers::SlotData<void(bool, const QString &, double, const QString &)>(15, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::Bool, 16 }, { QMetaType::QString, 17 }, { QMetaType::Double, 18 }, { QMetaType::QString, 19 },
        }}),
        // Slot 'onLogMessage'
        QtMocHelpers::SlotData<void(const QString &)>(20, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 17 },
        }}),
        // Slot 'onElapsedTick'
        QtMocHelpers::SlotData<void()>(21, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRemoveSelectedQueueItem'
        QtMocHelpers::SlotData<void()>(22, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onClearQueue'
        QtMocHelpers::SlotData<void()>(23, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onRunDiagnosticsClicked'
        QtMocHelpers::SlotData<void()>(24, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onDiagnosticsReportReady'
        QtMocHelpers::SlotData<void(const QString &)>(25, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 26 },
        }}),
        // Slot 'onDiagnosticsFailed'
        QtMocHelpers::SlotData<void(const QString &)>(27, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 17 },
        }}),
        // Slot 'onGenerateThumbnailsClicked'
        QtMocHelpers::SlotData<void()>(28, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'onThumbnailReady'
        QtMocHelpers::SlotData<void(const QString &, bool, const QString &)>(29, 2, QMC::AccessPrivate, QMetaType::Void, {{
            { QMetaType::QString, 30 }, { QMetaType::Bool, 16 }, { QMetaType::QString, 19 },
        }}),
        // Slot 'onThumbnailsAllDone'
        QtMocHelpers::SlotData<void()>(31, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'copyLogToClipboard'
        QtMocHelpers::SlotData<void()>(32, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'saveLogToFile'
        QtMocHelpers::SlotData<void()>(33, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'clearLog'
        QtMocHelpers::SlotData<void()>(34, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'showAboutDialog'
        QtMocHelpers::SlotData<void()>(35, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'copyDiagToClipboard'
        QtMocHelpers::SlotData<void()>(36, 2, QMC::AccessPrivate, QMetaType::Void),
        // Slot 'saveDiagReportToFile'
        QtMocHelpers::SlotData<void()>(37, 2, QMC::AccessPrivate, QMetaType::Void),
    };
    QtMocHelpers::UintData qt_properties {
    };
    QtMocHelpers::UintData qt_enums {
    };
    return QtMocHelpers::metaObjectData<MainWindow, qt_meta_tag_ZN10MainWindowE_t>(QMC::MetaObjectFlag{}, qt_stringData,
            qt_methods, qt_properties, qt_enums);
}
Q_CONSTINIT const QMetaObject MainWindow::staticMetaObject = { {
    QMetaObject::SuperData::link<QMainWindow::staticMetaObject>(),
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.stringdata,
    qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.data,
    qt_static_metacall,
    nullptr,
    qt_staticMetaObjectRelocatingContent<qt_meta_tag_ZN10MainWindowE_t>.metaTypes,
    nullptr
} };

void MainWindow::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    auto *_t = static_cast<MainWindow *>(_o);
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: _t->onRenderClicked(); break;
        case 1: _t->onStopClicked(); break;
        case 2: _t->onQualityPresetChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 3: _t->onCameraPresetChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 4: _t->onVideoPresetChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 5: _t->onCameraDistanceChanged((*reinterpret_cast<std::add_pointer_t<double>>(_a[1]))); break;
        case 6: _t->onSceneChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 7: _t->onModeChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 8: _t->onIntegratorChanged((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 9: _t->onProgressUpdate((*reinterpret_cast<std::add_pointer_t<int>>(_a[1]))); break;
        case 10: _t->onRenderComplete((*reinterpret_cast<std::add_pointer_t<bool>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<double>>(_a[3])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[4]))); break;
        case 11: _t->onLogMessage((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 12: _t->onElapsedTick(); break;
        case 13: _t->onRemoveSelectedQueueItem(); break;
        case 14: _t->onClearQueue(); break;
        case 15: _t->onRunDiagnosticsClicked(); break;
        case 16: _t->onDiagnosticsReportReady((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 17: _t->onDiagnosticsFailed((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1]))); break;
        case 18: _t->onGenerateThumbnailsClicked(); break;
        case 19: _t->onThumbnailReady((*reinterpret_cast<std::add_pointer_t<QString>>(_a[1])),(*reinterpret_cast<std::add_pointer_t<bool>>(_a[2])),(*reinterpret_cast<std::add_pointer_t<QString>>(_a[3]))); break;
        case 20: _t->onThumbnailsAllDone(); break;
        case 21: _t->copyLogToClipboard(); break;
        case 22: _t->saveLogToFile(); break;
        case 23: _t->clearLog(); break;
        case 24: _t->showAboutDialog(); break;
        case 25: _t->copyDiagToClipboard(); break;
        case 26: _t->saveDiagReportToFile(); break;
        default: ;
        }
    }
}

const QMetaObject *MainWindow::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->dynamicMetaObject() : &staticMetaObject;
}

void *MainWindow::qt_metacast(const char *_clname)
{
    if (!_clname) return nullptr;
    if (!strcmp(_clname, qt_staticMetaObjectStaticContent<qt_meta_tag_ZN10MainWindowE_t>.strings))
        return static_cast<void*>(this);
    return QMainWindow::qt_metacast(_clname);
}

int MainWindow::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 27)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 27;
    }
    if (_c == QMetaObject::RegisterMethodArgumentMetaType) {
        if (_id < 27)
            *reinterpret_cast<QMetaType *>(_a[0]) = QMetaType();
        _id -= 27;
    }
    return _id;
}
QT_WARNING_POP

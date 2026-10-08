// photo_import.h - the Scene Builder's "Object from a photo": finds the optional photo-to-mesh helper and runs it.
//
// The helper (tools/photo_to_mesh/photo_to_mesh.py, a TripoSR model run in its own Python environment made by
// scripts/setup_photo_to_mesh.ps1) is NOT part of the program: it needs PyTorch and about 5 GB. Everything here is
// about finding it and running it as a subprocess, so the GUI never links against any of it and a missing helper is
// just a message with the setup steps.
#pragma once

#include <QObject>
#include <QProcess>
#include <QString>
#include <QStringList>

namespace photo_import {

struct Setup {
	bool ready = false;
	QString python;   // the environment's python
	QString script;   // photo_to_mesh.py
	QString why;      // when not ready: what is missing, in a sentence
};

// <per-user data>/RayTracerPhoto (on Windows %LOCALAPPDATA%\RayTracerPhoto): where scripts/setup_photo_to_mesh.ps1 puts the Python environment.
QString environmentFolder();
// <per-user data>/photo_meshes: where results are kept (a scene saved with a photo object points into it).
QString resultsFolder();
// Looks for the helper. RAY_TRACER_PHOTO3D_PYTHON overrides the environment's python (for another install).
Setup locate();

// What `progress` lines look like: "PROGRESS 35 Guessing the 3D shape".
bool parseProgressLine(const QString &line, int *percent, QString *message);

}  // namespace photo_import

// The Diagnostics tab's "Photo helper" section: what the optional helper has and what is missing (script, Python environment,
// each package, the graphics card PyTorch sees, the model weights). The helper itself reports the Python side
// (photo_to_mesh.py --check); when there is no environment to ask, the section says so and how to set it up.
class PhotoHelperCheck : public QObject {
	Q_OBJECT
public:
	explicit PhotoHelperCheck(QObject *parent = nullptr) : QObject(parent) {}
	~PhotoHelperCheck() override;
	void start();
	bool isRunning() const { return m_running; }

signals:
	void finished(const QString &section);  // "=== Photo helper ... ===" followed by one "Key: value" fact per line

private:
	void finish(const QString &facts);

	QProcess *m_process = nullptr;
	QString m_head;
	bool m_running = false;
};

// Runs the helper on one photo. Emits progress while it works, then finished() once.
class PhotoToMeshJob : public QObject {
	Q_OBJECT
public:
	PhotoToMeshJob(const photo_import::Setup &setup, const QString &image, const QString &outFolder, bool removeBackground, QObject *parent = nullptr);
	~PhotoToMeshJob() override;
	void start();
	void cancel();

signals:
	void progress(int percent, const QString &message);
	// ok: mesh and texture are in outFolder. When not ok, `message` says why.
	void finished(bool ok, const QString &message);

private:
	void onOutput();
	void onFinished(int exitCode, QProcess::ExitStatus status);

	photo_import::Setup m_setup;
	QString m_image, m_out;
	bool m_removeBackground;
	QProcess *m_process = nullptr;
	QString m_pending;      // output not yet ended by a newline
	QString m_lastError;    // the helper's own "ERROR ..." line
	QString m_note;
	QStringList m_tail;     // the last few lines the helper printed that were not progress
	bool m_cancelled = false;
	bool m_done = false;
};

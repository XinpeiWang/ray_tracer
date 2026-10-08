#include "photo_import.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QStandardPaths>
#include <QStringList>

namespace photo_import {

QString environmentFolder() {
	// A short path on purpose: PyTorch and transformers have deep folders, and Windows stops at 260 characters.
	return QStandardPaths::writableLocation(QStandardPaths::GenericDataLocation) + "/RayTracerPhoto";
}

QString resultsFolder() {
	return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/photo_meshes";
}

static QString environmentPython() {
#ifdef Q_OS_WIN
	return environmentFolder() + "/venv/Scripts/python.exe";
#else
	return environmentFolder() + "/venv/bin/python";
#endif
}

static QString helperScript() {
	const QString app = QCoreApplication::applicationDirPath();
	for (const QString &c : {app + "/tools/photo_to_mesh/photo_to_mesh.py", app + "/../tools/photo_to_mesh/photo_to_mesh.py",
	                         app + "/../../tools/photo_to_mesh/photo_to_mesh.py", app + "/../../../tools/photo_to_mesh/photo_to_mesh.py"})
		if (QFileInfo::exists(c)) return QDir::cleanPath(c);
	return QString();
}

Setup locate() {
	Setup s;
	s.script = helperScript();
	const QString overridePython = qEnvironmentVariable("RAY_TRACER_PHOTO3D_PYTHON");
	s.python = overridePython.isEmpty() ? environmentPython() : overridePython;
	if (s.script.isEmpty())
		s.why = QObject::tr("The helper script (tools/photo_to_mesh/photo_to_mesh.py) was not found next to the program.");
	else if (!QFileInfo::exists(s.python))
		s.why = QObject::tr("The photo helper has not been set up on this computer yet.");
	else
		s.ready = true;
	return s;
}

bool parseProgressLine(const QString &line, int *percent, QString *message) {
	static const QString tag = QStringLiteral("PROGRESS ");
	if (!line.startsWith(tag)) return false;
	const QString rest = line.mid(tag.size());
	const int space = rest.indexOf(' ');
	bool ok = false;
	const int p = (space < 0 ? rest : rest.left(space)).toInt(&ok);
	if (!ok) return false;
	if (percent) *percent = qBound(0, p, 100);
	if (message) *message = space < 0 ? QString() : rest.mid(space + 1).trimmed();
	return true;
}

}  // namespace photo_import

PhotoToMeshJob::PhotoToMeshJob(const photo_import::Setup &setup, const QString &image, const QString &outFolder, bool removeBackground, QObject *parent)
    : QObject(parent), m_setup(setup), m_image(image), m_out(outFolder), m_removeBackground(removeBackground) {}

PhotoToMeshJob::~PhotoToMeshJob() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
}

void PhotoToMeshJob::start() {
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
	env.insert("PYTHONIOENCODING", "utf-8");
	env.insert("PYTHONUNBUFFERED", "1");
	m_process->setProcessEnvironment(env);
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() { onOutput(); });
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) { onFinished(code, st); });
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart && !m_done) {
			m_done = true;
			emit finished(false, tr("Could not start the photo helper (%1).").arg(m_setup.python));
		}
	});
	QStringList args{m_setup.script, m_image, "--out", m_out};
	if (!m_removeBackground) args << "--no-remove-bg";
	m_process->start(m_setup.python, args);
}

void PhotoToMeshJob::cancel() {
	m_cancelled = true;
	if (m_process) m_process->kill();
}

void PhotoToMeshJob::onOutput() {
	m_pending += QString::fromUtf8(m_process->readAllStandardOutput());
	int nl;
	while ((nl = m_pending.indexOf('\n')) >= 0) {
		const QString line = m_pending.left(nl).trimmed();
		m_pending.remove(0, nl + 1);
		int percent = 0;
		QString message;
		if (photo_import::parseProgressLine(line, &percent, &message))
			emit progress(percent, message);
		else if (line.startsWith("ERROR "))
			m_lastError = line.mid(6);
		else if (line.startsWith("NOTE "))
			m_note = line.mid(5);
		else if (!line.isEmpty()) {
			m_tail << line;  // the end of a Python traceback, when it crashes
			while (m_tail.size() > 6) m_tail.removeFirst();
		}
	}
}

void PhotoToMeshJob::onFinished(int exitCode, QProcess::ExitStatus status) {
	if (m_done) return;
	m_done = true;
	onOutput();
	if (m_cancelled) {
		emit finished(false, tr("Cancelled."));
		return;
	}
	const bool made = QFileInfo::exists(m_out + "/mesh.obj") && QFileInfo::exists(m_out + "/texture.png");
	if (status == QProcess::NormalExit && exitCode == 0 && made) {
		emit finished(true, m_note);
		return;
	}
	QString why = m_lastError;
	if (why.isEmpty()) why = tr("The helper stopped unexpectedly (exit code %1).").arg(exitCode) + (m_tail.isEmpty() ? QString() : "\n\n" + m_tail.join('\n'));
	emit finished(false, why);
}

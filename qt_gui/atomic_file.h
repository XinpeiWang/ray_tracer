#ifndef ATOMIC_FILE_H
#define ATOMIC_FILE_H

// Writes a file so a crash, a power cut or a full disk half-way through leaves the OLD file intact instead of a truncated one: the bytes go to a temporary
// file beside it, which replaces the target only when everything has been written (QSaveFile). Used for the files a user would be upset to lose: saved scenes,
// the Scene Builder's autosave, saved logs and reports.

#include <QByteArray>
#include <QSaveFile>
#include <QString>
#include <QtGlobal>

inline bool writeFileAtomically(const QString &path, const QByteArray &data, QString *error = nullptr) {
	QSaveFile file(path);
	if (!file.open(QIODevice::WriteOnly)) {
		if (error) *error = file.errorString();
		return false;
	}
	if (file.write(data) != data.size() || !file.commit()) {
		if (error) *error = file.errorString();
		return false;
	}
	return true;
}

// A text file in UTF-8 with the platform's line endings (what QFile::Text and QTextStream gave before): "\r\n" on Windows.
inline bool writeTextFileAtomically(const QString &path, QString text, QString *error = nullptr) {
#ifdef Q_OS_WIN
	text.replace(QLatin1Char('\n'), QStringLiteral("\r\n"));
#endif
	return writeFileAtomically(path, text.toUtf8(), error);
}

#endif  // ATOMIC_FILE_H

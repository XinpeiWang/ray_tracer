#include "theme.h"

#include "palette_file.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QSettings>
#include <QStandardPaths>

// ============================================================================
// User themes
// ============================================================================
// Loads *.theme files (INI syntax) from two places, in order:
//
//   1. <application dir>/themes         - ships alongside the exe
//   2. <AppConfigLocation>/themes       - per-user, survives reinstalling
//
// A loaded theme becomes the same PaletteData a built-in is, so it goes through
// the same registry, lookup, menu and persistence with no special cases. That
// is the whole reason palette_file parses into PaletteData rather than into
// some parallel "custom theme" type.
//
// Everything here is best-effort and loud: a malformed file is skipped with a
// warning naming the file and the offending key, never silently ignored and
// never fatal. A theme file is user content and the app has to start without
// it.
// ============================================================================
namespace theme {

QVector<palette_data::PaletteData> loadUserPalettes(QStringList *problems) {
	QVector<palette_data::PaletteData> loaded;

	// Reserved ids grow as we go, so two user files cannot collide with each
	// other either - not just with the built-ins.
	std::vector<std::string> reserved;
	for (const palette_data::PaletteData &d : palette_data::builtins())
		reserved.push_back(d.id);

	QStringList dirs;
	dirs << QDir(QCoreApplication::applicationDirPath()).filePath("themes");
	const QString config = QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation);
	if (!config.isEmpty())
		dirs << QDir(config).filePath("themes");

	for (const QString &dirPath : dirs) {
		QDir dir(dirPath);
		if (!dir.exists()) continue;

		const QStringList files = dir.entryList({"*.theme"}, QDir::Files, QDir::Name);
		for (const QString &file : files) {
			const QString path = dir.filePath(file);

			// INI parsing only - QSettings gives us the key/value map and the
			// Qt-free parser does everything that can actually be wrong.
			QSettings ini(path, QSettings::IniFormat);
			if (ini.status() != QSettings::NoError) {
				const QString msg = QString("%1: not readable as an INI file").arg(file);
				qWarning().noquote() << "Theme:" << msg;
				if (problems) problems->append(msg);
				continue;
			}

			std::map<std::string, std::string> values;
			for (const QString &key : ini.allKeys()) {
				// Keys are lower-cased so the file can use whichever casing
				// reads best; section prefixes are dropped so both a flat file
				// and one with a [theme] header work.
				const QString bare = key.section('/', -1).toLower();
				values[bare.toStdString()] = ini.value(key).toString().toStdString();
			}

			const palette_file::ParseResult r = palette_file::parse(values, reserved);
			if (!r.ok) {
				const QString msg = QString("%1: %2").arg(file, QString::fromStdString(r.error));
				qWarning().noquote() << "Theme:" << msg;
				if (problems) problems->append(msg);
				continue;
			}

			reserved.push_back(r.palette.id);
			loaded.push_back(r.palette);
		}
	}

	return loaded;
}

} // namespace theme

#ifndef STARTUP_PROFILE_H
#define STARTUP_PROFILE_H

// Where the start-up time goes. Scoped timers around the stages of launching (QApplication, translators, building the window, each tab, the theme...) are
// collected in memory - the log file does not exist yet for the first of them - and written to the log once the window is up: one "startup:" line per stage,
// indented by nesting, with how long it took and when it ended, and a summary naming the slowest. A start-up that has become slow is then visible in any
// bug report's log without a profiler.
//
//   startup_profile::begin();                 // first thing in main()
//   { startup_profile::Stage s("build tabs"); ... }
//   startup_profile::ready("window shown");   // writes it all to the log

#include <QElapsedTimer>
#include <QString>

#include <algorithm>
#include <vector>

#include "app_log.h"

namespace startup_profile {

struct Record {
	QString name;
	qint64 startMs, durationMs;
	int depth;
};

inline QElapsedTimer &clock() { static QElapsedTimer t; return t; }
inline std::vector<Record> &records() { static std::vector<Record> r; return r; }
inline int &depth() { static int d = 0; return d; }

inline void begin() { clock().start(); }

class Stage {
public:
	explicit Stage(const char *name) : m_name(QString::fromLatin1(name)), m_start(clock().isValid() ? clock().elapsed() : 0), m_depth(depth()++) {}
	~Stage() {
		--depth();
		if (clock().isValid()) records().push_back({m_name, m_start, clock().elapsed() - m_start, m_depth});
	}
	Stage(const Stage &) = delete;
	Stage &operator=(const Stage &) = delete;

private:
	QString m_name;
	qint64 m_start;
	int m_depth;
};

// Writes what was collected to the log (call once, when the window is on screen).
inline void ready(const char *what) {
	if (!clock().isValid()) return;
	const qint64 total = clock().elapsed();
	std::vector<Record> sorted = records();
	std::sort(records().begin(), records().end(), [](const Record &a, const Record &b) { return a.startMs + a.durationMs < b.startMs + b.durationMs || (a.startMs + a.durationMs == b.startMs + b.durationMs && a.depth > b.depth); });
	for (const Record &r : records())
		AppLog::info(QStringLiteral("startup"), QStringLiteral("%1%2: %3 ms (ended at %4 ms)").arg(QString(r.depth * 2, QLatin1Char(' ')), r.name).arg(r.durationMs).arg(r.startMs + r.durationMs));
	// The slowest top-level stages (nested ones are inside them).
	std::vector<Record> top;
	for (const Record &r : sorted) if (r.depth == 0) top.push_back(r);
	std::sort(top.begin(), top.end(), [](const Record &a, const Record &b) { return a.durationMs > b.durationMs; });
	QStringList slowest;
	for (std::size_t i = 0; i < top.size() && i < 3; ++i) slowest << QStringLiteral("%1 %2 ms").arg(top[i].name).arg(top[i].durationMs);
	AppLog::write(total > 3000 ? log_format::Level::Warn : log_format::Level::Info, QStringLiteral("startup"),
	              QStringLiteral("%1 %2 ms after the program started%3").arg(QLatin1String(what)).arg(total).arg(slowest.isEmpty() ? QString() : QStringLiteral("; slowest stages: ") + slowest.join(QStringLiteral(", "))));
	records().clear();
}

}  // namespace startup_profile

#endif  // STARTUP_PROFILE_H

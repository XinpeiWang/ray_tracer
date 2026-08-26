#include "mainwindow.h"
#include "settings_keys.h"

#include <QDateTime>
#include <QFileInfo>
#include <QSettings>

// ============================================================================
// Recent Renders persistence
// ============================================================================
// Lets the Preview tab's empty-state list (createPreviewTab(),
// mainwindow_tabs.cpp) reopen a past render after an app restart - closing
// a preview sub-tab never deletes its underlying file (closePreviewSubTab(),
// mainwindow_tabs.cpp), so the only thing missing across sessions was
// knowing the file existed at all. Same QSettings(settings_keys::kOrg,
// settings_keys::kApp) location as the theme/font/language prefs
// (theme_switch.cpp etc.), but its own array group
// (settings_keys::kRecentRendersGroup) via beginWriteArray/beginReadArray -
// the first list-shaped value this app persists, so there's no existing
// scalar-key precedent to extend.
// ============================================================================

namespace {

// A capped "recent files" length, not drawn from an existing house
// convention (this app has no other "how many X do we keep" precedent to
// match) - 20 is simply a reasonable default.
constexpr int kMaxRecentRenders = 20;

void writeEntry(QSettings &settings, const RecentRenderEntry &entry) {
	settings.setValue("outputPath", entry.outputPath);
	settings.setValue("previewPath", entry.previewPath);
	settings.setValue("isVideo", entry.isVideo);
	settings.setValue("sceneId", entry.sceneId);
	settings.setValue("displayTitle", entry.displayTitle);
	settings.setValue("sceneDescription", entry.sceneDescription);
	settings.setValue("width", entry.width);
	settings.setValue("height", entry.height);
	settings.setValue("samples", entry.samples);
	settings.setValue("useGPU", entry.useGPU);
	settings.setValue("useWavefront", entry.useWavefront);
	settings.setValue("integratorMode", static_cast<int>(entry.integratorMode));
	settings.setValue("timestampEpochSecs", entry.timestampEpochSecs);
}

RecentRenderEntry readEntry(QSettings &settings) {
	RecentRenderEntry entry;
	entry.outputPath = settings.value("outputPath").toString();
	entry.previewPath = settings.value("previewPath").toString();
	entry.isVideo = settings.value("isVideo").toBool();
	entry.sceneId = settings.value("sceneId").toString();
	entry.displayTitle = settings.value("displayTitle").toString();
	entry.sceneDescription = settings.value("sceneDescription").toString();
	entry.width = settings.value("width").toInt();
	entry.height = settings.value("height").toInt();
	entry.samples = settings.value("samples").toInt();
	entry.useGPU = settings.value("useGPU").toBool();
	entry.useWavefront = settings.value("useWavefront").toBool();
	entry.integratorMode = static_cast<IntegratorMode>(settings.value("integratorMode").toInt());
	entry.timestampEpochSecs = settings.value("timestampEpochSecs").toLongLong();
	return entry;
}

} // namespace

QList<RecentRenderEntry> MainWindow::loadRecentRenders() const {
	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	QList<RecentRenderEntry> entries;
	const int count = settings.beginReadArray(settings_keys::kRecentRendersGroup);
	entries.reserve(count);
	for (int i = 0; i < count; ++i) {
		settings.setArrayIndex(i);
		entries.append(readEntry(settings));
	}
	settings.endArray();

	// A file may have been moved or deleted since it was recorded (in a
	// different session, possibly by something outside this app entirely) -
	// filtering here, not at save time, is what keeps the list honest
	// without needing to watch the filesystem. previewPath is checked for
	// both images and video (it's the actual file a reopened tab would
	// display); outputPath is checked too for images, where it's a
	// distinct raw-render file from previewPath.
	QList<RecentRenderEntry> existing;
	existing.reserve(entries.size());
	for (const RecentRenderEntry &entry : entries) {
		if (!QFileInfo::exists(entry.previewPath)) continue;
		if (!entry.isVideo && !QFileInfo::exists(entry.outputPath)) continue;
		existing.append(entry);
	}
	return existing;
}

void MainWindow::saveRecentRender(const RenderJob &job, const QString &previewPath, bool isVideo,
                                   const QString &displayTitleOverride) const {
	RecentRenderEntry entry;
	// For video, addVideoPreviewTab() only ever takes one path (the .mp4
	// itself) - job.outputPath there is the per-frame base path, not the
	// assembled video, so outputPath/previewPath collapse to the same
	// value rather than holding a stale/unrelated path (matches
	// RecentRenderEntry::outputPath's own doc comment, mainwindow.h).
	entry.outputPath = isVideo ? previewPath : job.outputPath;
	entry.previewPath = previewPath;
	entry.isVideo = isVideo;
	entry.sceneId = job.sceneId;
	entry.displayTitle = displayTitleOverride.isEmpty() ? job.displayTitle : displayTitleOverride;
	entry.sceneDescription = job.sceneDescription;
	entry.width = job.width;
	entry.height = job.height;
	entry.samples = job.samples;
	entry.useGPU = job.useGPU;
	entry.useWavefront = job.useWavefront;
	entry.integratorMode = job.integratorOptions.mode;
	entry.timestampEpochSecs = QDateTime::currentDateTime().toSecsSinceEpoch();

	// Full rewrite rather than a partial update - simplest correct approach
	// at this size (<= kMaxRecentRenders entries), no risk of a stale
	// leftover entry from a previous, differently-sized array.
	QList<RecentRenderEntry> entries = loadRecentRenders();
	entries.prepend(entry);
	while (entries.size() > kMaxRecentRenders) entries.removeLast();

	QSettings settings(settings_keys::kOrg, settings_keys::kApp);
	settings.beginWriteArray(settings_keys::kRecentRendersGroup);
	for (int i = 0; i < entries.size(); ++i) {
		settings.setArrayIndex(i);
		writeEntry(settings, entries[i]);
	}
	settings.endArray();
}

// Same renderer/integrator-suffix shape as describeRenderJob()
// (mainwindow_slots.cpp) - reused as a template, not called directly,
// since RecentRenderEntry isn't a RenderJob - plus a relative-time suffix
// no other row format here needs.
QString MainWindow::describeRecentRenderEntry(const RecentRenderEntry &entry) const {
	const QString renderer = entry.useGPU ? (entry.useWavefront ? tr("GPU-WF") : tr("GPU")) : tr("CPU");
	const QString modeSuffix = entry.isVideo ? tr(" · Video") : QString();
	QString integratorSuffix;
	switch (entry.integratorMode) {
		case IntegratorMode::Default: break;
		case IntegratorMode::Sppm: integratorSuffix = tr(" · SPPM"); break;
		case IntegratorMode::Bdpt: integratorSuffix = tr(" · BDPT"); break;
		case IntegratorMode::Mlt: integratorSuffix = tr(" · MLT"); break;
		case IntegratorMode::RandomWalk: integratorSuffix = tr(" · RandomWalk"); break;
		case IntegratorMode::Ao: integratorSuffix = tr(" · AO"); break;
		case IntegratorMode::SimplePath: integratorSuffix = tr(" · SimplePath"); break;
		case IntegratorMode::SimpleVolPath: integratorSuffix = tr(" · SimpleVolPath"); break;
		case IntegratorMode::LightPath: integratorSuffix = tr(" · LightPath"); break;
	}

	const qint64 secsAgo = QDateTime::currentDateTime().toSecsSinceEpoch() - entry.timestampEpochSecs;
	QString whenText;
	if (secsAgo < 60) whenText = tr("just now");
	else if (secsAgo < 3600) whenText = tr("%1 min ago").arg(secsAgo / 60);
	else if (secsAgo < 86400) whenText = tr("%1 hr ago").arg(secsAgo / 3600);
	else if (secsAgo < 86400 * 14) whenText = tr("%1 days ago").arg(secsAgo / 86400);
	else whenText = QDateTime::fromSecsSinceEpoch(entry.timestampEpochSecs).date().toString(Qt::ISODate);

	return tr("%1 — %2×%3 · %4spp · %5%6%7 — %8")
		.arg(entry.displayTitle)
		.arg(entry.width).arg(entry.height)
		.arg(entry.samples)
		.arg(renderer, modeSuffix, integratorSuffix)
		.arg(whenText);
}

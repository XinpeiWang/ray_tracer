#include "mainwindow.h"
#include "scene_metadata_client.h"
#include "settings_keys.h"

#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QSet>
#include <QSettings>

#include <algorithm>

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
	settings.setValue("metadataKnown", entry.metadataKnown);
}

// Best-effort entry for a render_*.png/*_video.mp4 file found sitting in
// the default output folder that has no persisted RecentRenderEntry of its
// own - either because it predates this feature entirely, or its entry has
// since aged out past kMaxRecentRenders. Only what the filename itself
// encodes is recoverable: captureRenderJob() (mainwindow_slots.cpp) always
// names a file "<render|video>_<sceneId>_<timestamp...>.<ext>", and the
// video pipeline further appends "_video" before ".mp4" - see
// assembleVideoAutomatically()'s own comment on that naming. Everything
// else (resolution, samples, backend, integrator) stays at the struct's
// default/unknown state; describeRecentRenderEntry() knows to omit those
// rather than show fabricated zeros (metadataKnown = false).
RecentRenderEntry buildScannedEntry(const QFileInfo &fileInfo, bool isVideo) {
	RecentRenderEntry entry;
	entry.outputPath = fileInfo.absoluteFilePath();
	entry.previewPath = fileInfo.absoluteFilePath();
	entry.isVideo = isVideo;
	entry.metadataKnown = false;

	QString stem = fileInfo.completeBaseName(); // strips the final ".png"/".mp4" only
	if (isVideo && stem.endsWith(QLatin1String("_video"))) stem.chop(6);
	const QStringList parts = stem.split('_');
	// parts[0] = "render"/"video", parts[1] = sceneId (this app's scene
	// ids - "A1", "B23", "I5" - never contain an underscore themselves),
	// parts[2..] = timestamp fragments.
	const QString sceneId = parts.size() >= 2 ? parts[1] : QString();
	entry.sceneId = sceneId;
	const QString sceneName = sceneId.isEmpty() ? QString() : SceneMetadataClient::sceneName(sceneId);
	// Falls back to the raw filename when the id can't be parsed or isn't
	// recognized (e.g. a renamed/foreign file matching the glob by
	// accident) - still enough to identify which file double-clicking it
	// would open.
	entry.displayTitle = sceneName.isEmpty() ? fileInfo.fileName() : sceneName;
	entry.sceneDescription = sceneId.isEmpty() ? QString() : SceneMetadataClient::sceneDescription(sceneId);

	const QDateTime birth = fileInfo.birthTime();
	entry.timestampEpochSecs = (birth.isValid() ? birth : fileInfo.lastModified()).toSecsSinceEpoch();
	return entry;
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
	// Default true (not the QVariant-invalid-default false) when the key
	// is absent - an entry saved by a version of this app before
	// metadataKnown existed was always a real, fully-known
	// saveRecentRender() call, never a scanned best-effort one.
	entry.metadataKnown = settings.value("metadataKnown", true).toBool();
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
	QSet<QString> knownPaths;
	for (const RecentRenderEntry &entry : entries) {
		if (!QFileInfo::exists(entry.previewPath)) continue;
		if (!entry.isVideo && !QFileInfo::exists(entry.outputPath)) continue;
		existing.append(entry);
		knownPaths.insert(entry.previewPath);
	}

	// Best-effort backfill: scan the default output folder (Desktop - see
	// m_outputPathEdit's own default, mainwindow_tabs.cpp) for
	// render_*.png/*_video.mp4 files this app produced that have no
	// persisted entry of their own (predates this feature, or aged out
	// past kMaxRecentRenders) - see buildScannedEntry()'s own comment.
	// Anything Browse-saved outside Desktop is invisible to this scan, same
	// as it always was before Recent Renders existed at all; this only
	// covers the common default-path case.
	const QDir desktop(QDir::homePath() + QStringLiteral("/Desktop"));
	QList<RecentRenderEntry> scanned;
	for (const QFileInfo &fileInfo : desktop.entryInfoList(QStringList() << QStringLiteral("render_*.png"), QDir::Files)) {
		if (knownPaths.contains(fileInfo.absoluteFilePath())) continue;
		scanned.append(buildScannedEntry(fileInfo, /*isVideo=*/false));
	}
	for (const QFileInfo &fileInfo : desktop.entryInfoList(QStringList() << QStringLiteral("*_video.mp4"), QDir::Files)) {
		if (knownPaths.contains(fileInfo.absoluteFilePath())) continue;
		scanned.append(buildScannedEntry(fileInfo, /*isVideo=*/true));
	}

	QList<RecentRenderEntry> merged = existing + scanned;
	std::sort(merged.begin(), merged.end(), [](const RecentRenderEntry &a, const RecentRenderEntry &b) {
		return a.timestampEpochSecs > b.timestampEpochSecs;
	});
	while (merged.size() > kMaxRecentRenders) merged.removeLast();
	return merged;
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
	const qint64 secsAgo = QDateTime::currentDateTime().toSecsSinceEpoch() - entry.timestampEpochSecs;
	QString whenText;
	if (secsAgo < 60) whenText = tr("just now");
	else if (secsAgo < 3600) whenText = tr("%1 min ago").arg(secsAgo / 60);
	else if (secsAgo < 86400) whenText = tr("%1 hr ago").arg(secsAgo / 3600);
	else if (secsAgo < 86400 * 14) whenText = tr("%1 days ago").arg(secsAgo / 86400);
	else whenText = QDateTime::fromSecsSinceEpoch(entry.timestampEpochSecs).date().toString(Qt::ISODate);

	// A scanned (best-effort) entry only knows its file/scene id/timestamp -
	// showing "0×0 · 0spp · CPU" for it would look like real data rather
	// than the fact that this app never recorded it in the first place.
	if (!entry.metadataKnown) {
		return tr("%1%2 — %3")
			.arg(entry.displayTitle)
			.arg(entry.isVideo ? tr(" · Video") : QString())
			.arg(whenText);
	}

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

	return tr("%1 — %2×%3 · %4spp · %5%6%7 — %8")
		.arg(entry.displayTitle)
		.arg(entry.width).arg(entry.height)
		.arg(entry.samples)
		.arg(renderer, modeSuffix, integratorSuffix)
		.arg(whenText);
}

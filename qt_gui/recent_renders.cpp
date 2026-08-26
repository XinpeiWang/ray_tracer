#include "mainwindow.h"
#include "scene_metadata_client.h"
#include "settings_keys.h"

#include <QAbstractItemView>
#include <QDateTime>
#include <QDir>
#include <QFileInfo>
#include <QListWidget>
#include <QPixmap>
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
	settings.setValue("integratorMode", static_cast<int>(entry.integratorOptions.mode));
	// IntegratorOptions sub-fields - only the ones RenderController::start()
	// (mainwindow.cpp) would actually emit differently-from-default matter
	// for describeRecentRenderEntry()/renderTechniqueHtml(), but all are
	// persisted (not just the non-default ones) since reading a plain
	// value back is simpler than reconstructing "was this key omitted
	// because it was default, or never written at all" on read.
	settings.setValue("sppmIterations", entry.integratorOptions.sppmIterations);
	settings.setValue("sppmPhotons", entry.integratorOptions.sppmPhotons);
	settings.setValue("bdptMaxDepth", entry.integratorOptions.bdptMaxDepth);
	settings.setValue("mltBootstrap", entry.integratorOptions.mltBootstrap);
	settings.setValue("mltMutations", entry.integratorOptions.mltMutations);
	settings.setValue("mltMaxDepth", entry.integratorOptions.mltMaxDepth);
	settings.setValue("aoMaxDist", entry.integratorOptions.aoMaxDist);
	settings.setValue("aoUniform", entry.integratorOptions.aoUniform);
	settings.setValue("aoIllumScale", entry.integratorOptions.aoIllumScale);
	settings.setValue("aoIllumR", entry.integratorOptions.aoIllumR);
	settings.setValue("aoIllumG", entry.integratorOptions.aoIllumG);
	settings.setValue("aoIllumB", entry.integratorOptions.aoIllumB);
	settings.setValue("simplepathNoLights", entry.integratorOptions.simplepathNoLights);
	settings.setValue("simplepathNoBsdf", entry.integratorOptions.simplepathNoBsdf);
	// AdvancedRenderFlags.
	settings.setValue("denoise", entry.advancedFlags.denoise);
	settings.setValue("stats", entry.advancedFlags.stats);
	settings.setValue("optixValidate", entry.advancedFlags.optixValidate);
	settings.setValue("exposure", entry.advancedFlags.exposure);
	settings.setValue("sampler", entry.advancedFlags.sampler);
	settings.setValue("spectral", entry.advancedFlags.spectral);
	settings.setValue("tonemap", entry.advancedFlags.tonemap);
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
	entry.integratorOptions.mode = static_cast<IntegratorMode>(settings.value("integratorMode").toInt());
	// Guarded by contains(), not settings.value(key, <literal default>):
	// `entry.integratorOptions`/`entry.advancedFlags` are already
	// default-constructed to the correct struct defaults above (matching
	// mainwindow.h's own member-initializers, no duplicated literals to
	// drift out of sync) - an entry saved before these keys existed
	// should keep those defaults, not read back a fabricated 0/false/
	// empty-string from a missing QVariant.
	if (settings.contains("sppmIterations")) entry.integratorOptions.sppmIterations = settings.value("sppmIterations").toInt();
	if (settings.contains("sppmPhotons")) entry.integratorOptions.sppmPhotons = settings.value("sppmPhotons").toInt();
	if (settings.contains("bdptMaxDepth")) entry.integratorOptions.bdptMaxDepth = settings.value("bdptMaxDepth").toInt();
	if (settings.contains("mltBootstrap")) entry.integratorOptions.mltBootstrap = settings.value("mltBootstrap").toInt();
	if (settings.contains("mltMutations")) entry.integratorOptions.mltMutations = settings.value("mltMutations").toLongLong();
	if (settings.contains("mltMaxDepth")) entry.integratorOptions.mltMaxDepth = settings.value("mltMaxDepth").toInt();
	if (settings.contains("aoMaxDist")) entry.integratorOptions.aoMaxDist = settings.value("aoMaxDist").toDouble();
	if (settings.contains("aoUniform")) entry.integratorOptions.aoUniform = settings.value("aoUniform").toBool();
	if (settings.contains("aoIllumScale")) entry.integratorOptions.aoIllumScale = settings.value("aoIllumScale").toDouble();
	if (settings.contains("aoIllumR")) entry.integratorOptions.aoIllumR = settings.value("aoIllumR").toDouble();
	if (settings.contains("aoIllumG")) entry.integratorOptions.aoIllumG = settings.value("aoIllumG").toDouble();
	if (settings.contains("aoIllumB")) entry.integratorOptions.aoIllumB = settings.value("aoIllumB").toDouble();
	if (settings.contains("simplepathNoLights")) entry.integratorOptions.simplepathNoLights = settings.value("simplepathNoLights").toBool();
	if (settings.contains("simplepathNoBsdf")) entry.integratorOptions.simplepathNoBsdf = settings.value("simplepathNoBsdf").toBool();
	if (settings.contains("denoise")) entry.advancedFlags.denoise = settings.value("denoise").toBool();
	if (settings.contains("stats")) entry.advancedFlags.stats = settings.value("stats").toBool();
	if (settings.contains("optixValidate")) entry.advancedFlags.optixValidate = settings.value("optixValidate").toBool();
	if (settings.contains("exposure")) entry.advancedFlags.exposure = settings.value("exposure").toDouble();
	if (settings.contains("sampler")) entry.advancedFlags.sampler = settings.value("sampler").toString();
	if (settings.contains("spectral")) entry.advancedFlags.spectral = settings.value("spectral").toBool();
	if (settings.contains("tonemap")) entry.advancedFlags.tonemap = settings.value("tonemap").toString();
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
	entry.integratorOptions = job.integratorOptions;
	entry.advancedFlags = job.advancedFlags;
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

// Shares renderer/integrator-suffix formatting with describeRenderJob()
// (mainwindow_slots.cpp) via rendererLabel()/integratorSuffixTag()
// (mainwindow.h) - plus a relative-time suffix no other row format here
// needs.
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

	const QString renderer = rendererLabel(entry.useGPU, entry.useWavefront);
	const QString modeSuffix = entry.isVideo ? tr(" · Video") : QString();
	const QString integratorSuffix = integratorSuffixTag(entry.integratorOptions.mode);

	return tr("%1 — %2×%3 · %4spp · %5%6%7 — %8")
		.arg(entry.displayTitle)
		.arg(entry.width).arg(entry.height)
		.arg(entry.samples)
		.arg(renderer, modeSuffix, integratorSuffix)
		.arg(whenText);
}

// Rebuilds m_recentRendersList from a fresh loadRecentRenders() call - see
// createPreviewTab() (mainwindow_tabs.cpp), which calls this once to build
// the list initially, and the saveRecentRender() call sites
// (mainwindow_slots.cpp) / closePreviewSubTab() (mainwindow_tabs.cpp),
// which call it again so a render completed - or a tab closed - earlier
// this session shows up without an app restart. Text-only rows, no
// thumbnail decode - the real image only loads on demand, on double-click,
// same as the initial build did.
void MainWindow::refreshRecentRendersList() {
	if (!m_previewSubTabs) return;
	const QList<RecentRenderEntry> recents = loadRecentRenders();

	if (recents.isEmpty()) {
		if (m_recentRendersList) m_recentRendersList->clear();
		return;
	}

	if (!m_recentRendersList) {
		m_recentRendersList = new QListWidget();
		m_recentRendersList->setSelectionMode(QAbstractItemView::SingleSelection);
		m_recentRendersList->setMaximumHeight(200);
		m_recentRendersList->viewport()->installEventFilter(new ListEmptyAreaDeselectFilter(m_recentRendersList));
		m_previewSubTabs->addToEmptyState(m_recentRendersList);
	}

	m_recentRendersList->clear();
	// The double-click handler captures `recents` by value, so it's rewired
	// fresh on every refresh - disconnect the previous refresh's connection
	// first, or double-clicking an item would fire once per past refresh
	// with an increasingly stale `recents` list each time.
	disconnect(m_recentRendersList, &QListWidget::itemDoubleClicked, this, nullptr);
	for (int i = 0; i < recents.size(); ++i) {
		QListWidgetItem *item = new QListWidgetItem(describeRecentRenderEntry(recents[i]), m_recentRendersList);
		item->setData(Qt::UserRole, i);
	}
	connect(m_recentRendersList, &QListWidget::itemDoubleClicked, this, [this, recents](QListWidgetItem *item) {
		const RecentRenderEntry &entry = recents[item->data(Qt::UserRole).toInt()];
		// A scanned/best-effort entry (metadataKnown == false) never had
		// real settings recorded - leave techniqueHtml empty rather than
		// building one from fabricated default-constructed struct values,
		// same reasoning as describeRecentRenderEntry()'s own metadataKnown
		// branch above.
		const PreviewTechniqueInfo technique{
			entry.sceneId,
			entry.metadataKnown ? renderTechniqueHtml(entry.integratorOptions, entry.advancedFlags) : QString()};
		if (entry.isVideo) {
			addVideoPreviewTab(entry.displayTitle, entry.sceneDescription, entry.previewPath,
			                    describeRecentRenderEntry(entry), technique);
		} else {
			QPixmap pixmap(entry.previewPath);
			if (!pixmap.isNull()) {
				addImagePreviewTab(entry.displayTitle, entry.sceneDescription, pixmap,
				                    describeRecentRenderEntry(entry), entry.outputPath, entry.previewPath, technique);
			}
		}
		if (m_previewTabIndex >= 0) m_tabWidget->setCurrentIndex(m_previewTabIndex);
	});
}

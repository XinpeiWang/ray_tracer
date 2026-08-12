#include "mainwindow.h"
#include "icon_tint.h"

#include <QAction>
#include <QActionGroup>
#include <QComboBox>
#include <QMenu>
#include <QMenuBar>
#include <QSettings>
#include <QStatusBar>

// ============================================================================
// Theme selection and persistence
// ============================================================================
// Switching is live: applyTheme() re-applies the app palette and stylesheet,
// and restyleThemedWidgets() refreshes the handful of widgets that carry their
// own stylesheet rather than inheriting the global one. Asking the user to
// restart to see a colour change would be a poor trade for the small amount of
// work this costs.
// ============================================================================

namespace {
// Organisation/app names give QSettings a stable location; without them it
// falls back to names derived from the executable, which move if it is renamed.
constexpr const char *kSettingsOrg = "RayTracer";
constexpr const char *kSettingsApp = "RayTracerGUI";
constexpr const char *kThemeKey = "ui/theme";
} // namespace

QString MainWindow::loadSavedThemeId() const {
	QSettings settings(kSettingsOrg, kSettingsApp);
	return settings.value(kThemeKey, theme::defaultPalette().id).toString();
}

void MainWindow::saveThemeId(const QString &themeId) const {
	QSettings settings(kSettingsOrg, kSettingsApp);
	settings.setValue(kThemeKey, themeId);
}

void MainWindow::switchTheme(const QString &themeId) {
	const theme::Palette &p = theme::byId(themeId);
	applyTheme(p);
	restyleThemedWidgets();
	saveThemeId(p.id);

	// byId() falls back to the default for an unknown id, so re-sync the menu
	// against what was actually applied rather than what was requested.
	for (QAction *action : m_themeActions)
		action->setChecked(action->data().toString() == p.id);

	statusBar()->showMessage(QString("Theme: %1").arg(p.name), 3000);
}

void MainWindow::createThemeMenu(QMenu *viewMenu) {
	QMenu *themeMenu = viewMenu->addMenu("&Theme");

	// Exclusive check marks: exactly one scheme is in force at a time.
	auto *group = new QActionGroup(this);
	group->setExclusive(true);

	for (const theme::Palette &p : theme::all()) {
		QAction *action = themeMenu->addAction(p.name);
		action->setCheckable(true);
		action->setData(p.id);
		action->setStatusTip(p.origin);
		action->setChecked(p.id == m_activeTheme.id);
		group->addAction(action);
		m_themeActions.push_back(action);

		connect(action, &QAction::triggered, this, [this, id = p.id]() {
			switchTheme(id);
		});
	}
}

// Widgets that carry their own stylesheet rather than inheriting the global
// one. These used to hold hardcoded hex values, which is exactly what made the
// app un-themeable: the global sheet could change but these stayed cyberpunk.
void MainWindow::restyleThemedWidgets() {
	const theme::Palette &p = m_activeTheme;
	const QString body = p.textBody.name();
	const QString muted = p.textMuted.name();
	const QString border = p.border.name();
	const QString surface0 = p.surface0.name();
	const QString surface2 = p.surface2.name();

	if (m_sceneInfoLabel) {
		m_sceneInfoLabel->setStyleSheet(QString(
			"QLabel { color: %1; background-color: %2; border: 1px solid %3;"
			" border-radius: 6px; padding: 8px 12px; font-size: 11px; }")
			.arg(muted, surface2, border));
	}
	if (m_previewLabel) {
		m_previewLabel->setStyleSheet(QString(
			"ScaledImageLabel { background-color: %1; border: 1px solid %2;"
			" border-radius: 6px; color: %3; font-size: 11pt; }")
			.arg(surface0, border, muted));
	}
	if (m_previewInfoLabel)
		m_previewInfoLabel->setStyleSheet(QString("color: %1; font-size: 9pt;").arg(muted));
	if (m_logTextEdit) {
		m_logTextEdit->setStyleSheet(QString(
			"QTextEdit { background-color: %1; border: 1px solid %2; border-radius: 6px; }")
			.arg(surface0, border));
	}
	if (m_videoInfoLabel)
		m_videoInfoLabel->setStyleSheet(QString("QLabel { color: %1; font-style: italic; padding: 10px; }").arg(muted));
	if (m_statusDevice)
		m_statusDevice->setStyleSheet(QString("color: %1; padding: 0 8px;").arg(muted));
	if (m_statusSettings)
		m_statusSettings->setStyleSheet(QString("color: %1; padding: 0 8px;").arg(muted));

	// Combo popups are separate top-level widgets with their own stylesheet, so
	// the global sheet never reaches them. findChildren() rather than a list of
	// members: every combo in the window needs this and a hand-maintained list
	// would silently miss the next one added.
	for (QComboBox *combo : findChildren<QComboBox *>())
		applyComboPopupPalette(combo);

	// Icons are monochrome silhouettes recoloured at runtime, so they follow
	// the scheme like any other painted element. Without this a light theme
	// would show near-white icons on a near-white surface.
	icon_tint::retint(this, p.textBody, p.accentPrimary);

	// Combo ITEM icons are copies held by the model, not by a widget, so
	// retint() cannot reach them - they have to be set through the model.
	const auto retintItems = [&](QComboBox *combo, std::initializer_list<const char *> paths) {
		if (!combo) return;
		int index = 0;
		for (const char *path : paths) {
			if (index < combo->count())
				combo->setItemIcon(index, icon_tint::tinted(path, p.textBody));
			++index;
		}
	};
	retintItems(m_modeCombo, {":/icons/image.svg", ":/icons/video.svg"});
	retintItems(m_renderModeCombo, {":/icons/gpu.svg", ":/icons/cpu.svg"});

	statusBar()->setStyleSheet(QString(
		"QStatusBar { background-color: %1; color: %2; }"
		"QStatusBar::item { border: none; }").arg(surface0, muted));

	// The log pane's existing contents were written as HTML with the previous
	// scheme's colours baked into each line, so they cannot be recoloured in
	// place. Say so rather than leaving the user wondering why old lines look
	// wrong next to new ones.
	if (m_logTextEdit && !m_logTextEdit->document()->isEmpty())
		onLogMessage("[INFO] Theme changed - existing log lines keep their previous colours");

	Q_UNUSED(body);
}

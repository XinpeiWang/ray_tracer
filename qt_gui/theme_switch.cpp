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

// The parts of the UI the global stylesheet cannot reach on its own.
//
// Everything that CAN be expressed as a rule in the global sheet lives there
// instead, addressed by object name (QLabel#sceneInfo, #previewInfo, #videoInfo,
// #mutedInfo, #statusInfo) or class name (ScaledImageLabel). Those re-theme for
// free when applyTheme() rebuilds the sheet, and nothing has to be listed here.
// What remains are the two categories that genuinely cannot: widgets Qt styles
// outside the global sheet, and pixmaps that have to be regenerated.
void MainWindow::restyleThemedWidgets() {
	const theme::Palette &p = m_activeTheme;

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

	// The log pane's existing contents were written as HTML with the previous
	// scheme's colours baked into each line, so they cannot be recoloured in
	// place. Say so rather than leaving the user wondering why old lines look
	// wrong next to new ones.
	if (m_logTextEdit && !m_logTextEdit->document()->isEmpty())
		onLogMessage("[INFO] Theme changed - existing log lines keep their previous colours");
}

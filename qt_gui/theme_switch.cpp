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

	statusBar()->showMessage(tr("Theme: %1").arg(p.name), 3000);
}

// A top-level menu rather than a submenu of View. Themes are a first-class
// preference here - twelve of them, several with their own artwork - and
// burying a twelve-item list one level down made it both harder to reach and
// harder to discover. "T" is free as a mnemonic alongside File/Render/View/Help.
void MainWindow::createThemeMenu() {
	QMenu *themeMenu = menuBar()->addMenu(tr("&Theme"));

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
	icon_tint::retint(this, p.textBody, p.accentPrimary, p.error);

	// Combo ITEM icons are copies held by the model, not by a widget, so
	// retint() above cannot reach them. Each item carries its own resource path
	// (see icon_tint::addItem), so this needs no list of paths to keep in step
	// with the code that created them - and it picks up any combo added later
	// for free.
	for (QComboBox *combo : findChildren<QComboBox *>())
		icon_tint::retintItems(combo, p.textBody);

	// The pane holds HTML with the previous scheme's colours already written
	// into each span, so its existing lines cannot be recoloured in place -
	// they are re-rendered from the kept history instead. This used to print an
	// apology telling the user the old lines would stay the wrong colour.
	rebuildLogPane();
	rebuildDiagPane();  // same constraint, same fix - see its own comment

	// The progress bar's glow effect (startProgressGlow(), mainwindow_slots.cpp)
	// is a QGraphicsDropShadowEffect outside the global stylesheet's reach,
	// created once on the first render and left in place afterward - without
	// this, it would keep pulsing in whichever theme's accent colour was active
	// the first time a render ever started, for the rest of the session.
	if (auto *glow = qobject_cast<QGraphicsDropShadowEffect *>(
			m_progressBar ? m_progressBar->graphicsEffect() : nullptr))
		glow->setColor(p.accentPrimary);

	// Same "baked into inline HTML, QSS can't reach it" problem as the log/diag
	// panes above, for the scene-info warning badges - see refreshSceneInfoLabel()'s
	// own comment.
	refreshSceneInfoLabel();
}

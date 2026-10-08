// scene_builder_widget.cpp - see scene_builder_widget.h. The layout view is scene_layout_view.cpp, the property panel scene_builder_inspector.cpp.
#include "scene_builder_widget.h"

#include "scene_builder_common.h"
#include "../src/shared/pbrt_asset_check.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDir>
#include <QAbstractSpinBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QHBoxLayout>
#include <QKeySequence>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QTimer>
#include <QVBoxLayout>
#include <QWheelEvent>

#include <algorithm>
#include <cmath>

using scene_doc::Document;
using scene_doc::Float3;
using scene_doc::Light;
using scene_doc::LightKind;
using scene_doc::MaterialKind;
using scene_doc::Object;
using scene_doc::Rgb;
using scene_doc::ShapeKind;

using namespace scene_builder_ui;

// =====================================================================================================================================
// SceneBuilderWidget
// =====================================================================================================================================

SceneBuilderWidget::SceneBuilderWidget(QWidget *parent) : QWidget(parent) {
	m_autosaveEnabled = !qEnvironmentVariableIsSet("RT_GUI_SELFTEST");
	m_autosaveTimer = new QTimer(this);
	m_autosaveTimer->setSingleShot(true);
	m_autosaveTimer->setInterval(1500);
	connect(m_autosaveTimer, &QTimer::timeout, this, &SceneBuilderWidget::writeAutosave);

	buildUi();
	if (!(m_autosaveEnabled && loadAutosave())) newScene();
}

void SceneBuilderWidget::showEvent(QShowEvent *e) {
	QWidget::showEvent(e);
	// The panel can be dragged as narrow as the three buttons above the list allow, whatever the language and theme padding: that width comes from
	// their (now styled) size hints, not from a fixed number.
	if (!m_leftPanel) return;
	m_leftPanel->setMinimumWidth(m_addButton->sizeHint().width() + m_duplicateButton->sizeHint().width() + m_deleteButton->sizeHint().width() + 2 * 6);
}

SceneBuilderWidget::~SceneBuilderWidget() {
	if (m_process) {
		m_process->disconnect(this);
		m_process->kill();
		m_process->waitForFinished(2000);
	}
	if (m_dirty && m_autosaveEnabled) writeAutosave();
}

QString SceneBuilderWidget::workFolder() const {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::TempLocation) + "/ray_tracer_scene_builder";
	QDir().mkpath(dir);
	return dir;
}

QString SceneBuilderWidget::launcherPath() const {
#ifdef Q_OS_WIN
	return QCoreApplication::applicationDirPath() + "/ray_tracer.exe";
#else
	return QCoreApplication::applicationDirPath() + "/ray_tracer";
#endif
}

void SceneBuilderWidget::buildUi() {
	auto *root = new QVBoxLayout(this);

	// ---- top bar
	auto *bar = new QHBoxLayout;
	auto button = [this](const QString &text, const QString &tip) {
		auto *b = new QPushButton(text, this);
		b->setToolTip(tip);
		b->setAutoDefault(false);
		return b;
	};
	auto *newB = button(tr("New"), tr("Start again from the example scene"));
	auto *openB = button(tr("Open..."), tr("Open a .pbrt file saved by the Scene Builder"));
	auto *saveB = button(tr("Save"), tr("Save the scene as a .pbrt file"));
	auto *saveAsB = button(tr("Save As..."), tr("Save the scene under a new name"));
	auto *listB = button(tr("Add to scene list"), tr("Save the scene into the scenes folder so it shows up in the Settings tab"));
	m_undoButton = button(tr("Undo"), tr("Undo the last change (%1)").arg(shortcutText(QKeySequence::Undo)));
	m_redoButton = button(tr("Redo"), tr("Redo (%1)").arg(shortcutText(QKeySequence::Redo)));
	m_titleLabel = new QLabel(this);
	m_titleLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
	// The scene's name: what the scene list shows, and what a copy added to the list is named after. Edited right here (undoable like any change).
	m_titleEdit = new QLineEdit(this);
	m_titleEdit->setPlaceholderText(tr("Scene name"));
	m_titleEdit->setToolTip(tr("The name of this scene, shown in the scene list"));
	m_titleEdit->setMinimumWidth(180);
	m_titleEdit->setMaximumWidth(280);
	for (QPushButton *b : {newB, openB, saveB, saveAsB, listB}) bar->addWidget(b);
	bar->addSpacing(12);
	bar->addWidget(m_undoButton);
	bar->addWidget(m_redoButton);
	bar->addStretch(1);
	bar->addWidget(new QLabel(tr("Name:"), this));
	bar->addWidget(m_titleEdit);
	bar->addWidget(m_titleLabel);
	root->addLayout(bar);
	connect(m_titleEdit, &QLineEdit::textEdited, this, &SceneBuilderWidget::setSceneName);
	connect(newB, &QPushButton::clicked, this, [this]() { if (confirmDiscard()) newScene(); });
	connect(openB, &QPushButton::clicked, this, &SceneBuilderWidget::onOpenClicked);
	connect(saveB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveClicked);
	connect(saveAsB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveAsClicked);
	connect(listB, &QPushButton::clicked, this, &SceneBuilderWidget::onSaveToSceneListClicked);
	connect(m_undoButton, &QPushButton::clicked, this, [this]() { undo(); });
	connect(m_redoButton, &QPushButton::clicked, this, [this]() { redo(); });

	auto *split = new QSplitter(Qt::Horizontal, this);
	root->addWidget(split, 1);

	// ---- left: scene contents
	auto *left = new QWidget(split);
	auto *leftLayout = new QVBoxLayout(left);
	leftLayout->setContentsMargins(0, 0, 0, 0);
	auto *addB = new QPushButton(tr("Add"), left);
	addB->setAutoDefault(false);
	auto *addMenu = new QMenu(addB);
	addMenu->addSection(tr("Objects"));
	for (ShapeKind k : scene_doc::allShapeKinds())
		addMenu->addAction(shapeLabel(k), this, [this, k]() { addObject(k); });
	addMenu->addSection(tr("Props (several objects at once)"));
	for (scene_doc::PropKind k : scene_doc::allPropKinds()) addMenu->addAction(propLabel(k), this, [this, k]() { addProp(k); });
	addMenu->addSection(tr("More"));
	addMenu->addAction(tr("Object from a photo..."), this, [this]() { addObjectFromPhoto(); });
	addMenu->addAction(tr("Light panel (emitting quad)"), this, [this]() {
		edit(QString(), [this]() {
			scene_doc::Object o = scene_doc::makeAreaLightPanel(QString("Light panel %1").arg(m_doc.objects.size() + 1).toStdString());
			m_doc.objects.push_back(o);
			m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
		});
		rebuildList();
		setSelection(m_sel);
	});
	addMenu->addSection(tr("Lights"));
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite})
		addMenu->addAction(lightLabel(k), this, [this, k]() { addLight(k); });
	addB->setMenu(addMenu);
	m_duplicateButton = new QPushButton(tr("Duplicate"), left);
	m_deleteButton = new QPushButton(tr("Delete"), left);
	m_duplicateButton->setAutoDefault(false);
	m_deleteButton->setAutoDefault(false);
	for (QPushButton *b : {addB, m_duplicateButton, m_deleteButton}) b->setStyleSheet("padding: 6px 10px;");  // compact, so the list column can be narrow
	auto *row = new QHBoxLayout;
	row->addWidget(addB);
	row->addWidget(m_duplicateButton);
	row->addWidget(m_deleteButton);
	leftLayout->addLayout(row);
	m_list = new QListWidget(left);
	m_list->setObjectName("sceneBuilderList");  // sized by the application stylesheet (mainwindow_style.cpp)
	leftLayout->addWidget(m_list, 1);
	left->setMinimumWidth(0);  // showEvent() sets the real minimum from the buttons
	m_leftPanel = left;
	m_addButton = addB;
	connect(m_list, &QListWidget::currentRowChanged, this, [this](int) { onListSelectionChanged(); });
	connect(m_deleteButton, &QPushButton::clicked, this, [this]() { deleteSelected(); });
	connect(m_duplicateButton, &QPushButton::clicked, this, [this]() {
		if (m_sel.kind == SelKind::Object && m_sel.index < static_cast<int>(m_doc.objects.size())) {
			edit(QString(), [this]() {
				Object o = m_doc.objects[m_sel.index];
				o.name += " copy";
				o.position.x += 0.5;
				m_doc.objects.push_back(o);
				m_sel.index = static_cast<int>(m_doc.objects.size()) - 1;
			});
		} else if (m_sel.kind == SelKind::Light && m_sel.index < static_cast<int>(m_doc.lights.size())) {
			edit(QString(), [this]() {
				Light l = m_doc.lights[m_sel.index];
				l.name += " copy";
				l.position.x += 0.5;
				m_doc.lights.push_back(l);
				m_sel.index = static_cast<int>(m_doc.lights.size()) - 1;
			});
		} else {
			return;
		}
		rebuildList();
		setSelection(m_sel);
	});
	// Delete and undo work from the list and the layout view (not inside a text box, where Delete edits text).
	// (Backspace too: a Mac's "delete" key is Backspace, which the standard Delete shortcut does not include there.)
	for (const QKeySequence &key : {QKeySequence(QKeySequence::Delete), QKeySequence(Qt::Key_Backspace)}) {
		auto *delList = new QShortcut(key, m_list);
		delList->setContext(Qt::WidgetShortcut);
		connect(delList, &QShortcut::activated, this, [this]() { deleteSelected(); });
	}

	// ---- centre: layout view above the preview
	auto *centre = new QSplitter(Qt::Vertical, split);
	auto *layoutBox = new QWidget(centre);
	auto *layoutLayout = new QVBoxLayout(layoutBox);
	layoutLayout->setContentsMargins(0, 0, 0, 0);
	createViews(layoutBox, layoutLayout);  // scene_builder_views.cpp: the Top / Front / Side / 3D views and what they share

	auto *previewBox = new QWidget;
	auto *previewLayout = new QVBoxLayout(previewBox);
	previewLayout->setContentsMargins(0, 0, 0, 0);
	auto *renderRow = new QHBoxLayout;
	m_qualityCombo = new QComboBox(previewBox);
	m_qualityCombo->addItem(tr("Draft"), 0);
	m_qualityCombo->addItem(tr("Good"), 1);
	m_qualityCombo->addItem(tr("Best"), 2);
	m_qualityCombo->setCurrentIndex(0);
	m_qualityCombo->setToolTip(tr("Draft: 320 pixels wide, 16 samples. Good: 480 wide, 64 samples. Best: 640 wide, 256 samples."));
	m_gpuCheck = new QCheckBox(tr("Use the GPU"), previewBox);
	m_gpuCheck->setToolTip(tr("Render on the graphics card (NVIDIA OptiX on Windows, Metal on a Mac). Much faster for large pictures; needs a supported GPU."));
	m_previewButton = new QPushButton(tr("Preview"), previewBox);
	m_previewButton->setAutoDefault(false);
	m_finalButton = new QPushButton(tr("Render picture..."), previewBox);
	m_finalButton->setAutoDefault(false);
	m_finalButton->setToolTip(tr("Render at the image size and sample count set under Camera, and save the picture as a PNG"));
	m_previewStatus = new QLabel(previewBox);
	renderRow->addWidget(new QLabel(tr("Quality:"), previewBox));
	renderRow->addWidget(m_qualityCombo);
	renderRow->addWidget(m_gpuCheck);
	renderRow->addWidget(m_previewButton, 1);
	renderRow->addWidget(m_finalButton, 1);
	m_previewStatus->setWordWrap(true);
	previewLayout->addLayout(renderRow);
	m_previewLabel = new QLabel(previewBox);
	m_previewLabel->setAlignment(Qt::AlignCenter);
	m_previewLabel->setMinimumHeight(120);
	m_previewLabel->installEventFilter(this);  // rescale the picture when the pane is resized
	m_previewLabel->setText(tr("Press Preview to see the scene."));
	m_previewLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
	previewLayout->addWidget(m_previewLabel, 1);
	previewLayout->addWidget(m_previewStatus);
	connect(m_previewButton, &QPushButton::clicked, this, [this]() {
		if (m_process) {
			m_cancelRequested = true;
			m_process->kill();  // the button reads Cancel while a render runs
			return;
		}
		startPreview();
	});
	connect(m_finalButton, &QPushButton::clicked, this, &SceneBuilderWidget::onRenderFinalClicked);

	// ---- right: inspector
	auto *right = new QWidget(split);
	auto *inspectorPane = right;
	auto *rightLayout = new QVBoxLayout(inspectorPane);
	rightLayout->setContentsMargins(0, 0, 0, 0);
	m_inspectorScroll = new QScrollArea(inspectorPane);
	m_inspectorScroll->setWidgetResizable(true);
	m_inspectorScroll->setFrameShape(QFrame::NoFrame);
	rightLayout->addWidget(m_inspectorScroll, 1);
	m_problemsLabel = new QLabel(inspectorPane);
	m_problemsLabel->setWordWrap(true);
	m_problemsLabel->setTextFormat(Qt::RichText);
	rightLayout->addWidget(m_problemsLabel);
	m_inspectorPanel = right;  // rebuildInspector() sets its minimum width from the page it shows
	centre->addWidget(previewBox);
	centre->setStretchFactor(0, 3);
	centre->setStretchFactor(1, 2);
	centre->setSizes({500, 280});
	// The view and the preview keep a usable height: in a window too short for both, the column scrolls instead of squeezing them.
	layoutBox->setMinimumHeight(520);
	previewBox->setMinimumHeight(360);

	split->addWidget(left);
	auto *centreScroll = new QScrollArea(split);
	centreScroll->setWidget(centre);
	centreScroll->setWidgetResizable(true);
	centreScroll->setFrameShape(QFrame::NoFrame);
	centreScroll->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);  // always there, so it is clear the column scrolls when the window is short
	split->addWidget(centreScroll);
	split->addWidget(right);
	split->setStretchFactor(0, 0);
	split->setStretchFactor(1, 1);
	split->setStretchFactor(2, 0);
	split->setSizes({270, 640, 420});
}

// ---- document state -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::newScene() {
	m_doc = scene_doc::makeStarterScene();
	m_path.clear();
	m_listedPath.clear();
	m_dirty = false;
	m_undo.clear();
	m_redo.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	frameViews();
	refreshProblems();
	updateTitle();
	updateActions();
}

bool SceneBuilderWidget::openFile(const QString &path, QString *error) {
	QFile f(path);
	if (!f.open(QIODevice::ReadOnly)) {
		if (error) *error = tr("Cannot open %1.").arg(path);
		return false;
	}
	const std::string text = f.readAll().toStdString();
	Document d;
	std::string err;
	if (!scene_doc::fromPbrt(text, d, err)) {
		if (error) *error = QString::fromStdString(err);
		return false;
	}
	m_doc = std::move(d);
	m_path = path;
	// A file opened from the scene-list folder is a listed scene: adding it again offers to update it.
	m_listedPath = QFileInfo(path).absolutePath() == QFileInfo(sceneListFolder() + "/x").absolutePath() ? path : QString();
	m_dirty = false;
	m_undo.clear();
	m_redo.clear();
	m_lastEditKey.clear();
	m_sel = {SelKind::None, 0};
	clearAutosave();
	rebuildList();
	setSelection(m_sel);
	frameViews();
	refreshProblems();
	updateTitle();
	updateActions();
	return true;
}

// Writes the scene as pbrt text to `path`, without touching which file the document belongs to.
static bool writeSceneText(const Document &doc, const QString &path) {
	QFile f(path);
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return false;
	const std::string text = scene_doc::toPbrt(doc);
	if (f.write(text.data(), static_cast<qint64>(text.size())) != static_cast<qint64>(text.size())) return false;
	f.close();
	return true;
}

bool SceneBuilderWidget::saveFile(const QString &path) {
	if (!writeSceneText(m_doc, path)) return false;
	m_path = path;
	m_dirty = false;
	clearAutosave();
	updateTitle();
	return true;
}

bool SceneBuilderWidget::confirmDiscard() {
	if (!m_dirty) return true;
	const auto answer = QMessageBox::question(this, tr("Unsaved changes"), tr("The scene has changes that are not saved. Save them first?"),
	                                          QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel, QMessageBox::Save);
	if (answer == QMessageBox::Cancel) return false;
	if (answer == QMessageBox::Save) {
		onSaveClicked();
		return !m_dirty;
	}
	return true;
}

void SceneBuilderWidget::onOpenClicked() {
	if (!confirmDiscard()) return;
	const QString path = QFileDialog::getOpenFileName(this, tr("Open a Scene Builder scene"), m_path.isEmpty() ? QDir::homePath() : QFileInfo(m_path).absolutePath(),
	                                                  tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	QString error;
	if (!openFile(path, &error)) QMessageBox::warning(this, tr("Cannot open the scene"), error);
}

void SceneBuilderWidget::onSaveClicked() {
	if (m_path.isEmpty()) {
		onSaveAsClicked();
		return;
	}
	if (!saveFile(m_path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(m_path));
	else emit statusMessage(tr("Saved %1").arg(m_path));
}

void SceneBuilderWidget::onSaveAsClicked() {
	QString start = m_path;
	if (start.isEmpty()) start = QDir::homePath() + "/" + QString::fromStdString(m_doc.title).replace(QRegularExpression("[^A-Za-z0-9_-]+"), "-") + ".pbrt";
	QString path = QFileDialog::getSaveFileName(this, tr("Save the scene"), start, tr("pbrt scenes (*.pbrt)"));
	if (path.isEmpty()) return;
	if (!path.endsWith(".pbrt", Qt::CaseInsensitive)) path += ".pbrt";
	if (!saveFile(path)) QMessageBox::warning(this, tr("Cannot save"), tr("Could not write %1.").arg(path));
	else emit statusMessage(tr("Saved %1").arg(path));
}

// Where "Add to scene list" saves: the per-user folder <data>/user_scenes (pbrt_discover::userSceneDir), always. That is the only folder the scene list can grow
// from while the program runs (refresh_user_scenes() rescans it, and gives each scene a persistent id), it is always writable, and it never writes into the
// program's own folder (a macOS .app bundle's seal breaks if you do; a disk image is read-only). A scene saved into the program's pbrt_scenes folder would
// only be listed after a restart, and its id would shift the ids of the scenes found after it. RAY_TRACER_PBRT_DIR still names a folder to use instead.
QString SceneBuilderWidget::sceneListFolder() {
	const QString env = qEnvironmentVariable("RAY_TRACER_PBRT_DIR");
	if (!env.isEmpty() && QDir(env).exists() && QFileInfo(env).isWritable()) return QDir(env).absolutePath();
	const QString user = QString::fromStdString(pbrt_asset_check::userSceneDir());
	if (!user.isEmpty() && QDir().mkpath(user)) return user;
	return QString();
}

QString SceneBuilderWidget::addToSceneList(QString *error, bool update) {
	const QString folder = sceneListFolder();
	if (folder.isEmpty()) {
		if (error) *error = tr("The scenes folder (pbrt_scenes) was not found next to the program. Use Save As to put the file where you like, and set the "
		                       "environment variable RAY_TRACER_PBRT_DIR to that folder to have the program list it.");
		return QString();
	}
	QString name = QString::fromStdString(m_doc.title).trimmed().toLower().replace(QRegularExpression("[^a-z0-9]+"), "-");
	name.remove(QRegularExpression("^-+|-+$"));
	if (name.isEmpty()) name = "my-scene";
	QString path = folder + "/" + name + ".pbrt";
	if (update && !m_listedPath.isEmpty() && QFileInfo::exists(m_listedPath)) {
		path = m_listedPath;   // the listing this document already has
	} else {
		// A name already taken (by another scene, or by an earlier listing of this one) is never overwritten: the copy gets the next free name.
		for (int n = 2; QFileInfo::exists(path); ++n) path = folder + "/" + name + "-" + QString::number(n) + ".pbrt";
	}
	// A copy for the scene list: the document keeps its own file and its unsaved state.
	if (!writeSceneText(m_doc, path)) {
		if (error) *error = tr("Could not write %1.").arg(path);
		return QString();
	}
	m_listedPath = path;
	return path;
}

void SceneBuilderWidget::onSaveToSceneListClicked() {
	const QString folder = sceneListFolder();
	if (folder.isEmpty()) {
		QString why;
		addToSceneList(&why);
		QMessageBox::information(this, tr("No scenes folder"), why);
		return;
	}
	// Already listed once (or opened from the list): ask whether this is an update of that scene or a new one. Otherwise there is nothing to ask - a
	// name that is taken just gets the next free one.
	bool update = false;
	if (!m_listedPath.isEmpty() && QFileInfo::exists(m_listedPath)) {
		QMessageBox box(QMessageBox::Question, tr("Add to the scene list"),
		                tr("This scene is already in the list as \"%1\".").arg(QFileInfo(m_listedPath).completeBaseName()), QMessageBox::NoButton, this);
		QPushButton *asNew = box.addButton(tr("Add as a new scene"), QMessageBox::AcceptRole);
		QPushButton *updateIt = box.addButton(tr("Update the existing one"), QMessageBox::DestructiveRole);
		box.addButton(QMessageBox::Cancel);
		box.setDefaultButton(asNew);
		box.exec();
		if (box.clickedButton() == updateIt) update = true;
		else if (box.clickedButton() != asNew) return;
	}
	QString error;
	const QString path = addToSceneList(&error, update);
	if (path.isEmpty()) {
		QMessageBox::warning(this, tr("Cannot save"), error);
		return;
	}
	emit sceneListed(path);   // the main window lists it and selects it - no restart needed
	QMessageBox::information(this, tr("Added to the scene list"),
	                         tr("Saved a copy as %1.\n\nIt is in the scene list now (Settings tab, My Scenes).").arg(path));
}

// ---- undo, autosave -------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::pushUndo() {
	m_undo.append(QString::fromStdString(scene_doc::toJson(m_doc)));
	if (m_undo.size() > 200) m_undo.removeFirst();
	m_redo.clear();
}

void SceneBuilderWidget::edit(const QString &key, const std::function<void()> &mutate) {
	const bool merge = !key.isEmpty() && key == m_lastEditKey && (key.startsWith("drag#") || (m_editClock.isValid() && m_editClock.elapsed() < 1200));
	if (!merge) pushUndo();
	mutate();
	m_lastEditKey = key;
	m_editClock.restart();
	documentChanged();
}

void SceneBuilderWidget::restore(const QString &json) {
	Document d;
	std::string err;
	if (!scene_doc::fromJson(json.toStdString(), d, err)) return;
	m_doc = std::move(d);
	if (m_sel.kind == SelKind::Object && m_sel.index >= static_cast<int>(m_doc.objects.size())) m_sel = {SelKind::None, 0};
	if (m_sel.kind == SelKind::Light && m_sel.index >= static_cast<int>(m_doc.lights.size())) m_sel = {SelKind::None, 0};
	m_lastEditKey.clear();
	rebuildList();
	setSelection(m_sel);
	documentChanged();
}

bool SceneBuilderWidget::undo() {
	if (m_undo.isEmpty()) return false;
	const QString now = QString::fromStdString(scene_doc::toJson(m_doc));
	const QString prev = m_undo.takeLast();
	m_redo.append(now);
	restore(prev);
	return true;
}

bool SceneBuilderWidget::redo() {
	if (m_redo.isEmpty()) return false;
	const QString now = QString::fromStdString(scene_doc::toJson(m_doc));
	const QString next = m_redo.takeLast();
	m_undo.append(now);
	restore(next);
	return true;
}

void SceneBuilderWidget::documentChanged() {
	m_dirty = true;
	refreshListLabels();
	refreshProblems();
	updateTitle();
	updateActions();
	updateViews();
	scheduleAutosave();
}

static QString autosavePath() {
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
	QDir().mkpath(dir);
	return dir + "/scene_builder_autosave.pbrt";
}

void SceneBuilderWidget::scheduleAutosave() {
	if (m_autosaveEnabled) m_autosaveTimer->start();
}

void SceneBuilderWidget::writeAutosave() {
	if (!m_autosaveEnabled || !m_dirty) return;
	QFile f(autosavePath());
	if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return;
	const std::string text = scene_doc::toPbrt(m_doc);
	f.write(text.data(), static_cast<qint64>(text.size()));
}

void SceneBuilderWidget::clearAutosave() {
	if (m_autosaveEnabled) QFile::remove(autosavePath());
}

// A scene that was being edited when the program closed comes back, marked unsaved.
bool SceneBuilderWidget::loadAutosave() {
	QFile f(autosavePath());
	if (!f.exists() || !f.open(QIODevice::ReadOnly)) return false;
	Document d;
	std::string err;
	if (!scene_doc::fromPbrt(f.readAll().toStdString(), d, err)) return false;
	m_doc = std::move(d);
	m_path.clear();
	m_dirty = true;
	rebuildList();
	setSelection({SelKind::None, 0});
	frameViews();
	refreshProblems();
	updateTitle();
	updateActions();
	return true;
}

// ---- list, selection ------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::rebuildList() {
	QSignalBlocker block(m_list);
	m_list->clear();
	auto add = [this](const QString &text, SelKind kind, int index) {
		auto *item = new QListWidgetItem(text, m_list);
		item->setData(Qt::UserRole, static_cast<int>(kind));
		item->setData(Qt::UserRole + 1, index);
	};
	add(tr("Camera and image"), SelKind::Camera, 0);
	for (int i = 0; i < static_cast<int>(m_doc.objects.size()); ++i) add(QString(), SelKind::Object, i);
	for (int i = 0; i < static_cast<int>(m_doc.lights.size()); ++i) add(QString(), SelKind::Light, i);
	refreshListLabels();
	// reselect
	for (int r = 0; r < m_list->count(); ++r) {
		const auto *it = m_list->item(r);
		if (it->data(Qt::UserRole).toInt() == static_cast<int>(m_sel.kind) && it->data(Qt::UserRole + 1).toInt() == m_sel.index) {
			m_list->setCurrentRow(r);
			break;
		}
	}
}

void SceneBuilderWidget::refreshListLabels() {
	for (int r = 0; r < m_list->count(); ++r) {
		QListWidgetItem *it = m_list->item(r);
		const auto kind = static_cast<SelKind>(it->data(Qt::UserRole).toInt());
		const int i = it->data(Qt::UserRole + 1).toInt();
		if (kind == SelKind::Object && i < static_cast<int>(m_doc.objects.size())) {
			const Object &o = m_doc.objects[i];
			it->setText(QString("%1  -  %2%3").arg(QString::fromStdString(o.name), shapeLabel(o.shape).section(' ', 0, 0), o.emissive ? tr(", light") : QString()));
		} else if (kind == SelKind::Light && i < static_cast<int>(m_doc.lights.size())) {
			const Light &l = m_doc.lights[i];
			const QString type = lightLabel(l.kind).section(" (", 0, 0);
			const QString name = QString::fromStdString(l.name);
			it->setText(name.compare(type, Qt::CaseInsensitive) == 0 || name.startsWith(type.section(' ', 0, 0)) ? name : QString("%1  -  %2").arg(name, type));
		}
	}
}

void SceneBuilderWidget::onListSelectionChanged() {
	QListWidgetItem *it = m_list->currentItem();
	if (!it) {
		setSelection({SelKind::None, 0}, true);
		return;
	}
	setSelection({static_cast<SelKind>(it->data(Qt::UserRole).toInt()), it->data(Qt::UserRole + 1).toInt()}, true);
}

void SceneBuilderWidget::setSelection(const BuilderSelection &s, bool fromList) {
	m_sel = s;
	if (!fromList) {
		QSignalBlocker block(m_list);
		m_list->setCurrentRow(-1);
		for (int r = 0; r < m_list->count(); ++r) {
			const auto *it = m_list->item(r);
			if (it->data(Qt::UserRole).toInt() == static_cast<int>(s.kind) && it->data(Qt::UserRole + 1).toInt() == s.index) {
				m_list->setCurrentRow(r);
				break;
			}
		}
	}
	selectInViews(s);
	rebuildInspector();
	updateActions();
}

// Presses on object `index` in the layout view, drags by `deltaPx` pixels and releases, with real mouse events; returns whether the object moved.
bool SceneBuilderWidget::dragObjectForTest(int index, const QPointF &deltaPx) {
	if (index < 0 || index >= static_cast<int>(m_doc.objects.size())) return false;
	const Float3 before = m_doc.objects[index].position;
	const QPointF start = m_view->itemScreenPos({SelKind::Object, index});
	auto send = [this](QEvent::Type type, const QPointF &pos, Qt::MouseButton button, Qt::MouseButtons buttons) {
		QMouseEvent e(type, pos, m_view->mapToGlobal(pos), button, buttons, Qt::NoModifier);
		QApplication::sendEvent(m_view, &e);
	};
	send(QEvent::MouseButtonPress, start, Qt::LeftButton, Qt::LeftButton);
	for (int step = 1; step <= 4; ++step) send(QEvent::MouseMove, start + deltaPx * (step / 4.0), Qt::NoButton, Qt::LeftButton);
	send(QEvent::MouseButtonRelease, start + deltaPx, Qt::LeftButton, Qt::NoButton);
	const Float3 after = m_doc.objects[index].position;
	return before.x != after.x || before.y != after.y || before.z != after.z;
}

void SceneBuilderWidget::selectObject(int index) {
	setSelection({SelKind::Object, index});
}

// The prop's objects go in together at the drop point, named alike ("Table top", "Table leg 1", ...; a second table gets "Table top 2"...), and the first
// is selected. They are ordinary objects from then on.
void SceneBuilderWidget::addProp(scene_doc::PropKind kind) {
	edit(QString(), [&]() {
		std::vector<Object> parts = scene_doc::makeProp(kind);
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		int copy = 1;   // the first number whose suffix leaves every part's name free
		for (;; ++copy) {
			bool free = true;
			for (const Object &p : parts) free = free && !names.contains(QString::fromStdString(p.name) + (copy > 1 ? QString(" %1").arg(copy) : QString()));
			if (free) break;
		}
		const Float3 c = dropPoint();
		const int first = static_cast<int>(m_doc.objects.size());
		for (Object &p : parts) {
			if (copy > 1) p.name += " " + std::to_string(copy);
			p.position.x += c.x;
			p.position.y += c.y;
			p.position.z += c.z;
			m_doc.objects.push_back(p);
		}
		m_sel = {SelKind::Object, first};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::addObject(ShapeKind shape) {
	std::string fileName;
	if (shape == ShapeKind::Mesh) {
		const QString f = QFileDialog::getOpenFileName(this, tr("Choose a mesh"), QString(), tr("Meshes (*.ply *.obj)"));
		if (f.isEmpty()) return;
		fileName = f.toStdString();
	}
	edit(QString(), [&]() {
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		const QString name = uniqueName(shapeLabel(shape).section(' ', 0, 0), names);
		Object o = scene_doc::makeObject(shape, name.toStdString());
		o.meshFile = fileName;
		// Drop it at the middle of the layout view, keeping the shape's own height above the floor where the view does not show it.
		const Float3 c = dropPoint();
		o.position.x = c.x;
		o.position.z = c.z;
		if (c.y != 0.0) o.position.y = c.y;
		m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::addLight(LightKind kind) {
	edit(QString(), [&]() {
		Light l;
		l.kind = kind;
		QStringList names;
		for (const Light &existing : m_doc.lights) names << QString::fromStdString(existing.name);
		l.name = uniqueName(lightLabel(kind).section(' ', 0, 0), names).toStdString();
		switch (kind) {
			case LightKind::Point: l.position = {1.0, 4.0, 2.0}; l.intensity = 40.0; break;
			case LightKind::Spot: l.position = {0.0, 5.0, 2.0}; l.target = {0.0, 0.0, 0.0}; l.intensity = 120.0; break;
			case LightKind::Distant: l.position = {4.0, 6.0, 3.0}; l.target = {0.0, 0.0, 0.0}; l.color = {1.0, 0.95, 0.85}; l.intensity = 3.0; break;
			case LightKind::Infinite: l.color = {0.55, 0.70, 1.0}; l.intensity = 0.5; break;
		}
		const Float3 c = dropPoint();
		if (kind == LightKind::Point || kind == LightKind::Spot) { l.position.x = c.x; l.position.z = c.z; }
		m_doc.lights.push_back(l);
		m_sel = {SelKind::Light, static_cast<int>(m_doc.lights.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::deleteSelected() {
	if (m_sel.kind == SelKind::Object && m_sel.index < static_cast<int>(m_doc.objects.size())) {
		edit(QString(), [this]() { m_doc.objects.erase(m_doc.objects.begin() + m_sel.index); });
	} else if (m_sel.kind == SelKind::Light && m_sel.index < static_cast<int>(m_doc.lights.size())) {
		edit(QString(), [this]() { m_doc.lights.erase(m_doc.lights.begin() + m_sel.index); });
	} else {
		return;
	}
	m_sel = {SelKind::None, 0};
	rebuildList();
	setSelection(m_sel);
}

void SceneBuilderWidget::updateActions() {
	const bool item = (m_sel.kind == SelKind::Object || m_sel.kind == SelKind::Light);
	m_deleteButton->setEnabled(item);
	m_duplicateButton->setEnabled(item);
	m_undoButton->setEnabled(!m_undo.isEmpty());
	m_redoButton->setEnabled(!m_redo.isEmpty());
	const bool ok = !scene_doc::hasErrors(scene_doc::validate(m_doc));
	m_previewButton->setEnabled(ok || m_process);
	m_finalButton->setEnabled(ok && !m_process);
}

void SceneBuilderWidget::updateTitle() {
	const QString file = m_path.isEmpty() ? tr("not saved yet") : QFileInfo(m_path).fileName();
	const QString name = QString::fromStdString(m_doc.title);
	if (m_titleEdit && m_titleEdit->text() != name) {   // an undo, a new scene, an open: not the user typing
		const QSignalBlocker blocker(m_titleEdit);
		m_titleEdit->setText(name);
	}
	m_titleLabel->setText(tr("(%1)%2  |  %3 objects, %4 lights")
	                          .arg(file, m_dirty ? " *" : "")
	                          .arg(m_doc.objects.size())
	                          .arg(m_doc.lights.size()));
}

QString SceneBuilderWidget::problemsText() const {
	QString s;
	for (const auto &p : scene_doc::validate(m_doc))
		s += (p.severity == scene_doc::Problem::Severity::Error ? "Error: " : "Note: ") + QString::fromStdString(p.message) + "\n";
	return s;
}

void SceneBuilderWidget::refreshProblems() {
	const auto problems = scene_doc::validate(m_doc);
	if (problems.empty()) {
		m_problemsLabel->setText(tr("No problems found."));
		return;
	}
	QString html;
	for (const auto &p : problems) {
		const bool error = p.severity == scene_doc::Problem::Severity::Error;
		html += QString("<p style='margin:2px 0'><b>%1</b> %2</p>").arg(error ? tr("Fix this:") : tr("Note:"), QString::fromStdString(p.message).toHtmlEscaped());
	}
	m_problemsLabel->setText(html);
}

// ---- inspector ------------------------------------------------------------------------------------------------------------------

// ---- rendering -------------------------------------------------------------------------------------------------------------------

void SceneBuilderWidget::setUseGpu(bool on) { m_gpuCheck->setChecked(on); }

void SceneBuilderWidget::setSceneName(const QString &text) {
	const std::string name = text.toStdString();
	if (name == m_doc.title) return;
	edit(QStringLiteral("title"), [&]() { m_doc.title = name; });
}
void SceneBuilderWidget::startPreview(const std::function<void(bool, const QString &)> &done) {
	static const int widths[3] = {320, 480, 640};
	static const int spps[3] = {16, 64, 256};
	const int q = std::clamp(m_qualityCombo->currentData().toInt(), 0, 2);
	const double aspect = m_doc.render.width > 0 ? double(m_doc.render.height) / m_doc.render.width : 0.75;
	const int w = widths[q];
	const int h = std::max(1, static_cast<int>(std::lround(w * aspect)));
	runRender(w, h, spps[q], false, QString(), done);
}

void SceneBuilderWidget::onRenderFinalClicked() {
	QString start = m_path.isEmpty() ? QDir::homePath() + "/render.png" : QFileInfo(m_path).absolutePath() + "/" + QFileInfo(m_path).completeBaseName() + ".png";
	QString png = QFileDialog::getSaveFileName(this, tr("Save the rendered picture"), start, tr("PNG images (*.png)"));
	if (png.isEmpty()) return;
	if (!png.endsWith(".png", Qt::CaseInsensitive)) png += ".png";
	runRender(m_doc.render.width, m_doc.render.height, m_doc.render.samples, true, png, [this](bool ok, const QString &msg) {
		if (!ok) QMessageBox::warning(this, tr("The render failed"), msg);
	});
}

void SceneBuilderWidget::runRender(int width, int height, int samples, bool toFinalFile, const QString &finalPng,
                                   const std::function<void(bool, const QString &)> &done) {
	auto fail = [&](const QString &msg) {
		m_previewStatus->setText(msg);
		if (done) done(false, msg);
	};
	if (m_process) return fail(tr("A render is already running."));
	const auto problems = scene_doc::validate(m_doc);
	if (scene_doc::hasErrors(problems)) return fail(tr("Fix the problems listed under the properties first."));
	if (!QFileInfo::exists(launcherPath())) return fail(tr("The renderer (%1) was not found next to the program.").arg(launcherPath()));

	const QString dir = workFolder();
	const QString scene = dir + "/scene.pbrt";
	const QString base = dir + (toFinalFile ? "/final" : "/preview");
	QFile::remove(base + ".ppm");
	QFile::remove(base + ".png");
	{
		QFile f(scene);
		if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate)) return fail(tr("Could not write %1.").arg(scene));
		const std::string text = scene_doc::toPbrt(m_doc);
		f.write(text.data(), static_cast<qint64>(text.size()));
	}

	QStringList args;
	args << (m_gpuCheck->isChecked() ? "--gpu" : "--cpu") << "--output" << base + ".ppm" << "--height" << QString::number(height)
	     << QString::number(width) << QString::number(samples) << QString::number(m_doc.render.maxDepth) << scene;

	m_pendingFinalPng = toFinalFile ? finalPng : QString();
	m_previewPng = base + ".png";
	m_pendingDone = done;
	m_renderLog.clear();
	m_cancelRequested = false;
	m_renderClock.start();
	m_process = new QProcess(this);
	m_process->setProcessChannelMode(QProcess::MergedChannels);
	m_process->setWorkingDirectory(QCoreApplication::applicationDirPath());
	connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
		m_renderLog += QString::fromUtf8(m_process->readAll());
		if (m_renderLog.size() > 20000) m_renderLog.remove(0, m_renderLog.size() - 20000);
	});
	connect(m_process, &QProcess::errorOccurred, this, [this](QProcess::ProcessError e) {
		if (e == QProcess::FailedToStart) onPreviewFinished(-1);
	});
	connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), this, [this](int code, QProcess::ExitStatus st) {
		onPreviewFinished(st == QProcess::NormalExit ? code : -2);
	});
	m_previewButton->setText(tr("Cancel"));
	m_previewButton->setEnabled(true);
	m_finalButton->setEnabled(false);
	m_previewStatus->setText(tr("Rendering %1 x %2, %3 samples...").arg(width).arg(height).arg(samples));
	m_process->start(launcherPath(), args);
}

void SceneBuilderWidget::onPreviewFinished(int exitCode) {
	QProcess *p = m_process;
	if (!p) return;
	m_process = nullptr;
	p->disconnect(this);
	p->deleteLater();
	const auto done = m_pendingDone;
	m_pendingDone = nullptr;
	m_previewButton->setText(tr("Preview"));
	updateActions();

	const double secs = m_renderClock.elapsed() / 1000.0;
	QPixmap pix(m_previewPng);
	if (exitCode != 0 || pix.isNull()) {
		QString tail;
		const QStringList lines = m_renderLog.split(QRegularExpression("[\r\n]+"), Qt::SkipEmptyParts);
		for (int i = std::max(0, static_cast<int>(lines.size()) - 6); i < lines.size(); ++i) tail += lines[i] + "\n";
		const QString msg = (exitCode == -2 && m_cancelRequested) ? tr("The render was cancelled.") : exitCode == -2 ? tr("The renderer stopped unexpectedly.") : tr("The renderer did not produce a picture (exit code %1).\n%2").arg(exitCode).arg(tail.trimmed());
		m_previewStatus->setText((exitCode == -2 && m_cancelRequested) ? msg : tr("The render failed."));
		if (done) done(false, msg);
		return;
	}
	m_previewPixmap = pix;
	updatePreviewPixmap();
	QString message = tr("Done in %1 s (%2 x %3).").arg(secs, 0, 'f', 1).arg(pix.width()).arg(pix.height());
	if (!m_pendingFinalPng.isEmpty()) {
		QFile::remove(m_pendingFinalPng);
		if (QFile::copy(m_previewPng, m_pendingFinalPng)) message += " " + tr("Saved %1.").arg(m_pendingFinalPng);
		else message += " " + tr("Could not save to %1.").arg(m_pendingFinalPng);
	}
	m_previewStatus->setText(message);
	if (done) done(true, message);
}

void SceneBuilderWidget::updatePreviewPixmap() {
	if (m_previewPixmap.isNull()) return;
	m_previewLabel->setPixmap(m_previewPixmap.scaled(m_previewLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

bool SceneBuilderWidget::eventFilter(QObject *watched, QEvent *event) {
	if (watched == m_previewLabel && event->type() == QEvent::Resize) updatePreviewPixmap();
	return QWidget::eventFilter(watched, event);
}

void SceneBuilderWidget::resizeEvent(QResizeEvent *e) {
	QWidget::resizeEvent(e);
	updatePreviewPixmap();
}

// The Scene Builder's model library (Add > Model library...): ready-made meshes from the models/ folder (src/shared/model_library.h), shown with thumbnails.
// A model is added at a handy size (about 1.6 units across), centred on the drop point and standing on the floor, as one undo step.
#include "scene_builder_widget.h"

#include "app_log.h"
#include "scene_builder_common.h"

#include "../src/shared/mesh_preview.h"
#include "../src/shared/model_library.h"

#include <QCoreApplication>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMessageBox>
#include <QPushButton>
#include <QVBoxLayout>

using scene_doc::Object;
using scene_doc::ShapeKind;
using namespace scene_builder_ui;

// The folder the library's files are in: next to the installed app, or the source tree's models/ folder when running from a build inside it.
// Empty when none is found.
static QString modelsFolder() {
	const QString app = QCoreApplication::applicationDirPath();
	const QStringList candidates = {app + "/models", app + "/../models", app + "/../../models", QDir::currentPath() + "/models", QDir::currentPath() + "/../models"};
	for (const QString &c : candidates)
		if (QFileInfo(c + "/spot.obj").exists()) return QDir(c).absolutePath();
	for (const QString &c : candidates)
		for (const model_library::Entry &e : model_library::catalog())
			if (QFileInfo(c + "/" + e.stem + ".obj").exists()) return QDir(c).absolutePath();
	return QString();
}

bool SceneBuilderWidget::addLibraryModel(const QString &stem) {
	const model_library::Entry *entry = model_library::findEntry(stem.toUtf8().constData());
	const QString folder = modelsFolder();
	if (!entry || folder.isEmpty()) return false;
	const QString path = folder + "/" + stem + ".obj";
	if (!QFileInfo(path).exists()) return false;
	const mesh_preview::MeshPreview preview = mesh_preview::load(path.toStdString(), 1);
	if (!preview.ok) {
		AppLog::write(log_format::Level::Warn, QStringLiteral("models"), QStringLiteral("%1: %2").arg(path, QString::fromStdString(preview.error)));
		return false;
	}
	edit(QString(), [&]() {
		QStringList names;
		for (const Object &existing : m_doc.objects) names << QString::fromStdString(existing.name);
		Object o = scene_doc::makeObject(ShapeKind::Mesh, uniqueName(QString::fromUtf8(entry->name).section(" (", 0, 0), names).toStdString());
		o.meshFile = QDir::toNativeSeparators(path).toStdString();
		const scene_doc::Float3 c = dropPoint();
		const model_library::Placement p = model_library::placeOnFloor(preview.lo, preview.hi, c.x, c.z);
		o.meshScale = p.scale;
		o.position = {p.offset[0], p.offset[1], p.offset[2]};
		m_doc.objects.push_back(o);
		m_sel = {SelKind::Object, static_cast<int>(m_doc.objects.size()) - 1};
	});
	rebuildList();
	setSelection(m_sel);
	emit statusMessage(tr("Added %1 (%2 triangles). It stands on the floor, about 1.6 units across; use Scale to resize it.")
	                       .arg(QString::fromUtf8(entry->name), QString::number(static_cast<qulonglong>(preview.triangleCount))));
	return true;
}

void SceneBuilderWidget::showModelLibrary() {
	const QString folder = modelsFolder();
	QDialog dialog(this);
	dialog.setWindowTitle(tr("Model library"));
	dialog.resize(760, 560);
	auto *layout = new QVBoxLayout(&dialog);

	auto *filter = new QLineEdit(&dialog);
	filter->setPlaceholderText(tr("Search by name"));
	filter->setClearButtonEnabled(true);
	layout->addWidget(filter);

	auto *list = new QListWidget(&dialog);
	list->setViewMode(QListView::IconMode);
	list->setIconSize(QSize(128, 128));
	list->setResizeMode(QListView::Adjust);
	list->setMovement(QListView::Static);
	list->setSpacing(10);
	list->setWordWrap(true);
	list->setUniformItemSizes(true);
	list->setGridSize(QSize(150, 168));
	layout->addWidget(list, 1);

	int available = 0;
	if (!folder.isEmpty()) {
		for (const model_library::Entry &e : model_library::catalog()) {
			if (!QFileInfo(folder + "/" + e.stem + ".obj").exists()) continue;
			auto *item = new QListWidgetItem(QString::fromUtf8(e.name), list);
			const QString thumb = folder + "/thumbnails/" + e.stem + ".png";
			if (QFileInfo(thumb).exists()) item->setIcon(QIcon(thumb));
			item->setData(Qt::UserRole, QString::fromUtf8(e.stem));
			item->setToolTip(tr("%1\nAbout %2 thousand triangles\nFrom: %3").arg(QString::fromUtf8(e.name)).arg(e.kiloTriangles).arg(QString::fromUtf8(e.source)));
			++available;
		}
	}

	auto *note = new QLabel(&dialog);
	note->setWordWrap(true);
	if (available == 0)
		note->setText(tr("No models were found next to the app (a models folder). Use \"Choose another file...\" to add any .obj or .ply file."));
	else
		note->setText(tr("%1 models. A model is added standing on the floor, about 1.6 units across; use Scale to resize it. Each model keeps the licence of its source "
		                 "(docs/MODELS.md): check it before you publish pictures or scenes made with one.").arg(available));
	layout->addWidget(note);

	auto *buttons = new QDialogButtonBox(&dialog);
	QPushButton *addButton = buttons->addButton(tr("Add"), QDialogButtonBox::AcceptRole);
	QPushButton *otherButton = buttons->addButton(tr("Choose another file..."), QDialogButtonBox::ActionRole);
	buttons->addButton(QDialogButtonBox::Close);
	layout->addWidget(buttons);
	addButton->setEnabled(false);
	addButton->setDefault(true);

	connect(filter, &QLineEdit::textChanged, &dialog, [list](const QString &text) {
		for (int i = 0; i < list->count(); ++i) list->item(i)->setHidden(!list->item(i)->text().contains(text, Qt::CaseInsensitive));
	});
	connect(list, &QListWidget::itemSelectionChanged, &dialog, [list, addButton]() { addButton->setEnabled(!list->selectedItems().isEmpty()); });
	auto addSelected = [&]() {
		const QList<QListWidgetItem *> sel = list->selectedItems();
		if (sel.isEmpty()) return;
		if (addLibraryModel(sel.first()->data(Qt::UserRole).toString())) dialog.accept();
		else QMessageBox::warning(&dialog, tr("Model library"), tr("That model could not be read."));
	};
	connect(addButton, &QPushButton::clicked, &dialog, addSelected);
	connect(list, &QListWidget::itemDoubleClicked, &dialog, [&](QListWidgetItem *) { addSelected(); });
	connect(otherButton, &QPushButton::clicked, &dialog, [&]() {
		dialog.reject();
		addObject(ShapeKind::Mesh);
	});
	connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
	dialog.exec();
}

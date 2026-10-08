// scene_builder_inspector.cpp - the Scene Builder's property panel (SceneBuilderWidget::rebuildInspector and the inspect*/add* builders).
#include "scene_builder_widget.h"

#include "scene_builder_common.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QColorDialog>
#include <QComboBox>
#include <QCoreApplication>
#include <QDateTime>
#include <QImage>
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
#include <QImageReader>
#include <QPainter>
#include <QPixmap>
#include <QProcess>
#include <QRegularExpression>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
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

void SceneBuilderWidget::rebuildInspector() {
	m_refreshers.clear();
	auto *host = new QWidget;
	// The application stylesheet pads spin boxes generously; three of them in a row need less.
	host->setStyleSheet("QAbstractSpinBox { padding: 3px 3px; min-width: 40px; }");
	auto *form = new QFormLayout(host);
	form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
	form->setRowWrapPolicy(QFormLayout::WrapLongRows);  // in a narrow panel a label moves above its field instead of pushing the field off the edge
	m_loading = true;
	switch (m_sel.kind) {
		case SelKind::Camera: inspectCamera(form); break;
		case SelKind::Object:
			if (m_sel.index < static_cast<int>(m_doc.objects.size())) inspectObject(form, m_sel.index);
			break;
		case SelKind::Light:
			if (m_sel.index < static_cast<int>(m_doc.lights.size())) inspectLight(form, m_sel.index);
			break;
		case SelKind::None: {
			auto *help = new QLabel(tr("Pick something in the list or the layout view to edit it.\n\n"
			                           "Add shapes and lights with the Add button. Drag them in the layout view, then press Preview to see the picture. "
			                           "Save writes an ordinary .pbrt file that the renderer (and this tab) can open."),
			                        host);
			help->setWordWrap(true);
			form->addRow(help);
			break;
		}
	}
	m_loading = false;
	m_inspectorScroll->setWidget(host);  // the scroll area deletes the previous page
	// The panel can be dragged as narrow as this page's widest row (with labels wrapped above their fields) plus the scroll bar, so nothing is cut off.
	if (m_inspectorPanel) m_inspectorPanel->setMinimumWidth(host->minimumSizeHint().width() + m_inspectorScroll->verticalScrollBar()->sizeHint().width() + 6);
}

void SceneBuilderWidget::refreshInspectorValues() {
	m_loading = true;
	for (const auto &r : m_refreshers) r();
	m_loading = false;
}

static void addHeading(QFormLayout *f, const QString &text) {
	auto *l = new QLabel("<b>" + text.toHtmlEscaped() + "</b>");
	l->setContentsMargins(0, 8, 0, 0);
	f->addRow(l);
}

void SceneBuilderWidget::addNum(QFormLayout *f, const QString &label, const std::function<double *()> &ref, double lo, double hi, double step, int decimals,
                                const QString &suffix) {
	auto *box = new QDoubleSpinBox;
	box->setRange(lo, hi);
	box->setDecimals(decimals);
	box->setSingleStep(step);
	box->setKeyboardTracking(false);
	if (!suffix.isEmpty()) box->setSuffix(suffix);
	if (double *v = ref()) box->setValue(*v);
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, ref, key](double x) {
		if (m_loading) return;
		edit(key, [&]() { if (double *v = ref()) *v = x; });
	});
	m_refreshers.push_back([box, ref]() {
		QSignalBlocker b(box);
		if (double *v = ref()) box->setValue(*v);
	});
	f->addRow(label, box);
}

void SceneBuilderWidget::addInt(QFormLayout *f, const QString &label, const std::function<int *()> &ref, int lo, int hi) {
	auto *box = new QSpinBox;
	box->setRange(lo, hi);
	box->setKeyboardTracking(false);
	if (int *v = ref()) box->setValue(*v);
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(box, QOverload<int>::of(&QSpinBox::valueChanged), this, [this, ref, key](int x) {
		if (m_loading) return;
		edit(key, [&]() { if (int *v = ref()) *v = x; });
	});
	m_refreshers.push_back([box, ref]() {
		QSignalBlocker b(box);
		if (int *v = ref()) box->setValue(*v);
	});
	f->addRow(label, box);
}

void SceneBuilderWidget::addVec3(QFormLayout *f, const QString &label, const std::function<Float3 *()> &ref, double step) {
	auto *row = new QWidget;
	auto *h = new QHBoxLayout(row);
	h->setContentsMargins(0, 0, 0, 0);
	h->setSpacing(4);
	const char *axes[3] = {"X", "Y", "Z"};
	double Float3::*members[3] = {&Float3::x, &Float3::y, &Float3::z};
	for (int a = 0; a < 3; ++a) {
		auto *box = new QDoubleSpinBox;
		box->setRange(-10000, 10000);
		box->setDecimals(3);
		box->setSingleStep(step);
		box->setKeyboardTracking(false);
		box->setPrefix(QString(axes[a]) + " ");
		box->setButtonSymbols(QAbstractSpinBox::NoButtons);  // three boxes in a row have no room for arrows; type, or use the wheel
		box->setMinimumWidth(60);
		auto member = members[a];
		if (Float3 *v = ref()) box->setValue((*v).*member);
		const QString key = QString("%1:%2:%3:%4").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label).arg(a);
		connect(box, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, [this, ref, member, key](double x) {
			if (m_loading) return;
			edit(key, [&]() { if (Float3 *v = ref()) (*v).*member = x; });
		});
		m_refreshers.push_back([box, ref, member]() {
			QSignalBlocker b(box);
			if (Float3 *v = ref()) box->setValue((*v).*member);
		});
		h->addWidget(box, 1);
	}
	f->addRow(label, row);
}

void SceneBuilderWidget::addColor(QFormLayout *f, const QString &label, const std::function<Rgb *()> &ref) {
	auto *b = new QPushButton;
	b->setAutoDefault(false);
	b->setToolTip(tr("Click to choose a colour. Colours are picked as ordinary (sRGB) colours and stored as linear values for the renderer."));
	auto paint = [b, ref]() {
		if (Rgb *c = ref()) {
			const QColor q = toQColor(*c);
			const QColor fg = q.lightness() > 128 ? Qt::black : Qt::white;
			b->setText(q.name().toUpper());
			b->setStyleSheet(QString("QPushButton { background-color: %1; color: %2; border: 1px solid #666; padding: 4px; }").arg(q.name(), fg.name()));
		}
	};
	paint();
	connect(b, &QPushButton::clicked, this, [this, ref, paint, b]() {
		Rgb *c = ref();
		if (!c) return;
		const QColor picked = QColorDialog::getColor(toQColor(*c), this, tr("Choose a colour"));
		if (!picked.isValid()) return;
		edit(QString(), [&]() { if (Rgb *v = ref()) *v = fromQColor(picked); });
		paint();
	});
	m_refreshers.push_back(paint);
	f->addRow(label, b);
}

void SceneBuilderWidget::addBool(QFormLayout *f, const QString &label, const std::function<bool *()> &ref, bool rebuildAfter) {
	auto *c = new QCheckBox;
	if (bool *v = ref()) c->setChecked(*v);
	connect(c, &QCheckBox::toggled, this, [this, ref, rebuildAfter](bool on) {
		if (m_loading) return;
		edit(QString(), [&]() { if (bool *v = ref()) *v = on; });
		if (rebuildAfter) QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(label, c);
}

void SceneBuilderWidget::addText(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref) {
	auto *e = new QLineEdit;
	if (std::string *v = ref()) e->setText(QString::fromStdString(*v));
	const QString key = QString("%1:%2:%3").arg(static_cast<int>(m_sel.kind)).arg(m_sel.index).arg(label);
	connect(e, &QLineEdit::textEdited, this, [this, ref, key](const QString &t) {
		if (m_loading) return;
		edit(key, [&]() { if (std::string *v = ref()) *v = t.toStdString(); });
	});
	f->addRow(label, e);
}

// A phone photo is often stored sideways with an EXIF "rotate me" tag. The renderer ignores that tag, so the picture would show sideways (and a quad
// would take the wrong shape). When the tag asks for a rotation, this writes an upright PNG copy under <data>/pictures and returns its path;
// otherwise (no rotation, or a format Qt cannot read, such as .exr/.hdr/.tga) it returns `path` unchanged. `size` gets the upright size when known.
QString SceneBuilderWidget::uprightPictureCopy(const QString &path, QSize *size) {
	QImageReader reader(path);
	reader.setAutoTransform(true);
	const bool rotated = reader.transformation() != QImageIOHandler::TransformationNone;
	const QImage img = reader.read();
	if (img.isNull()) return path;
	if (size) *size = img.size();
	if (!rotated) return path;
	const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/pictures";
	const QFileInfo info(path);
	const QString key = QString::number(qHash(info.absoluteFilePath() + QString::number(info.lastModified().toSecsSinceEpoch())), 16);
	const QString copy = dir + "/" + info.completeBaseName() + "-" + key + ".png";
	if (QFileInfo::exists(copy)) return copy;
	if (!QDir().mkpath(dir) || !img.save(copy)) return path;
	return copy;
}

void SceneBuilderWidget::addFile(QFormLayout *f, const QString &label, const std::function<std::string *()> &ref, const QString &filter, bool clearable,
                                 const std::function<void(const QString &)> &alsoApply) {
	auto *row = new QWidget;
	auto *h = new QHBoxLayout(row);
	h->setContentsMargins(0, 0, 0, 0);
	auto *e = new QLineEdit;
	e->setReadOnly(true);
	e->setMinimumWidth(40);  // the buttons keep their size; the path gives way
	if (std::string *v = ref()) e->setText(QString::fromStdString(*v));
	auto *browse = new QPushButton(tr("Browse..."));
	browse->setStyleSheet("padding: 6px 10px;");  // compact: the row shares a narrow panel with a label and a second button
	browse->setAutoDefault(false);
	connect(browse, &QPushButton::clicked, this, [this, ref, e, filter, alsoApply]() {
		const QString p = QFileDialog::getOpenFileName(this, tr("Choose a file"), e->text(), filter);
		if (p.isEmpty()) return;
		edit(QString(), [&]() {
			if (std::string *v = ref()) *v = p.toStdString();
			if (alsoApply) alsoApply(p);
		});
		e->setText(p);
		if (alsoApply) QTimer::singleShot(0, this, [this]() { rebuildInspector(); });  // what the choice changed is shown in other rows
	});
	h->addWidget(e, 1);
	h->addWidget(browse);
	if (clearable) {
		auto *clear = new QPushButton(tr("Clear"));
		clear->setStyleSheet("padding: 6px 10px;");
		clear->setAutoDefault(false);
		clear->setEnabled(!e->text().isEmpty());
		connect(clear, &QPushButton::clicked, this, [this, ref]() {
			edit(QString(), [&]() { if (std::string *v = ref()) v->clear(); });
			QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
		});
		h->addWidget(clear);
	}
	f->addRow(label, row);
}

void SceneBuilderWidget::inspectCamera(QFormLayout *f) {
	addHeading(f, tr("Scene"));
	addText(f, tr("Title"), [this]() { return &m_doc.title; });
	addHeading(f, tr("Camera"));
	addVec3(f, tr("Position"), [this]() { return &m_doc.camera.position; }, 0.25);
	addVec3(f, tr("Looks at"), [this]() { return &m_doc.camera.target; }, 0.25);
	addNum(f, tr("Field of view"), [this]() { return &m_doc.camera.fov; }, 1, 170, 1, 1, QString::fromUtf8(" \xC2\xB0"));
	addNum(f, tr("Lens radius"), [this]() { return &m_doc.camera.lensRadius; }, 0, 10, 0.01, 3);
	addNum(f, tr("Focus distance"), [this]() { return &m_doc.camera.focusDistance; }, 0.01, 10000, 0.5, 2);
	auto *hint = new QLabel(tr("A lens radius above 0 blurs what is not at the focus distance (depth of field)."));
	hint->setWordWrap(true);
	f->addRow(hint);
	addHeading(f, tr("Picture"));
	addInt(f, tr("Width (pixels)"), [this]() { return &m_doc.render.width; }, 1, 16384);
	addInt(f, tr("Height (pixels)"), [this]() { return &m_doc.render.height; }, 1, 16384);
	auto *aspect = new QComboBox;
	aspect->addItem(tr("Set height from width..."), QVariant());
	for (const char *a : {"4:3", "16:9", "3:2", "1:1", "21:9", "9:16"}) aspect->addItem(a, a);
	connect(aspect, QOverload<int>::of(&QComboBox::activated), this, [this, aspect](int i) {
		if (i <= 0) return;
		const QStringList parts = aspect->itemData(i).toString().split(':');
		const double w = parts[0].toDouble(), h = parts[1].toDouble();
		edit(QString(), [&]() { m_doc.render.height = std::max(1, static_cast<int>(std::lround(m_doc.render.width * h / w))); });
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Aspect ratio"), aspect);
	addInt(f, tr("Samples per pixel"), [this]() { return &m_doc.render.samples; }, 1, 100000);
	addInt(f, tr("Light bounces (max depth)"), [this]() { return &m_doc.render.maxDepth; }, 1, 100);
}

void SceneBuilderWidget::inspectMaterial(QFormLayout *f, int i) {
	addHeading(f, tr("Material"));
	auto *kind = new QComboBox;
	for (MaterialKind k : {MaterialKind::Diffuse, MaterialKind::Conductor, MaterialKind::Dielectric, MaterialKind::CoatedDiffuse, MaterialKind::DiffuseTransmission})
		kind->addItem(materialLabel(k), static_cast<int>(k));
	kind->setCurrentIndex(kind->findData(static_cast<int>(m_doc.objects[i].material.kind)));
	connect(kind, QOverload<int>::of(&QComboBox::activated), this, [this, i, kind](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.objects.size())) return;
		edit(QString(), [&]() {
			scene_doc::Material &m = m_doc.objects[i].material;
			m.kind = static_cast<MaterialKind>(kind->itemData(idx).toInt());
			// Starting values that look right for the new kind.
			if (m.kind == MaterialKind::Conductor && m.roughness == 0.0) m.roughness = 0.1;
			if (m.kind == MaterialKind::CoatedDiffuse && m.roughness == 0.0) m.roughness = 0.05;
		});
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Type"), kind);
	auto mat = [this, i]() -> scene_doc::Material * { return i < static_cast<int>(m_doc.objects.size()) ? &m_doc.objects[i].material : nullptr; };
	const scene_doc::Material &m = m_doc.objects[i].material;
	// A picture can colour a diffuse or glossy-paint surface. Choosing one for a quad also gives the quad the picture's shape.
	const auto addPicture = [&]() {
		addFile(f, tr("Picture"), [mat]() { return mat() ? &mat()->imageFile : nullptr; }, tr("Images (*.png *.jpg *.jpeg *.bmp *.tga *.exr *.hdr)"), true,
		        [this, i](const QString &path) {
			        if (i >= static_cast<int>(m_doc.objects.size())) return;
			        QSize px;
			        scene_doc::Object &o = m_doc.objects[i];
			        o.material.imageFile = uprightPictureCopy(path, &px).toStdString();
			        if (o.shape == ShapeKind::Quad && px.isValid() && px.width() > 0) o.size.z = o.size.x * px.height() / px.width();
		        });
	};
	switch (m.kind) {
		case MaterialKind::Diffuse:
			addPicture();
			if (!m.imageFile.empty()) break;  // the picture is the colour
			addColor(f, m.checker ? tr("Colour A") : tr("Colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addBool(f, tr("Checker pattern"), [mat]() { return mat() ? &mat()->checker : nullptr; }, true);
			if (m.checker) {
				addColor(f, tr("Colour B"), [mat]() { return mat() ? &mat()->color2 : nullptr; });
				addNum(f, tr("Checks across"), [mat]() { return mat() ? &mat()->checkerCount : nullptr; }, 1, 1000, 1, 0);
			}
			break;
		case MaterialKind::Conductor:
			addColor(f, tr("Colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addNum(f, tr("Roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::Dielectric:
			addNum(f, tr("Index of refraction"), [mat]() { return mat() ? &mat()->ior : nullptr; }, 1.0, 3.0, 0.05, 2);
			addNum(f, tr("Roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::CoatedDiffuse:
			addPicture();
			if (m.imageFile.empty()) addColor(f, tr("Paint colour"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addNum(f, tr("Coat index of refraction"), [mat]() { return mat() ? &mat()->ior : nullptr; }, 1.0, 3.0, 0.05, 2);
			addNum(f, tr("Coat roughness"), [mat]() { return mat() ? &mat()->roughness : nullptr; }, 0, 1, 0.02, 2);
			break;
		case MaterialKind::DiffuseTransmission:
			addColor(f, tr("Reflects"), [mat]() { return mat() ? &mat()->color : nullptr; });
			addColor(f, tr("Lets through"), [mat]() { return mat() ? &mat()->transmittance : nullptr; });
			break;
	}
}

void SceneBuilderWidget::inspectObject(QFormLayout *f, int i) {
	auto obj = [this, i]() -> Object * { return i < static_cast<int>(m_doc.objects.size()) ? &m_doc.objects[i] : nullptr; };
	addHeading(f, tr("Object"));
	addText(f, tr("Name"), [obj]() { return obj() ? &obj()->name : nullptr; });
	auto *shape = new QComboBox;
	for (ShapeKind k : {ShapeKind::Sphere, ShapeKind::Box, ShapeKind::Quad, ShapeKind::Disk, ShapeKind::Cylinder, ShapeKind::Cone, ShapeKind::Mesh})
		shape->addItem(shapeLabel(k), static_cast<int>(k));
	shape->setCurrentIndex(shape->findData(static_cast<int>(m_doc.objects[i].shape)));
	connect(shape, QOverload<int>::of(&QComboBox::activated), this, [this, i, shape](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.objects.size())) return;
		edit(QString(), [&]() { m_doc.objects[i].shape = static_cast<ShapeKind>(shape->itemData(idx).toInt()); });
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Shape"), shape);
	addVec3(f, tr("Position"), [obj]() { return obj() ? &obj()->position : nullptr; }, 0.25);
	addVec3(f, tr("Rotation (degrees)"), [obj]() { return obj() ? &obj()->rotation : nullptr; }, 5.0);

	const Object &o = m_doc.objects[i];
	switch (o.shape) {
		case ShapeKind::Sphere:
		case ShapeKind::Disk:
			addNum(f, tr("Radius"), [obj]() { return obj() ? &obj()->radius : nullptr; }, 0.001, 10000, 0.1);
			break;
		case ShapeKind::Cylinder:
		case ShapeKind::Cone:
			addNum(f, tr("Radius"), [obj]() { return obj() ? &obj()->radius : nullptr; }, 0.001, 10000, 0.1);
			addNum(f, tr("Height"), [obj]() { return obj() ? &obj()->height : nullptr; }, 0.001, 10000, 0.1);
			break;
		case ShapeKind::Box:
			addVec3(f, tr("Size"), [obj]() { return obj() ? &obj()->size : nullptr; }, 0.25);
			break;
		case ShapeKind::Quad:
			addNum(f, tr("Width (X)"), [obj]() { return obj() ? &obj()->size.x : nullptr; }, 0.001, 10000, 0.5);
			addNum(f, tr("Depth (Z)"), [obj]() { return obj() ? &obj()->size.z : nullptr; }, 0.001, 10000, 0.5);
			break;
		case ShapeKind::Mesh:
			addFile(f, tr("Mesh file"), [obj]() { return obj() ? &obj()->meshFile : nullptr; }, tr("Meshes (*.ply *.obj)"));
			addNum(f, tr("Scale"), [obj]() { return obj() ? &obj()->meshScale : nullptr; }, 0.0001, 10000, 0.1, 4);
			break;
	}

	inspectMaterial(f, i);

	addHeading(f, tr("Light"));
	addBool(f, tr("Gives off light"), [obj]() { return obj() ? &obj()->emissive : nullptr; }, true);
	if (o.emissive) {
		addColor(f, tr("Light colour"), [obj]() { return obj() ? &obj()->emission : nullptr; });
		addNum(f, tr("Strength"), [obj]() { return obj() ? &obj()->emissionStrength : nullptr; }, 0, 100000, 1, 2);
		addBool(f, tr("Both sides"), [obj]() { return obj() ? &obj()->twoSided : nullptr; });
		if (o.shape == ShapeKind::Quad || o.shape == ShapeKind::Disk) {
			auto *hint = new QLabel(tr("A quad or disk lights the side that faces up. Rotate it 180 degrees about X to make a ceiling light."));
			hint->setWordWrap(true);
			f->addRow(hint);
		}
	}
}

void SceneBuilderWidget::inspectLight(QFormLayout *f, int i) {
	auto lt = [this, i]() -> Light * { return i < static_cast<int>(m_doc.lights.size()) ? &m_doc.lights[i] : nullptr; };
	addHeading(f, tr("Light"));
	addText(f, tr("Name"), [lt]() { return lt() ? &lt()->name : nullptr; });
	auto *kind = new QComboBox;
	for (LightKind k : {LightKind::Point, LightKind::Spot, LightKind::Distant, LightKind::Infinite}) kind->addItem(lightLabel(k), static_cast<int>(k));
	kind->setCurrentIndex(kind->findData(static_cast<int>(m_doc.lights[i].kind)));
	connect(kind, QOverload<int>::of(&QComboBox::activated), this, [this, i, kind](int idx) {
		if (m_loading || i >= static_cast<int>(m_doc.lights.size())) return;
		edit(QString(), [&]() { m_doc.lights[i].kind = static_cast<LightKind>(kind->itemData(idx).toInt()); });
		rebuildList();
		QTimer::singleShot(0, this, [this]() { rebuildInspector(); });
	});
	f->addRow(tr("Type"), kind);
	const Light &l = m_doc.lights[i];
	if (l.kind == LightKind::Point || l.kind == LightKind::Spot) addVec3(f, tr("Position"), [lt]() { return lt() ? &lt()->position : nullptr; }, 0.25);
	if (l.kind == LightKind::Spot) {
		addVec3(f, tr("Aims at"), [lt]() { return lt() ? &lt()->target : nullptr; }, 0.25);
		addNum(f, tr("Cone angle"), [lt]() { return lt() ? &lt()->coneAngle : nullptr; }, 1, 89, 1, 1, QString::fromUtf8(" \xC2\xB0"));
		addNum(f, tr("Soft edge"), [lt]() { return lt() ? &lt()->coneDelta : nullptr; }, 0, 89, 1, 1, QString::fromUtf8(" \xC2\xB0"));
	}
	if (l.kind == LightKind::Distant) {
		addVec3(f, tr("Shines from"), [lt]() { return lt() ? &lt()->position : nullptr; }, 0.25);
		addVec3(f, tr("Towards"), [lt]() { return lt() ? &lt()->target : nullptr; }, 0.25);
	}
	if (l.kind == LightKind::Infinite) {
		addFile(f, tr("Sky image"), [lt]() { return lt() ? &lt()->imageFile : nullptr; }, tr("Images (*.exr *.hdr *.png *.jpg *.jpeg)"));
		auto *hint = new QLabel(tr("Leave the image empty for a plain colour sky. An image is an equirectangular (lat-long) panorama."));
		hint->setWordWrap(true);
		f->addRow(hint);
	}
	if (l.kind != LightKind::Infinite || l.imageFile.empty()) addColor(f, tr("Colour"), [lt]() { return lt() ? &lt()->color : nullptr; });
	addNum(f, l.kind == LightKind::Point || l.kind == LightKind::Spot ? tr("Strength") : tr("Brightness"), [lt]() { return lt() ? &lt()->intensity : nullptr; }, 0, 100000, 0.5, 2);
}


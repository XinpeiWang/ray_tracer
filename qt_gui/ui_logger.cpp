#include "ui_logger.h"

#include "app_log.h"

#include <QAbstractButton>
#include <QAbstractItemView>
#include <QAbstractSpinBox>
#include <QAction>
#include <QApplication>
#include <QComboBox>
#include <QDialog>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QMenu>
#include <QMessageBox>
#include <QPointer>
#include <QShortcutEvent>
#include <QSlider>
#include <QSpinBox>
#include <QTabBar>
#include <QTabWidget>

namespace {

const char *kHooked = "rtLogHooked";
const char *kLastValue = "rtLogLast";

QString clean(QString text) {
	text.remove(QLatin1Char('&'));
	return text.simplified();
}

QString shorten(const QString &text, int n = 160) { return text.size() > n ? text.left(n) + QStringLiteral("...") : text; }

// "Scene Builder": the page of the outermost tab control that contains `w`.
QString tabOf(QWidget *w) {
	QString name;
	for (QWidget *p = w; p; p = p->parentWidget()) {
		if (auto *tabs = qobject_cast<QTabWidget *>(p)) {
			// A page inside this tab widget: the tab whose page is the ancestor we came from is the current one.
			name = clean(tabs->tabText(tabs->currentIndex()));
		}
	}
	return name;
}

// What the user calls this control: its own text, else the label beside it in a form, else its tooltip or name.
QString nameOf(QWidget *w) {
	if (auto *b = qobject_cast<QAbstractButton *>(w); b && !clean(b->text()).isEmpty()) return clean(b->text());
	if (auto *g = qobject_cast<QGroupBox *>(w)) return clean(g->title());
	for (QWidget *p = w; p && p->parentWidget(); p = p->parentWidget()) {
		auto *form = qobject_cast<QFormLayout *>(p->parentWidget()->layout());
		if (!form) continue;
		int row = -1;
		QFormLayout::ItemRole role;
		form->getWidgetPosition(p, &row, &role);
		if (row < 0) continue;
		if (QLayoutItem *label = form->itemAt(row, QFormLayout::LabelRole)) {
			if (auto *l = qobject_cast<QLabel *>(label->widget())) return clean(l->text());
		}
	}
	if (!w->accessibleName().isEmpty()) return clean(w->accessibleName());
	if (!w->toolTip().isEmpty()) return shorten(clean(w->toolTip()), 50);
	if (!w->objectName().isEmpty() && !w->objectName().startsWith(QLatin1String("qt_"))) return w->objectName();
	return QString::fromLatin1(w->metaObject()->className());
}

void logUi(QWidget *w, const QString &text) {
	const QString tab = tabOf(w);
	AppLog::info(QStringLiteral("ui"), tab.isEmpty() ? text : tab + QStringLiteral(" > ") + text);
}

}  // namespace

void UiLogger::install() {
	if (!qApp) return;
	qApp->installEventFilter(new UiLogger(qApp));
}

// Only the first Polish of each widget matters: that is when it exists fully constructed, and about to be shown.
bool UiLogger::eventFilter(QObject *watched, QEvent *event) {
	switch (event->type()) {
		case QEvent::Polish:
			if (auto *w = qobject_cast<QWidget *>(watched)) hook(w);
			break;
		case QEvent::Shortcut: {
			const auto *s = static_cast<QShortcutEvent *>(event);
			AppLog::info(QStringLiteral("ui"), QStringLiteral("shortcut %1").arg(s->key().toString(QKeySequence::PortableText)));
			break;
		}
		case QEvent::Show:
			if (auto *d = qobject_cast<QDialog *>(watched); d && d->isWindow()) {
				QString text;
				auto level = log_format::Level::Info;
				if (auto *m = qobject_cast<QMessageBox *>(d)) {
					text = QStringLiteral(": ") + shorten(clean(m->text()), 600);
					if (m->icon() == QMessageBox::Warning) level = log_format::Level::Warn;
					if (m->icon() == QMessageBox::Critical) level = log_format::Level::Error;
				}
				AppLog::write(level, QStringLiteral("ui"), QStringLiteral("dialog \"%1\" opened%2").arg(clean(d->windowTitle()), text));
			}
			break;
		case QEvent::Hide:
			if (auto *d = qobject_cast<QDialog *>(watched); d && d->isWindow() && !d->isVisible())
				AppLog::info(QStringLiteral("ui"), QStringLiteral("dialog \"%1\" closed (%2)").arg(clean(d->windowTitle()), d->result() == QDialog::Accepted ? QStringLiteral("accepted") : QStringLiteral("rejected/other")));
			break;
		default: break;
	}
	return false;
}

void UiLogger::hook(QWidget *w) {
	if (w->property(kHooked).toBool()) return;
	w->setProperty(kHooked, true);
	QPointer<QWidget> guard(w);

	if (auto *menu = qobject_cast<QMenu *>(w)) {
		QObject::connect(menu, &QMenu::triggered, w, [menu](QAction *a) {
			if (a && !a->isSeparator()) AppLog::info(QStringLiteral("ui"), QStringLiteral("menu \"%1\" > \"%2\"").arg(clean(menu->title()).isEmpty() ? QStringLiteral("(context)") : clean(menu->title()), clean(a->text())));
		});
		return;
	}
	if (auto *button = qobject_cast<QAbstractButton *>(w)) {
		QObject::connect(button, &QAbstractButton::clicked, w, [button](bool checked) {
			const QString state = button->isCheckable() ? QStringLiteral(" -> %1").arg(checked ? QStringLiteral("on") : QStringLiteral("off")) : QString();
			logUi(button, QStringLiteral("click \"%1\"%2").arg(nameOf(button), state));
		});
		return;
	}
	if (auto *combo = qobject_cast<QComboBox *>(w)) {
		QObject::connect(combo, QOverload<int>::of(&QComboBox::activated), w, [combo](int) {
			logUi(combo, QStringLiteral("\"%1\" = %2").arg(nameOf(combo), clean(combo->currentText())));
		});
		return;
	}
	if (auto *spin = qobject_cast<QAbstractSpinBox *>(w)) {
		spin->setProperty(kLastValue, spin->text());
		QObject::connect(spin, &QAbstractSpinBox::editingFinished, w, [spin]() {
			if (spin->property(kLastValue).toString() == spin->text()) return;
			spin->setProperty(kLastValue, spin->text());
			logUi(spin, QStringLiteral("\"%1\" = %2").arg(nameOf(spin), spin->text()));
		});
		return;
	}
	if (auto *slider = qobject_cast<QSlider *>(w)) {
		QObject::connect(slider, &QSlider::sliderReleased, w, [slider]() { logUi(slider, QStringLiteral("\"%1\" = %2").arg(nameOf(slider)).arg(slider->value())); });
		return;
	}
	if (auto *edit = qobject_cast<QLineEdit *>(w)) {
		// The editor inside a spin box or an editable combo box is reported by that control.
		if (qobject_cast<QAbstractSpinBox *>(edit->parentWidget()) || qobject_cast<QComboBox *>(edit->parentWidget())) return;
		if (edit->echoMode() != QLineEdit::Normal) return;
		edit->setProperty(kLastValue, edit->text());
		QObject::connect(edit, &QLineEdit::editingFinished, w, [edit]() {
			if (edit->property(kLastValue).toString() == edit->text()) return;
			edit->setProperty(kLastValue, edit->text());
			logUi(edit, QStringLiteral("\"%1\" = \"%2\"").arg(nameOf(edit), shorten(edit->text())));
		});
		return;
	}
	if (auto *bar = qobject_cast<QTabBar *>(w)) {
		QObject::connect(bar, &QTabBar::currentChanged, w, [bar](int index) {
			if (index >= 0) AppLog::info(QStringLiteral("ui"), QStringLiteral("tab \"%1\"").arg(clean(bar->tabText(index))));
		});
		return;
	}
	if (auto *view = qobject_cast<QAbstractItemView *>(w)) {
		QObject::connect(view, &QAbstractItemView::clicked, w, [view](const QModelIndex &i) { logUi(view, QStringLiteral("select \"%1\"").arg(shorten(clean(i.data().toString()), 100))); });
		QObject::connect(view, &QAbstractItemView::doubleClicked, w, [view](const QModelIndex &i) { logUi(view, QStringLiteral("open \"%1\"").arg(shorten(clean(i.data().toString()), 100))); });
		return;
	}
}

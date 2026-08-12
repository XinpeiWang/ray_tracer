#include "icon_tint.h"

#include <QAbstractButton>
#include <QAction>
#include <QPainter>
#include <QPixmap>
#include <QWidget>

namespace icon_tint {
namespace {

// Dynamic properties rather than a registry keyed on pointers: they travel with
// the object, so a widget destroyed between themes cannot leave a dangling
// entry behind, and there is no bookkeeping to keep in sync at the call sites.
constexpr const char *kPathProperty = "themedIconPath";
constexpr const char *kRoleProperty = "themedIconRole";

// The sizes Qt asks for in this app (menu items, buttons, combo items) plus
// their 2x hi-dpi equivalents. QIcon picks the closest and scales, so a few
// pre-rendered sizes are enough; rendering every possible size is not.
constexpr int kSizes[] = {16, 24, 32, 48};

QColor colourForRole(Role role, const QColor &body, const QColor &primary) {
	return role == Role::Primary ? primary : body;
}

void remember(QObject *target, const QString &path, Role role) {
	target->setProperty(kPathProperty, path);
	target->setProperty(kRoleProperty, static_cast<int>(role));
}

} // namespace

QIcon tinted(const QString &path, const QColor &colour) {
	const QIcon source(path);
	QIcon result;
	for (int size : kSizes) {
		QPixmap pixmap = source.pixmap(size, size);
		if (pixmap.isNull()) continue;

		// SourceIn keeps the destination's alpha and takes the source's colour,
		// which is exactly "recolour the silhouette, leave the holes alone".
		QPainter painter(&pixmap);
		painter.setCompositionMode(QPainter::CompositionMode_SourceIn);
		painter.fillRect(pixmap.rect(), colour);
		painter.end();

		result.addPixmap(pixmap);
	}
	// A missing resource yields an empty QIcon and no pixmaps, which renders as
	// nothing - the same as before tinting existed. Returning the untinted
	// source instead would be worse: it would paint the authored colour, which
	// is only correct by accident on one theme.
	return result;
}

void apply(QAction *action, const QString &path, Role role, const QColor &colour) {
	remember(action, path, role);
	action->setIcon(tinted(path, colour));
}

void apply(QAbstractButton *button, const QString &path, Role role, const QColor &colour) {
	remember(button, path, role);
	button->setIcon(tinted(path, colour));
}

void retint(QWidget *root, const QColor &bodyColour, const QColor &primaryColour) {
	const auto redo = [&](QObject *object, auto setter) {
		const QVariant path = object->property(kPathProperty);
		if (!path.isValid()) return;
		const Role role = static_cast<Role>(object->property(kRoleProperty).toInt());
		setter(tinted(path.toString(), colourForRole(role, bodyColour, primaryColour)));
	};

	for (QAction *action : root->findChildren<QAction *>())
		redo(action, [action](const QIcon &icon) { action->setIcon(icon); });
	for (QAbstractButton *button : root->findChildren<QAbstractButton *>())
		redo(button, [button](const QIcon &icon) { button->setIcon(icon); });
}

} // namespace icon_tint

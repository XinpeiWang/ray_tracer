#ifndef ICON_TINT_H
#define ICON_TINT_H

#include <QColor>
#include <QIcon>
#include <QString>

class QAbstractButton;
class QAction;
class QWidget;

// ============================================================================
// Runtime icon tinting
// ============================================================================
// The SVGs in resources.qrc are monochrome silhouettes drawn in one arbitrary
// colour. Which colour that is does not matter, because nothing ever paints
// them as authored: every icon is recoloured to a role from the active theme
// before it reaches a widget. A near-white icon baked into the artwork would
// be invisible the moment a light scheme is selected, and there were two
// hand-made "_accent" duplicates existing purely to hardcode a second colour -
// tinting removes the need for both.
//
// This only works because the artwork carries real transparency. Holes are cut
// with fill-rule="evenodd", not painted in the background colour; a painted
// hole survives the tint as a solid blob of whatever colour it was authored in.
// ============================================================================
namespace icon_tint {

// Which theme role an icon takes its colour from. Stored alongside the icon so
// a theme switch can re-tint without the call site being consulted again.
enum class Role {
	Body,       // ordinary chrome: menus, secondary buttons, combo items
	Primary,    // sits on the primary action's gradient, matching its text
};

// Recolours the resource at `path`, preserving its alpha.
QIcon tinted(const QString &path, const QColor &colour);

// Sets the icon and records (path, role) so retint() can redo it later.
void apply(QAction *action, const QString &path, Role role, const QColor &colour);
void apply(QAbstractButton *button, const QString &path, Role role, const QColor &colour);

// Re-tints every action and button under `root` that went through apply().
// bodyColour and primaryColour supply the two roles.
void retint(QWidget *root, const QColor &bodyColour, const QColor &primaryColour);

} // namespace icon_tint

#endif // ICON_TINT_H

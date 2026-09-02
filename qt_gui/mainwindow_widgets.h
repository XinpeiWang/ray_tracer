#ifndef MAINWINDOW_WIDGETS_H
#define MAINWINDOW_WIDGETS_H
// mainwindow_widgets.h -- small standalone UI helper widget/event-filter
// classes used by MainWindow (event filters, a toast notification, a
// themed scroll area, custom tab bars/widgets). Split out of mainwindow.h,
// which #includes this file at the point this content used to live.
//
// Every Q_OBJECT class here needs its own moc output - see qt_gui/
// RayTracerGUI.pro's HEADERS list, which lists this file alongside
// mainwindow.h for exactly that reason (qmake only runs moc on headers
// explicitly listed there, not on anything textually #include'd) - and
// that in turn means THIS file must be self-sufficient (its own full Qt
// include list, not "borrowed" from mainwindow.h's), since moc generates
// and compiles moc_mainwindow_widgets.cpp as its own independent
// translation unit that #includes only this header, standalone. Kept the
// same include list mainwindow.h's own top used before the split, rather
// than trimming to each class's exact minimum - over-including a Qt
// header is harmless, and this avoids re-deriving per-class minimums by
// hand only to get one wrong.
#include <QMainWindow>
#include <QTabWidget>
#include <QComboBox>
#include <QAbstractSpinBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>
#include <QAction>
#include <QGroupBox>
#include <QCheckBox>
#include <QLineEdit>
#include <QProcess>
#include <QVector3D>
#include <QTextEdit>
#include <QEvent>
#include <QMouseEvent>
#include <QTimer>
#include <QDateTime>
#include <QElapsedTimer>
#include <QPixmap>
#include <QResizeEvent>
#include <QSystemTrayIcon>
#include <QStackedWidget>
#include <QSlider>
#include <QMediaPlayer>
#include <QAudioOutput>
#include <QVideoWidget>
#include <QMap>
#include <QStylePainter>
#include <QStyleOptionTab>
#include <QHBoxLayout>
#include <QSplitter>
#include <QSignalBlocker>
#include <QQueue>
#include <QListWidget>
#include <QToolButton>
#include <QScrollArea>
#include <QDebug>
#include <QFile>
#include <functional>

#include "camera_math.h"
#include "render_output_parser.h"
#include "theme.h"
#include "icon_tint.h"

#include <QGraphicsDropShadowEffect>
#include <QGraphicsOpacityEffect>
#include <QPropertyAnimation>
#include <QEasingCurve>

class QMenu;
class QTabBar;

// ============================================================================
// WheelIgnoreFilter
// ============================================================================
// Prevents accidental value changes when the user scrolls the page.
// Install on any QComboBox / QSpinBox / QDoubleSpinBox to block wheel events
// unless the widget already has keyboard focus.
// ============================================================================
class WheelIgnoreFilter : public QObject {
    Q_OBJECT
public:
    explicit WheelIgnoreFilter(QObject *parent = nullptr) : QObject(parent) {}
protected:
    bool eventFilter(QObject *obj, QEvent *event) override {
        if (event->type() == QEvent::Wheel)
            return true;  // always block wheel on these controls
        return QObject::eventFilter(obj, event);
    }
};

// ============================================================================
// ListEmptyAreaDeselectFilter
// ============================================================================
// QListWidget's SingleSelection mode has no built-in way to click empty space
// below the items to clear the selection - once a row is selected it stays
// selected no matter where else in the list you click. Install on the list's
// viewport (see the Render Queue list in createProgressTab()) to add that.
// ============================================================================
class ListEmptyAreaDeselectFilter : public QObject {
    Q_OBJECT
public:
    explicit ListEmptyAreaDeselectFilter(QListWidget *list) : QObject(list), m_list(list) {}
protected:
    bool eventFilter(QObject *obj, QEvent *event) override {
        if (event->type() == QEvent::MouseButtonPress &&
            !m_list->indexAt(static_cast<QMouseEvent *>(event)->pos()).isValid()) {
            m_list->clearSelection();
        }
        return QObject::eventFilter(obj, event);
    }
private:
    QListWidget *m_list;
};

// ============================================================================
// HoverLiftFilter
// ============================================================================
// QSS :hover/:pressed snap their colours instantly - Qt style sheets have no
// transition property. This animates a widget's QGraphicsDropShadowEffect
// (see MainWindow::applyElevation) between rest/hover/pressed blur radii
// instead, so an elevated button (Render, Stop, Clear Queue) reads as
// lifting toward the cursor on hover and settling back down on press,
// smoothly, alongside its own QSS colour change rather than replacing it.
// Install on a widget that already has an elevation effect set via
// applyElevation() - a widget with no QGraphicsDropShadowEffect is a no-op.
//
// idlePulse (opt-in, only used for the primary Render button - a pulsing
// Stop/Clear Queue would read as "click me" on a destructive action, the
// wrong invitation) adds a slow, continuous "breathing" glow at rest: the
// main-menu-button pulse every game's "Start"/"Play" control has, built
// from the same shadow effect rather than a second one.
// ============================================================================
class HoverLiftFilter : public QObject {
    Q_OBJECT
public:
    HoverLiftFilter(QWidget *target, qreal restBlur, qreal hoverBlur, qreal pressedBlur,
                     bool idlePulse = false, QObject *parent = nullptr)
        : QObject(parent), m_target(target),
          m_restBlur(restBlur), m_hoverBlur(hoverBlur), m_pressedBlur(pressedBlur),
          m_idlePulse(idlePulse)
    {
        if (m_idlePulse) restartIdlePulse();
    }

protected:
    bool eventFilter(QObject *obj, QEvent *event) override {
        auto *shadow = qobject_cast<QGraphicsDropShadowEffect *>(m_target->graphicsEffect());
        if (shadow) {
            switch (event->type()) {
                case QEvent::Enter:
                    if (m_idleAnim) m_idleAnim->stop();
                    animateTo(shadow, m_hoverBlur);
                    break;
                case QEvent::Leave:
                    animateTo(shadow, m_restBlur);
                    // Given a moment to settle back to rest before the idle
                    // loop resumes, rather than fighting the leave animation
                    // for control of the same property mid-transition.
                    if (m_idlePulse)
                        QTimer::singleShot(180, this, [this]() { restartIdlePulse(); });
                    break;
                case QEvent::MouseButtonPress:
                    if (m_idleAnim) m_idleAnim->stop();
                    animateTo(shadow, m_pressedBlur);
                    break;
                // Mouse may have already left before release fires (e.g. a
                // drag off the button) - Leave's own animation already
                // handles that case, this only needs to cover the ordinary
                // "released while still hovering" click.
                case QEvent::MouseButtonRelease: animateTo(shadow, m_hoverBlur); break;
                default: break;
            }
        }
        return QObject::eventFilter(obj, event);
    }

private:
    // One animation object per shadow effect, created lazily and reused
    // (parented to `shadow`, found again via findChild rather than stored
    // here - this filter has no per-target storage of its own) - a rapid
    // hover/leave/hover flicker just retargets whatever animation already
    // exists instead of allocating a new one each time, restarting cleanly
    // from the shadow's CURRENT blur so it never jumps. m_idleAnim (below)
    // is parented to `this`, not `shadow`, so it never collides with this
    // lookup.
    void animateTo(QGraphicsDropShadowEffect *shadow, qreal blur) {
        auto *anim = shadow->findChild<QPropertyAnimation *>(QString(), Qt::FindDirectChildrenOnly);
        if (!anim) {
            anim = new QPropertyAnimation(shadow, "blurRadius", shadow);
            anim->setDuration(150);
            anim->setEasingCurve(QEasingCurve::OutCubic);
        }
        anim->stop();
        anim->setStartValue(shadow->blurRadius());
        anim->setEndValue(blur);
        anim->start();
    }

    // A seamless rest -> peak -> rest loop (one QPropertyAnimation with 3
    // keyframes, repeated via setLoopCount(-1)) rather than two animations
    // ping-ponging - a real back-and-forth "breathe", not a sawtooth reset.
    void restartIdlePulse() {
        auto *shadow = qobject_cast<QGraphicsDropShadowEffect *>(m_target->graphicsEffect());
        if (!shadow) return;
        if (!m_idleAnim) {
            m_idleAnim = new QPropertyAnimation(shadow, "blurRadius", this);
            m_idleAnim->setDuration(2200);
            m_idleAnim->setLoopCount(-1);
            m_idleAnim->setEasingCurve(QEasingCurve::InOutSine);
            m_idleAnim->setKeyValueAt(0.0, m_restBlur);
            m_idleAnim->setKeyValueAt(0.5, m_restBlur * 1.6);
            m_idleAnim->setKeyValueAt(1.0, m_restBlur);
        }
        m_idleAnim->stop();
        m_idleAnim->start();
    }

    QWidget *m_target;
    qreal m_restBlur, m_hoverBlur, m_pressedBlur;
    bool m_idlePulse;
    QPropertyAnimation *m_idleAnim = nullptr;
};

// ============================================================================
// ToastNotification
// ============================================================================
// A single, self-dismissing completion toast - not Fluent Widgets' ~600-line
// InfoBar/InfoBarManager (icon variants, 5 screen-corner docking positions, a
// stack of several simultaneous bars): a render finishing is one event at a
// time, never concurrent, so there is nothing to stack or dock.
//
// A frameless always-on-top Qt::Tool window rather than a child widget
// floating over MainWindow's own layout, so showing it needs no manual
// z-order/resize bookkeeping relative to the tab widget beneath it - it is
// positioned once, near the parent window's top edge, each time it is shown.
// WA_TranslucentBackground is what lets the rounded corners painted by
// styleSheet() below show through as transparent instead of a square window
// frame - a frameless top-level widget is otherwise still rectangular.
// ============================================================================
class ToastNotification : public QWidget {
    Q_OBJECT
public:
    explicit ToastNotification(QWidget *parent)
        : QWidget(parent, Qt::Tool | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint)
    {
        setAttribute(Qt::WA_ShowWithoutActivating);   // never steals focus from the main window
        setAttribute(Qt::WA_TranslucentBackground);

        auto *layout = new QHBoxLayout(this);
        layout->setContentsMargins(18, 12, 18, 12);
        m_label = new QLabel(this);
        m_label->setWordWrap(true);
        layout->addWidget(m_label);

        m_opacityEffect = new QGraphicsOpacityEffect(this);
        m_opacityEffect->setOpacity(0.0);
        setGraphicsEffect(m_opacityEffect);

        m_fadeAnim = new QPropertyAnimation(m_opacityEffect, "opacity", this);
        connect(m_fadeAnim, &QPropertyAnimation::finished, this, [this]() {
            if (m_opacityEffect->opacity() < 0.01) hide();
        });

        // The entrance "pop" - starts slightly smaller than final size, same
        // centre, and overshoots past it before settling (QEasingCurve::
        // OutBack's whole point) - an achievement-banner bounce rather than
        // a flat fade landing at rest immediately. geometry is a genuine
        // QWidget Q_PROPERTY (QRect-valued), directly animatable.
        m_popAnim = new QPropertyAnimation(this, "geometry", this);
        m_popAnim->setDuration(320);
        m_popAnim->setEasingCurve(QEasingCurve::OutBack);

        m_dismissTimer = new QTimer(this);
        m_dismissTimer->setSingleShot(true);
        connect(m_dismissTimer, &QTimer::timeout, this, [this]() { fadeTo(0.0, 300); });
    }

    // background/foreground drive the fill and text colour directly (the
    // caller already knows which theme role - success/error/muted - fits
    // the outcome; this widget has no theme knowledge of its own).
    void showToast(const QString &text, const QColor &background, const QColor &foreground,
                   int durationMs = 3500) {
        m_label->setText(text);
        setStyleSheet(QString(
            "QWidget { background-color: %1; border-radius: 8px; } "
            "QLabel { color: %2; font-size: 11pt; font-weight: bold; background: transparent; }")
            .arg(background.name(), foreground.name()));

        if (QWidget *p = parentWidget()) {
            adjustSize();
            const QRect pg = p->geometry();
            const QSize finalSize = size();
            const QRect finalGeom(pg.center().x() - finalSize.width() / 2, pg.top() + 48,
                                   finalSize.width(), finalSize.height());
            const QSize startSize = finalSize * 0.85;
            const QRect startGeom(finalGeom.center().x() - startSize.width() / 2,
                                   finalGeom.center().y() - startSize.height() / 2,
                                   startSize.width(), startSize.height());

            setGeometry(startGeom);
            m_popAnim->stop();
            m_popAnim->setStartValue(startGeom);
            m_popAnim->setEndValue(finalGeom);
            m_popAnim->start();
        }

        show();
        raise();
        fadeTo(1.0, 200);
        m_dismissTimer->start(durationMs);
    }

private:
    void fadeTo(qreal opacity, int durationMs) {
        m_fadeAnim->stop();
        m_fadeAnim->setDuration(durationMs);
        m_fadeAnim->setStartValue(m_opacityEffect->opacity());
        m_fadeAnim->setEndValue(opacity);
        m_fadeAnim->start();
    }

    QLabel *m_label;
    QGraphicsOpacityEffect *m_opacityEffect;
    QPropertyAnimation *m_fadeAnim;
    QPropertyAnimation *m_popAnim;
    QTimer *m_dismissTimer;
};

// ============================================================================
// ScaledImageLabel
// ============================================================================
// A QLabel that shows an image scaled to fit its current size (preserving
// aspect ratio, re-scaled on every resize) instead of QLabel's default
// native-size-or-nothing behavior. Used by the Preview tab to show the
// rendered image inline - see MainWindow::createPreviewTab() - instead of
// shelling out to the OS's default image viewer for every render.
// ============================================================================
class ScaledImageLabel : public QLabel {
	Q_OBJECT
public:
	explicit ScaledImageLabel(QWidget *parent = nullptr) : QLabel(parent) {
		setAlignment(Qt::AlignCenter);
		setMinimumSize(1, 1);
	}

	void setPreviewPixmap(const QPixmap &pixmap) {
		m_original = pixmap;
		updateScaledPixmap();
	}

	// Drops the currently displayed image (if any) and restores whatever
	// text was last set via setPlaceholderText().
	void clearPreviewPixmap() {
		m_original = QPixmap();
		setPixmap(QPixmap());
		setText(m_placeholderText);
	}

	void setPlaceholderText(const QString &text) {
		m_placeholderText = text;
		if (m_original.isNull()) setText(text);
	}

protected:
	void resizeEvent(QResizeEvent *event) override {
		QLabel::resizeEvent(event);
		updateScaledPixmap();
	}

private:
	void updateScaledPixmap() {
		if (m_original.isNull()) return;
		setPixmap(m_original.scaled(size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
	}

	QPixmap m_original;
	QString m_placeholderText;
};

// ============================================================================
// ThemedScrollArea
// ============================================================================
// A QScrollArea that paints an optional theme motif on its viewport via a
// PER-WIDGET stylesheet (viewport()->setStyleSheet(...)), not the app-wide
// one. mainwindow_style.cpp's global stylesheet used to declare this same
// background-image/-repeat/-position rule for QScrollArea#tabScroll, and it
// never visibly worked - confirmed by direct experiment: a paintEvent
// override on the viewport (custom widget, replacing Qt's default one
// entirely via setViewport()) never fires AT ALL despite the widget being
// properly constructed, assigned, and shown with correct geometry - a real,
// unexplained Qt behavior in this environment, not a logic bug in this
// class's own code (verified across autoFillBackground/WA_OpaquePaintEvent/
// WA_NoSystemBackground, event-filter Paint-event consumption, real window
// resizes, and forced repaint() calls - none of it made the override run).
// A per-widget stylesheet set directly on the SAME viewport, by contrast,
// DOES visibly take effect (verified the same way) - Qt re-polishes a
// widget's style immediately when setStyleSheet() is called directly on it,
// which the app-wide stylesheet path apparently doesn't reliably do for a
// deeply-nested selector like this one when the active theme changes at
// runtime. So: same QSS property or degradation this app's design accepted
// - a non-tiled motif still draws at native pixel size (no background-size
// in Qt QSS), anchored to a stylesheet corner - just applied where it
// actually works.
// ============================================================================
class ThemedScrollArea : public QScrollArea {
	Q_OBJECT
public:
	explicit ThemedScrollArea(QWidget *parent = nullptr) : QScrollArea(parent) {}

	// svgPath empty clears the motif (the eight schemes with none - most
	// calls). tiled: repeated edge-to-edge across the whole viewport
	// (matches the four topical themes' own backgroundTiled - theme.h);
	// otherwise drawn once at the SVG's own raw pixel size and anchored to
	// the corner named in `position` ("top left"/"bottom right", the exact
	// strings Palette::backgroundPosition already uses). themeId is only
	// for the missing-resource warning below (theme::Palette::id) - optional
	// since a caller with no theme object handy (there is none today, but a
	// future one might not) can still get a working, just less-labeled,
	// warning.
	void setMotif(const QString &svgPath, bool tiled, const QString &position,
				  const QString &themeId = QString()) {
		if (svgPath.isEmpty()) {
			viewport()->setStyleSheet(QString());
			return;
		}
		// A missing or unregistered resource makes QSS silently skip the
		// declaration - the same failure mode that once made the SVG icons
		// render as nothing at all. Check rather than trust, and say so.
		if (!QFile::exists(svgPath)) {
			qWarning() << "ThemedScrollArea: theme" << themeId << "motif" << svgPath
					   << "is not in the resource bundle - check resources.qrc";
			viewport()->setStyleSheet(QString());
			return;
		}
		// background-attachment: fixed anchors the motif to the viewport
		// rather than to the scrolled content, so it stays put while the
		// settings panel scrolls past it.
		viewport()->setStyleSheet(
			QString("background-image: url(%1); background-repeat: %2; "
					"background-position: %3; background-attachment: fixed;")
				.arg(svgPath, tiled ? "repeat" : "no-repeat", tiled ? "top left" : position));
	}
};

// ============================================================================
// HorizontalTabBar / SplitPreviewTabs
// ============================================================================
// A West-positioned (left-side) QTabBar whose labels stay upright/horizontal
// instead of Qt's default rendering for that position, which rotates the
// text 90 degrees to read top-to-bottom - readable for a handful of tabs,
// but not what "tabs on the left" means for the Preview tab's per-render
// sub-tabs (createPreviewTab()), which are titled with real scene/preset
// names.
//
// The label is painted by hand (setColors() + paintEvent() below) rather
// than by counter-rotating the painter and calling the style's own
// CE_TabBarTabLabel - that is the textbook technique for this, but with a
// stylesheet active it draws nothing: QStyleSheetStyle's CE_TabBarTabLabel
// matches each tab against its own cached geometry to pick a rule, and this
// bar's very own transposed opt.rect (needed for the rotation) is exactly
// what breaks that match, silently producing invisible (not just
// mis-rotated) text. Drawing the shape via CE_TabBarTabShape still works
// fine - only the label needs the hand-rolled path - so setColors() is fed
// the same three colours the (still QSS-driven) shape rule already keys
// off of (mainwindow_style.cpp's own QTabBar#previewSubTabsBar::tab rule),
// applied by MainWindow::applyTheme() whenever the scheme changes.
//
// tabSizeHint()/sizing is hand-rolled too, for the same "stylesheet breaks
// the textbook approach" reason as the label: routed through
// QStyleSheetStyle for a West-shaped bar under this app's stylesheet, it
// came out far taller per tab than a single-line label needs, and
// overriding just its height (or width) changed nothing - evidence
// something downstream (most likely "expanding", which stretches a lone
// tab to fill whatever vertical room the widget has) wasn't consulting it
// the way the docs imply. tabSizeHint() now returns a nominal per-row
// height with expanding turned off, and updateFixedHeight() pins the
// widget's own height directly to (tab count) rows - see both methods'
// own comments.
class HorizontalTabBar : public QTabBar {
	Q_OBJECT
public:
	explicit HorizontalTabBar(QWidget *parent = nullptr) : QTabBar(parent) {
		// Tabs are hand-painted at a fixed per-row height (see
		// updateFixedHeight()), not sized through Qt's own tabSizeHint/
		// layout machinery - QTabBar::sizeHint()'s handling of that
		// machinery for a West-shaped bar under this app's stylesheet came
		// out far taller than a single-line label needs, and neither
		// overriding tabSizeHint()'s height nor width actually changed the
		// rendered result (evidence it isn't consulted the way the docs
		// imply here). "Expanding" would stretch a lone tab to fill
		// whatever extra vertical room the widget has anyway, which is
		// exactly the failure mode - turned off so there is nothing left
		// for it to stretch into once the widget's real height is pinned
		// down directly instead.
		setExpanding(false);
	}

	// Kept in sync with the active theme's palette by MainWindow::
	// applyTheme() - see this class's own comment on why the label can't
	// just read colours out of the stylesheet the way the shape does.
	void setColors(const QColor &normal, const QColor &hover, const QColor &selected) {
		m_normalColor = normal;
		m_hoverColor = hover;
		m_selectedColor = selected;
		update();
	}

	// Pins the widget's own height to exactly (tab count) rows - called by
	// SplitPreviewTabs after every addTab()/removeTab(), once the bar's own
	// insertion/removal bookkeeping has fully returned rather than from
	// inside a tabInserted()/tabRemoved() hook, since resizing the widget
	// from within one of those (still on QTabBarPrivate's own call stack)
	// risks reentering its still-in-progress layout.
	void updateFixedHeight() { setFixedHeight(qMax(1, count()) * rowHeight()); }

signals:
	// Emitted on a left-click inside a tab's close glyph. Connected to
	// MainWindow::closePreviewSubTab() by createPreviewTab() - a plain
	// signal/slot connection, in place of the QTabBar::tabCloseRequested a
	// normal closable tab bar would emit, since this bar manages its close
	// affordance by hand (see closeGlyphRect()'s own comment on why).
	void closeRequested(int index);

protected:
	// Width is irrelevant here - paintEvent()/closeGlyphRect() both read
	// stretchedTabRect() instead, which substitutes the bar's own actual
	// current width at paint/hit-test time (see that method's own comment).
	// Height matches rowHeight(), the same value updateFixedHeight() uses
	// for the widget's own fixed height, so Qt's internal per-tab stacking
	// (tabRect(i)'s y/height, still read as-is by stretchedTabRect()) lines
	// up with one tab per row.
	QSize tabSizeHint(int index) const override {
		Q_UNUSED(index);
		return QSize(100, rowHeight());
	}

	void paintEvent(QPaintEvent * /*event*/) override {
		QStylePainter painter(this);
		for (int i = 0; i < count(); ++i) {
			QStyleOptionTab opt;
			initStyleOption(&opt, i);
			// Widened to the bar's actual current width rather than Qt's own
			// (natural, text-width-only) tabRect(i) - stretchedTabRect()'s
			// own comment explains why this can't be done by feeding
			// width() back through tabSizeHint() instead. Background/border
			// only here - QSS-aware (QTabBar#previewSubTabsBar::tab in
			// mainwindow_style.cpp), and unaffected by the label's own
			// rect-mismatch problem since there is no text to hide here.
			opt.rect = stretchedTabRect(i);
			painter.drawControl(QStyle::CE_TabBarTabShape, opt);

			QColor color = m_normalColor;
			QFont font = painter.font();
			if (opt.state & QStyle::State_Selected) {
				color = m_selectedColor;
				font.setBold(true);
			} else if (opt.state & QStyle::State_MouseOver) {
				color = m_hoverColor;
			}
			painter.setPen(color);
			painter.setFont(font);

			// Close glyph, drawn (and hit-tested - see mousePressEvent())
			// by hand for the same reason the label is: Qt's automatic
			// close-button placement (setTabButton()/tabsClosable(true))
			// positions relative to the tab bar's shape/orientation
			// semantics, not the screen - for a West-shaped bar,
			// QTabBar::LeftSide lands the button at the tab's physical
			// TOP edge, not its visual left, no matter what this bar's own
			// painting does to make the tab read as a normal horizontal
			// one. A real child widget positioned by Qt's layout can't be
			// told "screen-left" directly, so the glyph is drawn as plain
			// text instead, exactly where this bar's own coordinate space
			// says left actually is.
			painter.setPen(color);
			painter.drawText(closeGlyphRect(i), Qt::AlignCenter, QStringLiteral("×"));

			// Drawn upright in the bar's own (unrotated) coordinate space -
			// there is no rotation to undo here in the first place, unlike
			// the style's own CE_TabBarTabLabel handling for West tabs.
			// Left margin clears the close glyph; right margin is just
			// breathing room before the tab's own edge. Elided against
			// stretchedTabRect()'s width, not tabRect()'s, so widening the
			// bar (dragging the splitter - see SplitPreviewTabs) actually
			// reveals more of the label instead of leaving it truncated in
			// newly opened dead space to its right.
			const QRect textRect = stretchedTabRect(i).adjusted(30, 0, -8, 0);
			const QString elided = QFontMetrics(font).elidedText(tabText(i), Qt::ElideRight, textRect.width());
			painter.drawText(textRect, Qt::AlignVCenter | Qt::AlignLeft, elided);
		}
	}

	void mousePressEvent(QMouseEvent *event) override {
		if (event->button() == Qt::LeftButton) {
			for (int i = 0; i < count(); ++i) {
				if (closeGlyphRect(i).contains(event->pos())) {
					emit closeRequested(i);
					return; // Swallowed - a close click is not a tab-select click.
				}
			}
		}
		QTabBar::mousePressEvent(event);
	}

private:
	// tabSizeHint() reports a nominal width of 1px (see above) since Qt's
	// own per-tab width bookkeeping isn't used for anything - painting,
	// close-glyph hit-testing, and text elision all read this instead: same
	// rect as tabRect(index) (y/height still Qt's own per-row stacking),
	// width swapped for the bar's actual current width. A pure paint/
	// hit-test-time adjustment, not fed back through tabSizeHint(): an
	// earlier attempt did that (self-referential width() feeding the very
	// hint Qt uses to decide this bar's width) and it turned live splitter
	// drags into a feedback loop that collapsed the bar instead of growing
	// it.
	QRect stretchedTabRect(int index) const {
		QRect r = tabRect(index);
		r.setWidth(qMax(60, width() - r.x()));
		return r;
	}

	QRect closeGlyphRect(int index) const {
		const QRect r = stretchedTabRect(index);
		return QRect(r.left() + 6, r.center().y() - 8, 18, 18);
	}

	int rowHeight() const { return fontMetrics().height() + 20; }

	QColor m_normalColor;
	QColor m_hoverColor;
	QColor m_selectedColor;
};

// A hand-rolled stand-in for QTabWidget, built from a HorizontalTabBar and a
// QStackedWidget placed in a QSplitter instead of QTabWidget's fixed internal
// layout. QTabWidget offers no way to make the boundary between its tab bar
// and its pages user-draggable - that boundary isn't a QSplitter at all, just
// a plain layout - so there was no way to give the tab strip's width the same
// drag-to-resize affordance as the splitter on the sidebar's edge (see
// createPreviewTab()). Splitting the two halves out into real QSplitter
// children gets that same handle "for free", matching the sidebar one
// exactly since both come from the same default QSplitter::handle styling.
// Only the subset of QTabWidget's API createPreviewTab() and friends
// actually use is reimplemented here.
class SplitPreviewTabs : public QWidget {
	Q_OBJECT
public:
	explicit SplitPreviewTabs(QWidget *parent = nullptr) : QWidget(parent) {
		auto *layout = new QHBoxLayout(this);
		layout->setContentsMargins(0, 0, 0, 0);
		layout->setSpacing(0);

		// Swaps between the empty-state prompt (no renders yet, or every
		// tab closed) and the real tab bar/page splitter below - rather
		// than leaving the whole pane blank but for the sidebar, which is
		// what a bare, always-present splitter would do while count() == 0.
		m_outerStack = new QStackedWidget(this);
		layout->addWidget(m_outerStack);

		m_emptyState = new QWidget(m_outerStack);
		QVBoxLayout *emptyLayout = new QVBoxLayout(m_emptyState);
		emptyLayout->addStretch(1);
		m_emptyIcon = new QLabel(m_emptyState);
		m_emptyIcon->setAlignment(Qt::AlignCenter);
		emptyLayout->addWidget(m_emptyIcon);
		m_emptyTitle = new QLabel(tr("No renders yet"), m_emptyState);
		m_emptyTitle->setAlignment(Qt::AlignCenter);
		QFont titleFont = m_emptyTitle->font();
		titleFont.setPointSizeF(titleFont.pointSizeF() * 1.3);
		titleFont.setBold(true);
		m_emptyTitle->setFont(titleFont);
		emptyLayout->addWidget(m_emptyTitle);
		m_emptySubtitle = new QLabel(
			tr("Start a render from Basic Settings - each finished image\n"
			   "or video opens in its own tab here, so past renders stay\n"
			   "around while you compare or tweak settings."),
			m_emptyState);
		m_emptySubtitle->setAlignment(Qt::AlignCenter);
		m_emptySubtitle->setWordWrap(true);
		emptyLayout->addWidget(m_emptySubtitle);
		emptyLayout->addStretch(1);
		m_outerStack->addWidget(m_emptyState);

		m_splitter = new QSplitter(Qt::Horizontal, m_outerStack);
		m_splitter->setChildrenCollapsible(false);
		m_outerStack->addWidget(m_splitter);

		// QSplitter always stretches a pane to the splitter's full
		// perpendicular extent (here, full height) - but QTabBar, given more
		// height than its tabs need, centers the tab group in the extra
		// space rather than pinning it to the top the way QTabWidget's own
		// (non-QSplitter) internal layout does. Wrapping the bar in its own
		// top-anchoring container - tab bar, then a stretch - keeps the
		// splitter pane full height (so the drag handle still runs the
		// whole column) while restoring the top-anchored tab list.
		QWidget *tabBarContainer = new QWidget(m_splitter);
		QVBoxLayout *tabBarLayout = new QVBoxLayout(tabBarContainer);
		tabBarLayout->setContentsMargins(0, 0, 0, 0);
		tabBarLayout->setSpacing(0);
		m_tabBar = new HorizontalTabBar(tabBarContainer);
		m_tabBar->setShape(QTabBar::RoundedWest);
		tabBarLayout->addWidget(m_tabBar);
		tabBarLayout->addStretch(1);
		m_splitter->addWidget(tabBarContainer);

		m_stack = new QStackedWidget(m_splitter);
		m_splitter->addWidget(m_stack);

		// The tab strip keeps its own width on a splitter drag; the page
		// area absorbs the rest - mirrors setStretchFactor() on the outer
		// splitter between this widget and the sidebar.
		m_splitter->setStretchFactor(0, 0);
		m_splitter->setStretchFactor(1, 1);
		// setStretchFactor() only governs how space is redistributed on a
		// live resize; it does nothing for the very first layout, which
		// QSplitter seeds from each pane's sizeHint() - and m_stack's hint
		// is whatever an empty QStackedWidget reports (tiny), since this
		// runs at construction time, long before the first render ever adds
		// a page. Without this, that tiny initial split sticks: the page
		// area never claims the rest of the width on its own, so the first
		// image/video added would render small and centered in a pane far
		// smaller than what's actually available. A lopsided requested
		// split (mostly to the stack) forces QSplitter to hand it virtually
		// all the space up front instead.
		m_splitter->setSizes({1, 1'000'000});

		connect(m_tabBar, &QTabBar::currentChanged, this, [this](int index) {
			m_stack->setCurrentIndex(index);
			emit currentChanged(index);
		});
	}

	HorizontalTabBar *tabBar() const { return m_tabBar; }

	int addTab(QWidget *page, const QString &label) {
		const int stackIndex = m_stack->addWidget(page);
		const int tabIndex = m_tabBar->addTab(label);
		Q_ASSERT(stackIndex == tabIndex);
		m_tabBar->updateFixedHeight();
		m_outerStack->setCurrentWidget(m_splitter);
		return tabIndex;
	}

	void setTabToolTip(int index, const QString &tip) { m_tabBar->setTabToolTip(index, tip); }
	void setCurrentIndex(int index) { m_tabBar->setCurrentIndex(index); }
	QWidget *currentWidget() const { return m_stack->currentWidget(); }
	QWidget *widget(int index) const { return m_stack->widget(index); }
	void setElideMode(Qt::TextElideMode mode) { m_tabBar->setElideMode(mode); }

	// Removes the tab/page pair at index without deleting the page - same
	// ownership contract as QTabWidget::removeTab(), so closePreviewSubTab()
	// (mainwindow_tabs.cpp) still does its own deleteLater() afterward.
	// The tab bar's own removeTab() can auto-select and emit currentChanged
	// before the stack has caught up (index i+1's old page still sitting at
	// index i) - blocked here and re-emitted once both sides agree.
	void removeTab(int index) {
		QWidget *page = m_stack->widget(index);
		{
			const QSignalBlocker blocker(m_tabBar);
			m_tabBar->removeTab(index);
		}
		if (page) m_stack->removeWidget(page);
		m_tabBar->updateFixedHeight();
		const int newIndex = m_tabBar->currentIndex();
		m_stack->setCurrentIndex(newIndex);
		if (m_tabBar->count() == 0) m_outerStack->setCurrentWidget(m_emptyState);
		emit currentChanged(newIndex);
	}

	// Kept in sync with the active theme by MainWindow::applyTheme(), same
	// as HorizontalTabBar::setColors() - the empty-state prompt is plain
	// QLabels, not QSS-styled chrome, so it needs its colours (and the
	// icon's tint, baked into the pixmap at paint time rather than
	// stylesheet-recolourable) hand-fed the same way.
	void setEmptyStateColors(const QColor &iconColor, const QColor &titleColor, const QColor &subtitleColor) {
		m_emptyIcon->setPixmap(icon_tint::tinted(":/icons/image.svg", iconColor).pixmap(56, 56));
		m_emptyTitle->setStyleSheet(QStringLiteral("color: %1;").arg(titleColor.name()));
		m_emptySubtitle->setStyleSheet(QStringLiteral("color: %1;").arg(subtitleColor.name()));
	}

	// Inserts `widget` into the empty-state prompt, just before its
	// trailing stretch - used by createPreviewTab() to add the Recent
	// Renders list beneath the subtitle text, so it's visible in exactly
	// the state (no tabs open) where recovering a past render matters.
	void addToEmptyState(QWidget *widget) {
		auto *emptyLayout = qobject_cast<QVBoxLayout*>(m_emptyState->layout());
		emptyLayout->insertWidget(emptyLayout->count() - 1, widget);
	}

signals:
	void currentChanged(int index);

private:
	QStackedWidget *m_outerStack;
	QWidget *m_emptyState;
	QLabel *m_emptyIcon;
	QLabel *m_emptyTitle;
	QLabel *m_emptySubtitle;
	QSplitter *m_splitter;
	HorizontalTabBar *m_tabBar;
	QStackedWidget *m_stack;
};
#endif // MAINWINDOW_WIDGETS_H

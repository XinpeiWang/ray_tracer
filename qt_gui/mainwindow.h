#ifndef MAINWINDOW_H
#define MAINWINDOW_H

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

// ============================================================================
// RenderController
// ============================================================================
// Spawns ray_tracer.exe as a subprocess and turns its stdout into progress
// and log signals.
//
// This deliberately is NOT a QThread. QProcess is already fully asynchronous
// (it's a QObject that emits readyReadStandardOutput/finished/errorOccurred
// on the event loop), so it needs no worker thread of its own - this class
// lives on the GUI thread and is driven entirely by those signals. The
// earlier design subclassed QThread and ran a blocking
// `while (running) { waitForReadyRead(50); readAll(); }` loop, which cost a
// fixed up-to-50ms latency on every progress update and forced the
// stop-request flag to be an atomic polled across threads. Being
// single-threaded now, stopRender() can just kill the process directly.
// ============================================================================
class RenderController : public QObject {
	Q_OBJECT

public:
	explicit RenderController(QObject *parent = nullptr);

	// Set all render parameters before calling start()
	// Camera parameters (camX, camY, camZ) define the camera position (lookfrom)
	// The camera always looks at the Cornell box center (278, 278, 278) - lookat is fixed
	// sceneId selects which scene to render (category letter + number, e.g.
	// "A1"=Cornell Box, "A2"=Bouncing Spheres - see
	// src/TheRestOfYourLife/scene_registry.h's SceneDescriptor::id)
	// camExplicit: true if camX/Y/Z differ from sceneId's own recommended camera
	// (computed by the caller via SceneMetadataClient - see onRenderClicked) - only
	// then are cam_x/y/z actually passed on the command line, so ray_tracer.exe's
	// own "use the scene's recommended camera" fallback (main.cpp, driven by
	// LaunchArgs::cam_explicit) stays a real code path for untouched renders
	// instead of being permanently short-circuited by this GUI.
	void setParameters(bool useGPU, int width, int height, int samples, int maxDepth,
					   const QString &sceneId, double camX, double camY, double camZ, bool camExplicit,
					   const QString &outputPath = QString(), bool useWavefront = false);

	// Set video generation parameters
	void setVideoParameters(bool enabled, int frames, int fps, const QString &cameraPath, double speed = 1.0);

	// Set the "Render Options" tab's flags - a separate setter rather than
	// growing setParameters()'s already-large signature further. Each
	// parameter maps 1:1 to a CLI flag (see start()'s own arg-building);
	// defaults here match the CLI's own defaults, so a caller that never
	// invokes this (ThumbnailGenerator, in particular) renders exactly as
	// it always has. sampler/tonemap empty = use the CLI's own default
	// (sobol/aces) rather than passing the flag at all.
	void setAdvancedFlags(bool denoise, bool stats, bool optixValidate, double exposure,
						   const QString &sampler, bool spectral, const QString &tonemap);

	// Builds the command line and launches ray_tracer.exe. Returns immediately;
	// everything after this point is driven by the process's own signals.
	void start();

	// Kill the render process. Safe to call when nothing is running.
	void stopRender();

	bool isRunning() const;

signals:
	void progressUpdate(int percentage);
	void renderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void logMessage(const QString &message);

private slots:
	void onReadyRead();
	void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void onProcessErrorOccurred(QProcess::ProcessError error);

private:
	// Splits a raw output chunk into complete lines (holding any trailing
	// partial line back in m_lineBuffer until the rest arrives), emitting
	// each as a log message and feeding it to parseProgressLine().
	void handleOutputChunk(const QString &chunk);
	void parseProgressLine(const QString &line);
	void finish(bool success, const QString &message, const QString &outputPath);

	// Render configuration
	bool m_useGPU;          // true = GPU renderer, false = CPU renderer
	bool m_useWavefront;    // true = wavefront (queue-based) GPU backend; ignored unless m_useGPU
	int m_width;            // Image width in pixels
	int m_height;           // Image height in pixels
	int m_samples;          // Samples per pixel (anti-aliasing quality)
	int m_maxDepth;         // Max ray bounce depth (lighting quality)
	QString m_sceneId;      // Scene selector ID (category letter + number, e.g. "A1")

	// Camera position (lookfrom) - always looking at center (278, 278, 278)
	double m_camX;          // Camera X coordinate
	double m_camY;          // Camera Y coordinate
	double m_camZ;          // Camera Z coordinate
	bool m_camExplicit;     // See setParameters()'s comment

	QString m_outputPath;   // Output file path for rendered image
	QProcess *m_renderProcess = nullptr;  // Subprocess handle (child of this object)

	// "Render Options" tab flags - see setAdvancedFlags()'s own comment.
	// Defaults here match every existing caller's prior behavior (nothing
	// passed = CLI default), so ThumbnailGenerator (which never calls
	// setAdvancedFlags()) is unaffected.
	bool m_denoise = false;
	bool m_stats = false;
	bool m_optixValidate = false;
	double m_exposure = 1.0;
	QString m_sampler;
	bool m_spectral = false;
	QString m_tonemap;

	// Whether the user asked to stop, tracked explicitly rather than inferred
	// from exit status/code - a real crash (e.g. access violation) also
	// produces CrashExit + a nonzero exit code on Windows, so that heuristic
	// can't tell the two apart and used to mislabel genuine crashes as
	// "stopped by user", hiding the error dialog.
	bool m_stopRequested = false;

	// Guards against emitting renderComplete twice when both errorOccurred
	// and finished fire for the same run.
	bool m_finished = false;

	QElapsedTimer m_elapsed;   // Wall-clock timer for the render
	QString m_lineBuffer;      // Incomplete trailing line awaiting the rest of its chunk
	int m_lastProgress = 0;    // Last emitted percentage (progress only moves forward)

	// Video generation parameters
	bool m_videoMode;       // true = video generation, false = single image
	int m_videoFrames;      // Number of frames to render
	int m_videoFPS;         // Target frames per second
	double m_videoSpeed;    // Camera movement speed multiplier (1.0 = default path speed)
	QString m_cameraPath;   // Camera animation path (orbit, linear, figure8, spiral)
};

// ============================================================================
// DiagnosticsRunner
// ============================================================================
// Spawns `ray_tracer.exe --diagnose` and captures its full stdout as one
// report string. A stripped-down sibling of RenderController above (same
// QProcess-is-already-async reasoning, same not-a-QThread shape) - no
// progress parsing, no video/scene parameters, just "run it, hand back
// whatever it printed."
// ============================================================================
class DiagnosticsRunner : public QObject {
	Q_OBJECT

public:
	explicit DiagnosticsRunner(QObject *parent = nullptr);

	// Builds the command line and launches ray_tracer.exe --diagnose.
	// Returns immediately; everything after is driven by the process's own
	// signals.
	void start();

	// Kill the diagnostics process. Safe to call when nothing is running.
	// Mirrors RenderController::stopRender()/isRunning() - added so
	// ~MainWindow() has a graceful way to end an in-flight diagnostics run
	// at app exit instead of relying solely on QProcess's own destructor
	// (which does forcibly kill a still-running child, but with a blocking
	// wait and a logged warning rather than this app's own quieter path).
	void stop();

	bool isRunning() const;

signals:
	void reportReady(const QString &report);
	void reportFailed(const QString &message);

private slots:
	void onProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void onProcessErrorOccurred(QProcess::ProcessError error);

private:
	QProcess *m_process = nullptr;
	bool m_finished = false;   // guards against double-emitting on error+finished
};

// ============================================================================
// ThumbnailGenerator
// ============================================================================
// Renders the scene gallery grid's preview PNGs, one scene at a time, via its
// own RenderController - deliberately NOT m_renderController/m_renderQueue.
// Those are wired end-to-end into the visible progress bar, status label,
// queue panel and stop button; routing thumbnail jobs through them would
// hijack that chrome for invisible background work. This class owns a
// private RenderController instead and only ever READS the real render
// state (see MainWindow::onGenerateThumbnailsClicked()) to decide whether
// it's safe to start - it never competes with a user-requested render.
//
// Generation is manual/one-shot (triggered by "Generate Thumbnails"), not a
// background daemon, so there is no pause/resume logic here: start() walks
// its whole queue to completion or until stop() is called.
// ============================================================================
class ThumbnailGenerator : public QObject {
	Q_OBJECT

public:
	explicit ThumbnailGenerator(QObject *parent = nullptr);

	// Renders every id in `sceneIds` that doesn't already have a cached PNG
	// at outputPathForId(id), one at a time, low-res/low-spp/CPU-only (see
	// .cpp for the exact RenderController::setParameters() call). Ignored if
	// already running.
	void start(const QStringList &sceneIds, std::function<QString(const QString &)> outputPathForId);

	// Kills the in-flight render (if any) and drops the rest of the queue.
	void stop();

	bool isRunning() const;

signals:
	// One per finished scene (success or failure) - MainWindow uses this to
	// update that scene's grid item icon in place and advance status text.
	void thumbnailReady(const QString &sceneId, bool success, const QString &outputPath);
	// Fires once after the last queued scene finishes (or stop() is called).
	void allDone();

private:
	void startNext();

	RenderController *m_controller;
	QQueue<QString> m_pending;
	std::function<QString(const QString &)> m_outputPathForId;
	QString m_currentId;
	bool m_stopRequested = false;
};

// A snapshot of every render-affecting UI field at the moment "Start Render"
// was clicked, so a queued job is unaffected by any further changes the user
// makes to the form while an earlier job is still running. Mirrors
// RenderController::setParameters()/setVideoParameters()'s own parameter
// list exactly - see MainWindow::captureRenderJob()/startRenderJob()
// (mainwindow_slots.cpp).
struct RenderJob {
	bool useGPU = false;
	bool useWavefront = false;
	int width = 0;
	int height = 0;
	int samples = 0;
	int maxDepth = 0;
	QString sceneId;
	QString outputPath;
	double camX = 0.0;
	double camY = 0.0;
	double camZ = 0.0;
	bool camExplicit = true;
	bool videoMode = false;
	int videoFrames = 0;
	int videoFPS = 0;
	double videoSpeed = 1.0;
	QString cameraPath;
	QString displayTitle;      // Scene name, for the queue list row - see describeRenderJob()
	QString sceneDescription;  // Scene's own description, for the finished Preview tab's tooltip
	QString videoPresetName;   // Named video preset's display name, if one was selected; empty otherwise

	// "Render Options" tab fields - see RenderController::setAdvancedFlags()'s
	// own comment. sampler/tonemap empty = use the CLI's own default.
	bool denoise = false;
	bool stats = false;
	bool optixValidate = false;
	double exposure = 1.0;
	QString sampler;
	bool spectral = false;
	QString tonemap;
};

// ============================================================================
// ExpandingTabBar / ExpandingTabWidget
// ============================================================================
// Makes the main tab strip's 7 tabs fill the window's full width instead of
// sitting left-aligned with blank space to the right of "Diagnostics" once
// the window is wider than the tabs' own natural size.
//
// QTabBar::setExpanding(true) - the documented-sounding way to do this - does
// NOT do this: confirmed empirically (a standalone repro, logging every
// tabSizeHint() call) that it only equalizes tab widths when tabs must
// SHRINK to fit an overflowing bar, never grows them to fill idle space.
// QTabWidget also never stretches its tab bar to the widget's own width in
// the first place - the bar only ever claims its own sizeHint (the sum of
// its tabs' natural widths), which is the real reason the strip left-aligns.
//
// The fix is a tabSizeHint() override that hands out width()/count() per
// tab whenever that's wider than the tab's natural hint - but width() itself
// is circular (the bar's width is DERIVED from summing these same hints), so
// naively reading width() here just converges to some in-between value, not
// the full window width (also confirmed empirically). Reading
// parentWidget()->width() instead - the QTabWidget itself, stable and set
// independently of the tab bar's own size - breaks that circularity.
class ExpandingTabBar : public QTabBar {
public:
	explicit ExpandingTabBar(QWidget *parent = nullptr) : QTabBar(parent) {}

protected:
	QSize tabSizeHint(int index) const override {
		QSize hint = QTabBar::tabSizeHint(index);
		const int n = count();
		if (n > 0 && parentWidget()) {
			// The "- n * 4" isn't cosmetic: each tab's own QSS margin/border
			// (mainwindow_style.cpp's QTabBar::tab rule) adds a few pixels
			// this per-tab hint doesn't otherwise account for. Without this
			// slack, the summed hints land a handful of pixels OVER the
			// parent's actual width, tipping the whole bar into scroll-arrow
			// mode - which then reserves its own space for the arrows and
			// never revisits this hint, so only the first few (oversized)
			// tabs end up visible at all. A few pixels of unused margin at
			// the right edge is a far smaller cost than that cliff.
			const int evenWidth = (parentWidget()->width() - n * 4) / n;
			if (evenWidth > hint.width()) hint.setWidth(evenWidth);
		}
		return hint;
	}
};

class ExpandingTabWidget : public QTabWidget {
public:
	explicit ExpandingTabWidget(QWidget *parent = nullptr) : QTabWidget(parent) {
		setTabBar(new ExpandingTabBar(this));
	}
};

// ============================================================================
// MainWindow
// ============================================================================
// Main GUI window with tabbed interface for render controls
// Basic Tab: Quick presets and render mode selection
// Advanced Tab: Detailed controls including camera position presets
// ============================================================================
class MainWindow : public QMainWindow {
	Q_OBJECT

public:
	explicit MainWindow(QWidget *parent = nullptr);
	~MainWindow();

private slots:
	void onRenderClicked();
	void onStopClicked();
	void onQualityPresetChanged(int index);
	void onCameraPresetChanged(int index);  // Updates camera spinboxes when preset changes
	void onVideoPresetChanged(int index);   // Points scene/camera-path/frames/fps/speed at a named preset
	void onCameraDistanceChanged(double distance);  // Repositions camera X/Y/Z along its current direction from lookat
	void onSceneChanged(int index);         // Updates UI when scene selection changes
	void onModeChanged(int index);          // Switches between Image and Video modes
	void onProgressUpdate(int percentage);
	void onRenderComplete(bool success, const QString &message, double totalTime, const QString &outputPath);
	void onLogMessage(const QString &message);
	void onElapsedTick();        // fires every second during render to update status label
	void onRemoveSelectedQueueItem();  // Removes the currently-selected row from m_renderQueue
	void onClearQueue();               // Empties m_renderQueue entirely
	void onRunDiagnosticsClicked();
	void onDiagnosticsReportReady(const QString &report);
	void onDiagnosticsFailed(const QString &message);
	void onGenerateThumbnailsClicked();
	void onThumbnailReady(const QString &sceneId, bool success, const QString &outputPath);
	void onThumbnailsAllDone();

	// Shared by the log tab's buttons and the File menu's actions.
	void copyLogToClipboard();
	void saveLogToFile();
	void clearLog();
	void showAboutDialog();

	// Shared by the Diagnostics tab's buttons.
	void copyDiagToClipboard();
	void saveDiagReportToFile();

private:
	void setupUI();
	void createBasicTab();
	void createAdvancedTab();
	void createRenderOptionsTab();
	void createVideoTab();
	void createPreviewTab();
	void createProgressTab();
	void createLogTab();
	void createDiagnosticsTab();

	// ------------------------------------------------------------------
	// Action layer
	// ------------------------------------------------------------------
	// Every command is a QAction first. A QAction carries its shortcut,
	// tooltip and status tip in one object, so the menu entry, the keyboard
	// shortcut and the status-bar hint can't drift out of sync the way three
	// separate hand-wired copies would. Qt Creator, OBS, Kate and Wireshark
	// are all built this way.
	void createActions();
	void createMenus();
	void createStatusBar();
	// Enables/disables the render-related actions to match m_isRendering.
	void updateActionStates();

	QAction *m_actRender = nullptr;
	QAction *m_actRenderVideo = nullptr;
	QAction *m_actStop = nullptr;
	QAction *m_actOpenFolder = nullptr;
	QAction *m_actOpenViewer = nullptr;
	QAction *m_actCopyLog = nullptr;
	QAction *m_actSaveLog = nullptr;
	QAction *m_actClearLog = nullptr;
	QAction *m_actAbout = nullptr;
	QAction *m_actAboutQt = nullptr;
	QAction *m_actQuit = nullptr;

	// Status bar: permanent widgets carry ambient state that would otherwise
	// have no home (which renderer, what size/quality). Deliberately NOT the
	// progress bar - that stays large and central where it belongs.
	QLabel *m_statusDevice = nullptr;
	QLabel *m_statusSettings = nullptr;

	// The scheme currently in force. Widgets that are styled individually
	// (the log pane, preview, scene info panel) re-read this when it changes.
	theme::Palette m_activeTheme;
	QVector<QAction *> m_themeActions;
	void restyleThemedWidgets();
	void refreshStatusBarInfo();
	// Applies a colour scheme to the whole app. Safe to call repeatedly: the
	// theme menu re-invokes it to switch live rather than asking for a restart.
	void applyTheme(const theme::Palette &palette);
	void switchTheme(const QString &themeId);
	void createThemeMenu();
	QString loadSavedThemeId() const;
	void saveThemeId(const QString &themeId) const;
	void styleComboBox(QComboBox *combo);
	void applyComboPopupPalette(QComboBox *combo);
	void styleSpinBox(QAbstractSpinBox *spinBox);
	void styleGroupBox(QGroupBox *box);
	void styleCheckBox(QCheckBox *box);
	// A subtle "elevated card" drop shadow (QSS alone cannot do box-shadow) -
	// neutral black at low alpha rather than theme-tinted, the same choice
	// every real elevation system (Material, Fluent, CSS itself) makes,
	// since a shadow reads as "surface above surface" on any hue, dark or
	// light theme alike, without needing a per-palette shadow colour.
	void applyElevation(QWidget *widget, qreal blurRadius, qreal offsetY, int alpha);
	// Same QGraphicsDropShadowEffect mechanism as applyElevation(), but
	// centred (no offset) and theme-coloured rather than neutral black - a
	// glow of energy around the widget rather than a cast shadow beneath
	// it. Used for the progress bar's own pulse while actively rendering
	// (see startProgressGlow()) - a different visual language deliberately
	// kept separate from applyElevation() rather than folding a colour
	// parameter into that one, since "surface depth" and "energy glow" read
	// as two different things even though the underlying Qt mechanism is
	// identical.
	void applyGlow(QWidget *widget, qreal blurRadius, const QColor &color);
	// Starts/stops the progress bar's pulsing glow - see applyGlow()'s own
	// comment. Called from startRenderJob()/onRenderComplete() so the pulse
	// runs exactly while m_isRendering is true, settling to a fixed glow
	// once finished (the QSS resultState colouring already handles success/
	// error at that point - this only ever controls whether it's animating).
	void startProgressGlow();
	void stopProgressGlow();
	// Smoothly animates m_progressBar's own `value` property (a real
	// QProgressBar Q_PROPERTY) toward `value` instead of snapping - Qt
	// redraws the bar's built-in percentage text from that same live
	// property each animation frame, so this animates the fill AND the
	// number simultaneously for free, no separate text animation needed.
	void animateProgressTo(int value);
	// baseOutputPath is the render's own --output argument (see
	// onRenderClicked()) so the assembled video's expected path
	// ("<stem>_video.mp4") can be derived directly instead of guessing at a
	// shared directory - necessary now that every render's base path is
	// unique. Automatically assembles/locates the video after frames are
	// rendered and adds it as a new Preview sub-tab. `job` is the RenderJob
	// that was actually rendered, captured by value into the caller's
	// deferred QTimer::singleShot rather than read back from m_currentJob -
	// this runs ~500ms after onRenderComplete() returns, by which time a
	// queued next job may already have overwritten it (see m_currentJob's
	// own comment).
	void assembleVideoAutomatically(const QString &baseOutputPath, const RenderJob &job);
	void refreshCameraDistanceDisplay(); // Recomputes m_cameraDistance's shown value from X/Y/Z and m_currentLookat*, without re-triggering onCameraDistanceChanged

	// Preview sub-tabs (see m_previewSubTabs's own comment). Both add*
	// functions build a self-contained tab page, register it, and make it
	// current; tooltip is the full scene/preset description, shown on
	// hover since the tab bar itself only has room for a short title.
	void addImagePreviewTab(const QString &title, const QString &tooltip, const QPixmap &pixmap,
							 const QString &infoText, const QString &outputPath, const QString &previewPath);
	void addVideoPreviewTab(const QString &title, const QString &tooltip, const QString &videoPath,
							 const QString &infoText);
	// First use of a title returns it unchanged; each repeat appends " (N)".
	QString uniquePreviewTabTitle(const QString &baseTitle);
	// Reads a property (see m_previewSubTabs's comment) off the currently
	// active sub-tab's page widget; empty string if there is no active tab.
	QString currentPreviewProperty(const char *name) const;
	// Refreshes m_previewInfoLabel and the Open Folder/Viewer actions from
	// whichever sub-tab just became active - connected to m_previewSubTabs's
	// currentChanged signal.
	void updatePreviewSidebarForActiveTab();
	// Removes and deletes a Preview sub-tab. Called from each tab's own
	// left-side close button (see addImagePreviewTab()/addVideoPreviewTab())
	// rather than QTabWidget::tabCloseRequested, since Qt's own auto-managed
	// close button is turned off entirely - see createPreviewTab()'s comment
	// on why.
	void closePreviewSubTab(int index);

	// Camera arithmetic lives in camera_math.h (Qt-free, unit tested); these
	// just read the current values out of the widgets for it.
	camera_math::Vec3 currentCameraPosition() const;
	camera_math::Vec3 currentLookAt() const;

	// UI Components
	QTabWidget *m_tabWidget;

	// Mode selector (Image vs Video)
	QComboBox *m_modeCombo;             // Render mode: Image or Video

	// Basic Tab
	QComboBox *m_renderModeCombo;       // GPU vs CPU selection
	QComboBox *m_gpuBackendCombo;       // Recursive vs wavefront GPU path tracer (only meaningful under GPU)
	QComboBox *m_qualityPresetCombo;    // Quality preset dropdown
	QComboBox *m_resolutionCombo;       // Resolution preset dropdown
	QLineEdit *m_outputPathEdit;        // Output file path (timestamped by default)
	QPushButton *m_browseButton;
	QPushButton *m_renderButton;
	QPushButton *m_stopButton;
	QProgressBar *m_progressBar;
	// See applyGlow()/startProgressGlow()'s own comments. Both lazily
	// created on first use, not at construction - the progress bar's
	// QGraphicsDropShadowEffect only needs to exist once rendering has
	// actually started once.
	QPropertyAnimation *m_progressGlowAnim = nullptr;
	QPropertyAnimation *m_progressValueAnim = nullptr;
	QLabel *m_statusLabel;
	// A secondary caveat line below m_statusLabel (e.g. "render succeeded but
	// the preview image couldn't be shown") - kept separate rather than
	// appended into m_statusLabel's own text so it can carry the theme's
	// warning colour without needing m_statusLabel's other ~15 call sites
	// (success/failure/progress messages) to switch to rich text and start
	// HTML-escaping arbitrary renderer output. Hidden whenever there is
	// nothing to caveat - see setStatusWarning()/clearStatusWarning().
	QLabel *m_statusWarningLabel;
	QLabel *m_currentJobLabel;          // Which job is actually rendering - see startRenderJob()/describeRenderJob()
	int m_progressTabIndex = -1;        // Index of the Progress tab within m_tabWidget (see createProgressTab())
	int m_videoTabIndex = -1;           // Index of the Video Settings tab within m_tabWidget (see createVideoTab()/onModeChanged())

	// Advanced Tab - Manual Controls
	QSpinBox *m_widthSpinBox;           // Custom width
	QSpinBox *m_heightSpinBox;          // Custom height
	QSpinBox *m_samplesSpinBox;         // Samples per pixel
	QSpinBox *m_maxDepthSpinBox;        // Max ray depth

	// Render Options tab (createRenderOptionsTab()) - one widget per CLI
	// flag RenderController::start() can emit; see setAdvancedFlags()'s own
	// comment. m_samplerCombo/m_spectralCheck are CPU-default-path-tracer
	// only; m_denoiseCheck/m_optixValidateCheck are GPU-only (m_denoiseCheck
	// further disabled under wavefront) - enabled state kept in sync with
	// m_renderModeCombo/m_gpuBackendCombo, see the constructor's own
	// connect() lambda for m_gpuBackendCombo's existing enable/disable
	// pattern (mainwindow.cpp).
	QComboBox *m_samplerCombo;          // --sampler (CPU default path tracer only)
	QCheckBox *m_spectralCheck;         // --spectral (CPU default path tracer only)
	QDoubleSpinBox *m_exposureSpin;     // --exposure (both backends, default path tracer only)
	QComboBox *m_tonemapCombo;          // --tonemap (both backends, default path tracer only)
	QCheckBox *m_statsCheck;            // --stats (always applicable)
	QCheckBox *m_denoiseCheck;          // --denoise (GPU recursive backend only)
	QCheckBox *m_optixValidateCheck;    // --optix-validate (GPU only)

	// Camera controls
	// Camera position (lookfrom) can be set via presets or custom X/Y/Z
	// values; lookat is fixed per-scene in the renderer (not the same point
	// for every scene - see scene_registry.h's CameraConfig::lookat_x/y/z,
	// queried live via SceneMetadataClient::recommendedCamera() and
	// mirrored into m_currentLookatX/Y/Z below whenever the scene changes).
	QComboBox *m_cameraPresetCombo;     // Preset camera positions (includes "Custom" option)
	QDoubleSpinBox *m_cameraPosX;       // Camera X position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosY;       // Camera Y position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraPosZ;       // Camera Z position (enabled only for "Custom" preset)
	QDoubleSpinBox *m_cameraDistance;   // Distance from the current scene's look-at point (enabled only for "Custom")
	// Currently selected scene's look-at point (updated in onSceneChanged).
	// Used by onCameraDistanceChanged to reposition the camera along its
	// existing viewing direction from this point, rather than needing to
	// know the scene's geometry to type new X/Y/Z values by hand.
	double m_currentLookatX = 278.0, m_currentLookatY = 278.0, m_currentLookatZ = 278.0;
	// Distance from m_currentLookat* to the CURRENT scene's own recommended
	// camera (updated in onSceneChanged alongside the lookat fields above).
	// m_cameraPresetCombo's items store direction*ratio vectors (see its
	// setup comment in mainwindow_tabs.cpp) that get scaled by this value in
	// onCameraPresetChanged, so a named preset like "Right Wall" lands at a
	// sensible position for whatever scene is active instead of always
	// landing at Cornell Box's own literal (500,278,278), which used to be
	// wildly outside the geometry of any other scene (e.g. scene 1's
	// spheres sit within roughly +-15 units of the origin).
	double m_currentSceneCamDistance = 1078.0;

	// Scene selection
	// Two-level filter: m_sceneAvailabilityTabs splits every scene into
	// "Self-Contained" (renders in a fresh checkout, no extra downloads) vs.
	// "Requires External Files" (SceneMetadataClient::sceneRequiresFiles()),
	// independently of SceneCategories - a category like Basics or Geometry
	// has scenes in both buckets (e.g. Basics' A4 Earth needs an image file,
	// A1-A9 mostly don't), so this is a per-scene split layered on top of the
	// existing letter categories, not a coarser replacement for them.
	// m_sceneCategoryTabs is then rebuilt to show only the letter categories
	// that have at least one scene in whichever availability bucket is
	// selected (rebuildCategoryTabs()) - mirroring how it already skips any
	// category with zero scenes at all.
	QTabBar *m_sceneAvailabilityTabs = nullptr;  // "Self-Contained" / "Requires External Files"
	QTabBar *m_sceneCategoryTabs = nullptr;  // Category filter above the scene dropdown
	// Narrows the current category's own combo list further by name/id/
	// description substring - independent of, and applied on top of, the
	// availability/category tab filters above. Scenes went from 78 to 154
	// (curated pbrt examples added under real topic categories, see
	// scene_registry.h), so a category tab alone can now hold 20+ scenes -
	// exactly the point a flat dropdown starts wanting search.
	QLineEdit *m_sceneSearchBox = nullptr;
	QComboBox *m_sceneCombo;            // Scene selector dropdown, showing one category at a time

	// Grid ("gallery") alternative to m_sceneCombo - same filtered scene set,
	// shown as thumbnail tiles instead of text rows. Toggled against
	// m_sceneCombo via m_sceneViewStack/m_sceneViewToggle; see
	// populateSceneGrid()'s own comment. Thumbnails come from
	// thumbnailCachePath() - a generated PNG if one exists, else a shared
	// placeholder icon.
	QListWidget *m_sceneGrid = nullptr;
	QStackedWidget *m_sceneViewStack = nullptr;   // page 0 = m_sceneCombo, page 1 = m_sceneGrid
	QToolButton *m_sceneViewToggle = nullptr;     // checked = grid page showing
	QPushButton *m_generateThumbnailsButton = nullptr;
	// Lazily created (and reused across multiple "Generate Thumbnails"
	// clicks) by onGenerateThumbnailsClicked() the first time it's needed -
	// see that slot's own comment.
	ThumbnailGenerator *m_thumbnailGenerator = nullptr;

	// Rebuilds m_sceneCategoryTabs' tab set for the given availability filter
	// (true = only scenes with sceneRequiresFiles()==true, false = only
	// scenes with sceneRequiresFiles()==false), skipping any letter category
	// left with zero matching scenes - same skip-if-empty rule
	// createBasicTab() already applies for categories with zero scenes at
	// all. Does not touch m_sceneCombo; callers follow up with
	// populateSceneCombo() for whichever category tab ends up selected.
	void rebuildCategoryTabs(bool requiresFiles);

	// The scene ids matching `category` and m_sceneAvailabilityTabs' current
	// selection, further narrowed by m_sceneSearchBox's text if any (name/id/
	// description substring match) - the shared filter both
	// populateSceneCombo() and populateSceneGrid() build their contents from,
	// so category/availability/search apply identically to both views with
	// no duplicated filter logic.
	QStringList filteredSceneIds(const QString &category) const;

	// Refills m_sceneCombo with just the scenes in `category` that also match
	// m_sceneAvailabilityTabs' current selection. Does NOT emit
	// currentIndexChanged per insertion - callers apply the resulting selection
	// themselves with a single onSceneChanged() call.
	void populateSceneCombo(const QString &category);

	// Refills m_sceneGrid the same way populateSceneCombo() refills
	// m_sceneCombo - same filteredSceneIds(category), one QListWidgetItem per
	// scene (Qt::UserRole holds the scene id, mirroring the combo's item-data
	// convention), icon set to the cached thumbnail if thumbnailCachePath(id)
	// exists on disk, else a shared placeholder built once in
	// createBasicTab(). Does NOT emit itemSelectionChanged - callers behave
	// like populateSceneCombo()'s own callers.
	void populateSceneGrid(const QString &category);

	// Calls populateSceneCombo() and populateSceneGrid() together - every
	// call site that used to call populateSceneCombo() alone now calls this,
	// so both views always agree on what's currently filtered in.
	void populateSceneViews(const QString &category);

	// Absolute path a scene's cached thumbnail PNG would live at, under
	// QStandardPaths::CacheLocation + "/thumbnails" (mirrors theme_load.cpp's
	// own QStandardPaths convention for per-user files) - does not check
	// whether the file actually exists.
	QString thumbnailCachePath(const QString &sceneId) const;

	// Drives all three scene-selection widgets (availability tab, category
	// tab, m_sceneCombo) to the scene matching `id`, as if the user had
	// clicked through to it by hand - switching availability/category tabs
	// first if `id` isn't already visible under the current ones. Used by
	// the video preset combo (see onVideoPresetChanged()) to point the
	// scene picker at a preset's scene without duplicating this tab-
	// cascading logic. No-op (logs a warning) if `id` isn't a real scene id.
	void selectSceneById(const QString &id);
	QLabel *m_sceneInfoLabel;           // Scene description and performance info

	// Rebuilds m_sceneInfoLabel's text (description/performance/SPP/GPU-
	// support, plus the requires-files/CPU-only warning badges) for
	// whichever scene m_sceneCombo currently has selected. Split out of
	// onSceneChanged() so restyleThemedWidgets() can call this alone on a
	// theme switch - the warning badges' colours are baked into inline HTML
	// at build time (QSS can't reach them), so without a way to rebuild just
	// the label, an already-shown badge would keep the previous theme's
	// colour until the user reselected a scene. No-op if nothing is selected
	// or the scene's metadata can't be queried.
	void refreshSceneInfoLabel();

	// Video Tab
	QComboBox *m_videoPresetCombo;      // Named scene+path+frames/fps/speed bundle - see video_preset.h
	QComboBox *m_cameraPathCombo;       // Camera animation path selector
	QSpinBox *m_videoFramesSpinBox;     // Number of frames to render
	QSpinBox *m_videoFPSSpinBox;        // Target FPS for video
	QDoubleSpinBox *m_videoSpeedSpinBox; // Camera movement speed multiplier
	QLabel *m_videoInfoLabel;           // Video duration and path info
	// Visible whenever Output Mode (Basic Settings) isn't "Generate Video" -
	// see createVideoTab()'s own comment for why this tab stays enabled and
	// clickable rather than being disabled outright. Toggled by onModeChanged().
	QLabel *m_videoModeWarningLabel = nullptr;

	// Preview tab - each completed render gets its own closable sub-tab
	// (see addImagePreviewTab()/addVideoPreviewTab()) instead of a single
	// shared pane that the next render overwrites, so switching between
	// sub-tabs keeps every past render's image/video around. Each sub-tab
	// page widget carries its own "outputPath"/"previewPath"/"infoText"
	// Qt properties (image mode: a ScaledImageLabel; video mode: its own
	// QVideoWidget/QMediaPlayer/QAudioOutput, all parented to the page so
	// closing the tab tears them down too) - see createPreviewTab() and
	// currentPreviewProperty()/updatePreviewSidebarForActiveTab().
	SplitPreviewTabs *m_previewSubTabs = nullptr;
	QWidget *m_previewSidebar = nullptr; // Info/buttons pane; hidden while there are no sub-tabs - see updatePreviewSidebarForActiveTab()
	QLabel *m_previewInfoLabel;         // Filename / resolution / size / render time - reflects whichever sub-tab is active
	QLabel *m_previewSceneDescLabel;    // Selected scene's description - see onSceneChanged()
	int m_previewTabIndex = -1;         // Index of the Preview tab within m_tabWidget
	// Counts repeat sub-tab titles ("Cornell Box" -> "Cornell Box (2)") so
	// re-rendering the same scene/preset in one session doesn't produce
	// indistinguishable tabs - see uniquePreviewTabTitle().
	QMap<QString, int> m_previewTitleCounts;

	// Log output
	QTextEdit *m_logTextEdit;           // Log output display
	int m_logTabIndex = -1;             // Index of the Log Output tab within m_tabWidget

	// Every line the log has shown, kept so a theme change can re-render it.
	// The QTextEdit holds only the RESULT of styling - HTML with one scheme's
	// colours already baked into each span - which cannot be recoloured after
	// the fact. Keeping the inputs (what the line said, and how it classified)
	// is what makes a rebuild possible; the classification is cached rather
	// than recomputed because it is regex work and a long render log has tens
	// of thousands of lines.
	struct LoggedLine {
		QString timestamp;
		QString escaped;
		render_output::LogCategory category;
	};
	QVector<LoggedLine> m_logHistory;
	void rebuildLogPane();

	// Diagnostics
	QTextEdit *m_diagTextEdit;              // Diagnostics report display
	QPushButton *m_runDiagnosticsButton;    // Disabled while a probe is running
	int m_diagnosticsTabIndex = -1;         // Index of the Diagnostics tab within m_tabWidget
	DiagnosticsRunner *m_diagnosticsRunner = nullptr;  // nullptr when not running
	// Raw (unstyled) report text, kept for the same reason m_logHistory is:
	// the pane holds styled HTML with the previous scheme's colours already
	// baked in, so a theme change re-renders from this rather than trying to
	// recolour existing spans in place. Empty when the pane is showing a
	// plain (non-report) message, e.g. "Running diagnostics..." or a failure
	// - see onDiagnosticsReportReady()/onDiagnosticsFailed().
	QString m_lastDiagReport;
	void rebuildDiagPane();

	// Render driver (nullptr when not rendering)
	RenderController *m_renderController;

	// State
	bool m_isRendering;                 // true when a render is in progress
	bool m_videoMode;                   // true = video generation mode, false = single image mode

	// ------------------------------------------------------------------
	// Render queue
	// ------------------------------------------------------------------
	// onRenderClicked() (mainwindow_slots.cpp) always enqueues a captured
	// RenderJob and then calls processQueueIfIdle() - which starts the front
	// job only if nothing is currently running. The everyday single-render
	// case is just "enqueue one job into an empty, immediately-idle queue",
	// so there is no separate "start now" code path to keep in sync with
	// the queued one.
	QQueue<RenderJob> m_renderQueue;
	// The job actually passed to the RenderController currently running (or
	// most recently run). onRenderComplete() reads its videoMode/sceneId/
	// displayTitle from here, not from the live UI widgets - once renders
	// can queue, the user may have already changed the scene/mode combo for
	// the *next* job by the time an earlier one's completion signal arrives.
	RenderJob m_currentJob;
	QGroupBox *m_queueGroup = nullptr;       // Hidden whenever m_renderQueue is empty
	QListWidget *m_queueListWidget = nullptr;
	RenderJob captureRenderJob();             // Snapshots every render field currently in the UI
	void startRenderJob(const RenderJob &job); // Builds a RenderController for `job` and starts it
	void processQueueIfIdle();                // Dequeues and starts the front job if nothing is running
	void refreshQueuePanel();                 // Rebuilds m_queueListWidget from m_renderQueue
	static QString describeRenderJob(const RenderJob &job); // One-line queue-row summary

	// Elapsed render timer
	QTimer *m_elapsedTimer;             // fires every second during render
	QDateTime m_renderStartTime;        // wall-clock time when render began

	// ------------------------------------------------------------------
	// Progress sampling for the time-remaining estimate
	// ------------------------------------------------------------------
	// Modelled on HandBrake's UpdateState() (libhb/sync.c), which is the
	// most carefully-built ETA of the render/encode tools surveyed. Two
	// separate rates are derived from the same samples:
	//
	//   * an INSTANTANEOUS rate over the short sliding window, which is
	//     responsive enough to be worth showing, but far too noisy to
	//     divide by; and
	//   * a CUMULATIVE rate over the whole render, whose sensitivity to
	//     new noise keeps shrinking, so the ETA drifts smoothly instead
	//     of oscillating every tick.
	//
	// The ETA uses the cumulative rate for exactly that reason. Nothing is
	// shown at all until kEtaWarmupMs has passed - an early estimate from
	// two samples is worse than admitting we don't know yet, so the label
	// reads "--:--" rather than a wild (or zero) guess.
	struct ProgressSample {
		qint64 elapsedMs = 0;
		int percent = 0;
	};
	static constexpr int kProgressSamples = 4;     // ring depth (~3s window at 1Hz)
	static constexpr qint64 kEtaWarmupMs = 4000;   // no estimate before this
	ProgressSample m_progressRing[kProgressSamples];
	int m_progressRingCount = 0;        // how many slots are actually filled

	void resetProgressSamples();
	// Appends a sample and returns the formatted "elapsed / ETA" status
	// text for the current tick.
	QString formatProgressStatus(qint64 elapsedMs, int percent);
	// Paints the progress bar green on success / red on failure, matching
	// Qt Creator's ProgressBarColorFinished / ProgressBarColorError.
	void setProgressResultState(const char *state);
	// Shows/hides m_statusWarningLabel - see that member's own comment.
	void setStatusWarning(const QString &text);
	void clearStatusWarning();
	// Clears or reddens the taskbar button and, if the window isn't the
	// active one, raises a tray notification.
	void notifyRenderFinished(bool success, const QString &message, double totalTime);

	// Last percentage pushed to the taskbar button, so the COM call only
	// fires when the integer percent actually changes.
	int m_lastTaskbarPercent = -1;

	// Used only for completion notifications - the app has no tray UI.
	// Null if the platform has no system tray.
	QSystemTrayIcon *m_trayIcon = nullptr;
	// The active-window complement to m_trayIcon's showMessage() below - see
	// notifyRenderFinished()'s own comment for which one fires when.
	ToastNotification *m_toast = nullptr;
	// Shared event filter that blocks accidental wheel-scroll on controls
	WheelIgnoreFilter *m_wheelFilter;
};

#endif // MAINWINDOW_H

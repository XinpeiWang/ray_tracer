#ifndef ORBIT_PREVIEW_LABEL_H
#define ORBIT_PREVIEW_LABEL_H
// orbit_preview_label.h -- the Live Preview's picture widget: ScaledImageLabel (an image scaled to fit its label) and OrbitPreviewLabel (mouse and key input for
// orbiting the camera, plus the Move objects mode). Split out of mainwindow_widgets.h, which includes it, so existing includes keep working.

#include <QKeyEvent>
#include <QLabel>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPainter>
#include <QPen>
#include <QPixmap>
#include <QPointF>
#include <QResizeEvent>
#include <QVector>
#include <QWheelEvent>

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

	// Where the image is drawn inside the label (it is scaled to fit and centred), in the label's own coordinates; an empty rectangle without an image.
	QRect displayedImageRect() const {
		if (m_original.isNull()) return QRect();
		const QSize shown = m_original.size().scaled(size(), Qt::KeepAspectRatio);
		return QRect(QPoint((width() - shown.width()) / 2, (height() - shown.height()) / 2), shown);
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
// OrbitPreviewLabel
// ============================================================================
// ScaledImageLabel plus click-drag-to-orbit and wheel-to-zoom mouse input -
// used by Live Preview's sub-tab (MainWindow::addLivePreviewTab()) to drive
// the GPU progressive-refinement preview's camera interactively. Deliberately
// dumb: this widget only reports raw drag deltas (in pixels) and wheel
// deltas - it owns no camera/orbit state itself (no lookAt point, no
// spherical coordinates). MainWindow (onLivePreviewOrbitDragged()/
// onLivePreviewZoomRequested()) is what turns those deltas into an actual
// new camera position and forwards it to RealtimePreviewSession::
// setCamera() - keeping this widget reusable/testable independent of the
// live-preview feature's own camera math, the same "widget stays dumb,
// MainWindow owns the logic" split ScaledImageLabel/SplitPreviewTabs above
// already follow.
// ============================================================================
class OrbitPreviewLabel : public ScaledImageLabel {
	Q_OBJECT
public:
	explicit OrbitPreviewLabel(QWidget *parent = nullptr) : ScaledImageLabel(parent) {
		// Otherwise a drag that leaves the widget's bounds (easy to do with a
		// fast mouse move) stops delivering move events entirely until the
		// cursor re-enters - mouseGrabber-style tracking (grabMouse() below)
		// needs a defined release point regardless of where the button
		// physically comes up, which setMouseTracking() alone doesn't give.
		setCursor(Qt::OpenHandCursor);
		// StrongFocus (not the default NoFocus, and not ClickFocus) so arrow-
		// key/+/- navigation works the instant a preview starts - MainWindow
		// calls setFocus() right after creating this widget (addLivePreviewTab())
		// - without requiring an extra click first, which would itself begin
		// an orbit drag rather than just moving focus here.
		setFocusPolicy(Qt::StrongFocus);
	}

	// Defensively ends an in-progress drag without waiting for a
	// mouseReleaseEvent that may never arrive - called by MainWindow's
	// Live-Preview-tab-switch handler, since switching tabs while the
	// mouse button is still held (e.g. via a keyboard tab-switch shortcut
	// mid-drag) is not a scenario Qt is documented to guarantee cleans up
	// an active grabMouse() on its own. Safe to call even when not
	// currently dragging - releaseMouse() on a widget that isn't the
	// current mouse grabber is a harmless no-op.
	void cancelDrag() {
		const bool wasObjectDrag = m_objectGrabbed;
		m_objectGrabbed = false;
		setCursor(m_objectMode ? Qt::PointingHandCursor : Qt::OpenHandCursor);
		releaseMouse();
		// A drag ended without a mouse release (the tab was switched mid-drag): whoever follows an object drag must hear that it is over, or it keeps waiting.
		if (wasObjectDrag) emit objectReleased();
	}

	// Object mode: a press on the picture first asks MainWindow (objectPressed) whether an object is under the cursor. If MainWindow answers by calling
	// setObjectGrabbed(true) while handling the signal, the drag moves that object (objectDragged) instead of orbiting; otherwise it orbits as usual.
	void setObjectMode(bool on) {
		m_objectMode = on;
		if (!on) clearSelection();
		setCursor(on ? Qt::PointingHandCursor : Qt::OpenHandCursor);
	}
	bool objectMode() const { return m_objectMode; }
	void setObjectGrabbed(bool grabbed) { m_objectGrabbed = grabbed; }

	// The box drawn round the selected object: line segments as pairs of picture positions (s left to right, t bottom to top, both in [0, 1]). Empty clears it.
	void setSelectionSegments(const QVector<QPointF> &segmentEnds) {
		m_selection = segmentEnds;
		update();
	}
	void clearSelection() { setSelectionSegments({}); }

signals:
	// Raw pixel deltas since the last mouse-move event during an active
	// drag - not accumulated, not scaled by any sensitivity here (that's
	// MainWindow's own concern, since it depends on the orbit radius/FOV
	// this widget knows nothing about).
	void orbitDragged(int dxPixels, int dyPixels);

	// Positive = wheel scrolled "up/away" (this widget's own convention;
	// MainWindow decides what that means for zoom direction), one signal
	// per wheel event's angleDelta().y(), not normalized to a fixed step.
	void zoomRequested(int angleDeltaY);

	// Keyboard equivalents of the two signals above - kept SEPARATE from
	// them (rather than synthesizing fake pixel/wheel deltas and reusing
	// orbitDragged()/zoomRequested()) so MainWindow can apply an
	// independent keyboard sensitivity multiplier instead of the mouse's -
	// see onLivePreviewKeyOrbit()/onLivePreviewKeyZoom()'s own comments.
	// Each step is already normalized to -1/0/+1, unlike the mouse
	// signals' raw unnormalized deltas, since a key press has no
	// "distance" of its own the way a mouse move does.
	void keyOrbitRequested(int azimuthSteps, int elevationSteps);
	void keyZoomRequested(int radiusSteps);

	// WASD/Up/Down free-fly translation - forwardSteps/rightSteps/upSteps
	// are each already normalized to -1/0/+1, same shape as
	// keyOrbitRequested() above. MainWindow (onLivePreviewTranslate())
	// turns these into an actual world-space move, since this widget has
	// no notion of the camera's current facing direction either.
	void translateRequested(int forwardSteps, int rightSteps, int upSteps);

	// Object mode, picture positions as in setSelectionSegments(). `vertical` is true while Shift is held.
	void objectPressed(double s, double t);
	void objectDragged(double s, double t, bool vertical);
	void objectReleased();

protected:
	void paintEvent(QPaintEvent *event) override {
		ScaledImageLabel::paintEvent(event);
		const QRect image = displayedImageRect();
		if (m_selection.size() < 2 || image.isEmpty()) return;
		QPainter painter(this);
		painter.setRenderHint(QPainter::Antialiasing);
		auto toWidget = [&image](const QPointF &p) { return QPointF(image.left() + p.x() * image.width(), image.top() + (1.0 - p.y()) * image.height()); };
		painter.setPen(QPen(QColor(0, 0, 0, 160), 4));   // a dark line under a bright one: readable on any picture
		for (int i = 0; i + 1 < m_selection.size(); i += 2) painter.drawLine(toWidget(m_selection[i]), toWidget(m_selection[i + 1]));
		painter.setPen(QPen(QColor(255, 220, 60), 2));
		for (int i = 0; i + 1 < m_selection.size(); i += 2) painter.drawLine(toWidget(m_selection[i]), toWidget(m_selection[i + 1]));
	}

	void mousePressEvent(QMouseEvent *event) override {
		if (event->button() == Qt::LeftButton && m_objectMode) {
			const QRect image = displayedImageRect();
			if (image.contains(event->pos())) {
				m_objectGrabbed = false;
				emit objectPressed((event->pos().x() - image.left() + 0.5) / image.width(), 1.0 - (event->pos().y() - image.top() + 0.5) / image.height());
				if (m_objectGrabbed) {
					setCursor(Qt::SizeAllCursor);
					grabMouse();
					return;
				}
			}
		}
		if (event->button() == Qt::LeftButton) {
			m_lastPos = event->pos();
			setCursor(Qt::ClosedHandCursor);
			// grabMouse() (not just setMouseTracking()) is what keeps
			// delivering move/release events to THIS widget even once the
			// cursor leaves its bounds mid-drag - a fast orbit drag
			// routinely does. No separate "am I dragging" flag needed:
			// mouseTracking is never enabled on this widget, so
			// mouseMoveEvent is only ever delivered at all while a grab is
			// held - mouseGrabber() == this already IS that flag.
			grabMouse();
		}
		ScaledImageLabel::mousePressEvent(event);
	}

	void mouseMoveEvent(QMouseEvent *event) override {
		if (mouseGrabber() == this && m_objectGrabbed) {
			const QRect image = displayedImageRect();
			if (!image.isEmpty())
				emit objectDragged((event->pos().x() - image.left() + 0.5) / image.width(), 1.0 - (event->pos().y() - image.top() + 0.5) / image.height(),
				                   (event->modifiers() & Qt::ShiftModifier) != 0);
			return;
		}
		if (mouseGrabber() == this) {
			const QPoint delta = event->pos() - m_lastPos;
			m_lastPos = event->pos();
			if (!delta.isNull()) emit orbitDragged(delta.x(), delta.y());
		}
		ScaledImageLabel::mouseMoveEvent(event);
	}

	void mouseReleaseEvent(QMouseEvent *event) override {
		if (event->button() == Qt::LeftButton && mouseGrabber() == this) {
			cancelDrag();
		}
		ScaledImageLabel::mouseReleaseEvent(event);
	}

	void wheelEvent(QWheelEvent *event) override {
		emit zoomRequested(event->angleDelta().y());
		event->accept();
	}

	// WASD translates (forward/back/strafe); Up/Down arrows translate
	// vertically - true free-fly movement, not orbit, so the camera can end
	// up looking anywhere rather than always facing the scene's subject
	// (see MainWindow::applyTranslateDelta()'s own comment). Left/Right
	// arrows still orbit (azimuth) - mouse-drag already covers full
	// rotation, and there's no natural free-fly meaning left over for them
	// once WASD+Up/Down cover all 6 translation directions, so they keep
	// their pre-existing behavior rather than being left doing nothing.
	// +/-/= still zoom (radius), unchanged. event->accept() on every
	// recognized key is required so it never reaches SplitPreviewTabs/
	// HorizontalTabBar (this widget lives inside that tab strip's stack)
	// and gets reinterpreted as a tab-switch key.
	void keyPressEvent(QKeyEvent *event) override {
		switch (event->key()) {
		case Qt::Key_W: emit translateRequested(+1, 0, 0); event->accept(); return;
		case Qt::Key_S: emit translateRequested(-1, 0, 0); event->accept(); return;
		case Qt::Key_A: emit translateRequested(0, -1, 0); event->accept(); return;
		case Qt::Key_D: emit translateRequested(0, +1, 0); event->accept(); return;
		case Qt::Key_Up:   emit translateRequested(0, 0, +1); event->accept(); return;
		case Qt::Key_Down: emit translateRequested(0, 0, -1); event->accept(); return;
		case Qt::Key_Left:  emit keyOrbitRequested(-1, 0); event->accept(); return;
		case Qt::Key_Right: emit keyOrbitRequested(+1, 0); event->accept(); return;
		case Qt::Key_Plus: case Qt::Key_Equal:
			emit keyZoomRequested(+1); event->accept(); return;
		case Qt::Key_Minus:
			emit keyZoomRequested(-1); event->accept(); return;
		default: break;
		}
		ScaledImageLabel::keyPressEvent(event);
	}

private:
	QPoint m_lastPos;
	bool m_objectMode = false;
	bool m_objectGrabbed = false;
	QVector<QPointF> m_selection;
};

#endif // ORBIT_PREVIEW_LABEL_H

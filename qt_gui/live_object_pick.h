#ifndef LIVE_OBJECT_PICK_H
#define LIVE_OBJECT_PICK_H
// live_object_pick.h -- the answer to "which object is at this point of the Live Preview picture".

#include <QMetaType>
#include <QString>

// What a click on the Live Preview picture found (RealtimePreviewSession::pickObjectAt()): the object under the cursor, where the click met its surface, and the
// camera that drew the picture, so a drag can be turned into a move with the same camera. All positions are in the scene's own (pbrt) coordinates.
struct LiveObjectPick {
	bool valid = false;
	int object = -1;
	QString label;                 // "sphere", "3 shapes"
	double hit[3] = {0, 0, 0};     // where the click met the surface
	double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};   // the object's box, where it is now
	double offset[3] = {0, 0, 0};  // how far it already is from where the file puts it
	double cameraBasis[12] = {0};  // origin, lower-left corner, horizontal, vertical of the camera that drew the picture
};

Q_DECLARE_METATYPE(LiveObjectPick)

#endif // LIVE_OBJECT_PICK_H

#ifndef RENDER_FORM_MEMORY_H
#define RENDER_FORM_MEMORY_H

// What the main render form remembers between runs (QSettings, settings_keys.h's location): the scene that was selected and the image size. Not samples,
// exposure or the camera - those are reset to each scene's recommendation on purpose (a value left over from another scene is wrong, not just suboptimal) -
// and not the renderer choice, since the GPU may not be there next time.

#include <QString>

namespace render_form_memory {

struct Form {
	QString sceneId;     // empty: nothing saved
	int width = 0;       // 0: nothing saved
	int height = 0;
};

Form load();
void save(const Form &form);

// True for a size the form can show (the spin boxes' range), so a damaged saved value is ignored rather than clamped to something surprising.
bool plausibleSize(int width, int height);

}  // namespace render_form_memory

#endif  // RENDER_FORM_MEMORY_H

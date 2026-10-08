#include "render_form_memory.h"

#include "settings_keys.h"

#include <QSettings>

namespace render_form_memory {
namespace {

const char *kSceneKey = "renderForm/sceneId";
const char *kWidthKey = "renderForm/width";
const char *kHeightKey = "renderForm/height";

}  // namespace

bool plausibleSize(int width, int height) { return width >= 100 && width <= 4096 && height >= 100 && height <= 4096; }

Form load() {
	QSettings s(settings_keys::kOrg, settings_keys::kApp);
	Form f;
	f.sceneId = s.value(kSceneKey).toString();
	const int w = s.value(kWidthKey, 0).toInt();
	const int h = s.value(kHeightKey, 0).toInt();
	if (plausibleSize(w, h)) {
		f.width = w;
		f.height = h;
	}
	return f;
}

void save(const Form &form) {
	QSettings s(settings_keys::kOrg, settings_keys::kApp);
	if (!form.sceneId.isEmpty()) s.setValue(kSceneKey, form.sceneId);
	if (plausibleSize(form.width, form.height)) {
		s.setValue(kWidthKey, form.width);
		s.setValue(kHeightKey, form.height);
	}
}

}  // namespace render_form_memory

// scene_3d_view_meshes.cpp - the 3D view's mesh-file previews: bounds and a vertex sample read on a worker thread, never while painting (see scene_3d_view.h).
#include "scene_3d_view_internal.h"

#include <QFileInfo>
#include <QMetaObject>
#include <QPointer>
#include <QThreadPool>

#include <algorithm>

using namespace scene_builder_ui;

// ---------------------------------------------------------------------------------------------------------------------------------
// Mesh files: read on a worker thread, so painting and picking never wait for a file
// ---------------------------------------------------------------------------------------------------------------------------------
const mesh_preview::MeshPreview *Scene3DView::meshPreview(const std::string &path) const {
	if (path.empty()) return nullptr;
	const qint64 now = m_clock.elapsed();
	auto it = m_meshes.find(path);
	if (it == m_meshes.end()) {
		const QFileInfo info(QString::fromStdString(path));
		startMeshLoad(path, info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1, info.size());
		return nullptr;
	}
	CachedMesh &c = it->second;
	if (!c.pending && now - c.checkedAt > kMeshRecheckMs) {
		c.checkedAt = now;
		const QFileInfo info(QString::fromStdString(path));
		const qint64 modified = info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1;
		if (modified != c.modified || info.size() != c.size) startMeshLoad(path, modified, info.size());
	}
	return (!c.pending && c.preview.ok) ? &c.preview : nullptr;
}

void Scene3DView::startMeshLoad(const std::string &path, qint64 modified, qint64 size) const {
	CachedMesh &c = m_meshes[path];
	c.pending = true;
	c.modified = modified;
	c.size = size;
	c.checkedAt = m_clock.elapsed();
	QPointer<Scene3DView> self(const_cast<Scene3DView *>(this));
	QThreadPool::globalInstance()->start([self, path, modified, size]() {
		mesh_preview::MeshPreview preview = mesh_preview::load(path);
		if (!self) return;
		QMetaObject::invokeMethod(
		    self.data(), [self, path, modified, size, p = std::move(preview)]() mutable {
			    if (self) self->meshLoaded(path, std::move(p), modified, size);
		    },
		    Qt::QueuedConnection);
	});
}

void Scene3DView::meshLoaded(const std::string &path, mesh_preview::MeshPreview preview, qint64 modified, qint64 size) {
	CachedMesh &c = m_meshes[path];
	if (c.modified != modified || c.size != size) return;  // the file changed again meanwhile: a newer read is on its way
	c.preview = std::move(preview);
	c.pending = false;
	c.checkedAt = m_clock.elapsed();
	++m_meshVersion;
	if (!m_userView) frameAll();
	update();
}

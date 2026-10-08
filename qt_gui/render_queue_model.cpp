#include "render_queue_model.h"

#include "app_log.h"

#include <QColor>
#include <QFont>

RenderQueueModel::RenderQueueModel(Describer renderer, Describer summary, QObject *parent)
    : QAbstractTableModel(parent), m_renderer(std::move(renderer)), m_summary(std::move(summary)) {}

namespace {
void logJob(const char *what, int id, const job_queue::Entry<RenderJob> *e) {
	AppLog::info(QStringLiteral("queue"), QStringLiteral("%1 job #%2%3").arg(QLatin1String(what)).arg(id).arg(e ? QStringLiteral(": %1").arg(e->job.displayTitle) : QString()));
}
}  // namespace

int RenderQueueModel::add(const RenderJob &job) {
	beginInsertRows(QModelIndex(), static_cast<int>(m_queue.size()), static_cast<int>(m_queue.size()));
	const int id = m_queue.add(job);
	endInsertRows();
	logJob("queued", id, m_queue.find(id));
	emit countsChanged();
	return id;
}

bool RenderQueueModel::takeNext(int &id, RenderJob &job) {
	if (!m_queue.takeNext(id, job)) return false;
	const int row = m_queue.rowOf(id);
	emit dataChanged(index(row, 0), index(row, ColumnCount - 1));
	logJob("started", id, m_queue.find(id));
	emit countsChanged();
	return true;
}

bool RenderQueueModel::finish(int id, job_queue::State result, double seconds, const QString &note) {
	const int before = static_cast<int>(m_queue.size());
	if (!m_queue.finish(id, result, seconds, note.toStdString())) return false;
	AppLog::info(QStringLiteral("queue"), QStringLiteral("job #%1 %2 after %3 s%4").arg(id).arg(QLatin1String(job_queue::name(result))).arg(seconds, 0, 'f', 1)
	                                          .arg(note.isEmpty() ? QString() : QStringLiteral(": %1").arg(note)));
	if (static_cast<int>(m_queue.size()) != before) {   // the history limit dropped the oldest finished rows
		beginResetModel();
		endResetModel();
	} else {
		const int row = m_queue.rowOf(id);
		emit dataChanged(index(row, 0), index(row, ColumnCount - 1));
	}
	emit countsChanged();
	return true;
}

bool RenderQueueModel::remove(int id) {
	const int row = m_queue.rowOf(id);
	if (row < 0 || m_queue.entries()[row].state == job_queue::State::Running) return false;
	beginRemoveRows(QModelIndex(), row, row);
	const bool ok = m_queue.remove(id);
	endRemoveRows();
	if (ok) { logJob("removed", id, nullptr); emit countsChanged(); }
	return ok;
}

bool RenderQueueModel::moveUp(int id) {
	if (!m_queue.moveUp(id)) return false;
	beginResetModel();   // rows swapped places; a table this small is simplest to just refresh
	endResetModel();
	logJob("moved up", id, nullptr);
	emit countsChanged();
	return true;
}

bool RenderQueueModel::moveDown(int id) {
	if (!m_queue.moveDown(id)) return false;
	beginResetModel();
	endResetModel();
	logJob("moved down", id, nullptr);
	emit countsChanged();
	return true;
}

int RenderQueueModel::retry(int id) {
	const job_queue::Entry<RenderJob> *e = m_queue.find(id);
	if (!e || (e->state != job_queue::State::Failed && e->state != job_queue::State::Cancelled)) return 0;
	return add(e->job);
}

int RenderQueueModel::clearFinished() {
	beginResetModel();
	const int n = static_cast<int>(m_queue.clearFinished());
	endResetModel();
	if (n) { AppLog::info(QStringLiteral("queue"), QStringLiteral("cleared %1 finished job(s)").arg(n)); emit countsChanged(); }
	return n;
}

int RenderQueueModel::clearWaiting() {
	beginResetModel();
	const int n = static_cast<int>(m_queue.clearWaiting());
	endResetModel();
	if (n) { AppLog::info(QStringLiteral("queue"), QStringLiteral("cleared %1 waiting job(s)").arg(n)); emit countsChanged(); }
	return n;
}

int RenderQueueModel::idAt(int row) const {
	return row >= 0 && row < static_cast<int>(m_queue.size()) ? m_queue.entries()[static_cast<std::size_t>(row)].id : 0;
}

job_queue::State RenderQueueModel::stateAt(int row) const {
	return row >= 0 && row < static_cast<int>(m_queue.size()) ? m_queue.entries()[static_cast<std::size_t>(row)].state : job_queue::State::Waiting;
}

int RenderQueueModel::rowCount(const QModelIndex &parent) const { return parent.isValid() ? 0 : static_cast<int>(m_queue.size()); }
int RenderQueueModel::columnCount(const QModelIndex &parent) const { return parent.isValid() ? 0 : ColumnCount; }

QString RenderQueueModel::stateText(job_queue::State state) const {
	switch (state) {
		case job_queue::State::Waiting: return tr("Waiting");
		case job_queue::State::Running: return tr("Running");
		case job_queue::State::Done: return tr("Done");
		case job_queue::State::Failed: return tr("Failed");
		case job_queue::State::Cancelled: return tr("Cancelled");
	}
	return QString();
}

QVariant RenderQueueModel::data(const QModelIndex &index, int role) const {
	if (!index.isValid() || index.row() < 0 || index.row() >= static_cast<int>(m_queue.size())) return QVariant();
	const job_queue::Entry<RenderJob> &e = m_queue.entries()[static_cast<std::size_t>(index.row())];
	if (role == JobStateRole) return static_cast<int>(e.state);
	if (role == JobIdRole) return e.id;
	switch (role) {
		case Qt::DisplayRole:
			switch (index.column()) {
				case StatusColumn: return stateText(e.state);
				case SceneColumn: return e.job.displayTitle;
				case SizeColumn: return e.job.videoMode ? tr("%1×%2 · video").arg(e.job.width).arg(e.job.height) : QStringLiteral("%1×%2").arg(e.job.width).arg(e.job.height);
				case SamplesColumn: return e.job.samples;
				case RendererColumn: return m_renderer ? m_renderer(e.job) : QString();
				case TimeColumn: return job_queue::isFinished(e.state) ? tr("%1 s").arg(e.seconds, 0, 'f', 1) : QString();
			}
			break;
		case Qt::ToolTipRole: {
			QString tip = m_summary ? m_summary(e.job) : e.job.displayTitle;
			if (!e.note.empty()) tip += QStringLiteral("\n") + QString::fromStdString(e.note);
			return tip;
		}
		case Qt::TextAlignmentRole:
			if (index.column() == SamplesColumn || index.column() == TimeColumn) return int(Qt::AlignRight | Qt::AlignVCenter);
			break;
		case Qt::FontRole:
			if (e.state == job_queue::State::Running) { QFont f; f.setBold(true); return f; }
			break;
		case Qt::ForegroundRole:
			if (index.column() == StatusColumn) {
				if (e.state == job_queue::State::Failed) return QColor(220, 90, 90);
				if (e.state == job_queue::State::Done) return QColor(90, 180, 110);
			}
			break;
	}
	return QVariant();
}

QVariant RenderQueueModel::headerData(int section, Qt::Orientation orientation, int role) const {
	if (orientation != Qt::Horizontal || role != Qt::DisplayRole) return QVariant();
	switch (section) {
		case StatusColumn: return tr("Status");
		case SceneColumn: return tr("Scene");
		case SizeColumn: return tr("Size");
		case SamplesColumn: return tr("Samples");
		case RendererColumn: return tr("Renderer");
		case TimeColumn: return tr("Time");
	}
	return QVariant();
}

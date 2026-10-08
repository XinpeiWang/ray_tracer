#ifndef RENDER_QUEUE_MODEL_H
#define RENDER_QUEUE_MODEL_H

// The Progress tab's render queue as a table: one row per job - waiting, running, and the finished ones with what became of them (done, failed, cancelled, how
// long it took) - instead of a list of text that only ever showed what was still waiting. The bookkeeping (order, one job at a time, history limit, moving,
// retrying) is src/shared/job_queue.h, unit-tested; this class puts it behind QAbstractTableModel so the view updates itself.

#include <QAbstractTableModel>
#include <QString>

#include <functional>

#include "../src/shared/job_queue.h"
#include "mainwindow_jobtypes.h"

class RenderQueueModel : public QAbstractTableModel {
	Q_OBJECT
public:
	enum Column { StatusColumn, SceneColumn, SizeColumn, SamplesColumn, RendererColumn, TimeColumn, ColumnCount };
	enum Role { JobStateRole = Qt::UserRole + 1, JobIdRole };

	using Describer = std::function<QString(const RenderJob &)>;

	// `renderer` gives the Renderer column's text (CPU, GPU...), `summary` the row's tooltip (the same one-line description the status area uses).
	RenderQueueModel(Describer renderer, Describer summary, QObject *parent = nullptr);

	int add(const RenderJob &job);
	bool takeNext(int &id, RenderJob &job);                                  // starts the first waiting job
	bool finish(int id, job_queue::State result, double seconds, const QString &note);
	bool remove(int id);
	bool moveUp(int id);
	bool moveDown(int id);
	int retry(int id);                                                       // queues a failed or cancelled job again; the new id, 0 if not retryable
	int clearFinished();
	int clearWaiting();

	int waitingCount() const { return static_cast<int>(m_queue.waitingCount()); }
	int finishedCount() const { return static_cast<int>(m_queue.finishedCount()); }
	bool hasWaiting() const { return m_queue.waitingCount() > 0; }
	bool isRunning() const { return m_queue.runningId() != 0; }
	int idAt(int row) const;                                                 // 0 for no such row
	job_queue::State stateAt(int row) const;
	const job_queue::JobQueue<RenderJob> &queue() const { return m_queue; }

	int rowCount(const QModelIndex &parent = QModelIndex()) const override;
	int columnCount(const QModelIndex &parent = QModelIndex()) const override;
	QVariant data(const QModelIndex &index, int role = Qt::DisplayRole) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role = Qt::DisplayRole) const override;

signals:
	void countsChanged();   // what is waiting or finished changed: the panel title and buttons follow

private:
	QString stateText(job_queue::State state) const;

	job_queue::JobQueue<RenderJob> m_queue;
	Describer m_renderer;
	Describer m_summary;
};

#endif  // RENDER_QUEUE_MODEL_H

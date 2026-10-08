#pragma once
// job_queue.h -- the render queue's bookkeeping, without Qt so it can be unit-tested: jobs with a state (waiting, running, done, failed, cancelled), the order
// they will run in, and the history of what happened to the finished ones. qt_gui/render_queue_model.h puts a table model over it.
//
// One job runs at a time. Rows are kept in insertion order, which is also the order they start in, so finished rows come first, then the running one, then the
// waiting ones; only waiting jobs can be moved. Finished rows stay (up to a limit) so the user can see what became of the last renders and run a failed one again.

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace job_queue {

enum class State { Waiting, Running, Done, Failed, Cancelled };

inline const char* name(State s) {
	switch (s) {
		case State::Waiting: return "waiting";
		case State::Running: return "running";
		case State::Done: return "done";
		case State::Failed: return "failed";
		case State::Cancelled: return "cancelled";
	}
	return "waiting";
}
inline bool isFinished(State s) { return s == State::Done || s == State::Failed || s == State::Cancelled; }

template <typename Job>
struct Entry {
	int id = 0;           // never reused during a run, so a row can be found again after others are removed
	Job job;
	State state = State::Waiting;
	double seconds = 0.0; // how long it ran (finished jobs)
	std::string note;     // why it failed or was cancelled, or a short result
};

template <typename Job>
class JobQueue {
public:
	explicit JobQueue(std::size_t historyLimit = 50) : m_historyLimit(historyLimit) {}

	const std::vector<Entry<Job>>& entries() const { return m_entries; }
	std::size_t size() const { return m_entries.size(); }
	bool empty() const { return m_entries.empty(); }

	// A new waiting job at the end; returns its id.
	int add(Job job) {
		Entry<Job> e;
		e.id = ++m_nextId;
		e.job = std::move(job);
		m_entries.push_back(std::move(e));
		return m_entries.back().id;
	}

	// Starts the first waiting job. False if one is already running or none waits.
	bool takeNext(int& id, Job& job) {
		if (runningId() != 0) return false;
		for (Entry<Job>& e : m_entries)
			if (e.state == State::Waiting) {
				e.state = State::Running;
				id = e.id;
				job = e.job;
				return true;
			}
		return false;
	}

	// Ends the running job with a result. False if `id` is not the running one.
	bool finish(int id, State result, double seconds, std::string note = std::string()) {
		Entry<Job>* e = findMutable(id);
		if (!e || e->state != State::Running || !isFinished(result)) return false;
		e->state = result;
		e->seconds = seconds;
		e->note = std::move(note);
		trimHistory();
		return true;
	}

	// Drops a waiting or finished job (a running one has to be stopped, not removed).
	bool remove(int id) {
		for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
			if (it->id == id) {
				if (it->state == State::Running) return false;
				m_entries.erase(it);
				return true;
			}
		return false;
	}

	// A waiting job swaps places with the waiting job before / after it.
	bool moveUp(int id) { return moveWaiting(id, -1); }
	bool moveDown(int id) { return moveWaiting(id, +1); }

	// A failed or cancelled job is queued again as a new waiting job (the old row stays as the record). Returns the new id, 0 if it was not retryable.
	int retry(int id) {
		const Entry<Job>* e = find(id);
		if (!e || (e->state != State::Failed && e->state != State::Cancelled)) return 0;
		return add(e->job);
	}

	std::size_t clearFinished() {
		const std::size_t before = m_entries.size();
		m_entries.erase(std::remove_if(m_entries.begin(), m_entries.end(), [](const Entry<Job>& e) { return isFinished(e.state); }), m_entries.end());
		return before - m_entries.size();
	}
	std::size_t clearWaiting() {
		const std::size_t before = m_entries.size();
		m_entries.erase(std::remove_if(m_entries.begin(), m_entries.end(), [](const Entry<Job>& e) { return e.state == State::Waiting; }), m_entries.end());
		return before - m_entries.size();
	}

	std::size_t waitingCount() const { return countOf(State::Waiting); }
	std::size_t finishedCount() const {
		return countOf(State::Done) + countOf(State::Failed) + countOf(State::Cancelled);
	}
	int runningId() const {
		for (const Entry<Job>& e : m_entries)
			if (e.state == State::Running) return e.id;
		return 0;
	}
	const Entry<Job>* find(int id) const {
		for (const Entry<Job>& e : m_entries)
			if (e.id == id) return &e;
		return nullptr;
	}
	int rowOf(int id) const {
		for (std::size_t i = 0; i < m_entries.size(); ++i)
			if (m_entries[i].id == id) return static_cast<int>(i);
		return -1;
	}

private:
	Entry<Job>* findMutable(int id) {
		for (Entry<Job>& e : m_entries)
			if (e.id == id) return &e;
		return nullptr;
	}
	std::size_t countOf(State s) const {
		std::size_t n = 0;
		for (const Entry<Job>& e : m_entries) n += e.state == s ? 1 : 0;
		return n;
	}
	bool moveWaiting(int id, int direction) {
		const int row = rowOf(id);
		if (row < 0 || m_entries[row].state != State::Waiting) return false;
		for (int other = row + direction; other >= 0 && other < static_cast<int>(m_entries.size()); other += direction)
			if (m_entries[other].state == State::Waiting) {
				std::swap(m_entries[row], m_entries[other]);
				return true;
			}
		return false;
	}
	// The oldest finished rows go first once there are more than the limit.
	void trimHistory() {
		while (finishedCount() > m_historyLimit)
			for (auto it = m_entries.begin(); it != m_entries.end(); ++it)
				if (isFinished(it->state)) { m_entries.erase(it); break; }
	}

	std::vector<Entry<Job>> m_entries;
	int m_nextId = 0;
	std::size_t m_historyLimit;
};

}  // namespace job_queue

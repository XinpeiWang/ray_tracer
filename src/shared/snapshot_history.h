#pragma once
// snapshot_history.h - the Scene Builder's undo and redo lists: whole-document snapshots (as text), newest last. Qt-free so it can be unit-tested.
//
// Use: record() the document BEFORE changing it. To undo, restore undoTop() and, if that worked, call undone() with the document as it was just
// before; redo is the same with redoTop() and redone(). Restoring first and only then moving the entry means a snapshot that cannot be read
// back is not lost. The undo list is kept to a number of steps and a total size, so a long session on a big scene cannot eat memory.

#include <cstddef>
#include <deque>
#include <string>
#include <utility>

class SnapshotHistory {
public:
	explicit SnapshotHistory(std::size_t maxSteps = 200, std::size_t maxBytes = std::size_t(32) << 20) : m_maxSteps(maxSteps), m_maxBytes(maxBytes) {}

	bool canUndo() const { return !m_undo.empty(); }
	bool canRedo() const { return !m_redo.empty(); }
	std::size_t undoSize() const { return m_undo.size(); }
	std::size_t redoSize() const { return m_redo.size(); }
	const std::string& undoTop() const { return m_undo.back(); }
	const std::string& redoTop() const { return m_redo.back(); }

	// A new change is about to be made: `before` becomes the newest undo step, and redo is forgotten.
	void record(std::string before) {
		push(m_undo, m_undoBytes, std::move(before));
		m_redo.clear();
		m_redoBytes = 0;
		while (m_undo.size() > 1 && (m_undo.size() > m_maxSteps || m_undoBytes > m_maxBytes)) {
			m_undoBytes -= m_undo.front().size();
			m_undo.pop_front();
		}
	}
	// undoTop() has been restored; `current` is the document as it was just before.
	void undone(std::string current) {
		pop(m_undo, m_undoBytes);
		push(m_redo, m_redoBytes, std::move(current));
	}
	void redone(std::string current) {
		pop(m_redo, m_redoBytes);
		push(m_undo, m_undoBytes, std::move(current));
	}
	void clear() {
		m_undo.clear();
		m_redo.clear();
		m_undoBytes = m_redoBytes = 0;
	}

private:
	static void push(std::deque<std::string>& d, std::size_t& bytes, std::string s) {
		bytes += s.size();
		d.push_back(std::move(s));
	}
	static void pop(std::deque<std::string>& d, std::size_t& bytes) {
		bytes -= d.back().size();
		d.pop_back();
	}

	std::deque<std::string> m_undo, m_redo;
	std::size_t m_undoBytes = 0, m_redoBytes = 0;
	std::size_t m_maxSteps, m_maxBytes;
};

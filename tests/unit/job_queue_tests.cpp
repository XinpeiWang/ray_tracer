/**
 * @file job_queue_tests.cpp
 * @brief The render queue's bookkeeping (src/shared/job_queue.h): order, one running job, results, history, moving, retrying
 */

#include <gtest/gtest.h>

#include "../../src/shared/job_queue.h"

#include <string>

using job_queue::JobQueue;
using job_queue::State;

namespace {
std::string order(const JobQueue<std::string>& q) {
	std::string s;
	for (const auto& e : q.entries()) s += e.job + std::string(1, job_queue::name(e.state)[0]) + " ";
	return s;
}
}  // namespace

TEST(JobQueueTest, JobsRunOneAtATimeInTheOrderTheyWereAdded) {
	JobQueue<std::string> q;
	const int a = q.add("A"), b = q.add("B");
	EXPECT_NE(a, b);
	EXPECT_EQ(q.waitingCount(), 2u);
	int id = 0;
	std::string job;
	ASSERT_TRUE(q.takeNext(id, job));
	EXPECT_EQ(job, "A");
	EXPECT_EQ(id, a);
	EXPECT_EQ(q.runningId(), a);
	EXPECT_FALSE(q.takeNext(id, job)) << "one job runs at a time";
	EXPECT_TRUE(q.finish(a, State::Done, 12.5, "ok"));
	EXPECT_EQ(q.runningId(), 0);
	ASSERT_TRUE(q.takeNext(id, job));
	EXPECT_EQ(job, "B");
	EXPECT_EQ(order(q), "Ad Br ");
	EXPECT_EQ(q.find(a)->seconds, 12.5);
	EXPECT_EQ(q.find(a)->note, "ok");
}

TEST(JobQueueTest, OnlyTheRunningJobCanFinishAndOnlyWithAFinishedState) {
	JobQueue<std::string> q;
	const int a = q.add("A"), b = q.add("B");
	EXPECT_FALSE(q.finish(a, State::Done, 1)) << "not running yet";
	int id;
	std::string job;
	q.takeNext(id, job);
	EXPECT_FALSE(q.finish(b, State::Done, 1)) << "another job";
	EXPECT_FALSE(q.finish(a, State::Waiting, 1)) << "not a result";
	EXPECT_TRUE(q.finish(a, State::Failed, 3, "boom"));
	EXPECT_FALSE(q.finish(a, State::Done, 1)) << "already finished";
	EXPECT_EQ(q.find(a)->state, State::Failed);
}

TEST(JobQueueTest, RemoveDropsWaitingAndFinishedJobsButNotTheRunningOne) {
	JobQueue<std::string> q;
	const int a = q.add("A"), b = q.add("B"), c = q.add("C");
	int id;
	std::string job;
	q.takeNext(id, job);
	EXPECT_FALSE(q.remove(a)) << "running";
	EXPECT_TRUE(q.remove(b));
	q.finish(a, State::Done, 1);
	EXPECT_TRUE(q.remove(a));
	EXPECT_EQ(order(q), "Cw ");
	EXPECT_FALSE(q.remove(b)) << "already gone";
	EXPECT_EQ(q.rowOf(c), 0);
}

TEST(JobQueueTest, WaitingJobsMoveAmongThemselvesAndNeverPastFinishedOnes) {
	JobQueue<std::string> q;
	const int a = q.add("A"), b = q.add("B"), c = q.add("C"), d = q.add("D");
	int id;
	std::string job;
	q.takeNext(id, job);
	q.finish(a, State::Done, 1);
	EXPECT_EQ(order(q), "Ad Bw Cw Dw ");
	EXPECT_FALSE(q.moveUp(b)) << "nothing waiting above B";
	EXPECT_TRUE(q.moveDown(b));
	EXPECT_EQ(order(q), "Ad Cw Bw Dw ");
	EXPECT_TRUE(q.moveUp(d));
	EXPECT_EQ(order(q), "Ad Cw Dw Bw ");
	EXPECT_FALSE(q.moveDown(b)) << "last";
	EXPECT_FALSE(q.moveUp(a)) << "finished jobs do not move";
	EXPECT_FALSE(q.moveUp(999));
	q.takeNext(id, job);
	EXPECT_EQ(job, "C");
	EXPECT_FALSE(q.moveUp(c)) << "a running job does not move";
	(void)c;
}

TEST(JobQueueTest, AFailedOrCancelledJobCanBeQueuedAgainAsANewJob) {
	JobQueue<std::string> q;
	const int a = q.add("A"), b = q.add("B"), c = q.add("C");
	int id;
	std::string job;
	q.takeNext(id, job);
	q.finish(a, State::Failed, 1, "x");
	q.takeNext(id, job);
	q.finish(b, State::Done, 1);
	q.takeNext(id, job);
	q.finish(c, State::Cancelled, 1);
	EXPECT_EQ(q.retry(b), 0) << "a finished job is not retried";
	const int again = q.retry(a);
	EXPECT_GT(again, c);
	EXPECT_EQ(q.find(again)->job, "A");
	EXPECT_EQ(q.find(again)->state, State::Waiting);
	EXPECT_EQ(q.find(a)->state, State::Failed) << "the old row stays as the record";
	EXPECT_NE(q.retry(c), 0);
	EXPECT_EQ(q.retry(999), 0);
}

TEST(JobQueueTest, HistoryIsCappedAndClearsWork) {
	JobQueue<std::string> q(3);
	for (int i = 0; i < 6; ++i) {
		const int id0 = q.add(std::string(1, static_cast<char>('A' + i)));
		int id;
		std::string job;
		q.takeNext(id, job);
		q.finish(id0, i % 2 ? State::Failed : State::Done, 1);
	}
	EXPECT_EQ(q.finishedCount(), 3u) << "only the newest three are kept";
	EXPECT_EQ(order(q), "Df Ed Ff ");
	q.add("G");
	q.add("H");
	EXPECT_EQ(q.clearWaiting(), 2u);
	EXPECT_EQ(q.finishedCount(), 3u);
	EXPECT_EQ(q.clearFinished(), 3u);
	EXPECT_TRUE(q.empty());
	JobQueue<std::string> fresh;
	EXPECT_EQ(fresh.clearFinished(), 0u);
}

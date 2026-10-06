// optix_module_parallel.h -- optixModuleCreate, with OptiX's own compile work spread over every core.
//
// optixModuleCreate compiles a PTX module on the calling thread. The recursive module has six closest-hit programs that each inline the
// whole material system, and a cold OptiX cache meant ~5 minutes of one core (the wavefront and SPPM modules take a fraction of that). The same call
// split into tasks (optixModuleCreateWithTasks / optixTaskExecute) lets OptiX compile independent entry functions at the same time. The result is the
// same module, the compiled code is cached the same way, and a warm cache makes this as quick as before: nothing is recompiled.
#pragma once

#include <optix.h>
#include <optix_stubs.h>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

inline OptixResult optixModuleCreateParallel(OptixDeviceContext context,
                                             const OptixModuleCompileOptions* moduleCompileOptions,
                                             const OptixPipelineCompileOptions* pipelineCompileOptions,
                                             const char* input, size_t inputSize,
                                             char* logString, size_t* logStringSize,
                                             OptixModule* module) {
	OptixTask firstTask = nullptr;
	OptixResult result = optixModuleCreateWithTasks(context, moduleCompileOptions, pipelineCompileOptions, input, inputSize,
	                                                logString, logStringSize, module, &firstTask);
	if (result != OPTIX_SUCCESS || firstTask == nullptr) return result;

	const unsigned int workers = std::clamp(std::thread::hardware_concurrency(), 1u, 32u);
	const unsigned int maxAdditional = workers * 2;

	std::mutex mutex;
	std::condition_variable wake;
	std::vector<OptixTask> queue{firstTask};
	size_t unfinished = 1;  // queued + executing
	OptixResult firstError = OPTIX_SUCCESS;

	auto work = [&]() {
		std::vector<OptixTask> additional(maxAdditional);
		for (;;) {
			OptixTask task;
			{
				std::unique_lock<std::mutex> lock(mutex);
				wake.wait(lock, [&] { return !queue.empty() || unfinished == 0; });
				if (queue.empty()) return;  // nothing queued and nothing running: all done
				task = queue.back();
				queue.pop_back();
			}
			unsigned int created = 0;
			const OptixResult r = optixTaskExecute(task, additional.data(), maxAdditional, &created);
			{
				std::lock_guard<std::mutex> lock(mutex);
				if (r != OPTIX_SUCCESS && firstError == OPTIX_SUCCESS) firstError = r;
				if (r == OPTIX_SUCCESS) {
					queue.insert(queue.end(), additional.begin(), additional.begin() + created);
					unfinished += created;
				}
				--unfinished;
			}
			wake.notify_all();
		}
	};

	std::vector<std::thread> threads;
	for (unsigned int i = 1; i < workers; ++i) threads.emplace_back(work);
	work();
	for (std::thread& t : threads) t.join();
	return firstError;
}

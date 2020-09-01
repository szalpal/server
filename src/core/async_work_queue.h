// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/core/status.h"
#include "src/core/sync_queue.h"

namespace nvidia { namespace inferenceserver {

class AsyncWorkQueue {
 public:
  // Should only be called once to set the number of worker threads.
  static void SetWorkerCount(size_t thread_count);

  // Get the number of worker threads.
  static size_t GetWorkerCount();

  // Add a task to the queue.
  static void AddTask(const std::function<void(void)> task);

 private:
  AsyncWorkQueue() = default;
  ~AsyncWorkQueue();
  static AsyncWorkQueue* GetSingleton();
  static void Initialize();

  size_t thread_count_;
  std::vector<std::unique_ptr<std::thread>> worker_threads_;
  SyncQueue<std::function<void(void)>> task_queue_;
  std::condition_variable queue_pending;
  std::mutex mutex_;
  bool exit_;
};

}}  // namespace nvidia::inferenceserver

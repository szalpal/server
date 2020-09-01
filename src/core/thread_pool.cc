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
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>

#include "src/core/thread_pool.h"

namespace nvidia { namespace inferenceserver {

AsyncWorkQueue::~AsyncWorkQueue()
{
  {
    std::lock_guard<std::mutex> lock(mutex_);
    exit_ = true;
    queue_pending.notify_one();
  }

  for (auto& worker_thread : worker_threads_) {
    if (worker_thread->joinable()) {
      worker_thread->join();
    }
  }
}

AsyncWorkQueue*
AsyncWorkQueue::GetSingleton()
{
  static AsyncWorkQueue singleton;
  return &singleton;
}

void
AsyncWorkQueue::SetWorkerCount(size_t thread_count)
{
  GetSingleton()->thread_count_ = thread_count;
  if (thread_count > 1) {
    GetSingleton()->exit_ = false;
    for (size_t id = 0; id < thread_count; id++) {
      GetSingleton()->worker_threads_.push_back(
          std::unique_ptr<std::thread>(new std::thread([] { Initialize(); })));
    }
  }
}

size_t
AsyncWorkQueue::GetWorkerCount()
{
  return GetSingleton()->thread_count_;
}

void
AsyncWorkQueue::AddTask(const std::function<void(void)> task)
{
  std::lock_guard<std::mutex> lock(GetSingleton()->mutex_);
  GetSingleton()->task_queue_.Put(task);
  GetSingleton()->queue_pending.notify_one();
}

void
AsyncWorkQueue::Initialize()
{
  std::function<void(void)> task;

  while (true) {
    {
      std::unique_lock<std::mutex> lock(GetSingleton()->mutex_);
      GetSingleton()->queue_pending.wait(lock, [&]() {
        return GetSingleton()->exit_ || !GetSingleton()->task_queue_.Empty();
      });

      if (GetSingleton()->exit_)
        return;

      task = GetSingleton()->task_queue_.Get();
    }
    task();
  }
}

}}  // namespace nvidia::inferenceserver

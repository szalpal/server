// The MIT License (MIT)
//
// Copyright (c) 2020 NVIDIA CORPORATION
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef DALI_BACKEND_DALI_BACKEND_H_
#define DALI_BACKEND_DALI_BACKEND_H_

#include <core/model_config.h>
#include <core/model_config.pb.h>
#include <custom/sdk/custom_instance.h>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <unordered_map>
#include "model_provider/model_provider.h"
#include "dali_executor/dali_executor.h"
#include "utils/dali.h"
#include "common.h"

namespace triton { namespace backend { namespace dali {

using nvidia::inferenceserver::DataType;
using nvidia::inferenceserver::GetDataTypeByteSize;

const std::string DALI_PIPELINE_KEY = "dali_pipeline";  // NOLINT

class DaliBackend : public nvidia::inferenceserver::custom::CustomInstance {
 public:
  DaliBackend(const std::string &instance_name, const nvidia::inferenceserver::ModelConfig &config,
              const int gpu_device, const std::string &dali_pipeline_path);


  ~DaliBackend() override = default;

  int
  Execute(const uint32_t payload_cnt, CustomPayload *payloads, CustomGetNextInputV2Fn_t input_fn,
          CustomGetOutputV2Fn_t output_fn) override;

 private:
  void
  TryExecute(const uint32_t payload_cnt, CustomPayload *payloads, CustomGetNextInputV2Fn_t input_fn,
             CustomGetOutputV2Fn_t output_fn);

  std::vector<DaliExecutor::InputDscr>
  ReadInputs(CustomPayload &payload, CustomGetNextInputV2Fn_t input_fn, uint32_t batch_size);

  ::dali::span<char> ReserveInputBuffer(const std::string &name, size_t n_elements, DataType type);

  std::vector<std::pair<void *, device_type_t>>
  FetchOutputBuffers(CustomPayload &payload, CustomGetOutputV2Fn_t output_fn,
                     const std::vector<::dali::TensorListShape<>> &shapes);

  const std::string kDaliErrorMsg_ = "DALI error occurred. Check server logs for more info.";
  const int kDaliErrorCode_;
  std::unique_ptr<ModelProvider> model_provider_;
  std::unordered_map<std::string, std::vector<char>> inputs_buffers_;
  DaliExecutor executor_;
};

}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_BACKEND_H_

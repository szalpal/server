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

#include "dali_backend.h"  // NOLINT
#include <custom/sdk/error_codes.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <numeric>
#include "utils/error_handling.h"
#include "utils/utils.h"
#include "utils/logging.h"
#include "common.h"

using std::cout;
using std::endl;

namespace dali_backend = ::triton::backend::dali;

namespace nvidia {
namespace inferenceserver {
namespace custom {


extern "C" DLL_PUBLIC uint32_t CustomVersion() {
  return 2;
}


int CustomInstance::Create(CustomInstance **instance, const std::string &name,
                           const ModelConfig &model_config,
                           int gpu_device, const CustomInitializeData *data) {
  std::stringstream dali_pipeline_path;
  const char sep = '/';
  const char model_version = '1';  // TODO(mszolucha) acquire
  dali_pipeline_path << data->server_parameters[MODEL_REPOSITORY_PATH] << sep << model_config.name()
                     << sep << model_version << sep
                     << model_config.parameters().at(
                             dali_backend::DALI_PIPELINE_KEY).string_value();
  auto *dali_backend = new dali_backend::DaliBackend(name, model_config, gpu_device,
                                                     dali_pipeline_path.str());
  *instance = dali_backend;  // TODO(mszolucha) memleak?
  return 0;
}

}  // namespace custom
}  // namespace inferenceserver
}  // namespace nvidia

namespace triton { namespace backend { namespace dali {

namespace {

dali_data_type_t to_dali(DataType t) {
  assert(t >= 0 && t <= 12);
  if (t == 0) return static_cast<dali_data_type_t>(-1);
  if (t == 1) return static_cast<dali_data_type_t>(11);
  return static_cast<dali_data_type_t>(t - 2);
}


device_type_t to_dali(CustomMemoryType t) {
  switch (t) {
    case CUSTOM_MEMORY_CPU:
      return device_type_t::CPU;
    case CUSTOM_MEMORY_GPU:
      return device_type_t::GPU;
    case CUSTOM_MEMORY_CPU_PINNED:
      return device_type_t::CPU;
    default:
      throw std::invalid_argument("Unknown memory type");
  }
}


struct TritonInOutException : public dali_backend::DaliBackendException {
  explicit TritonInOutException(const std::string &err) : DaliBackendException(err) {}
};

template<typename T = void>
struct TritonIODescriptor {
  T *data;
  CustomMemoryType memory_type;
};


void CopyInput(::dali::span<char> input_buffer, CustomGetNextInputV2Fn_t input_fn,
               void *input_ctx, const std::string &input_name) {
  CustomMemoryType src_memory_type = CUSTOM_MEMORY_CPU_PINNED;
  int64_t src_memory_type_id = 0;
  const void *content;
  uint64_t content_byte_size = -1;
  auto fetch_input = [&]() {
      auto success = input_fn(input_ctx, input_name.c_str(), &content, &content_byte_size,
                              &src_memory_type, &src_memory_type_id);
      if (!success) throw TritonInOutException("Acquiring input failed");
      return content != nullptr;
  };
  char *ibuffer = input_buffer.data();
  while (fetch_input()) {
    ENFORCE(src_memory_type != CUSTOM_MEMORY_GPU, "GPU input is currently unavailable");
    assert(ibuffer + content_byte_size <= input_buffer.end());
    std::memcpy(ibuffer, content, content_byte_size);
    ibuffer += content_byte_size;
  }
}


/**
 * Returns non-owning descriptor
 */
TritonIODescriptor<>
FetchOutput(CustomGetOutputV2Fn_t output_fn, void *output_ctx, const std::string &output_name,
            ::dali::TensorShape<> shape, DataType data_type) {
  TritonIODescriptor<> ret = {};

  int64_t src_memory_type_id = 0;

  auto success =
          output_fn(output_ctx, output_name.c_str(), shape.size(), shape.data(),
                    shape.num_elements() * GetDataTypeByteSize(data_type),
                    &ret.data, &ret.memory_type, &src_memory_type_id);
  if (!success) {
    throw TritonInOutException("Acquiring output failed");
  }
  return ret;
}

}  // namespace

::dali::span<char>
DaliBackend::ReserveInputBuffer(const std::string &name, size_t n_elements, DataType type) {
  auto &buffer = inputs_buffers_[name];
  auto input_size = n_elements * GetDataTypeByteSize(type);
  buffer.resize(input_size);
  return ::dali::make_span(buffer);
}


std::vector<DaliExecutor::InputDscr>
DaliBackend::ReadInputs(CustomPayload &payload, CustomGetNextInputV2Fn_t input_fn,
                        uint32_t batch_size) {
  std::vector<DaliExecutor::InputDscr> inputs;
  for (uint32_t input_idx = 0; input_idx < payload.input_cnt; input_idx++) {
    ENFORCE(payload.input_names[input_idx] == model_config_.input(input_idx).name(),
            "Names don't match");
    auto name = model_config_.input(input_idx).name();
    ENFORCE(static_cast<size_t>(model_config_.input(input_idx).dims().size()) ==
            payload.input_shape_dim_cnts[input_idx],
            make_string("Number of dimensions specified for input:",
                        model_config_.input(input_idx).name(),
                        ", doesn't match the received data: ",
                        payload.input_shape_dim_cnts[input_idx]));
    auto dims = payload.input_shape_dim_cnts[input_idx];

    auto sample_shape = ::dali::TensorShape<>(payload.input_shape_dims[input_idx],
                                              payload.input_shape_dims[input_idx] + dims);

    auto shape = ::dali::TensorListShape<>::make_uniform(batch_size, sample_shape);
    auto pb_type = model_config_.input(input_idx).data_type();
    auto input_buffer = ReserveInputBuffer(name, shape.num_elements(), pb_type);
    CopyInput(input_buffer, input_fn, payload.input_context, name);
    DaliExecutor::InputDscr dscr;
    dscr.name = name;
    dscr.buffer = input_buffer;
    dscr.shape = shape;
    dscr.type = to_dali(pb_type);
    inputs.push_back(dscr);
  }
  return inputs;
}


std::vector<std::pair<void *, device_type_t>>
DaliBackend::FetchOutputBuffers(CustomPayload &payload, CustomGetOutputV2Fn_t output_fn,
                                const std::vector<::dali::TensorListShape<>> &shapes) {
  std::vector<std::pair<void *, device_type_t>> outputs;
  for (uint32_t output_idx = 0; output_idx < payload.output_cnt; output_idx++) {
    auto batch_shape = ::dali::shape_cat(shapes[output_idx].num_samples(), shapes[output_idx][0]);
    auto output = FetchOutput(output_fn, payload.output_context,
                              payload.required_output_names[output_idx], batch_shape,
                              model_config_.output(output_idx).data_type());
    outputs.emplace_back(output.data, to_dali(output.memory_type));
  }
  return outputs;
}


DaliBackend::DaliBackend(const std::string &instance_name,
                         const nvidia::inferenceserver::ModelConfig &config, const int gpu_device,
                         const std::string &dali_pipeline_path)
        : CustomInstance(instance_name, config, gpu_device),
          kDaliErrorCode_(RegisterError(kDaliErrorMsg_)),
          model_provider_(std::make_unique<FileModelProvider>(dali_pipeline_path)),
          executor_(model_provider_->GetModel(), gpu_device_) {}


void DaliBackend::TryExecute(const uint32_t payload_cnt, CustomPayload *payloads,
                             CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn) {
  auto pipe = model_provider_->GetModel();
  for (uint32_t pidx = 0; pidx < payload_cnt; ++pidx) {
    auto &payload = payloads[pidx];
    uint32_t batch_size = payload.batch_size == 0 ? 1 : payload.batch_size;
    ENFORCE(batch_size > 0, make_string("Incorrect batch size: ", batch_size));
    auto inputs = ReadInputs(payload, input_fn, batch_size);
    auto output_shapes = executor_.Run(inputs);
    for (const auto &shape : output_shapes) {
      ENFORCE(::dali::is_uniform(shape),
              "Pipeline returned an output batch of a non-uniform shape.");
    }
    auto outputs = FetchOutputBuffers(payload, output_fn, output_shapes);
    executor_.PutOutputs(outputs);
  }
}


int DaliBackend::Execute(const uint32_t payload_cnt, CustomPayload *payloads,
                         CustomGetNextInputV2Fn_t input_fn, CustomGetOutputV2Fn_t output_fn) {
  int error_code = 0;  // success
  try {
    TryExecute(payload_cnt, payloads, input_fn, output_fn);
  } catch (DaliBackendException &e) {
    LOG_ERROR << e.what();
    error_code = kDaliErrorCode_;
  } catch (...) {
    LOG_ERROR << "Unknown error occurred";
    error_code = nvidia::inferenceserver::custom::ErrorCodes::Unknown;
  }
  return error_code;
}

}}}  // namespace triton::backend::dali

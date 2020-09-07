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

#ifndef DALI_BACKEND_UTILS_DALI_H_
#define DALI_BACKEND_UTILS_DALI_H_

#include <dali/c_api.h>
#include <dali/operators.h>
#include <dali/core/tensor_shape.h>
#include <dali/core/format.h>
#include <dali/core/tensor_shape_print.h>
#include <dali/core/span.h>


namespace triton { namespace backend { namespace dali {

static int64_t dali_type_size(dali_data_type_t type) {
  if (type == DALI_BOOL || type == DALI_UINT8 || type == DALI_INT8)
    return 1;
  if (type == DALI_UINT16 || type == DALI_INT16 || type == DALI_FLOAT16)
    return 2;
  if (type == DALI_UINT32 || type == DALI_INT32 || type == DALI_FLOAT)
    return 4;
  else
    return 8;
}

}}}  // namespace triton::backend::dali


#endif  // DALI_BACKEND_UTILS_DALI_H_

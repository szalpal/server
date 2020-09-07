# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

set(
    DALI_BACKEND_SRCS
        dali_executor.cc
        pipeline_group.cc
        dali_pipeline.cc
)

add_library(dali_executor STATIC ${DALI_BACKEND_SRCS})

target_include_directories(dali_executor PUBLIC BEFORE
        ${DaliBackend_SOURCE_DIR}/dali_backend
        ${CUDA_INCLUDE_DIRS}
        ${DALI_INCLUDE_DIR})

target_link_directories(dali_executor PUBLIC
        ${DALI_LIB_DIR})

target_link_libraries(dali_executor PUBLIC
        ${CUDA_LIBRARIES}
        ${DALI_LIBRARIES})

add_executable(unittests main.test.cc executor.test.cc)

target_link_libraries(unittests Catch2::Catch2 dali_executor)

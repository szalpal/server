#!/bin/bash -e

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

mkdir -p model_repository/resnet50_netdef/1
mkdir -p model_repository/dali_backend/1
mkdir -p model_repository/ensemble_dali_resnet/1

docker pull gitlab-master.nvidia.com/dl/dali/dali_triton_backend:latest

wget -O model_repository/resnet50_netdef/1/model.netdef \
  http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/predict_net.pb
wget -O model_repository/resnet50_netdef/1/init_model.netdef \
  http://download.caffe2.ai.s3.amazonaws.com/models/resnet50/init_net.pb

docker run --rm --entrypoint cat gitlab-master.nvidia.com/dl/dali/dali_triton_backend:latest /libdali_backend.so >model_repository/dali_backend/1/libdali_backend.so

echo "Model ready. Remember to include serialized pipelines."
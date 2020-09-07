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

import nvidia.dali.pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import argparse


class ResizePipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self, batch_size=2, num_threads=1, device_id=0):
        super().__init__(batch_size, num_threads, device_id)
        file_root = "/opt/dali_extra/db/single/jpeg/"
        self.read = ops.FileReader(device="cpu", file_root=file_root,
                                   file_list=file_root + "image_list.txt")
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.resize = ops.Resize(device="cpu", resize_x=224, resize_y=224, image_type=types.RGB)

    def define_graph(self):
        images, labels = self.read()
        images = self.decode(images)
        images = self.resize(images)
        return images, labels


class ResizePipelineExtSrc(nvidia.dali.pipeline.Pipeline):
    def __init__(self, batch_size=2, num_threads=1, device_id=0):
        super().__init__(batch_size, num_threads, device_id)
        self.read = ops.ExternalSource(device="cpu", name="INPUT0")
        self.decode = ops.ImageDecoder(device="cpu", output_type=types.RGB)
        self.resize = ops.Resize(device="cpu", resize_x=224, resize_y=224, image_type=types.RGB)

    def define_graph(self):
        images = self.read()
        images = self.decode(images)
        images = self.resize(images)
        return images


def main(pipeline_path):
    pipe = ResizePipelineExtSrc()
    pipe.serialize(filename=pipeline_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Serialize pipeline and save it to file")
    parser.add_argument('file_path', type=str, help='Path, where to save serialized pipeline')
    args = parser.parse_args()
    main(args.file_path)

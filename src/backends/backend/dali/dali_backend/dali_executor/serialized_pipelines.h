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

#ifndef DALI_BACKEND_DALI_EXECUTOR_SERIALIZED_PIPELINES_H_
#define DALI_BACKEND_DALI_EXECUTOR_SERIALIZED_PIPELINES_H_

namespace triton { namespace backend { namespace dali {
namespace test {
namespace pipelines {

unsigned char rn50_gpu_dali_chr[] = {
        0x08, 0x01, 0x10, 0x03, 0x2a, 0x51, 0x0a, 0x0f, 0x5f, 0x45, 0x78, 0x74,
        0x65, 0x72, 0x6e, 0x61, 0x6c, 0x53, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x1a,
        0x15, 0x0a, 0x0c, 0x44, 0x41, 0x4c, 0x49, 0x5f, 0x49, 0x4e, 0x50, 0x55,
        0x54, 0x5f, 0x30, 0x12, 0x03, 0x63, 0x70, 0x75, 0x18, 0x00, 0x22, 0x17,
        0x0a, 0x06, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x12, 0x06, 0x73, 0x74,
        0x72, 0x69, 0x6e, 0x67, 0x2a, 0x03, 0x63, 0x70, 0x75, 0x40, 0x00, 0x2a,
        0x0c, 0x44, 0x41, 0x4c, 0x49, 0x5f, 0x49, 0x4e, 0x50, 0x55, 0x54, 0x5f,
        0x30, 0x30, 0x00, 0x2a, 0x9f, 0x01, 0x0a, 0x0c, 0x49, 0x6d, 0x61, 0x67,
        0x65, 0x44, 0x65, 0x63, 0x6f, 0x64, 0x65, 0x72, 0x12, 0x15, 0x0a, 0x0c,
        0x44, 0x41, 0x4c, 0x49, 0x5f, 0x49, 0x4e, 0x50, 0x55, 0x54, 0x5f, 0x30,
        0x12, 0x03, 0x63, 0x70, 0x75, 0x18, 0x00, 0x1a, 0x19, 0x0a, 0x10, 0x5f,
        0x5f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x44, 0x65, 0x63, 0x6f, 0x64, 0x65,
        0x72, 0x5f, 0x31, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x22, 0x18,
        0x0a, 0x0b, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x74, 0x79, 0x70,
        0x65, 0x12, 0x05, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x20, 0x00, 0x40, 0x00,
        0x22, 0x19, 0x0a, 0x06, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x12, 0x06,
        0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a, 0x05, 0x6d, 0x69, 0x78, 0x65,
        0x64, 0x40, 0x00, 0x22, 0x14, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x73, 0x65,
        0x72, 0x76, 0x65, 0x12, 0x04, 0x62, 0x6f, 0x6f, 0x6c, 0x30, 0x00, 0x40,
        0x00, 0x2a, 0x10, 0x5f, 0x5f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x44, 0x65,
        0x63, 0x6f, 0x64, 0x65, 0x72, 0x5f, 0x31, 0x30, 0x01, 0x2a, 0xc2, 0x01,
        0x0a, 0x06, 0x52, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x12, 0x19, 0x0a, 0x10,
        0x5f, 0x5f, 0x49, 0x6d, 0x61, 0x67, 0x65, 0x44, 0x65, 0x63, 0x6f, 0x64,
        0x65, 0x72, 0x5f, 0x31, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x1a,
        0x13, 0x0a, 0x0a, 0x5f, 0x5f, 0x52, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x5f,
        0x32, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x22, 0x17, 0x0a, 0x0a,
        0x69, 0x6d, 0x61, 0x67, 0x65, 0x5f, 0x74, 0x79, 0x70, 0x65, 0x12, 0x05,
        0x69, 0x6e, 0x74, 0x36, 0x34, 0x20, 0x00, 0x40, 0x00, 0x22, 0x18, 0x0a,
        0x08, 0x72, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x5f, 0x79, 0x12, 0x05, 0x66,
        0x6c, 0x6f, 0x61, 0x74, 0x1d, 0x00, 0x00, 0x60, 0x43, 0x40, 0x00, 0x22,
        0x18, 0x0a, 0x08, 0x72, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x5f, 0x78, 0x12,
        0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d, 0x00, 0x00, 0x60, 0x43, 0x40,
        0x00, 0x22, 0x17, 0x0a, 0x06, 0x64, 0x65, 0x76, 0x69, 0x63, 0x65, 0x12,
        0x06, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a, 0x03, 0x67, 0x70, 0x75,
        0x40, 0x00, 0x22, 0x14, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x73, 0x65, 0x72,
        0x76, 0x65, 0x12, 0x04, 0x62, 0x6f, 0x6f, 0x6c, 0x30, 0x00, 0x40, 0x00,
        0x2a, 0x0a, 0x5f, 0x5f, 0x52, 0x65, 0x73, 0x69, 0x7a, 0x65, 0x5f, 0x32,
        0x30, 0x02, 0x2a, 0xf4, 0x03, 0x0a, 0x13, 0x43, 0x72, 0x6f, 0x70, 0x4d,
        0x69, 0x72, 0x72, 0x6f, 0x72, 0x4e, 0x6f, 0x72, 0x6d, 0x61, 0x6c, 0x69,
        0x7a, 0x65, 0x12, 0x13, 0x0a, 0x0a, 0x5f, 0x5f, 0x52, 0x65, 0x73, 0x69,
        0x7a, 0x65, 0x5f, 0x32, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x1a,
        0x20, 0x0a, 0x17, 0x5f, 0x5f, 0x43, 0x72, 0x6f, 0x70, 0x4d, 0x69, 0x72,
        0x72, 0x6f, 0x72, 0x4e, 0x6f, 0x72, 0x6d, 0x61, 0x6c, 0x69, 0x7a, 0x65,
        0x5f, 0x33, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x22, 0x60, 0x0a,
        0x04, 0x6d, 0x65, 0x61, 0x6e, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74,
        0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20,
        0x30, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d, 0x9a, 0x59, 0xf7,
        0x42, 0x40, 0x00, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x65,
        0x6e, 0x74, 0x20, 0x31, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d,
        0x5c, 0x8f, 0xe8, 0x42, 0x40, 0x00, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c,
        0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20, 0x32, 0x12, 0x05, 0x66, 0x6c, 0x6f,
        0x61, 0x74, 0x1d, 0x5c, 0x0f, 0xcf, 0x42, 0x40, 0x00, 0x40, 0x01, 0x22,
        0x14, 0x0a, 0x08, 0x70, 0x72, 0x65, 0x73, 0x65, 0x72, 0x76, 0x65, 0x12,
        0x04, 0x62, 0x6f, 0x6f, 0x6c, 0x30, 0x00, 0x40, 0x00, 0x22, 0x5f, 0x0a,
        0x03, 0x73, 0x74, 0x64, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x3a,
        0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20, 0x30,
        0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d, 0x7b, 0x94, 0x69, 0x42,
        0x40, 0x00, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e,
        0x74, 0x20, 0x31, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d, 0xe1,
        0x7a, 0x64, 0x42, 0x40, 0x00, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65,
        0x6d, 0x65, 0x6e, 0x74, 0x20, 0x32, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61,
        0x74, 0x1d, 0x00, 0x80, 0x65, 0x42, 0x40, 0x00, 0x40, 0x01, 0x22, 0x1e,
        0x0a, 0x0d, 0x6f, 0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x6c, 0x61, 0x79,
        0x6f, 0x75, 0x74, 0x12, 0x06, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a,
        0x03, 0x43, 0x48, 0x57, 0x40, 0x00, 0x22, 0x17, 0x0a, 0x06, 0x64, 0x65,
        0x76, 0x69, 0x63, 0x65, 0x12, 0x06, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67,
        0x2a, 0x03, 0x67, 0x70, 0x75, 0x40, 0x00, 0x22, 0x19, 0x0a, 0x0c, 0x6f,
        0x75, 0x74, 0x70, 0x75, 0x74, 0x5f, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12,
        0x05, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x20, 0x09, 0x40, 0x00, 0x22, 0x45,
        0x0a, 0x04, 0x63, 0x72, 0x6f, 0x70, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61,
        0x74, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74,
        0x20, 0x30, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74, 0x1d, 0x00, 0x00,
        0x60, 0x43, 0x40, 0x00, 0x3a, 0x19, 0x0a, 0x09, 0x65, 0x6c, 0x65, 0x6d,
        0x65, 0x6e, 0x74, 0x20, 0x31, 0x12, 0x05, 0x66, 0x6c, 0x6f, 0x61, 0x74,
        0x1d, 0x00, 0x00, 0x60, 0x43, 0x40, 0x00, 0x40, 0x01, 0x22, 0x17, 0x0a,
        0x0a, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x5f, 0x74, 0x79, 0x70, 0x65, 0x12,
        0x05, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x20, 0x00, 0x40, 0x00, 0x2a, 0x17,
        0x5f, 0x5f, 0x43, 0x72, 0x6f, 0x70, 0x4d, 0x69, 0x72, 0x72, 0x6f, 0x72,
        0x4e, 0x6f, 0x72, 0x6d, 0x61, 0x6c, 0x69, 0x7a, 0x65, 0x5f, 0x33, 0x30,
        0x03, 0x3a, 0x20, 0x0a, 0x17, 0x5f, 0x5f, 0x43, 0x72, 0x6f, 0x70, 0x4d,
        0x69, 0x72, 0x72, 0x6f, 0x72, 0x4e, 0x6f, 0x72, 0x6d, 0x61, 0x6c, 0x69,
        0x7a, 0x65, 0x5f, 0x33, 0x12, 0x03, 0x67, 0x70, 0x75, 0x18, 0x00, 0x40,
        0x00, 0x48, 0xef, 0xfb, 0xe5, 0xce, 0xf9, 0xff, 0xff, 0xff, 0xff, 0x01
};
unsigned int rn50_gpu_dali_len = 996;

const unsigned char scale_pipeline_str[] = {
        0x8, 0x1, 0x10, 0x2, 0x2a, 0x45, 0xa, 0xf, 0x5f, 0x45, 0x78, 0x74, 0x65, 0x72, 0x6e, 0x61,
        0x6c, 0x53, 0x6f, 0x75, 0x72, 0x63, 0x65, 0x1a, 0xf, 0xa, 0x6, 0x49, 0x4e, 0x50, 0x55,
        0x54, 0x30, 0x12, 0x3, 0x63, 0x70, 0x75, 0x18, 0x0, 0x22, 0x17, 0xa, 0x6, 0x64, 0x65,
        0x76, 0x69, 0x63, 0x65, 0x12, 0x6, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a, 0x3, 0x63,
        0x70, 0x75, 0x40, 0x0, 0x2a, 0x6, 0x49, 0x4e, 0x50, 0x55, 0x54, 0x30, 0x30, 0x0, 0x2a,
        0xf7, 0x1, 0xa, 0x13, 0x41, 0x72, 0x69, 0x74, 0x68, 0x6d, 0x65, 0x74, 0x69, 0x63, 0x47,
        0x65, 0x6e, 0x65, 0x72, 0x69, 0x63, 0x4f, 0x70, 0x12, 0xf, 0xa, 0x6, 0x49, 0x4e, 0x50,
        0x55, 0x54, 0x30, 0x12, 0x3, 0x63, 0x70, 0x75, 0x18, 0x0, 0x1a, 0x20, 0xa, 0x17, 0x5f,
        0x5f, 0x41, 0x72, 0x69, 0x74, 0x68, 0x6d, 0x65, 0x74, 0x69, 0x63, 0x47, 0x65, 0x6e, 0x65,
        0x72, 0x69, 0x63, 0x4f, 0x70, 0x5f, 0x31, 0x12, 0x3, 0x63, 0x70, 0x75, 0x18, 0x0, 0x22,
        0x2d, 0xa, 0xf, 0x65, 0x78, 0x70, 0x72, 0x65, 0x73, 0x73, 0x69, 0x6f, 0x6e, 0x5f, 0x64,
        0x65, 0x73, 0x63, 0x12, 0x6, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a, 0x10, 0x6d, 0x75,
        0x6c, 0x28, 0x26, 0x30, 0x20, 0x24, 0x30, 0x3a, 0x69, 0x6e, 0x74, 0x33, 0x32, 0x29, 0x40,
        0x0, 0x22, 0x34, 0xa, 0x11, 0x69, 0x6e, 0x74, 0x65, 0x67, 0x65, 0x72, 0x5f, 0x63, 0x6f,
        0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x73, 0x12, 0x5, 0x69, 0x6e, 0x74, 0x36, 0x34, 0x3a,
        0x16, 0xa, 0x9, 0x65, 0x6c, 0x65, 0x6d, 0x65, 0x6e, 0x74, 0x20, 0x30, 0x12, 0x5, 0x69,
        0x6e, 0x74, 0x36, 0x34, 0x20, 0x2, 0x40, 0x0, 0x40, 0x1, 0x22, 0x17, 0xa, 0x6, 0x64, 0x65,
        0x76, 0x69, 0x63, 0x65, 0x12, 0x6, 0x73, 0x74, 0x72, 0x69, 0x6e, 0x67, 0x2a, 0x3, 0x63,
        0x70, 0x75, 0x40, 0x0, 0x22, 0x14, 0xa, 0x8, 0x70, 0x72, 0x65, 0x73, 0x65, 0x72, 0x76,
        0x65, 0x12, 0x4, 0x62, 0x6f, 0x6f, 0x6c, 0x30, 0x0, 0x40, 0x0, 0x2a, 0x17, 0x5f, 0x5f,
        0x41, 0x72, 0x69, 0x74, 0x68, 0x6d, 0x65, 0x74, 0x69, 0x63, 0x47, 0x65, 0x6e, 0x65, 0x72,
        0x69, 0x63, 0x4f, 0x70, 0x5f, 0x31, 0x30, 0x1, 0x3a, 0x20, 0xa, 0x17, 0x5f, 0x5f, 0x41,
        0x72, 0x69, 0x74, 0x68, 0x6d, 0x65, 0x74, 0x69, 0x63, 0x47, 0x65, 0x6e, 0x65, 0x72, 0x69,
        0x63, 0x4f, 0x70, 0x5f, 0x31, 0x12, 0x3, 0x63, 0x70, 0x75, 0x18, 0x0, 0x40, 0x0, 0x48,
        0xd4, 0x85, 0xf9, 0xf9, 0xfd, 0xff, 0xff, 0xff, 0xff, 0x1};
unsigned int scale_pipeline_len = 372;

}  // namespace pipelines
}  // namespace test
}}}  // namespace triton::backend::dali

#endif  // DALI_BACKEND_DALI_EXECUTOR_SERIALIZED_PIPELINES_H_

/* Copyright 2021 Google LLC
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include "Test/TestUtils.h"
#include "Simd/SimdImageLoad.h"

namespace {
  uint32_t ReadBe32(const uint8_t *data) {
    return (uint32_t(data[0]) << 24) | (uint32_t(data[1]) << 16) |
           (uint32_t(data[2]) << 8) | uint32_t(data[3]);
  }

  bool PngFitsBudget(const uint8_t *data, size_t size) {
    const uint8_t signature[] = {137, 80, 78, 71, 13, 10, 26, 10};
    const size_t maxDimension = 4096;
    const size_t maxPixels = 1024 * 1024;
    if (size < sizeof(signature) || memcmp(data, signature, sizeof(signature)) != 0)
      return true;
    size_t offset = sizeof(signature);
    while (size - offset >= 12) {
      const size_t length = ReadBe32(data + offset);
      if (length > size - offset - 12)
        return true;
      if (length == 13 && memcmp(data + offset + 4, "IHDR", 4) == 0) {
        const size_t width = ReadBe32(data + offset + 8);
        const size_t height = ReadBe32(data + offset + 12);
        if (width > maxDimension || height > maxDimension ||
            (width != 0 && height > maxPixels / width))
          return false;
      }
      offset += length + 12;
    }
    return true;
  }
}

extern "C"
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size){
  if (!PngFitsBudget(data, size))
    return 0;
  const Test::View::Format formats[] = {Test::View::Gray8,
                                   Test::View::Bgr24,
                                   Test::View::Bgra32,
                                   Test::View::Rgb24,
                                   Test::View::Rgba32,
                                   Test::View::None};
  for (size_t formatIndex = 0; formatIndex < sizeof(formats) / sizeof(formats[0]); ++formatIndex) {
    Test::View decoded;
    decoded.Load(data, size, formats[formatIndex]);
    decoded.Clear();

    size_t stride = 0, width = 0, height = 0;
    SimdPixelFormatType format = static_cast<SimdPixelFormatType>(formats[formatIndex]);
    uint8_t *baseDecoded = Simd::Base::ImageLoadFromMemory(data, size, &stride, &width, &height, &format);
    SimdFree(baseDecoded);
  }
  return 0;
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
    namespace Base
    {
        void Float32ToBFloat16(const float * src, size_t size, uint16_t * dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Float32ToBFloat16(src[i + 0]);
                dst[i + 1] = Float32ToBFloat16(src[i + 1]);
                dst[i + 2] = Float32ToBFloat16(src[i + 2]);
                dst[i + 3] = Float32ToBFloat16(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = Float32ToBFloat16(src[i]);
        }

        //---------------------------------------------------------------------------------------------

        void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = BFloat16ToFloat32(src[i + 0]);
                dst[i + 1] = BFloat16ToFloat32(src[i + 1]);
                dst[i + 2] = BFloat16ToFloat32(src[i + 2]);
                dst[i + 3] = BFloat16ToFloat32(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = BFloat16ToFloat32(src[i]);
        }
    }
}

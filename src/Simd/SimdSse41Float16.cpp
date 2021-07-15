/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        void Float32ToFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t size8 = Simd::AlignLo(size, 8);
            size_t size4 = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < size8; i += 8)
            {
                __m128i d0 = Float32ToFloat16(_mm_loadu_ps(src + i + 0));
                __m128i d1 = Float32ToFloat16(_mm_loadu_ps(src + i + 4));
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(d0, d1));
            }
            for (; i < size4; i += 4)
            {
                __m128i d0 = Float32ToFloat16(_mm_loadu_ps(src + i + 0));
                _mm_storeu_si64(dst + i, _mm_packus_epi32(d0, K_ZERO));
            }
            for (; i < size; ++i)
                dst[i] = Base::Float32ToFloat16(src[i]);
        }

        void Float16ToFloat32(const uint16_t* src, size_t size, float* dst)
        {
            size_t alignedSize = Simd::AlignLo(size, 4);
            size_t i = 0;
            for (; i < alignedSize; i += 4)
            {
                dst[i + 0] = Base::Float16ToFloat32(src[i + 0]);
                dst[i + 1] = Base::Float16ToFloat32(src[i + 1]);
                dst[i + 2] = Base::Float16ToFloat32(src[i + 2]);
                dst[i + 3] = Base::Float16ToFloat32(src[i + 3]);
            }
            for (; i < size; ++i)
                dst[i] = Base::Float16ToFloat32(src[i]);
        }
    }
#endif// SIMD_SSE2_ENABLE
}

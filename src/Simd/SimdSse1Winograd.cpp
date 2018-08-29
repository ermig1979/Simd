/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdWinograd.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        template<size_t step> SIMD_INLINE void Load4(const float * src, __m128 * dst)
        {
            __m128 a0 = _mm_loadu_ps(src + 0 * step);
            __m128 a1 = _mm_loadu_ps(src + 1 * step);
            __m128 a2 = _mm_loadu_ps(src + 2 * step);
            __m128 a3 = _mm_loadu_ps(src + 3 * step);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpackhi_ps(a0, a2);
            __m128 b2 = _mm_unpacklo_ps(a1, a3);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            dst[0] = _mm_unpacklo_ps(b0, b2);
            dst[1] = _mm_unpackhi_ps(b0, b2);
            dst[2] = _mm_unpacklo_ps(b1, b3);
            dst[3] = _mm_unpackhi_ps(b1, b3);
        }

        SIMD_INLINE void Winograd2x3SetFilter4(const float * src, float * dst, size_t stride)
        {
            const __m128 r2 = _mm_set1_ps(1.0f / 2.0f);
            const __m128 r4 = _mm_set1_ps(1.0f / 4.0f);
            __m128 _src[9];
            Load4<9>(src + 0, _src + 0);
            Load4<9>(src + 4, _src + 4);
            _src[8] = _mm_setr_ps(src[8], src[17], src[26], src[35]);
            _mm_storeu_ps(dst + 0 * stride, _src[0]);
            __m128 _0a2 = _mm_add_ps(_src[0], _src[2]);
            _mm_storeu_ps(dst + 1 * stride, _mm_mul_ps(_mm_add_ps(_0a2, _src[1]), r2));
            _mm_storeu_ps(dst + 2 * stride, _mm_mul_ps(_mm_sub_ps(_0a2, _src[1]), r2));
            _mm_storeu_ps(dst + 3 * stride, _src[2]);
            __m128 _0a6a3 = _mm_add_ps(_mm_add_ps(_src[0], _src[6]), _src[3]);
            _mm_storeu_ps(dst + 4 * stride, _mm_mul_ps(_0a6a3, r2));
            __m128 _2a8a5 = _mm_add_ps(_mm_add_ps(_src[2], _src[8]), _src[5]);
            __m128 _1a7a4 = _mm_add_ps(_mm_add_ps(_src[1], _src[7]), _src[4]);
            _mm_storeu_ps(dst + 5 * stride, _mm_mul_ps(_mm_add_ps(_mm_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm_storeu_ps(dst + 6 * stride, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(_0a6a3, _2a8a5), _1a7a4), r4));
            _mm_storeu_ps(dst + 7 * stride, _mm_mul_ps(_2a8a5, r2));
            __m128 _0a6s3 = _mm_sub_ps(_mm_add_ps(_src[0], _src[6]), _src[3]);
            _mm_storeu_ps(dst + 8 * stride, _mm_mul_ps(_0a6s3, r2));
            __m128 _2a8s5 = _mm_sub_ps(_mm_add_ps(_src[2], _src[8]), _src[5]);
            __m128 _1a7s4 = _mm_sub_ps(_mm_add_ps(_src[1], _src[7]), _src[4]);
            _mm_storeu_ps(dst + 9 * stride, _mm_mul_ps(_mm_add_ps(_mm_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm_storeu_ps(dst + 10 * stride, _mm_mul_ps(_mm_sub_ps(_mm_add_ps(_0a6s3, _2a8s5), _1a7s4), r4));
            _mm_storeu_ps(dst + 11 * stride, _mm_mul_ps(_2a8s5, r2));
            _mm_storeu_ps(dst + 12 * stride, _src[6]);
            __m128 _6a8 = _mm_add_ps(_src[6], _src[8]);
            _mm_storeu_ps(dst + 13 * stride, _mm_mul_ps(_mm_add_ps(_6a8, _src[7]), r2));
            _mm_storeu_ps(dst + 14 * stride, _mm_mul_ps(_mm_sub_ps(_6a8, _src[7]), r2));
            _mm_storeu_ps(dst + 15 * stride, _src[8]);
        }

        void Winograd2x3SetFilter(const float * src, size_t srcChannels, size_t dstChannels, float * dst, size_t dstStride)
        {
            size_t size = dstChannels * srcChannels;
            size_t size4 = AlignLo(size, 4);
            size_t i = 0;
            for (; i < size4; i += 4, src += 36, dst += 4)
                Winograd2x3SetFilter4(src, dst, dstStride);
            for (; i < size; i += 1, src += 9, dst += 1)
                Base::Winograd2x3SetFilter1(src, dst, dstStride);
        }
    }
#endif// SIMD_SSE_ENABLE
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i K8_SHUFFLE_BGRA_TO_B = SIMD_MM_SETR_EPI8(0x0, -1, -1, -1, 0x4, -1, -1, -1, 0x8, -1, -1, -1, 0xC, -1, -1, -1);
        const __m128i K8_SHUFFLE_BGRA_TO_G = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, 0x5, -1, -1, -1, 0x9, -1, -1, -1, 0xD, -1, -1, -1);
        const __m128i K8_SHUFFLE_BGRA_TO_R = SIMD_MM_SETR_EPI8(0x2, -1, -1, -1, 0x6, -1, -1, -1, 0xA, -1, -1, -1, 0xE, -1, -1, -1);
        const __m128i K8_SHUFFLE_BGRA_TO_A = SIMD_MM_SETR_EPI8(0x3, -1, -1, -1, 0x7, -1, -1, -1, 0xB, -1, -1, -1, 0xF, -1, -1, -1);

        SIMD_INLINE void AlphaUnpremultiply(const uint8_t* src, uint8_t* dst, __m128 _255)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128i b = _mm_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_B);
            __m128i g = _mm_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_G);
            __m128i r = _mm_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_R);
            __m128i a = _mm_shuffle_epi8(_src, K8_SHUFFLE_BGRA_TO_A);
            __m128 k = _mm_cvtepi32_ps(a);
            k = _mm_blendv_ps(_mm_div_ps(_255, k), k, _mm_cmpeq_ps(k, _mm_setzero_ps()));
            b = _mm_cvtps_epi32(_mm_min_ps(_mm_floor_ps(_mm_mul_ps(_mm_cvtepi32_ps(b), k)), _255));
            g = _mm_cvtps_epi32(_mm_min_ps(_mm_floor_ps(_mm_mul_ps(_mm_cvtepi32_ps(g), k)), _255));
            r = _mm_cvtps_epi32(_mm_min_ps(_mm_floor_ps(_mm_mul_ps(_mm_cvtepi32_ps(r), k)), _255));
            __m128i _dst = _mm_or_si128(b, _mm_slli_si128(g, 1));
            _dst = _mm_or_si128(_dst, _mm_slli_si128(r, 2));
            _dst = _mm_or_si128(_dst, _mm_slli_si128(a, 3));
            _mm_storeu_si128((__m128i*)dst, _dst);
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            __m128 _255 = _mm_set1_ps(255.0f);
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < sizeA; col += A)
                    AlphaUnpremultiply(src + col, dst + col, _255);
                for (; col < size; col += 4)
                    Base::AlphaUnpremultiply(src + col, dst + col);
                src += srcStride;
                dst += dstStride;
            }
        }
    }
#endif// SIMD_SSE41_ENABLE
}

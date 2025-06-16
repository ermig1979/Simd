/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSynetQuantizeLinear.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        SIMD_INLINE __m128 DequantizeLinear(__m128i value, __m128i bias, __m128 norm)
        {
            return _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(value, bias)), norm);
        }

        SIMD_INLINE void DequantizeLinear1(const uint8_t* src, __m128i bias, __m128 norm, float * dst)
        {
            __m128i _src = _mm_set1_epi32(src[0]);
            __m128 _dst = DequantizeLinear(_src, bias, norm);
            _mm_store_ss(dst, _dst);
        }

        SIMD_INLINE void DequantizeLinearF(const uint8_t* src, __m128i bias, __m128 norm, float* dst)
        {
            __m128i _src = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)src)[0]));
            __m128 _dst = DequantizeLinear(_src, bias, norm);
            _mm_storeu_ps(dst, _dst);
        }

        SIMD_INLINE void DequantizeLinearA(const uint8_t* src, __m128i bias, __m128 norm, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0 * F, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * F)), bias, norm));
            _mm_storeu_ps(dst + 1 * F, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * F)), bias, norm));
            _mm_storeu_ps(dst + 2 * F, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * F)), bias, norm));
            _mm_storeu_ps(dst + 3 * F, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * F)), bias, norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            __m128i _bias = _mm_set1_epi32(bias);
            __m128 _norm = _mm_set1_ps(norm[0]);
            size_t i = 0, sizeF = AlignLo(size, F), sizeA = AlignLo(size, A);
            for (; i < sizeA; i += A)
                DequantizeLinearA(src + i, _bias, _norm, dst + i);
            for (; i < sizeF; i += F)
                DequantizeLinearF(src + i, _bias, _norm, dst + i);
            for (; i < size; i += 1)
                DequantizeLinear1(src + i, _bias, _norm, dst + i);
        }
    }
#endif
}

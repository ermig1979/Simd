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
        SIMD_INLINE void DequantizeLinear16(const uint8_t* src, __m128i bias, __m128 norm, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm_storeu_ps(dst + 0 * 4, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 4)), bias, norm));
            _mm_storeu_ps(dst + 1 * 4, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 4)), bias, norm));
            _mm_storeu_ps(dst + 2 * 4, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 2 * 4)), bias, norm));
            _mm_storeu_ps(dst + 3 * 4, DequantizeLinear(_mm_cvtepu8_epi32(_mm_srli_si128(_src, 3 * 4)), bias, norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            __m128i _bias = _mm_set1_epi32(bias);
            __m128 _norm = _mm_set1_ps(norm[0]);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                DequantizeLinear16(src + i, _bias, _norm, dst + i);
            for (; i < size4; i += 4)
                DequantizeLinear4(src + i, _bias, _norm, dst + i);
            for (; i < size; i += 1)
                DequantizeLinear1(src + i, _bias, _norm, dst + i);
        }

        //--------------------------------------------------------------------------------------------------

        void SynetQuantizeLinear(const float* src, size_t size, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m128 _scale = _mm_set1_ps(scale[0]);
            __m128i _zero = _mm_set1_epi32(zero);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                QuantizeLinear16(src + i, _scale, _zero, dst + i);
            for (; i < size4; i += 4)
                QuantizeLinear4(src + i, _scale, _zero, dst + i);
            for (; i < size; i += 1)
                QuantizeLinear1(src + i, _scale, _zero, dst + i);
        }
    }
#endif
}

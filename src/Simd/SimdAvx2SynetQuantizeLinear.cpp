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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        SIMD_INLINE void DequantizeLinear16(const uint8_t* src, __m256i bias, __m256 norm, float* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            _mm256_storeu_ps(dst + 0 * 8, DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(_src, 0 * 8)), bias, norm));
            _mm256_storeu_ps(dst + 1 * 8, DequantizeLinear(_mm256_cvtepu8_epi32(_mm_srli_si128(_src, 1 * 8)), bias, norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            __m256i _bias = _mm256_set1_epi32(bias);
            __m256 _norm = _mm256_set1_ps(norm[0]);
            size_t i = 0, size4 = AlignLo(size, 4), size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
                DequantizeLinear16(src + i, _bias, _norm, dst + i);
            for (; i < size4; i += 4)
                Sse41::DequantizeLinear4(src + i, _mm256_castsi256_si128(_bias), _mm256_castps256_ps128(_norm), dst + i);
            for (; i < size; i += 1)
                Sse41::DequantizeLinear1(src + i, _mm256_castsi256_si128(_bias), _mm256_castps256_ps128(_norm), dst + i);
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void QuantizeLinear32(const float* src, __m256 scale, __m256i zero, uint8_t* dst)
        {
            __m256i i0 = QuantizeLinear(_mm256_loadu_ps(src + 0 * 8), scale, zero);
            __m256i i1 = QuantizeLinear(_mm256_loadu_ps(src + 1 * 8), scale, zero);
            __m256i i2 = QuantizeLinear(_mm256_loadu_ps(src + 2 * 8), scale, zero);
            __m256i i3 = QuantizeLinear(_mm256_loadu_ps(src + 3 * 8), scale, zero);
            _mm256_storeu_si256((__m256i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
        }

        void SynetQuantizeLinear(const float* src, size_t size, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m256 _scale = _mm256_set1_ps(scale[0]);
            __m256i _zero = _mm256_set1_epi32(zero);
            size_t i = 0, size4 = AlignLo(size, 4), size32 = AlignLo(size, 32);
            for (; i < size32; i += 32)
                QuantizeLinear32(src + i, _scale, _zero, dst + i);
            for (; i < size4; i += 4)
                Sse41::QuantizeLinear4(src + i, _mm256_castps256_ps128(_scale), _mm256_castsi256_si128(_zero), dst + i);
            for (; i < size; i += 1)
                Sse41::QuantizeLinear1(src + i, _mm256_castps256_ps128(_scale), _mm256_castsi256_si128(_zero), dst + i);
        }
    }
#endif
}

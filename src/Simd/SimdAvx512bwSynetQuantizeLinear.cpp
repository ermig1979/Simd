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
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        SIMD_INLINE void DequantizeLinear16(const uint8_t* src, __m512i bias, __m512 norm, float* dst, __mmask16 tail = __mmask16(-1))
        {
            __m128i _src = _mm_maskz_loadu_epi8(tail, src);
            _mm512_mask_storeu_ps(dst, tail, DequantizeLinear(_mm512_cvtepu8_epi32(_src), bias, norm));
        }

        void SynetDequantizeLinear(const uint8_t* src, size_t size, int32_t bias, const float* norm, float* dst)
        {
            __m512i _bias = _mm512_set1_epi32(bias);
            __m512 _norm = _mm512_set1_ps(norm[0]);
            size_t i = 0, size16 = AlignLo(size, 16);
            __mmask16 tail16 = TailMask16(size - size16);
            for (; i < size16; i += 16)
                DequantizeLinear16(src + i, _bias, _norm, dst + i);
            if (i < size)
                DequantizeLinear16(src + i, _bias, _norm, dst + i, tail16);
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void QuantizeLinear16(const float* src, __m512 scale, __m512i zero, uint8_t* dst, __mmask16 tail = __mmask16(-1))
        {
            __m512i i0 = QuantizeLinear(_mm512_loadu_ps(src), scale, zero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void QuantizeLinear64(const float* src, __m512 scale, __m512i zero, uint8_t* dst)
        {
            __m512i i0 = QuantizeLinear(_mm512_loadu_ps(src + 0 * 16), scale, zero);
            __m512i i1 = QuantizeLinear(_mm512_loadu_ps(src + 1 * 16), scale, zero);
            __m512i i2 = QuantizeLinear(_mm512_loadu_ps(src + 2 * 16), scale, zero);
            __m512i i3 = QuantizeLinear(_mm512_loadu_ps(src + 3 * 16), scale, zero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
        }

        void SynetQuantizeLinear(const float* src, size_t size, const float* scale, int32_t zero, uint8_t* dst)
        {
            __m512 _scale = _mm512_set1_ps(scale[0]);
            __m512i _zero = _mm512_set1_epi32(zero);
            size_t i = 0, size16 = AlignLo(size, 16), size64 = AlignLo(size, 64);
            __mmask16 tail16 = TailMask16(size - size16);
            for (; i < size64; i += 64)
                QuantizeLinear64(src + i, _scale, _zero, dst + i);
            for (; i < size16; i += 16)
                QuantizeLinear16(src + i, _scale, _zero, dst + i);
            if (i < size)
                QuantizeLinear16(src + i, _scale, _zero, dst + i, tail16);
        }
    }
#endif
}

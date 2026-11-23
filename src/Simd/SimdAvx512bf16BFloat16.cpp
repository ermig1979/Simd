/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAvx512bf16.h"

namespace Simd
{
#ifdef SIMD_AVX512BF16_ENABLE
    namespace Avx512bf16
    {
        void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
        {
            size_t size32 = AlignLo(size, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            size_t i = 0;
            for (; i < size32; i += 32)
                Float32ToBFloat16<false, false>(src + i, dst + i, srcMask, dstMask);
            if (size32 < size)
            {
                srcMask[0] = TailMask16(size - size32 - F * 0);
                srcMask[1] = TailMask16(size - size32 - F * 1);
                dstMask[0] = TailMask32(size - size32);
                Float32ToBFloat16<false, true>(src + i, dst + i, srcMask, dstMask);
            }
        }

        //------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE __m512i Float16ToBFloat16(const uint16_t* src, __mmask32 tail)
        {
            __m512 lo, hi;
            if (mask)
            {
                __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
                lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 0));
                hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 1));
            }
            else
            {
#if defined(_MSC_VER)
                __m512i _src = _mm512_loadu_si512(src);
                lo = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 0));
                hi = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_src, 1));
#else
                lo = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 0));
                hi = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)src + 1));
#endif
            }
            return (__m512i)_mm512_cvtne2ps_pbh(hi, lo);
        }

        SIMD_INLINE void Float16ToBFloat16(const uint16_t* src, uint16_t* dst)
        {
            _mm512_storeu_si512(dst, Float16ToBFloat16<false>(src, __mmask32(-1)));
        }

        SIMD_INLINE void Float16ToBFloat16x2(const uint16_t* src, uint16_t* dst)
        {
            Float16ToBFloat16(src + 0 * HA, dst + 0 * HA);
            Float16ToBFloat16(src + 1 * HA, dst + 1 * HA);
        }

        void Float16ToBFloat16(const uint16_t* src, size_t size, uint16_t* dst)
        {
            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
            __mmask32 tailMask = TailMask32(size - alignedSize);

            size_t i = 0;
            for (; i < fullAlignedSize; i += QF)
                Float16ToBFloat16x2(src + i, dst + i);
            for (; i < alignedSize; i += DF)
                Float16ToBFloat16(src + i, dst + i);
            if (i < size)
                _mm512_mask_storeu_epi16(dst + i, tailMask, Float16ToBFloat16<true>(src + i, tailMask));
        }
    }
#endif
}

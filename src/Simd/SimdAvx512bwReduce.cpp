/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512i Reduce16(const __m512i & s0, const __m512i & s1)
        {
            return _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(s0, K8_01), _mm512_maddubs_epi16(s1, K8_01)), K16_0002), 2);
        }

        SIMD_INLINE __m512i Reduce8(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11)
        {
            return _mm512_permutexvar_epi64(K64_PERMUTE_FOR_PACK, _mm512_packus_epi16(Reduce16(s00, s10), Reduce16(s01, s11)));
        }

        template <size_t channelCount> __m512i Reduce8(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11);

        template<> SIMD_INLINE __m512i Reduce8<1>(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11)
        {
            return Reduce8(s00, s01, s10, s11);
        }

        const __m512i K8_RC2 = SIMD_MM512_SETR_EPI8(
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF,
            0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        template<> SIMD_INLINE __m512i Reduce8<2>(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11)
        {
            return Reduce8(_mm512_shuffle_epi8(s00, K8_RC2), _mm512_shuffle_epi8(s01, K8_RC2), _mm512_shuffle_epi8(s10, K8_RC2), _mm512_shuffle_epi8(s11, K8_RC2));
        }

        const __m512i K8_RC4 = SIMD_MM512_SETR_EPI8(
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF,
            0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        template<> SIMD_INLINE __m512i Reduce8<4>(const __m512i & s00, const __m512i & s01, const __m512i & s10, const __m512i & s11)
        {
            return Reduce8(_mm512_shuffle_epi8(s00, K8_RC4), _mm512_shuffle_epi8(s01, K8_RC4), _mm512_shuffle_epi8(s10, K8_RC4), _mm512_shuffle_epi8(s11, K8_RC4));
        }

        template <size_t channelCount, bool mask> SIMD_INLINE void ReduceColor2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst, __mmask64 * tails)
        {
            __m512i s00 = mask ? _mm512_maskz_loadu_epi8(tails[0], src0 + 0) : _mm512_loadu_si512(src0 + 0);
            __m512i s01 = mask ? _mm512_maskz_loadu_epi8(tails[1], src0 + A) : _mm512_loadu_si512(src0 + A);
            __m512i s10 = mask ? _mm512_maskz_loadu_epi8(tails[0], src1 + 0) : _mm512_loadu_si512(src1 + 0);
            __m512i s11 = mask ? _mm512_maskz_loadu_epi8(tails[1], src1 + A) : _mm512_loadu_si512(src1 + A);
            if (mask)
                _mm512_mask_storeu_epi8(dst, tails[2], Reduce8<channelCount>(s00, s01, s10, s11));
            else
                _mm512_storeu_si512(dst, Reduce8<channelCount>(s00, s01, s10, s11));
        }

        template <size_t channelCount> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t evenSize = evenWidth * channelCount;
            size_t alignedSize = AlignLo(evenSize, DA);
            __mmask64 tailMasks[3];
            for (size_t c = 0; c < 2; ++c)
                tailMasks[c] = TailMask64(evenSize - alignedSize - A * c);
            tailMasks[2] = TailMask64((evenSize - alignedSize) / 2);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += DA, dstOffset += A)
                    ReduceColor2x2<channelCount, false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                if (srcOffset < evenSize)
                    ReduceColor2x2<channelCount, true>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < channelCount; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        const __m512i K8_BGR_SM0 = SIMD_MM512_SETR_EPI8(
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m512i K8_BGR_SM1 = SIMD_MM512_SETR_EPI8(
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m512i K8_BGR_SM2 = SIMD_MM512_SETR_EPI8(
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF,
            0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1,
            -1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1,
            -1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);

        const __m512i K64_BGR_PE0 = SIMD_MM512_SETR_EPI64(0x2, 0x0, 0x4, 0x1, 0x6, 0x3, 0x8, 0x5);
        const __m512i K64_BGR_PE1 = SIMD_MM512_SETR_EPI64(0x7, 0x9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF);
        const __m512i K64_BGR_PE2 = SIMD_MM512_SETR_EPI64(0xA, 0x7, 0xC, 0x9, 0xE, 0xB, 0xF, 0xD);

        const __m512i K8_BGR_SE0 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);
        const __m512i K8_BGR_SE1 = SIMD_MM512_SETR_EPI8(
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);
        const __m512i K8_BGR_SE2 = SIMD_MM512_SETR_EPI8(
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0,
            0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1,
            0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        template <bool mask> SIMD_INLINE void ReduceBgr2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst, __mmask64 * tails)
        {
            __m512i s00 = mask ? _mm512_maskz_loadu_epi8(tails[0], src0 + 0 * A) : _mm512_loadu_si512(src0 + 0 * A);
            __m512i s01 = mask ? _mm512_maskz_loadu_epi8(tails[1], src0 + 1 * A) : _mm512_loadu_si512(src0 + 1 * A);
            __m512i s02 = mask ? _mm512_maskz_loadu_epi8(tails[2], src0 + 2 * A) : _mm512_loadu_si512(src0 + 2 * A);
            __m512i s10 = mask ? _mm512_maskz_loadu_epi8(tails[0], src1 + 0 * A) : _mm512_loadu_si512(src1 + 0 * A);
            __m512i s11 = mask ? _mm512_maskz_loadu_epi8(tails[1], src1 + 1 * A) : _mm512_loadu_si512(src1 + 1 * A);
            __m512i s12 = mask ? _mm512_maskz_loadu_epi8(tails[2], src1 + 2 * A) : _mm512_loadu_si512(src1 + 2 * A);
            __m512i e00 = _mm512_permutex2var_epi64(s00, K64_BGR_PE0, s01);
            __m512i e01 = _mm512_permutex2var_epi64(_mm512_permutex2var_epi64(s00, K64_BGR_PE1, s01), K64_BGR_PE0, s02);
            __m512i e10 = _mm512_permutex2var_epi64(s10, K64_BGR_PE0, s11);
            __m512i e11 = _mm512_permutex2var_epi64(_mm512_permutex2var_epi64(s10, K64_BGR_PE1, s11), K64_BGR_PE0, s12);
            __m512i m00 = _mm512_or_si512(_mm512_shuffle_epi8(s00, K8_BGR_SM0), _mm512_shuffle_epi8(e00, K8_BGR_SE0));
            __m512i m01 = _mm512_or_si512(_mm512_shuffle_epi8(s01, K8_BGR_SM1), _mm512_shuffle_epi8(e01, K8_BGR_SE1));
            __m512i m10 = _mm512_or_si512(_mm512_shuffle_epi8(s10, K8_BGR_SM0), _mm512_shuffle_epi8(e10, K8_BGR_SE0));
            __m512i m11 = _mm512_or_si512(_mm512_shuffle_epi8(s11, K8_BGR_SM1), _mm512_shuffle_epi8(e11, K8_BGR_SE1));
            if (mask)
                _mm512_mask_storeu_epi8(dst + 0 * A, tails[6], Reduce8(m00, m01, m10, m11));
            else
                _mm512_storeu_si512(dst + 0 * A, Reduce8(m00, m01, m10, m11));
            __m512i s03 = mask ? _mm512_maskz_loadu_epi8(tails[3], src0 + 3 * A) : _mm512_loadu_si512(src0 + 3 * A);
            __m512i s04 = mask ? _mm512_maskz_loadu_epi8(tails[4], src0 + 4 * A) : _mm512_loadu_si512(src0 + 4 * A);
            __m512i s13 = mask ? _mm512_maskz_loadu_epi8(tails[3], src1 + 3 * A) : _mm512_loadu_si512(src1 + 3 * A);
            __m512i s14 = mask ? _mm512_maskz_loadu_epi8(tails[4], src1 + 4 * A) : _mm512_loadu_si512(src1 + 4 * A);
            __m512i e02 = _mm512_permutex2var_epi64(s01, K64_BGR_PE2, s02);
            __m512i e03 = _mm512_permutex2var_epi64(s03, K64_BGR_PE0, s04);
            __m512i e12 = _mm512_permutex2var_epi64(s11, K64_BGR_PE2, s12);
            __m512i e13 = _mm512_permutex2var_epi64(s13, K64_BGR_PE0, s14);
            __m512i m02 = _mm512_or_si512(_mm512_shuffle_epi8(s02, K8_BGR_SM2), _mm512_shuffle_epi8(e02, K8_BGR_SE2));
            __m512i m03 = _mm512_or_si512(_mm512_shuffle_epi8(s03, K8_BGR_SM0), _mm512_shuffle_epi8(e03, K8_BGR_SE0));
            __m512i m12 = _mm512_or_si512(_mm512_shuffle_epi8(s12, K8_BGR_SM2), _mm512_shuffle_epi8(e12, K8_BGR_SE2));
            __m512i m13 = _mm512_or_si512(_mm512_shuffle_epi8(s13, K8_BGR_SM0), _mm512_shuffle_epi8(e13, K8_BGR_SE0));
            if (mask)
                _mm512_mask_storeu_epi8(dst + 1 * A, tails[7], Reduce8(m02, m03, m12, m13));
            else
                _mm512_storeu_si512(dst + 1 * A, Reduce8(m02, m03, m12, m13));
            __m512i s05 = mask ? _mm512_maskz_loadu_epi8(tails[5], src0 + 5 * A) : _mm512_loadu_si512(src0 + 5 * A);
            __m512i s15 = mask ? _mm512_maskz_loadu_epi8(tails[5], src1 + 5 * A) : _mm512_loadu_si512(src1 + 5 * A);
            __m512i e04 = _mm512_permutex2var_epi64(_mm512_permutex2var_epi64(s03, K64_BGR_PE1, s04), K64_BGR_PE0, s05);
            __m512i e05 = _mm512_permutex2var_epi64(s04, K64_BGR_PE2, s05);
            __m512i e14 = _mm512_permutex2var_epi64(_mm512_permutex2var_epi64(s13, K64_BGR_PE1, s14), K64_BGR_PE0, s15);
            __m512i e15 = _mm512_permutex2var_epi64(s14, K64_BGR_PE2, s15);
            __m512i m04 = _mm512_or_si512(_mm512_shuffle_epi8(s04, K8_BGR_SM1), _mm512_shuffle_epi8(e04, K8_BGR_SE1));
            __m512i m05 = _mm512_or_si512(_mm512_shuffle_epi8(s05, K8_BGR_SM2), _mm512_shuffle_epi8(e05, K8_BGR_SE2));
            __m512i m14 = _mm512_or_si512(_mm512_shuffle_epi8(s14, K8_BGR_SM1), _mm512_shuffle_epi8(e14, K8_BGR_SE1));
            __m512i m15 = _mm512_or_si512(_mm512_shuffle_epi8(s15, K8_BGR_SM2), _mm512_shuffle_epi8(e15, K8_BGR_SE2));
            if (mask)
                _mm512_mask_storeu_epi8(dst + 2 * A, tails[8], Reduce8(m04, m05, m14, m15));
            else
                _mm512_storeu_si512(dst + 2 * A, Reduce8(m04, m05, m14, m15));
        }

        void ReduceBgr2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenSize = evenWidth * 3;
            size_t alignedSize = alignedWidth * 3;
            size_t srcStep = DA * 3, dstStep = A * 3;
            __mmask64 tailMasks[9];
            for (size_t c = 0; c < 6; ++c)
                tailMasks[c] = TailMask64(evenSize - alignedSize - A * c);
            for (size_t c = 0; c < 3; ++c)
                tailMasks[6 + c] = TailMask64((evenSize - alignedSize)/2 - A * c);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += srcStep, dstOffset += dstStep)
                    ReduceBgr2x2<false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                if (srcOffset < evenSize)
                    ReduceBgr2x2<true>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, tailMasks);
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < 3; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);

            switch (channelCount)
            {
            case 1: ReduceColor2x2<1>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 2: ReduceColor2x2<2>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceBgr2x2(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

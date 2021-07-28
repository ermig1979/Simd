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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i Average16(const __m128i & s0, const __m128i & s1)
        {
            return _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(_mm_maddubs_epi16(s0, K8_01), _mm_maddubs_epi16(s1, K8_01)), K16_0002), 2);
        }

        SIMD_INLINE __m128i Average8(const __m128i & s00, const __m128i & s01, const __m128i & s10, const __m128i & s11)
        {
            return _mm_packus_epi16(Average16(s00, s10), Average16(s01, s11));
        }

        template <size_t channelCount> __m128i Average8(const __m128i & s00, const __m128i & s01, const __m128i & s10, const __m128i & s11);

        template<> SIMD_INLINE __m128i Average8<1>(const __m128i & s00, const __m128i & s01, const __m128i & s10, const __m128i & s11)
        {
            return Average8(s00, s01, s10, s11);
        }

        const __m128i K8_RC2 = SIMD_MM_SETR_EPI8(0x0, 0x2, 0x1, 0x3, 0x4, 0x6, 0x5, 0x7, 0x8, 0xA, 0x9, 0xB, 0xC, 0xE, 0xD, 0xF);

        template<> SIMD_INLINE __m128i Average8<2>(const __m128i & s00, const __m128i & s01, const __m128i & s10, const __m128i & s11)
        {
            return Average8(_mm_shuffle_epi8(s00, K8_RC2), _mm_shuffle_epi8(s01, K8_RC2), _mm_shuffle_epi8(s10, K8_RC2), _mm_shuffle_epi8(s11, K8_RC2));
        }

        const __m128i K8_RC4 = SIMD_MM_SETR_EPI8(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7, 0x8, 0xC, 0x9, 0xD, 0xA, 0xE, 0xB, 0xF);

        template<> SIMD_INLINE __m128i Average8<4>(const __m128i & s00, const __m128i & s01, const __m128i & s10, const __m128i & s11)
        {
            return Average8(_mm_shuffle_epi8(s00, K8_RC4), _mm_shuffle_epi8(s01, K8_RC4), _mm_shuffle_epi8(s10, K8_RC4), _mm_shuffle_epi8(s11, K8_RC4));
        }

        template <size_t channelCount, bool align> SIMD_INLINE void ReduceColor2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            __m128i s00 = Load<align>((__m128i*)src0 + 0);
            __m128i s01 = Load<align>((__m128i*)src0 + 1);
            __m128i s10 = Load<align>((__m128i*)src1 + 0);
            __m128i s11 = Load<align>((__m128i*)src1 + 1);
            Store<align>((__m128i*)dst, Average8<channelCount>(s00, s01, s10, s11));
        }

        template <size_t channelCount, bool align> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t evenSize = evenWidth * channelCount;
            size_t alignedSize = AlignLo(evenSize, DA);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += DA, dstOffset += A)
                    ReduceColor2x2<channelCount, align>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - DA;
                    dstOffset = srcOffset / 2;
                    ReduceColor2x2<channelCount, false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                }
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < channelCount; ++c)
                        dst[evenSize/2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }                
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        const __m128i K8_BGR0 = SIMD_MM_SETR_EPI8(0x0, 0x3, 0x1, 0x4, 0x2, 0x5, 0x6, 0x9, 0x7, 0xA, 0x8, 0xB, 0xC, 0xF, 0xD, -1);
        const __m128i K8_BGR1 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0);
        const __m128i K8_BGR2 = SIMD_MM_SETR_EPI8(0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_BGR3 = SIMD_MM_SETR_EPI8(-1, 0x1, 0x2, 0x5, 0x3, 0x6, 0x4, 0x7, 0x8, 0xB, 0x9, 0xC, 0xA, 0xD, 0xE, -1);
        const __m128i K8_BGR4 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1);
        const __m128i K8_BGR5 = SIMD_MM_SETR_EPI8(0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_BGR6 = SIMD_MM_SETR_EPI8(-1, 0x2, 0x0, 0x3, 0x4, 0x7, 0x5, 0x8, 0x6, 0x9, 0xA, 0xD, 0xB, 0xE, 0xC, 0xF);

        template <bool align> SIMD_INLINE void ReduceBgr2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            __m128i s00 = Load<align>((__m128i*)src0 + 0);
            __m128i s01 = Load<align>((__m128i*)src0 + 1);
            __m128i s02 = Load<align>((__m128i*)src0 + 2);
            __m128i s10 = Load<align>((__m128i*)src1 + 0);
            __m128i s11 = Load<align>((__m128i*)src1 + 1);
            __m128i s12 = Load<align>((__m128i*)src1 + 2);
            __m128i m00 = _mm_or_si128(_mm_shuffle_epi8(s00, K8_BGR0), _mm_shuffle_epi8(s01, K8_BGR1));
            __m128i m01 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s00, K8_BGR2), _mm_shuffle_epi8(s01, K8_BGR3)), _mm_shuffle_epi8(s02, K8_BGR4));
            __m128i m10 = _mm_or_si128(_mm_shuffle_epi8(s10, K8_BGR0), _mm_shuffle_epi8(s11, K8_BGR1));
            __m128i m11 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s10, K8_BGR2), _mm_shuffle_epi8(s11, K8_BGR3)), _mm_shuffle_epi8(s12, K8_BGR4));
            Store<align>((__m128i*)dst + 0, Average8(m00, m01, m10, m11));
            __m128i s03 = Load<align>((__m128i*)src0 + 3);
            __m128i s04 = Load<align>((__m128i*)src0 + 4); 
            __m128i s13 = Load<align>((__m128i*)src1 + 3);
            __m128i s14 = Load<align>((__m128i*)src1 + 4);
            __m128i m02 = _mm_or_si128(_mm_shuffle_epi8(s01, K8_BGR5), _mm_shuffle_epi8(s02, K8_BGR6));
            __m128i m03 = _mm_or_si128(_mm_shuffle_epi8(s03, K8_BGR0), _mm_shuffle_epi8(s04, K8_BGR1));
            __m128i m12 = _mm_or_si128(_mm_shuffle_epi8(s11, K8_BGR5), _mm_shuffle_epi8(s12, K8_BGR6));
            __m128i m13 = _mm_or_si128(_mm_shuffle_epi8(s13, K8_BGR0), _mm_shuffle_epi8(s14, K8_BGR1));
            Store<align>((__m128i*)dst + 1, Average8(m02, m03, m12, m13));
            __m128i s05 = Load<align>((__m128i*)src0 + 5);
            __m128i s15 = Load<align>((__m128i*)src1 + 5);
            __m128i m04 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s03, K8_BGR2), _mm_shuffle_epi8(s04, K8_BGR3)), _mm_shuffle_epi8(s05, K8_BGR4));
            __m128i m05 = _mm_or_si128(_mm_shuffle_epi8(s04, K8_BGR5), _mm_shuffle_epi8(s05, K8_BGR6));
            __m128i m14 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s13, K8_BGR2), _mm_shuffle_epi8(s14, K8_BGR3)), _mm_shuffle_epi8(s15, K8_BGR4));
            __m128i m15 = _mm_or_si128(_mm_shuffle_epi8(s14, K8_BGR5), _mm_shuffle_epi8(s15, K8_BGR6));
            Store<align>((__m128i*)dst + 2, Average8(m04, m05, m14, m15));
        }

        template <bool align> void ReduceBgr2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenSize = evenWidth * 3;
            size_t alignedSize = alignedWidth*3;
            size_t srcStep = DA * 3, dstStep = A*3;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += srcStep, dstOffset += dstStep)
                    ReduceBgr2x2<align>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - srcStep;
                    dstOffset = srcOffset / 2;
                    ReduceBgr2x2<false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
                }
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < 3; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        template <bool align> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: ReduceColor2x2<1, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 2: ReduceColor2x2<2, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceBgr2x2<align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4, align>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }

        void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ReduceColor2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
            else
                ReduceColor2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
        }
    }
#endif
}

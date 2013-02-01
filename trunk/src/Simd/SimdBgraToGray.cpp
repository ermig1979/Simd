/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdInit.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdBgrToGray.h"
#include "Simd/SimdBgraToGray.h"

namespace Simd
{
    namespace Base
    {
        void BgraToGray(const uchar *bgra, size_t size, uchar *gray)
        {
            const uchar *end = gray + size;
            for(; gray < end; gray += 1, bgra += 4)
            {
                *gray = BgrToGray(bgra[0], bgra[1], bgra[2]);
            }
        }

        void BgraToGray(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *gray, size_t grayStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                BgraToGray(bgra, width, gray);
                bgra += bgraStride;
                gray += grayStride;
            }
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        const __m128i K8_FF_00 = SIMD_MM_SET2_EPI8(0xFF, 0x00);
        const __m128i K16_BLUE_RED = SIMD_MM_SET2_EPI16(Base::BLUE_TO_GRAY_WEIGHT, Base::RED_TO_GRAY_WEIGHT);        
        const __m128i K16_GREEN_0000 = SIMD_MM_SET2_EPI16(Base::GREEN_TO_GRAY_WEIGHT, 0x0000);
        const __m128i K32_ROUND_TERM = SIMD_MM_SET1_EPI32(Base::BGR_TO_GRAY_ROUND_TERM);

        SIMD_INLINE __m128i BgraToGray32(__m128i bgra)
        {
            const __m128i g0a0 = _mm_and_si128(_mm_srli_si128(bgra, 1), K8_FF_00);
            const __m128i b0r0 = _mm_and_si128(bgra, K8_FF_00);
            const __m128i weightedSum = _mm_add_epi32(_mm_madd_epi16(g0a0, K16_GREEN_0000), _mm_madd_epi16(b0r0, K16_BLUE_RED));
            return _mm_srli_epi32(_mm_add_epi32(weightedSum, K32_ROUND_TERM), Base::BGR_TO_GRAY_AVERAGING_SHIFT);
        }

        SIMD_INLINE __m128i BgraToGray(__m128i bgra[4])
        {
            const __m128i lo = _mm_packs_epi32(BgraToGray32(bgra[0]), BgraToGray32(bgra[1]));
            const __m128i hi = _mm_packs_epi32(BgraToGray32(bgra[2]), BgraToGray32(bgra[3]));
            return _mm_packus_epi16(lo, hi);
        }

        SIMD_INLINE void BlockLoadU(const uchar* p, __m128i block[4])
        {
            block[0] = _mm_loadu_si128((__m128i*)p + 0);
            block[1] = _mm_loadu_si128((__m128i*)p + 1);
            block[2] = _mm_loadu_si128((__m128i*)p + 2);
            block[3] = _mm_loadu_si128((__m128i*)p + 3);
        }

        void BgraToGrayA(const uchar *bgra, size_t size, uchar *gray)
        {
            assert(Aligned(bgra) && Aligned(size) && Aligned(gray));

            const uchar *end = gray + size;
            for(; gray < end; gray += A, bgra += QA)
            {
                _mm_store_si128((__m128i*)gray, BgraToGray((__m128i*)bgra));
            }
        }

        void BgraToGrayU(const uchar *bgra, size_t size, uchar *gray)
        {
            assert(size >= A);

            size_t mainSize = AlignLo(size, A);
            __m128i block[4];
            for(size_t col = 0; col < mainSize; col += A)
            {
                BlockLoadU(bgra + 4*col, block);
                _mm_storeu_si128((__m128i*)(gray + col), BgraToGray(block));
            }
            if(mainSize != size)
            {
                BlockLoadU(bgra + 4*(size - A), block);
                _mm_storeu_si128((__m128i*)(gray + size - A), BgraToGray(block));
            }
        }

        void BgraToGray(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *gray, size_t grayStride)
        {
            assert(width >= A);

            if(Aligned(bgra) && Aligned(width) && Aligned(gray) && Aligned(bgraStride) && Aligned(grayStride))
            {
                for(size_t row = 0; row < height; ++row)
                {
                    BgraToGrayA(bgra, width, gray);
                    bgra += bgraStride;
                    gray += grayStride;
                }
            }
            else
            {
                for(size_t row = 0; row < height; ++row)
                {
                    BgraToGrayU(bgra, width, gray);
                    bgra += bgraStride;
                    gray += grayStride;
                }
            }        
        }
    }
#endif// SIMD_SSE2_ENABLE

    FlatBgraToGrayPtr FlatBgraToGrayA = SIMD_SSE2_INIT_FUNCTION_PTR(FlatBgraToGrayPtr, Sse2::BgraToGrayA, Base::BgraToGray);

    void BgraToGray(const uchar *bgra, size_t width, size_t height, size_t bgraStride, uchar *gray, size_t grayStride)
    {
#ifdef SIMD_SSE2_ENABLE
        if(width >= Sse2::A)
            Sse2::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
        else
#endif//SIMD_SSE2_ENABLE       
            Base::BgraToGray(bgra, width, height, bgraStride, gray, grayStride);
    }
}
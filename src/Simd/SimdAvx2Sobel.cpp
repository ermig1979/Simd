/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE void LoadNoseDx(const uint8_t * p, __m256i a[2])
        {
            a[0] = LoadBeforeFirst<false, 1>(p);
            a[1] = _mm256_loadu_si256((__m256i*)(p + 1));
        }

        SIMD_INLINE void LoadBodyDx(const uint8_t * p, __m256i a[2])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
            a[1] = _mm256_loadu_si256((__m256i*)(p + 1));
        }

        SIMD_INLINE void LoadTailDx(const uint8_t * p, __m256i a[2])
        {
            a[0] = _mm256_loadu_si256((__m256i*)(p - 1));
            a[1] = LoadAfterLast<false, 1>(p);
        }

        template<bool align> SIMD_INLINE void SobelDx(__m256i a[3][2], int16_t * dst)
        {
            __m256i lo = BinomialSum16(
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[0][1], K_ZERO), _mm256_unpacklo_epi8(a[0][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[1][1], K_ZERO), _mm256_unpacklo_epi8(a[1][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[2][1], K_ZERO), _mm256_unpacklo_epi8(a[2][0], K_ZERO)));
            __m256i hi = BinomialSum16(
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[0][1], K_ZERO), _mm256_unpackhi_epi8(a[0][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[1][1], K_ZERO), _mm256_unpackhi_epi8(a[1][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[2][1], K_ZERO), _mm256_unpackhi_epi8(a[2][0], K_ZERO)));
            Store<align>((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20)); 
            Store<align>((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        template <bool align> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i a[3][2];

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                LoadNoseDx(src0 + 0, a[0]);
                LoadNoseDx(src1 + 0, a[1]);
                LoadNoseDx(src2 + 0, a[2]);
                SobelDx<align>(a, dst + 0);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBodyDx(src0 + col, a[0]);
                    LoadBodyDx(src1 + col, a[1]);
                    LoadBodyDx(src2 + col, a[2]);
                    SobelDx<align>(a, dst + col);
                }
                LoadTailDx(src0 + width - A, a[0]);
                LoadTailDx(src1 + width - A, a[1]);
                LoadTailDx(src2 + width - A, a[2]);
                SobelDx<false>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(dst) && Aligned(dstStride))
                SobelDx<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                SobelDx<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE void SobelDy(__m256i a[2][3], int16_t * dst)
        {
            __m256i lo = BinomialSum16(
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[1][0], K_ZERO), _mm256_unpacklo_epi8(a[0][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[1][1], K_ZERO), _mm256_unpacklo_epi8(a[0][1], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpacklo_epi8(a[1][2], K_ZERO), _mm256_unpacklo_epi8(a[0][2], K_ZERO)));
            __m256i hi = BinomialSum16(
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[1][0], K_ZERO), _mm256_unpackhi_epi8(a[0][0], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[1][1], K_ZERO), _mm256_unpackhi_epi8(a[0][1], K_ZERO)),
                _mm256_sub_epi16(_mm256_unpackhi_epi8(a[1][2], K_ZERO), _mm256_unpackhi_epi8(a[0][2], K_ZERO)));
            Store<align>((__m256i*)dst + 0, _mm256_permute2x128_si256(lo, hi, 0x20)); 
            Store<align>((__m256i*)dst + 1, _mm256_permute2x128_si256(lo, hi, 0x31));
        }

        template <bool align> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width >= A);
            if(align)
                assert(Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m256i a[2][3];

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[1]);
                SobelDy<align>(a, dst + 0);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[1]);
                    SobelDy<align>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src2 + width - A, a[1]);
                SobelDy<false>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                SobelDy<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }
    }
#endif// SIMD_AVX2_ENABLE
}
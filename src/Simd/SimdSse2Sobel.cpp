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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template<bool align> SIMD_INLINE void SobelDx(__m128i a[3][3], int16_t * dst)
        {
            Store<align>((__m128i*)dst + 0, BinomialSum16(
                _mm_sub_epi16(_mm_unpacklo_epi8(a[0][2], K_ZERO), _mm_unpacklo_epi8(a[0][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpacklo_epi8(a[1][2], K_ZERO), _mm_unpacklo_epi8(a[1][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpacklo_epi8(a[2][2], K_ZERO), _mm_unpacklo_epi8(a[2][0], K_ZERO))));
            Store<align>((__m128i*)dst + 1, BinomialSum16(
                _mm_sub_epi16(_mm_unpackhi_epi8(a[0][2], K_ZERO), _mm_unpackhi_epi8(a[0][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpackhi_epi8(a[1][2], K_ZERO), _mm_unpackhi_epi8(a[1][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpackhi_epi8(a[2][2], K_ZERO), _mm_unpackhi_epi8(a[2][0], K_ZERO))));
        }

        template <bool align> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m128i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNoseDx(src0 + 0, a[0]);
                LoadNoseDx(src1 + 0, a[1]);
                LoadNoseDx(src2 + 0, a[2]);
                SobelDx<align>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
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
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(dst) && Aligned(dstStride))
                SobelDx<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDx<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE void SobelDy(__m128i a[3][3], int16_t * dst)
        {
            Store<align>((__m128i*)dst + 0, BinomialSum16(
                _mm_sub_epi16(_mm_unpacklo_epi8(a[2][0], K_ZERO), _mm_unpacklo_epi8(a[0][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpacklo_epi8(a[2][1], K_ZERO), _mm_unpacklo_epi8(a[0][1], K_ZERO)),
                _mm_sub_epi16(_mm_unpacklo_epi8(a[2][2], K_ZERO), _mm_unpacklo_epi8(a[0][2], K_ZERO))));
            Store<align>((__m128i*)dst + 1, BinomialSum16(
                _mm_sub_epi16(_mm_unpackhi_epi8(a[2][0], K_ZERO), _mm_unpackhi_epi8(a[0][0], K_ZERO)),
                _mm_sub_epi16(_mm_unpackhi_epi8(a[2][1], K_ZERO), _mm_unpackhi_epi8(a[0][1], K_ZERO)),
                _mm_sub_epi16(_mm_unpackhi_epi8(a[2][2], K_ZERO), _mm_unpackhi_epi8(a[0][2], K_ZERO))));
        }

        template <bool align> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            __m128i a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                SobelDy<align>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    SobelDy<align>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SobelDy<false>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                SobelDy<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                SobelDy<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        template<bool align> SIMD_INLINE __m128i AnchorComponent(const int16_t * src, size_t step, const __m128i & current, const __m128i & threshold, const __m128i & mask)
        {
            __m128i last = _mm_srli_epi16(Load<align>((__m128i*)(src - step)), 1);
            __m128i next = _mm_srli_epi16(Load<align>((__m128i*)(src + step)), 1);
            return _mm_andnot_si128(_mm_or_si128(_mm_cmplt_epi16(_mm_sub_epi16(current, last), threshold),
                _mm_cmplt_epi16(_mm_sub_epi16(current, next), threshold)), mask);
        }

        template<bool align> SIMD_INLINE __m128i Anchor(const int16_t * src, size_t stride, const __m128i & threshold)
        {
            __m128i _src = Load<align>((__m128i*)src);
            __m128i direction = _mm_and_si128(_src, K16_0001);
            __m128i magnitude = _mm_srli_epi16(_src, 1);
            __m128i vertical = AnchorComponent<false>(src, 1, magnitude, threshold, _mm_cmpeq_epi16(direction, K16_0001));
            __m128i horizontal = AnchorComponent<align>(src, stride, magnitude, threshold, _mm_cmpeq_epi16(direction, K_ZERO));
            return _mm_andnot_si128(_mm_cmpeq_epi16(magnitude, K_ZERO), _mm_and_si128(_mm_or_si128(vertical, horizontal), K16_00FF));
        }

        template<bool align> SIMD_INLINE void Anchor(const int16_t * src, size_t stride, const __m128i & threshold, uint8_t * dst)
        {
            __m128i lo = Anchor<align>(src, stride, threshold);
            __m128i hi = Anchor<align>(src + HA, stride, threshold);
            Store<align>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template <bool align> void ContourAnchors(const int16_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride, HA) && Aligned(dst) && Aligned(dstStride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            __m128i _threshold = _mm_set1_epi16(threshold);
            memset(dst, 0, width);
            memset(dst + dstStride*(height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                Anchor<false>(src + 1, srcStride, _threshold, dst + 1);
                for (size_t col = A; col < bodyWidth; col += A)
                    Anchor<align>(src + col, srcStride, _threshold, dst + col);
                Anchor<false>(src + width - A - 1, srcStride, _threshold, dst + width - A - 1);
                dst[width - 1] = 0;
                src += step*srcStride;
                dst += step*dstStride;
            }
        }

        void ContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ContourAnchors<true>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
            else
                ContourAnchors<false>((const int16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif// SIMD_SSE2_ENABLE
}

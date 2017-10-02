/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        template <bool abs> int SobelDx(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2);

        template <> SIMD_INLINE int SobelDx<false>(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2)
        {
            return (s0[x2] + 2 * s1[x2] + s2[x2]) - (s0[x0] + 2 * s1[x0] + s2[x0]);
        }

        template <> SIMD_INLINE int SobelDx<true>(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x2)
        {
            return Simd::Abs(SobelDx<false>(s0, s1, s2, x0, x2));
        }

        template <bool abs> void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = SobelDx<abs>(src0, src1, src2, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    dst[col] = SobelDx<abs>(src0, src1, src2, col - 1, col + 1);

                dst[width - 1] = SobelDx<abs>(src0, src1, src2, width - 2, width - 1);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

#ifdef __GNUC__
                size_t rowSum = 0;
#else
                uint32_t rowSum = 0;
#endif

                rowSum += SobelDx<true>(src0, src1, src2, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    rowSum += SobelDx<true>(src0, src1, src2, col - 1, col + 1);

                rowSum += SobelDx<true>(src0, src1, src2, width - 2, width - 1);

                *sum += rowSum;
            }
        }

        template <bool abs> SIMD_INLINE int SobelDy(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2);

        template <> SIMD_INLINE int SobelDy<false>(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return (s2[x0] + 2 * s2[x1] + s2[x2]) - (s0[x0] + 2 * s0[x1] + s0[x2]);
        }

        template <> SIMD_INLINE int SobelDy<true>(const uint8_t *s0, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            return Simd::Abs(SobelDy<false>(s0, s2, x0, x1, x2));
        }

        template <bool abs> void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = SobelDy<abs>(src0, src2, 0, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    dst[col] = SobelDy<abs>(src0, src2, col - 1, col, col + 1);

                dst[width - 1] = SobelDy<abs>(src0, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void SobelDy(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDy<false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDyAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDy<true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void SobelDyAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

#ifdef __GNUC__
                size_t rowSum = 0;
#else
                uint32_t rowSum = 0;
#endif

                rowSum += SobelDy<true>(src0, src2, 0, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    rowSum += SobelDy<true>(src0, src2, col - 1, col, col + 1);

                rowSum += SobelDy<true>(src0, src2, width - 2, width - 1, width - 1);

                *sum += rowSum;
            }
        }

        SIMD_INLINE int ContourMetrics(const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, size_t x0, size_t x1, size_t x2)
        {
            int dx = SobelDx<true>(s0, s1, s2, x0, x2);
            int dy = SobelDy<true>(s0, s2, x0, x1, x2);
            return (dx + dy) * 2 + (dx >= dy ? 0 : 1);
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint16_t * dst, size_t dstStride)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = ContourMetrics(src0, src1, src2, 0, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    dst[col] = ContourMetrics(src0, src1, src2, col - 1, col, col + 1);

                dst[width - 1] = ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetrics(src, srcStride, width, height, (uint16_t *)dst, dstStride / sizeof(int16_t));
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint16_t * dst, size_t dstStride)
        {
            assert(width > 1);

            const uint8_t *src0, *src1, *src2;

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = mask[0] < indexMin ? 0 : ContourMetrics(src0, src1, src2, 0, 0, 1);

                for (size_t col = 1; col < width - 1; ++col)
                    dst[col] = mask[col] < indexMin ? 0 : ContourMetrics(src0, src1, src2, col - 1, col, col + 1);

                dst[width - 1] = mask[width - 1] < indexMin ? 0 : ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t indexMin, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, (uint16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE uint8_t Anchor(const uint16_t * src, ptrdiff_t stride, int16_t threshold)
        {
            uint16_t s = src[0];
            uint16_t a = s / 2;
            if (s & 1)
                return ((a > 0) && (a - src[+1] / 2 >= threshold) && (a - src[-1] / 2 >= threshold)) ? 255 : 0;
            else
                return ((a > 0) && (a - src[+stride] / 2 >= threshold) && (a - src[-stride] / 2 >= threshold)) ? 255 : 0;
        }

        void ContourAnchors(const uint16_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            memset(dst, 0, width);
            memset(dst + dstStride*(height - 1), 0, width);
            dst += dstStride;
            src += srcStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                for (size_t col = 1; col < width - 1; ++col)
                    dst[col] = Anchor(src + col, srcStride, threshold);
                dst[width - 1] = 0;
                dst += step*dstStride;
                src += step*srcStride;
            }
        }

        void ContourAnchors(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t * dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            ContourAnchors((const uint16_t *)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
}

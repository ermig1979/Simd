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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template <bool abs> SIMD_INLINE int16_t SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2);

        template <> SIMD_INLINE int16_t SobelDx<false>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2)
        {
            return (int16_t)((s0[x2] + 2 * s1[x2] + s2[x2]) - (s0[x0] + 2 * s1[x0] + s2[x0]));
        }

        template <> SIMD_INLINE int16_t SobelDx<true>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x2)
        {
            return (int16_t)Simd::Abs(SobelDx<false>(s0, s1, s2, x0, x2));
        }

        template <bool abs> SIMD_INLINE svint16_t SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask);

        template <> SIMD_INLINE svint16_t SobelDx<false>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t left = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1), 1)), svld1ub_s16(mask, s2));
            svint16_t right = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 2), 1)), svld1ub_s16(mask, s2 + 2));
            return svsub_s16_x(mask, right, left);
        }

        template <> SIMD_INLINE svint16_t SobelDx<true>(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            return svabs_s16_x(mask, SobelDx<false>(s0, s1, s2, mask));
        }

        template <bool abs> SIMD_INLINE void SobelDx(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, SobelDx<abs>(s0, s1, s2, mask));
        }

        template <bool abs> void SobelDx(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = SobelDx<abs>(src0, src1, src2, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    SobelDx<abs>(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = SobelDx<abs>(src0, src1, src2, width - 2, width - 1);

                dst += dstStride;
            }
        }

        void SobelDx(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<false>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        void SobelDxAbs(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            SobelDx<true>(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE int16_t ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, size_t x0, size_t x1, size_t x2)
        {
            int dx = Simd::Abs((s0[x2] + 2 * s1[x2] + s2[x2]) - (s0[x0] + 2 * s1[x0] + s2[x0]));
            int dy = Simd::Abs((s2[x0] + 2 * s2[x1] + s2[x2]) - (s0[x0] + 2 * s0[x1] + s0[x2]));
            return (int16_t)((dx + dy) * 2 + (dx >= dy ? 0 : 1));
        }

        SIMD_INLINE svint16_t SobelDxAbs(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t left = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1), 1)), svld1ub_s16(mask, s2));
            svint16_t right = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0 + 2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s1 + 2), 1)), svld1ub_s16(mask, s2 + 2));
            return svabs_s16_x(mask, svsub_s16_x(mask, right, left));
        }

        SIMD_INLINE svint16_t SobelDyAbs(const uint8_t* s0, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t top = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s0), svlsl_n_s16_x(mask, svld1ub_s16(mask, s0 + 1), 1)), svld1ub_s16(mask, s0 + 2));
            svint16_t bottom = svadd_s16_x(mask, svadd_s16_x(mask, svld1ub_s16(mask, s2), svlsl_n_s16_x(mask, svld1ub_s16(mask, s2 + 1), 1)), svld1ub_s16(mask, s2 + 2));
            return svabs_s16_x(mask, svsub_s16_x(mask, bottom, top));
        }

        SIMD_INLINE svint16_t ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, const svbool_t& mask)
        {
            svint16_t dx = SobelDxAbs(s0, s1, s2, mask);
            svint16_t dy = SobelDyAbs(s0, s2, mask);
            svint16_t sum = svlsl_n_s16_x(mask, svadd_s16_x(mask, dx, dy), 1);
            return svadd_s16_x(mask, sum, svsel_s16(svcmplt_s16(mask, dx, dy), svdup_n_s16(1), svdup_n_s16(0)));
        }

        SIMD_INLINE void ContourMetrics(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2, int16_t* dst, const svbool_t& mask)
        {
            svst1_s16(mask, dst, ContourMetrics(s0, s1, s2, mask));
        }

        void ContourMetrics(const uint8_t* src, size_t srcStride, size_t width, size_t height, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = ContourMetrics(src0, src1, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    ContourMetrics(src0 + col - 1, src1 + col - 1, src2 + col - 1, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
            }
        }

        void ContourMetrics(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetrics(src, srcStride, width, height, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void ContourMetricsMasked(const uint8_t* s0, const uint8_t* s1, const uint8_t* s2,
            const uint8_t* mask, uint8_t indexMin, int16_t* dst, const svbool_t& tail)
        {
            svuint16_t _mask = svld1ub_u16(tail, mask);
            svbool_t valid = svcmpge_n_u16(tail, _mask, indexMin);
            svst1_s16(tail, dst, svsel_s16(valid, ContourMetrics(s0, s1, s2, tail), svdup_n_s16(0)));
        }

        void ContourMetricsMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t indexMin, int16_t* dst, size_t dstStride)
        {
            assert(width > 1);

            const size_t A = svcnth();
            const uint8_t * src0, * src1, * src2;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride * (row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                dst[0] = mask[0] < indexMin ? 0 : ContourMetrics(src0, src1, src2, 0, 0, 1);
                for (size_t col = 1; col < width - 1; col += A)
                    ContourMetricsMasked(src0 + col - 1, src1 + col - 1, src2 + col - 1,
                        mask + col, indexMin, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = mask[width - 1] < indexMin ? 0 : ContourMetrics(src0, src1, src2, width - 2, width - 1, width - 1);

                dst += dstStride;
                mask += maskStride;
            }
        }

        void ContourMetricsMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t indexMin, uint8_t* dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            ContourMetricsMasked(src, srcStride, width, height, mask, maskStride, indexMin, (int16_t*)dst, dstStride / sizeof(int16_t));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svint16_t Half(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_s16_u16(svlsr_n_u16_x(mask, svld1_u16(mask, src), 1));
        }

        SIMD_INLINE void Anchor(const uint16_t* src, size_t stride, const svint16_t& threshold, uint8_t* dst, const svbool_t& mask)
        {
            svuint16_t _src = svld1_u16(mask, src);
            svuint16_t direction = svand_n_u16_x(mask, _src, 1);
            svuint16_t magnitude = svlsr_n_u16_x(mask, _src, 1);
            svint16_t value = svsub_s16_x(mask, svreinterpret_s16_u16(magnitude), threshold);

            svbool_t vertical = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 1),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - 1, mask)), svcmpge_s16(mask, value, Half(src + 1, mask))));
            svbool_t horizontal = svand_b_z(mask, svcmpeq_n_u16(mask, direction, 0),
                svand_b_z(mask, svcmpge_s16(mask, value, Half(src - stride, mask)), svcmpge_s16(mask, value, Half(src + stride, mask))));
            svbool_t anchor = svand_b_z(mask, svcmpgt_n_u16(mask, magnitude, 0), svorr_b_z(mask, vertical, horizontal));

            svst1b_u16(mask, dst, svsel_u16(anchor, svdup_n_u16(0xFF), svdup_n_u16(0)));
        }

        void ContourAnchors(const uint16_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcnth();
            const svint16_t _threshold = svdup_n_s16(threshold);

            memset(dst, 0, width);
            memset(dst + dstStride * (height - 1), 0, width);
            src += srcStride;
            dst += dstStride;
            for (size_t row = 1; row < height - 1; row += step)
            {
                dst[0] = 0;
                for (size_t col = 1; col < width - 1; col += A)
                    Anchor(src + col, srcStride, _threshold, dst + col, svwhilelt_b16(col, width - 1));
                dst[width - 1] = 0;
                src += step * srcStride;
                dst += step * dstStride;
            }
        }

        void ContourAnchors(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t step, int16_t threshold, uint8_t* dst, size_t dstStride)
        {
            assert(srcStride % sizeof(int16_t) == 0);

            ContourAnchors((const uint16_t*)src, srcStride / sizeof(int16_t), width, height, step, threshold, dst, dstStride);
        }
    }
#endif
}

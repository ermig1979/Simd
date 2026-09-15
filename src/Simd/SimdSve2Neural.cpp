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
        SIMD_INLINE void UpdateWeights(const svbool_t& mask, const svfloat32_t& a, const svfloat32_t& b, const float* x, float* d, float* w)
        {
            svfloat32_t _d = svmla_f32_m(mask, svmul_f32_x(mask, svld1_f32(mask, x), b), svld1_f32(mask, d), a);
            svst1_f32(mask, d, _d);
            svst1_f32(mask, w, svadd_f32_x(mask, svld1_f32(mask, w), _d));
        }

        void NeuralUpdateWeights(const float* x, size_t size, const float* a, const float* b, float* d, float* w)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _a = svdup_n_f32(a[0]);
            const svfloat32_t _b = svdup_n_f32(b[0]);

            for (; i + QF <= size; i += QF)
            {
                UpdateWeights(body, _a, _b, x + i + 0 * F, d + i + 0 * F, w + i + 0 * F);
                UpdateWeights(body, _a, _b, x + i + 1 * F, d + i + 1 * F, w + i + 1 * F);
                UpdateWeights(body, _a, _b, x + i + 2 * F, d + i + 2 * F, w + i + 2 * F);
                UpdateWeights(body, _a, _b, x + i + 3 * F, d + i + 3 * F, w + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                UpdateWeights(body, _a, _b, x + i, d + i, w + i);
            if (i < size)
                UpdateWeights(svwhilelt_b32(i, size), _a, _b, x + i, d + i, w + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AdaptiveGradientUpdate(const svbool_t& mask, const svfloat32_t& norm, const svfloat32_t& alpha, const svfloat32_t& epsilon, const float* delta, float* gradient, float* weight)
        {
            svfloat32_t d = svmul_f32_x(mask, svld1_f32(mask, delta), norm);
            svfloat32_t _gradient = svmla_f32_m(mask, svld1_f32(mask, gradient), d, d);
            svst1_f32(mask, gradient, _gradient);
            svst1_f32(mask, weight, svsub_f32_x(mask, svld1_f32(mask, weight),
                svdiv_f32_x(mask, svmul_f32_x(mask, alpha, d), svsqrt_f32_x(mask, svadd_f32_x(mask, _gradient, epsilon)))));
        }

        void NeuralAdaptiveGradientUpdate(const float* delta, size_t size, size_t batch, const float* alpha, const float* epsilon, float* gradient, float* weight)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _norm = svdup_n_f32((float)(1.0 / batch));
            const svfloat32_t _alpha = svdup_n_f32(alpha[0]);
            const svfloat32_t _epsilon = svdup_n_f32(epsilon[0]);

            for (; i + QF <= size; i += QF)
            {
                AdaptiveGradientUpdate(body, _norm, _alpha, _epsilon, delta + i + 0 * F, gradient + i + 0 * F, weight + i + 0 * F);
                AdaptiveGradientUpdate(body, _norm, _alpha, _epsilon, delta + i + 1 * F, gradient + i + 1 * F, weight + i + 1 * F);
                AdaptiveGradientUpdate(body, _norm, _alpha, _epsilon, delta + i + 2 * F, gradient + i + 2 * F, weight + i + 2 * F);
                AdaptiveGradientUpdate(body, _norm, _alpha, _epsilon, delta + i + 3 * F, gradient + i + 3 * F, weight + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AdaptiveGradientUpdate(body, _norm, _alpha, _epsilon, delta + i, gradient + i, weight + i);
            if (i < size)
                AdaptiveGradientUpdate(svwhilelt_b32(i, size), _norm, _alpha, _epsilon, delta + i, gradient + i, weight + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE float Max2(const float* src)
        {
            return Simd::Max(src[0], src[1]);
        }

        SIMD_INLINE float Max2x2(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Max2(src + stride));
        }

        SIMD_INLINE float Max2x3(const float* src, size_t stride)
        {
            return Simd::Max(Max2(src), Simd::Max(Max2(src + stride), Max2(src + 2 * stride)));
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x1(const svbool_t& mask, const float* src)
        {
            svfloat32_t src0 = svld1_f32(mask, src + 0);
            svfloat32_t src1 = svld1_f32(mask, src + 1);
            svfloat32_t src2 = svld1_f32(mask, src + 2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x2(const svbool_t& mask, const float* src, size_t stride)
        {
            svfloat32_t src0 = Pooling1x1Max3x1(mask, src);
            svfloat32_t src1 = Pooling1x1Max3x1(mask, src + stride);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling1x1Max3x3(const svbool_t& mask, const float* src, size_t stride)
        {
            svfloat32_t src0 = Pooling1x1Max3x1(mask, src);
            svfloat32_t src1 = Pooling1x1Max3x1(mask, src + stride);
            svfloat32_t src2 = Pooling1x1Max3x1(mask, src + 2 * stride);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE void Pooling1x1Max3x2(const float* src, size_t stride, size_t width, float* dst)
        {
            size_t F = svcntw(), bodyWidth = width - 1;
            for (size_t col = 1; col < bodyWidth; col += F)
            {
                svbool_t mask = svwhilelt_b32(col, bodyWidth);
                svst1_f32(mask, dst + col, Pooling1x1Max3x2(mask, src + col - 1, stride));
            }
        }

        SIMD_INLINE void Pooling1x1Max3x3(const float* src, size_t stride, size_t width, float* dst)
        {
            size_t F = svcntw(), bodyWidth = width - 1;
            for (size_t col = 1; col < bodyWidth; col += F)
            {
                svbool_t mask = svwhilelt_b32(col, bodyWidth);
                svst1_f32(mask, dst + col, Pooling1x1Max3x3(mask, src + col - 1, stride));
            }
        }

        void NeuralPooling1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width > 1 && height > 1);

            dst[0] = Max2x2(src, srcStride);
            Pooling1x1Max3x2(src, srcStride, width, dst);
            dst[width - 1] = Max2x2(src + width - 2, srcStride);
            dst += dstStride;

            for (size_t row = 1; row < height - 1; ++row)
            {
                const float* s = src + (row - 1) * srcStride;
                dst[0] = Max2x3(s, srcStride);
                Pooling1x1Max3x3(s, srcStride, width, dst);
                dst[width - 1] = Max2x3(s + width - 2, srcStride);
                dst += dstStride;
            }

            src += (height - 2) * srcStride;
            dst[0] = Max2x2(src, srcStride);
            Pooling1x1Max3x2(src, srcStride, width, dst);
            dst[width - 1] = Max2x2(src + width - 2, srcStride);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max1x3(const svbool_t& mask, const float* src, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = svld1_gather_u32index_f32(mask, src, offsets0);
            svfloat32_t src1 = svld1_gather_u32index_f32(mask, src, offsets1);
            svfloat32_t src2 = svld1_gather_u32index_f32(mask, src, offsets2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max1x2(const svbool_t& mask, const float* src, const svuint32_t& offsets0, const svuint32_t& offsets1)
        {
            svfloat32_t src0 = svld1_gather_u32index_f32(mask, src, offsets0);
            svfloat32_t src1 = svld1_gather_u32index_f32(mask, src, offsets1);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max2x2(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1)
        {
            svfloat32_t src0 = Pooling2x2Max1x2(mask, src, offsets0, offsets1);
            svfloat32_t src1 = Pooling2x2Max1x2(mask, src + stride, offsets0, offsets1);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max3x2(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = Pooling2x2Max1x3(mask, src, offsets0, offsets1, offsets2);
            svfloat32_t src1 = Pooling2x2Max1x3(mask, src + stride, offsets0, offsets1, offsets2);
            return svmax_f32_x(mask, src0, src1);
        }

        SIMD_INLINE svfloat32_t Pooling2x2Max3x3(const svbool_t& mask, const float* src, size_t stride, const svuint32_t& offsets0, const svuint32_t& offsets1, const svuint32_t& offsets2)
        {
            svfloat32_t src0 = Pooling2x2Max1x3(mask, src, offsets0, offsets1, offsets2);
            svfloat32_t src1 = Pooling2x2Max1x3(mask, src + stride, offsets0, offsets1, offsets2);
            svfloat32_t src2 = Pooling2x2Max1x3(mask, src + 2 * stride, offsets0, offsets1, offsets2);
            return svmax_f32_x(mask, svmax_f32_x(mask, src0, src1), src2);
        }

        void NeuralPooling2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), heightEven = Simd::AlignLo(height, 2), widthEven = Simd::AlignLo(width, 2), dstWidthEven = widthEven >> 1;
            const svuint32_t offsets0 = svindex_u32(0, 2);
            const svuint32_t offsets1 = svindex_u32(1, 2);

            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max2x2(mask, src + 2 * col, srcStride, offsets0, offsets1));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max1x2(mask, src + 2 * col, offsets0, offsets1));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = src[widthEven];
            }
        }

        void NeuralPooling2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t F = svcntw(), heightEven = Simd::AlignLo(height, 2), widthEven = Simd::AlignLo(width, 2), dstWidthEven = widthEven >> 1;
            const svuint32_t offsets0 = svindex_u32(0, 2);
            const svuint32_t offsets1 = svindex_u32(1, 2);
            const svuint32_t offsets2 = svindex_u32(2, 2);

            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max3x3(mask, src + 2 * col, srcStride, offsets0, offsets1, offsets2));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Max2x3(src + widthEven, srcStride);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < dstWidthEven; col += F)
                {
                    svbool_t mask = svwhilelt_b32(col, dstWidthEven);
                    svst1_f32(mask, dst + col, Pooling2x2Max3x2(mask, src + 2 * col, srcStride, offsets0, offsets1, offsets2));
                }
                if (width - widthEven)
                    dst[dstWidthEven] = Max2x2(src + widthEven, srcStride);
            }
        }
    }
#endif
}

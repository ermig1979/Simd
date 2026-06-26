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
        SIMD_INLINE svfloat32_t Convolution2x2Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3)
        {
            svfloat32_t sum = svmul_f32_x(mask, svld1_f32(mask, src), weight0);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src + 1), weight1);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src + stride), weight2);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src + stride + 1), weight3);
            return sum;
        }

        SIMD_INLINE void AddConvolution2x2Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3, float* dst)
        {
            svfloat32_t sum = Convolution2x2Forward(mask, src, stride, weight0, weight1, weight2, weight3);
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), sum));
        }

        void NeuralAddConvolution2x2Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution2x2Forward(body, src + col + 0 * F, srcStride, weight0, weight1, weight2, weight3, dst + col + 0 * F);
                    AddConvolution2x2Forward(body, src + col + 1 * F, srcStride, weight0, weight1, weight2, weight3, dst + col + 1 * F);
                    AddConvolution2x2Forward(body, src + col + 2 * F, srcStride, weight0, weight1, weight2, weight3, dst + col + 2 * F);
                    AddConvolution2x2Forward(body, src + col + 3 * F, srcStride, weight0, weight1, weight2, weight3, dst + col + 3 * F);
                }
                for (; col + F <= width; col += F)
                    AddConvolution2x2Forward(body, src + col, srcStride, weight0, weight1, weight2, weight3, dst + col);
                if (col < width)
                    AddConvolution2x2Forward(svwhilelt_b32(col, width), src + col, srcStride, weight0, weight1, weight2, weight3, dst + col);

                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddMultiplied(const svbool_t& mask, const float* src, const svfloat32_t& value, float* dst)
        {
            svst1_f32(mask, dst, svmla_f32_m(mask, svld1_f32(mask, dst), svld1_f32(mask, src), value));
        }

        SIMD_INLINE void AddMultiplied(const float* src, size_t size, const svfloat32_t& value, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();

            for (; i + QF <= size; i += QF)
            {
                AddMultiplied(body, src + i + 0 * F, value, dst + i + 0 * F);
                AddMultiplied(body, src + i + 1 * F, value, dst + i + 1 * F);
                AddMultiplied(body, src + i + 2 * F, value, dst + i + 2 * F);
                AddMultiplied(body, src + i + 3 * F, value, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddMultiplied(body, src + i, value, dst + i);
            if (i < size)
                AddMultiplied(svwhilelt_b32(i, size), src + i, value, dst + i);
        }

        void NeuralAddConvolution2x2Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);

            for (size_t row = 0; row < height; ++row)
            {
                float* dst0 = dst;
                float* dst1 = dst + dstStride;

                AddMultiplied(src, width, weight0, dst0 + 0);
                AddMultiplied(src, width, weight1, dst0 + 1);
                AddMultiplied(src, width, weight2, dst1 + 0);
                AddMultiplied(src, width, weight3, dst1 + 1);

                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddConvolution2x2Sum(const svbool_t& mask, const float* src, size_t stride, const svfloat32_t& dst,
            svfloat32_t& sum0, svfloat32_t& sum1, svfloat32_t& sum2, svfloat32_t& sum3)
        {
            sum0 = svmla_f32_m(mask, sum0, svld1_f32(mask, src), dst);
            sum1 = svmla_f32_m(mask, sum1, svld1_f32(mask, src + 1), dst);
            sum2 = svmla_f32_m(mask, sum2, svld1_f32(mask, src + stride), dst);
            sum3 = svmla_f32_m(mask, sum3, svld1_f32(mask, src + stride + 1), dst);
        }

        void NeuralAddConvolution2x2Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution2x2Sum(body, src + col + 0 * F, srcStride, svld1_f32(body, dst + col + 0 * F), sum0, sum1, sum2, sum3);
                    AddConvolution2x2Sum(body, src + col + 1 * F, srcStride, svld1_f32(body, dst + col + 1 * F), sum0, sum1, sum2, sum3);
                    AddConvolution2x2Sum(body, src + col + 2 * F, srcStride, svld1_f32(body, dst + col + 2 * F), sum0, sum1, sum2, sum3);
                    AddConvolution2x2Sum(body, src + col + 3 * F, srcStride, svld1_f32(body, dst + col + 3 * F), sum0, sum1, sum2, sum3);
                }
                for (; col + F <= width; col += F)
                    AddConvolution2x2Sum(body, src + col, srcStride, svld1_f32(body, dst + col), sum0, sum1, sum2, sum3);
                if (col < width)
                {
                    svbool_t tail = svwhilelt_b32(col, width);
                    AddConvolution2x2Sum(tail, src + col, srcStride, svld1_f32(tail, dst + col), sum0, sum1, sum2, sum3);
                }

                src += srcStride;
                dst += dstStride;
            }

            sums[0] += svaddv_f32(body, sum0);
            sums[1] += svaddv_f32(body, sum1);
            sums[2] += svaddv_f32(body, sum2);
            sums[3] += svaddv_f32(body, sum3);
        }
    }
#endif
}

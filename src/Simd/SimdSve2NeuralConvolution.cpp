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

        SIMD_INLINE svfloat32_t Convolution3x3Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2,
            const svfloat32_t& weight3, const svfloat32_t& weight4, const svfloat32_t& weight5,
            const svfloat32_t& weight6, const svfloat32_t& weight7, const svfloat32_t& weight8)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            svfloat32_t sum = svmul_f32_x(mask, svld1_f32(mask, src0 + 0), weight0);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 1), weight1);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 2), weight2);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 0), weight3);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 1), weight4);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 2), weight5);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 0), weight6);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 1), weight7);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 2), weight8);
            return sum;
        }

        SIMD_INLINE void AddConvolution3x3Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2,
            const svfloat32_t& weight3, const svfloat32_t& weight4, const svfloat32_t& weight5,
            const svfloat32_t& weight6, const svfloat32_t& weight7, const svfloat32_t& weight8, float* dst)
        {
            svfloat32_t sum = Convolution3x3Forward(mask, src, stride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8);
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), sum));
        }

        void NeuralAddConvolution3x3Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);
            const svfloat32_t weight4 = svdup_n_f32(weights[4]);
            const svfloat32_t weight5 = svdup_n_f32(weights[5]);
            const svfloat32_t weight6 = svdup_n_f32(weights[6]);
            const svfloat32_t weight7 = svdup_n_f32(weights[7]);
            const svfloat32_t weight8 = svdup_n_f32(weights[8]);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution3x3Forward(body, src + col + 0 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col + 0 * F);
                    AddConvolution3x3Forward(body, src + col + 1 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col + 1 * F);
                    AddConvolution3x3Forward(body, src + col + 2 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col + 2 * F);
                    AddConvolution3x3Forward(body, src + col + 3 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col + 3 * F);
                }
                for (; col + F <= width; col += F)
                    AddConvolution3x3Forward(body, src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col);
                if (col < width)
                    AddConvolution3x3Forward(svwhilelt_b32(col, width), src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, dst + col);

                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t Convolution4x4Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3,
            const svfloat32_t& weight4, const svfloat32_t& weight5, const svfloat32_t& weight6, const svfloat32_t& weight7,
            const svfloat32_t& weight8, const svfloat32_t& weight9, const svfloat32_t& weight10, const svfloat32_t& weight11,
            const svfloat32_t& weight12, const svfloat32_t& weight13, const svfloat32_t& weight14, const svfloat32_t& weight15)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            const float* src3 = src + 3 * stride;
            svfloat32_t sum = svmul_f32_x(mask, svld1_f32(mask, src0 + 0), weight0);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 1), weight1);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 2), weight2);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 3), weight3);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 0), weight4);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 1), weight5);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 2), weight6);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 3), weight7);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 0), weight8);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 1), weight9);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 2), weight10);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 3), weight11);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 0), weight12);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 1), weight13);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 2), weight14);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 3), weight15);
            return sum;
        }

        SIMD_INLINE void AddConvolution4x4Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3,
            const svfloat32_t& weight4, const svfloat32_t& weight5, const svfloat32_t& weight6, const svfloat32_t& weight7,
            const svfloat32_t& weight8, const svfloat32_t& weight9, const svfloat32_t& weight10, const svfloat32_t& weight11,
            const svfloat32_t& weight12, const svfloat32_t& weight13, const svfloat32_t& weight14, const svfloat32_t& weight15, float* dst)
        {
            svfloat32_t sum = Convolution4x4Forward(mask, src, stride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15);
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), sum));
        }

        void NeuralAddConvolution4x4Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);
            const svfloat32_t weight4 = svdup_n_f32(weights[4]);
            const svfloat32_t weight5 = svdup_n_f32(weights[5]);
            const svfloat32_t weight6 = svdup_n_f32(weights[6]);
            const svfloat32_t weight7 = svdup_n_f32(weights[7]);
            const svfloat32_t weight8 = svdup_n_f32(weights[8]);
            const svfloat32_t weight9 = svdup_n_f32(weights[9]);
            const svfloat32_t weight10 = svdup_n_f32(weights[10]);
            const svfloat32_t weight11 = svdup_n_f32(weights[11]);
            const svfloat32_t weight12 = svdup_n_f32(weights[12]);
            const svfloat32_t weight13 = svdup_n_f32(weights[13]);
            const svfloat32_t weight14 = svdup_n_f32(weights[14]);
            const svfloat32_t weight15 = svdup_n_f32(weights[15]);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution4x4Forward(body, src + col + 0 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col + 0 * F);
                    AddConvolution4x4Forward(body, src + col + 1 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col + 1 * F);
                    AddConvolution4x4Forward(body, src + col + 2 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col + 2 * F);
                    AddConvolution4x4Forward(body, src + col + 3 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col + 3 * F);
                }
                for (; col + F <= width; col += F)
                    AddConvolution4x4Forward(body, src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col);
                if (col < width)
                    AddConvolution4x4Forward(svwhilelt_b32(col, width), src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7,
                        weight8, weight9, weight10, weight11, weight12, weight13, weight14, weight15, dst + col);

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

        void NeuralAddConvolution3x3Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);
            const svfloat32_t weight4 = svdup_n_f32(weights[4]);
            const svfloat32_t weight5 = svdup_n_f32(weights[5]);
            const svfloat32_t weight6 = svdup_n_f32(weights[6]);
            const svfloat32_t weight7 = svdup_n_f32(weights[7]);
            const svfloat32_t weight8 = svdup_n_f32(weights[8]);

            for (size_t row = 0; row < height; ++row)
            {
                float* dst0 = dst;
                float* dst1 = dst + dstStride;
                float* dst2 = dst + 2 * dstStride;

                AddMultiplied(src, width, weight0, dst0 + 0);
                AddMultiplied(src, width, weight1, dst0 + 1);
                AddMultiplied(src, width, weight2, dst0 + 2);
                AddMultiplied(src, width, weight3, dst1 + 0);
                AddMultiplied(src, width, weight4, dst1 + 1);
                AddMultiplied(src, width, weight5, dst1 + 2);
                AddMultiplied(src, width, weight6, dst2 + 0);
                AddMultiplied(src, width, weight7, dst2 + 1);
                AddMultiplied(src, width, weight8, dst2 + 2);

                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void NeuralAddConvolution4x4Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);
            const svfloat32_t weight4 = svdup_n_f32(weights[4]);
            const svfloat32_t weight5 = svdup_n_f32(weights[5]);
            const svfloat32_t weight6 = svdup_n_f32(weights[6]);
            const svfloat32_t weight7 = svdup_n_f32(weights[7]);
            const svfloat32_t weight8 = svdup_n_f32(weights[8]);
            const svfloat32_t weight9 = svdup_n_f32(weights[9]);
            const svfloat32_t weight10 = svdup_n_f32(weights[10]);
            const svfloat32_t weight11 = svdup_n_f32(weights[11]);
            const svfloat32_t weight12 = svdup_n_f32(weights[12]);
            const svfloat32_t weight13 = svdup_n_f32(weights[13]);
            const svfloat32_t weight14 = svdup_n_f32(weights[14]);
            const svfloat32_t weight15 = svdup_n_f32(weights[15]);

            for (size_t row = 0; row < height; ++row)
            {
                float* dst0 = dst;
                float* dst1 = dst + dstStride;
                float* dst2 = dst + 2 * dstStride;
                float* dst3 = dst + 3 * dstStride;

                AddMultiplied(src, width, weight0, dst0 + 0);
                AddMultiplied(src, width, weight1, dst0 + 1);
                AddMultiplied(src, width, weight2, dst0 + 2);
                AddMultiplied(src, width, weight3, dst0 + 3);
                AddMultiplied(src, width, weight4, dst1 + 0);
                AddMultiplied(src, width, weight5, dst1 + 1);
                AddMultiplied(src, width, weight6, dst1 + 2);
                AddMultiplied(src, width, weight7, dst1 + 3);
                AddMultiplied(src, width, weight8, dst2 + 0);
                AddMultiplied(src, width, weight9, dst2 + 1);
                AddMultiplied(src, width, weight10, dst2 + 2);
                AddMultiplied(src, width, weight11, dst2 + 3);
                AddMultiplied(src, width, weight12, dst3 + 0);
                AddMultiplied(src, width, weight13, dst3 + 1);
                AddMultiplied(src, width, weight14, dst3 + 2);
                AddMultiplied(src, width, weight15, dst3 + 3);

                src += srcStride;
                dst += dstStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void NeuralAddConvolution5x5Backward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
        {
            const svfloat32_t weight0 = svdup_n_f32(weights[0]);
            const svfloat32_t weight1 = svdup_n_f32(weights[1]);
            const svfloat32_t weight2 = svdup_n_f32(weights[2]);
            const svfloat32_t weight3 = svdup_n_f32(weights[3]);
            const svfloat32_t weight4 = svdup_n_f32(weights[4]);
            const svfloat32_t weight5 = svdup_n_f32(weights[5]);
            const svfloat32_t weight6 = svdup_n_f32(weights[6]);
            const svfloat32_t weight7 = svdup_n_f32(weights[7]);
            const svfloat32_t weight8 = svdup_n_f32(weights[8]);
            const svfloat32_t weight9 = svdup_n_f32(weights[9]);
            const svfloat32_t weight10 = svdup_n_f32(weights[10]);
            const svfloat32_t weight11 = svdup_n_f32(weights[11]);
            const svfloat32_t weight12 = svdup_n_f32(weights[12]);
            const svfloat32_t weight13 = svdup_n_f32(weights[13]);
            const svfloat32_t weight14 = svdup_n_f32(weights[14]);
            const svfloat32_t weight15 = svdup_n_f32(weights[15]);
            const svfloat32_t weight16 = svdup_n_f32(weights[16]);
            const svfloat32_t weight17 = svdup_n_f32(weights[17]);
            const svfloat32_t weight18 = svdup_n_f32(weights[18]);
            const svfloat32_t weight19 = svdup_n_f32(weights[19]);
            const svfloat32_t weight20 = svdup_n_f32(weights[20]);
            const svfloat32_t weight21 = svdup_n_f32(weights[21]);
            const svfloat32_t weight22 = svdup_n_f32(weights[22]);
            const svfloat32_t weight23 = svdup_n_f32(weights[23]);
            const svfloat32_t weight24 = svdup_n_f32(weights[24]);

            for (size_t row = 0; row < height; ++row)
            {
                float* dst0 = dst;
                float* dst1 = dst + dstStride;
                float* dst2 = dst + 2 * dstStride;
                float* dst3 = dst + 3 * dstStride;
                float* dst4 = dst + 4 * dstStride;

                AddMultiplied(src, width, weight0, dst0 + 0);
                AddMultiplied(src, width, weight1, dst0 + 1);
                AddMultiplied(src, width, weight2, dst0 + 2);
                AddMultiplied(src, width, weight3, dst0 + 3);
                AddMultiplied(src, width, weight4, dst0 + 4);
                AddMultiplied(src, width, weight5, dst1 + 0);
                AddMultiplied(src, width, weight6, dst1 + 1);
                AddMultiplied(src, width, weight7, dst1 + 2);
                AddMultiplied(src, width, weight8, dst1 + 3);
                AddMultiplied(src, width, weight9, dst1 + 4);
                AddMultiplied(src, width, weight10, dst2 + 0);
                AddMultiplied(src, width, weight11, dst2 + 1);
                AddMultiplied(src, width, weight12, dst2 + 2);
                AddMultiplied(src, width, weight13, dst2 + 3);
                AddMultiplied(src, width, weight14, dst2 + 4);
                AddMultiplied(src, width, weight15, dst3 + 0);
                AddMultiplied(src, width, weight16, dst3 + 1);
                AddMultiplied(src, width, weight17, dst3 + 2);
                AddMultiplied(src, width, weight18, dst3 + 3);
                AddMultiplied(src, width, weight19, dst3 + 4);
                AddMultiplied(src, width, weight20, dst4 + 0);
                AddMultiplied(src, width, weight21, dst4 + 1);
                AddMultiplied(src, width, weight22, dst4 + 2);
                AddMultiplied(src, width, weight23, dst4 + 3);
                AddMultiplied(src, width, weight24, dst4 + 4);

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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddConvolution3x3Sum(const svbool_t& mask, const float* src, size_t stride, const svfloat32_t& dst,
            svfloat32_t& sum0, svfloat32_t& sum1, svfloat32_t& sum2, svfloat32_t& sum3, svfloat32_t& sum4,
            svfloat32_t& sum5, svfloat32_t& sum6, svfloat32_t& sum7, svfloat32_t& sum8)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            sum0 = svmla_f32_m(mask, sum0, svld1_f32(mask, src0 + 0), dst);
            sum1 = svmla_f32_m(mask, sum1, svld1_f32(mask, src0 + 1), dst);
            sum2 = svmla_f32_m(mask, sum2, svld1_f32(mask, src0 + 2), dst);
            sum3 = svmla_f32_m(mask, sum3, svld1_f32(mask, src1 + 0), dst);
            sum4 = svmla_f32_m(mask, sum4, svld1_f32(mask, src1 + 1), dst);
            sum5 = svmla_f32_m(mask, sum5, svld1_f32(mask, src1 + 2), dst);
            sum6 = svmla_f32_m(mask, sum6, svld1_f32(mask, src2 + 0), dst);
            sum7 = svmla_f32_m(mask, sum7, svld1_f32(mask, src2 + 1), dst);
            sum8 = svmla_f32_m(mask, sum8, svld1_f32(mask, src2 + 2), dst);
        }

        void NeuralAddConvolution3x3Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);
            svfloat32_t sum4 = svdup_n_f32(0.0f);
            svfloat32_t sum5 = svdup_n_f32(0.0f);
            svfloat32_t sum6 = svdup_n_f32(0.0f);
            svfloat32_t sum7 = svdup_n_f32(0.0f);
            svfloat32_t sum8 = svdup_n_f32(0.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution3x3Sum(body, src + col + 0 * F, srcStride, svld1_f32(body, dst + col + 0 * F), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                    AddConvolution3x3Sum(body, src + col + 1 * F, srcStride, svld1_f32(body, dst + col + 1 * F), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                    AddConvolution3x3Sum(body, src + col + 2 * F, srcStride, svld1_f32(body, dst + col + 2 * F), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                    AddConvolution3x3Sum(body, src + col + 3 * F, srcStride, svld1_f32(body, dst + col + 3 * F), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                }
                for (; col + F <= width; col += F)
                    AddConvolution3x3Sum(body, src + col, srcStride, svld1_f32(body, dst + col), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                if (col < width)
                {
                    svbool_t tail = svwhilelt_b32(col, width);
                    AddConvolution3x3Sum(tail, src + col, srcStride, svld1_f32(tail, dst + col), sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8);
                }

                src += srcStride;
                dst += dstStride;
            }

            sums[0] += svaddv_f32(body, sum0);
            sums[1] += svaddv_f32(body, sum1);
            sums[2] += svaddv_f32(body, sum2);
            sums[3] += svaddv_f32(body, sum3);
            sums[4] += svaddv_f32(body, sum4);
            sums[5] += svaddv_f32(body, sum5);
            sums[6] += svaddv_f32(body, sum6);
            sums[7] += svaddv_f32(body, sum7);
            sums[8] += svaddv_f32(body, sum8);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddConvolution4x4Sum(const svbool_t& mask, const float* src, size_t stride, const svfloat32_t& dst,
            svfloat32_t& sum0, svfloat32_t& sum1, svfloat32_t& sum2, svfloat32_t& sum3,
            svfloat32_t& sum4, svfloat32_t& sum5, svfloat32_t& sum6, svfloat32_t& sum7,
            svfloat32_t& sum8, svfloat32_t& sum9, svfloat32_t& sum10, svfloat32_t& sum11,
            svfloat32_t& sum12, svfloat32_t& sum13, svfloat32_t& sum14, svfloat32_t& sum15)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            const float* src3 = src + 3 * stride;
            sum0 = svmla_f32_m(mask, sum0, svld1_f32(mask, src0 + 0), dst);
            sum1 = svmla_f32_m(mask, sum1, svld1_f32(mask, src0 + 1), dst);
            sum2 = svmla_f32_m(mask, sum2, svld1_f32(mask, src0 + 2), dst);
            sum3 = svmla_f32_m(mask, sum3, svld1_f32(mask, src0 + 3), dst);
            sum4 = svmla_f32_m(mask, sum4, svld1_f32(mask, src1 + 0), dst);
            sum5 = svmla_f32_m(mask, sum5, svld1_f32(mask, src1 + 1), dst);
            sum6 = svmla_f32_m(mask, sum6, svld1_f32(mask, src1 + 2), dst);
            sum7 = svmla_f32_m(mask, sum7, svld1_f32(mask, src1 + 3), dst);
            sum8 = svmla_f32_m(mask, sum8, svld1_f32(mask, src2 + 0), dst);
            sum9 = svmla_f32_m(mask, sum9, svld1_f32(mask, src2 + 1), dst);
            sum10 = svmla_f32_m(mask, sum10, svld1_f32(mask, src2 + 2), dst);
            sum11 = svmla_f32_m(mask, sum11, svld1_f32(mask, src2 + 3), dst);
            sum12 = svmla_f32_m(mask, sum12, svld1_f32(mask, src3 + 0), dst);
            sum13 = svmla_f32_m(mask, sum13, svld1_f32(mask, src3 + 1), dst);
            sum14 = svmla_f32_m(mask, sum14, svld1_f32(mask, src3 + 2), dst);
            sum15 = svmla_f32_m(mask, sum15, svld1_f32(mask, src3 + 3), dst);
        }

        void NeuralAddConvolution4x4Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);
            svfloat32_t sum4 = svdup_n_f32(0.0f);
            svfloat32_t sum5 = svdup_n_f32(0.0f);
            svfloat32_t sum6 = svdup_n_f32(0.0f);
            svfloat32_t sum7 = svdup_n_f32(0.0f);
            svfloat32_t sum8 = svdup_n_f32(0.0f);
            svfloat32_t sum9 = svdup_n_f32(0.0f);
            svfloat32_t sum10 = svdup_n_f32(0.0f);
            svfloat32_t sum11 = svdup_n_f32(0.0f);
            svfloat32_t sum12 = svdup_n_f32(0.0f);
            svfloat32_t sum13 = svdup_n_f32(0.0f);
            svfloat32_t sum14 = svdup_n_f32(0.0f);
            svfloat32_t sum15 = svdup_n_f32(0.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution4x4Sum(body, src + col + 0 * F, srcStride, svld1_f32(body, dst + col + 0 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                    AddConvolution4x4Sum(body, src + col + 1 * F, srcStride, svld1_f32(body, dst + col + 1 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                    AddConvolution4x4Sum(body, src + col + 2 * F, srcStride, svld1_f32(body, dst + col + 2 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                    AddConvolution4x4Sum(body, src + col + 3 * F, srcStride, svld1_f32(body, dst + col + 3 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                }
                for (; col + F <= width; col += F)
                    AddConvolution4x4Sum(body, src + col, srcStride, svld1_f32(body, dst + col),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                if (col < width)
                {
                    svbool_t tail = svwhilelt_b32(col, width);
                    AddConvolution4x4Sum(tail, src + col, srcStride, svld1_f32(tail, dst + col),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15);
                }

                src += srcStride;
                dst += dstStride;
            }

            sums[0] += svaddv_f32(body, sum0);
            sums[1] += svaddv_f32(body, sum1);
            sums[2] += svaddv_f32(body, sum2);
            sums[3] += svaddv_f32(body, sum3);
            sums[4] += svaddv_f32(body, sum4);
            sums[5] += svaddv_f32(body, sum5);
            sums[6] += svaddv_f32(body, sum6);
            sums[7] += svaddv_f32(body, sum7);
            sums[8] += svaddv_f32(body, sum8);
            sums[9] += svaddv_f32(body, sum9);
            sums[10] += svaddv_f32(body, sum10);
            sums[11] += svaddv_f32(body, sum11);
            sums[12] += svaddv_f32(body, sum12);
            sums[13] += svaddv_f32(body, sum13);
            sums[14] += svaddv_f32(body, sum14);
            sums[15] += svaddv_f32(body, sum15);
        }
    }
#endif
}

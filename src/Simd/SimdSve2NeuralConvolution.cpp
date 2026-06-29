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

        SIMD_INLINE svfloat32_t Convolution5x5Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3, const svfloat32_t& weight4,
            const svfloat32_t& weight5, const svfloat32_t& weight6, const svfloat32_t& weight7, const svfloat32_t& weight8, const svfloat32_t& weight9,
            const svfloat32_t& weight10, const svfloat32_t& weight11, const svfloat32_t& weight12, const svfloat32_t& weight13, const svfloat32_t& weight14,
            const svfloat32_t& weight15, const svfloat32_t& weight16, const svfloat32_t& weight17, const svfloat32_t& weight18, const svfloat32_t& weight19,
            const svfloat32_t& weight20, const svfloat32_t& weight21, const svfloat32_t& weight22, const svfloat32_t& weight23, const svfloat32_t& weight24)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            const float* src3 = src + 3 * stride;
            const float* src4 = src + 4 * stride;
            svfloat32_t sum = svmul_f32_x(mask, svld1_f32(mask, src0 + 0), weight0);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 1), weight1);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 2), weight2);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 3), weight3);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src0 + 4), weight4);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 0), weight5);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 1), weight6);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 2), weight7);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 3), weight8);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src1 + 4), weight9);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 0), weight10);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 1), weight11);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 2), weight12);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 3), weight13);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src2 + 4), weight14);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 0), weight15);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 1), weight16);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 2), weight17);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 3), weight18);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src3 + 4), weight19);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src4 + 0), weight20);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src4 + 1), weight21);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src4 + 2), weight22);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src4 + 3), weight23);
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src4 + 4), weight24);
            return sum;
        }

        SIMD_INLINE void AddConvolution5x5Forward(const svbool_t& mask, const float* src, size_t stride,
            const svfloat32_t& weight0, const svfloat32_t& weight1, const svfloat32_t& weight2, const svfloat32_t& weight3, const svfloat32_t& weight4,
            const svfloat32_t& weight5, const svfloat32_t& weight6, const svfloat32_t& weight7, const svfloat32_t& weight8, const svfloat32_t& weight9,
            const svfloat32_t& weight10, const svfloat32_t& weight11, const svfloat32_t& weight12, const svfloat32_t& weight13, const svfloat32_t& weight14,
            const svfloat32_t& weight15, const svfloat32_t& weight16, const svfloat32_t& weight17, const svfloat32_t& weight18, const svfloat32_t& weight19,
            const svfloat32_t& weight20, const svfloat32_t& weight21, const svfloat32_t& weight22, const svfloat32_t& weight23, const svfloat32_t& weight24, float* dst)
        {
            svfloat32_t sum = Convolution5x5Forward(mask, src, stride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24);
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), sum));
        }

        void NeuralAddConvolution5x5Forward(const float* src, size_t srcStride, size_t width, size_t height, const float* weights, float* dst, size_t dstStride)
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
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution5x5Forward(body, src + col + 0 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col + 0 * F);
                    AddConvolution5x5Forward(body, src + col + 1 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col + 1 * F);
                    AddConvolution5x5Forward(body, src + col + 2 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col + 2 * F);
                    AddConvolution5x5Forward(body, src + col + 3 * F, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col + 3 * F);
                }
                for (; col + F <= width; col += F)
                    AddConvolution5x5Forward(body, src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col);
                if (col < width)
                    AddConvolution5x5Forward(svwhilelt_b32(col, width), src + col, srcStride, weight0, weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9,
                        weight10, weight11, weight12, weight13, weight14, weight15, weight16, weight17, weight18, weight19, weight20, weight21, weight22, weight23, weight24, dst + col);

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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddConvolution5x5Sum(const svbool_t& mask, const float* src, size_t stride, const svfloat32_t& dst,
            svfloat32_t& sum0, svfloat32_t& sum1, svfloat32_t& sum2, svfloat32_t& sum3, svfloat32_t& sum4,
            svfloat32_t& sum5, svfloat32_t& sum6, svfloat32_t& sum7, svfloat32_t& sum8, svfloat32_t& sum9,
            svfloat32_t& sum10, svfloat32_t& sum11, svfloat32_t& sum12, svfloat32_t& sum13, svfloat32_t& sum14,
            svfloat32_t& sum15, svfloat32_t& sum16, svfloat32_t& sum17, svfloat32_t& sum18, svfloat32_t& sum19,
            svfloat32_t& sum20, svfloat32_t& sum21, svfloat32_t& sum22, svfloat32_t& sum23, svfloat32_t& sum24)
        {
            const float* src0 = src;
            const float* src1 = src + stride;
            const float* src2 = src + 2 * stride;
            const float* src3 = src + 3 * stride;
            const float* src4 = src + 4 * stride;
            sum0 = svmla_f32_m(mask, sum0, svld1_f32(mask, src0 + 0), dst);
            sum1 = svmla_f32_m(mask, sum1, svld1_f32(mask, src0 + 1), dst);
            sum2 = svmla_f32_m(mask, sum2, svld1_f32(mask, src0 + 2), dst);
            sum3 = svmla_f32_m(mask, sum3, svld1_f32(mask, src0 + 3), dst);
            sum4 = svmla_f32_m(mask, sum4, svld1_f32(mask, src0 + 4), dst);
            sum5 = svmla_f32_m(mask, sum5, svld1_f32(mask, src1 + 0), dst);
            sum6 = svmla_f32_m(mask, sum6, svld1_f32(mask, src1 + 1), dst);
            sum7 = svmla_f32_m(mask, sum7, svld1_f32(mask, src1 + 2), dst);
            sum8 = svmla_f32_m(mask, sum8, svld1_f32(mask, src1 + 3), dst);
            sum9 = svmla_f32_m(mask, sum9, svld1_f32(mask, src1 + 4), dst);
            sum10 = svmla_f32_m(mask, sum10, svld1_f32(mask, src2 + 0), dst);
            sum11 = svmla_f32_m(mask, sum11, svld1_f32(mask, src2 + 1), dst);
            sum12 = svmla_f32_m(mask, sum12, svld1_f32(mask, src2 + 2), dst);
            sum13 = svmla_f32_m(mask, sum13, svld1_f32(mask, src2 + 3), dst);
            sum14 = svmla_f32_m(mask, sum14, svld1_f32(mask, src2 + 4), dst);
            sum15 = svmla_f32_m(mask, sum15, svld1_f32(mask, src3 + 0), dst);
            sum16 = svmla_f32_m(mask, sum16, svld1_f32(mask, src3 + 1), dst);
            sum17 = svmla_f32_m(mask, sum17, svld1_f32(mask, src3 + 2), dst);
            sum18 = svmla_f32_m(mask, sum18, svld1_f32(mask, src3 + 3), dst);
            sum19 = svmla_f32_m(mask, sum19, svld1_f32(mask, src3 + 4), dst);
            sum20 = svmla_f32_m(mask, sum20, svld1_f32(mask, src4 + 0), dst);
            sum21 = svmla_f32_m(mask, sum21, svld1_f32(mask, src4 + 1), dst);
            sum22 = svmla_f32_m(mask, sum22, svld1_f32(mask, src4 + 2), dst);
            sum23 = svmla_f32_m(mask, sum23, svld1_f32(mask, src4 + 3), dst);
            sum24 = svmla_f32_m(mask, sum24, svld1_f32(mask, src4 + 4), dst);
        }

        void NeuralAddConvolution5x5Sum(const float* src, size_t srcStride, const float* dst, size_t dstStride, size_t width, size_t height, float* sums)
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
            svfloat32_t sum16 = svdup_n_f32(0.0f);
            svfloat32_t sum17 = svdup_n_f32(0.0f);
            svfloat32_t sum18 = svdup_n_f32(0.0f);
            svfloat32_t sum19 = svdup_n_f32(0.0f);
            svfloat32_t sum20 = svdup_n_f32(0.0f);
            svfloat32_t sum21 = svdup_n_f32(0.0f);
            svfloat32_t sum22 = svdup_n_f32(0.0f);
            svfloat32_t sum23 = svdup_n_f32(0.0f);
            svfloat32_t sum24 = svdup_n_f32(0.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + QF <= width; col += QF)
                {
                    AddConvolution5x5Sum(body, src + col + 0 * F, srcStride, svld1_f32(body, dst + col + 0 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
                    AddConvolution5x5Sum(body, src + col + 1 * F, srcStride, svld1_f32(body, dst + col + 1 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
                    AddConvolution5x5Sum(body, src + col + 2 * F, srcStride, svld1_f32(body, dst + col + 2 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
                    AddConvolution5x5Sum(body, src + col + 3 * F, srcStride, svld1_f32(body, dst + col + 3 * F),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
                }
                for (; col + F <= width; col += F)
                    AddConvolution5x5Sum(body, src + col, srcStride, svld1_f32(body, dst + col),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
                if (col < width)
                {
                    svbool_t tail = svwhilelt_b32(col, width);
                    AddConvolution5x5Sum(tail, src + col, srcStride, svld1_f32(tail, dst + col),
                        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11, sum12, sum13, sum14,
                        sum15, sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sum24);
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
            sums[16] += svaddv_f32(body, sum16);
            sums[17] += svaddv_f32(body, sum17);
            sums[18] += svaddv_f32(body, sum18);
            sums[19] += svaddv_f32(body, sum19);
            sums[20] += svaddv_f32(body, sum20);
            sums[21] += svaddv_f32(body, sum21);
            sums[22] += svaddv_f32(body, sum22);
            sums[23] += svaddv_f32(body, sum23);
            sums[24] += svaddv_f32(body, sum24);
        }

        //-------------------------------------------------------------------------------------------------

        namespace Ncf
        {
            SIMD_INLINE void Add4ExtractedSums(const svfloat32_t& s0, const svfloat32_t& s1, const svfloat32_t& s2, const svfloat32_t& s3, float* dst)
            {
                const svbool_t body = svptrue_b32();
                dst[0] += svaddv_f32(body, s0);
                dst[1] += svaddv_f32(body, s1);
                dst[2] += svaddv_f32(body, s2);
                dst[3] += svaddv_f32(body, s3);
            }

            namespace Ver0
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY,
                    size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (dilationX * dilationY * strideX * strideY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow * strideY - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol * strideX - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow * dilationY;
                                        if (srcRow < srcHeight)
                                        {
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol * dilationX;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (kernelX * kernelY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow;
                                        if (srcRow < srcHeight)
                                        {
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < N; ++i)
                        {
                            for (size_t k = 0; k < K; ++k)
                                *(dst++) = src[k * N + i];
                        }
                    }
                }

                static SIMD_INLINE void Kernel1x4x4(const svfloat32_t& a, size_t K, const float* b,
                    svfloat32_t& s0, svfloat32_t& s1, svfloat32_t& s2, svfloat32_t& s3, const svbool_t& mask)
                {
                    s0 = svmla_f32_m(mask, s0, a, svld1_f32(mask, b + 0 * K));
                    s1 = svmla_f32_m(mask, s1, a, svld1_f32(mask, b + 1 * K));
                    s2 = svmla_f32_m(mask, s2, a, svld1_f32(mask, b + 2 * K));
                    s3 = svmla_f32_m(mask, s3, a, svld1_f32(mask, b + 3 * K));
                }

                static SIMD_INLINE void Kernel1x1x4(const svfloat32_t& a, const float* b, svfloat32_t& sum, const svbool_t& mask)
                {
                    sum = svmla_f32_m(mask, sum, a, svld1_f32(mask, b));
                }

                static SIMD_INLINE void Kernel3x4x4(const svfloat32_t& a0, const svfloat32_t& a1, const svfloat32_t& a2, size_t K, const float* b,
                    svfloat32_t& s00, svfloat32_t& s01, svfloat32_t& s02, svfloat32_t& s03,
                    svfloat32_t& s04, svfloat32_t& s05, svfloat32_t& s06, svfloat32_t& s07,
                    svfloat32_t& s08, svfloat32_t& s09, svfloat32_t& s0A, svfloat32_t& s0B, const svbool_t& mask)
                {
                    svfloat32_t _b;
                    _b = svld1_f32(mask, b + 0 * K);
                    s00 = svmla_f32_m(mask, s00, a0, _b);
                    s04 = svmla_f32_m(mask, s04, a1, _b);
                    s08 = svmla_f32_m(mask, s08, a2, _b);
                    _b = svld1_f32(mask, b + 1 * K);
                    s01 = svmla_f32_m(mask, s01, a0, _b);
                    s05 = svmla_f32_m(mask, s05, a1, _b);
                    s09 = svmla_f32_m(mask, s09, a2, _b);
                    _b = svld1_f32(mask, b + 2 * K);
                    s02 = svmla_f32_m(mask, s02, a0, _b);
                    s06 = svmla_f32_m(mask, s06, a1, _b);
                    s0A = svmla_f32_m(mask, s0A, a2, _b);
                    _b = svld1_f32(mask, b + 3 * K);
                    s03 = svmla_f32_m(mask, s03, a0, _b);
                    s07 = svmla_f32_m(mask, s07, a1, _b);
                    s0B = svmla_f32_m(mask, s0B, a2, _b);
                }

                static SIMD_INLINE void Kernel3x1x4(const svfloat32_t& a0, const svfloat32_t& a1, const svfloat32_t& a2, const float* b,
                    svfloat32_t& s0, svfloat32_t& s1, svfloat32_t& s2, const svbool_t& mask)
                {
                    svfloat32_t _b = svld1_f32(mask, b);
                    s0 = svmla_f32_m(mask, s0, a0, _b);
                    s1 = svmla_f32_m(mask, s1, a1, _b);
                    s2 = svmla_f32_m(mask, s2, a2, _b);
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t F = svcntw();
                    const svbool_t body = svptrue_b32();
                    const svfloat32_t zero = svdup_n_f32(0.0f);
                    size_t M3 = M / 3 * 3;
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t KF = Simd::AlignLo(K, F);
                    size_t i = 0;
                    for (; i < M3; i += 3)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            svfloat32_t s00 = zero, s01 = zero, s02 = zero, s03 = zero;
                            svfloat32_t s04 = zero, s05 = zero, s06 = zero, s07 = zero;
                            svfloat32_t s08 = zero, s09 = zero, s0A = zero, s0B = zero;
                            for (size_t k = 0; k < KF; k += F)
                            {
                                svfloat32_t a0 = svld1_f32(body, pa + k + 0 * K);
                                svfloat32_t a1 = svld1_f32(body, pa + k + 1 * K);
                                svfloat32_t a2 = svld1_f32(body, pa + k + 2 * K);
                                Kernel3x4x4(a0, a1, a2, K, pb + k, s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s0A, s0B, body);
                            }
                            if (KF < K)
                            {
                                size_t k = K - F;
                                const svbool_t tail = svwhilelt_b32(k, K);
                                svfloat32_t a0 = svld1_f32(tail, pa + k + 0 * K);
                                svfloat32_t a1 = svld1_f32(tail, pa + k + 1 * K);
                                svfloat32_t a2 = svld1_f32(tail, pa + k + 2 * K);
                                Kernel3x4x4(a0, a1, a2, K, pb + k, s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s0A, s0B, tail);
                            }
                            Add4ExtractedSums(s00, s01, s02, s03, pc + j + 0 * N);
                            Add4ExtractedSums(s04, s05, s06, s07, pc + j + 1 * N);
                            Add4ExtractedSums(s08, s09, s0A, s0B, pc + j + 2 * N);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            svfloat32_t s0 = zero, s1 = zero, s2 = zero;
                            for (size_t k = 0; k < KF; k += F)
                            {
                                svfloat32_t a0 = svld1_f32(body, pa + k + 0 * K);
                                svfloat32_t a1 = svld1_f32(body, pa + k + 1 * K);
                                svfloat32_t a2 = svld1_f32(body, pa + k + 2 * K);
                                Kernel3x1x4(a0, a1, a2, pb + k, s0, s1, s2, body);
                            }
                            if (KF < K)
                            {
                                size_t k = K - F;
                                const svbool_t tail = svwhilelt_b32(k, K);
                                svfloat32_t a0 = svld1_f32(tail, pa + k + 0 * K);
                                svfloat32_t a1 = svld1_f32(tail, pa + k + 1 * K);
                                svfloat32_t a2 = svld1_f32(tail, pa + k + 2 * K);
                                Kernel3x1x4(a0, a1, a2, pb + k, s0, s1, s2, tail);
                            }
                            pc[j + 0 * N] += svaddv_f32(body, s0);
                            pc[j + 1 * N] += svaddv_f32(body, s1);
                            pc[j + 2 * N] += svaddv_f32(body, s2);
                        }
                    }
                    for (; i < M; ++i)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            svfloat32_t s0 = zero, s1 = zero, s2 = zero, s3 = zero;
                            for (size_t k = 0; k < KF; k += F)
                            {
                                svfloat32_t _a = svld1_f32(body, pa + k);
                                Kernel1x4x4(_a, K, pb + k, s0, s1, s2, s3, body);
                            }
                            if (KF < K)
                            {
                                size_t k = K - F;
                                const svbool_t tail = svwhilelt_b32(k, K);
                                svfloat32_t _a = svld1_f32(tail, pa + k);
                                Kernel1x4x4(_a, K, pb + k, s0, s1, s2, s3, tail);
                            }
                            Add4ExtractedSums(s0, s1, s2, s3, pc + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            svfloat32_t sum = zero;
                            for (size_t k = 0; k < KF; k += F)
                            {
                                svfloat32_t _a = svld1_f32(body, pa + k);
                                Kernel1x1x4(_a, pb + k, sum, body);
                            }
                            if (KF < K)
                            {
                                size_t k = K - F;
                                const svbool_t tail = svwhilelt_b32(k, K);
                                svfloat32_t _a = svld1_f32(tail, pa + k);
                                Kernel1x1x4(_a, pb + k, sum, tail);
                            }
                            pc[j] += svaddv_f32(body, sum);
                        }
                    }
                }
            }

            namespace Ver1
            {
                void PrepareA(const float* src, size_t M, size_t K, size_t cell, float* dst)
                {
                    for (size_t i = 0; i < M; i += cell)
                    {
                        size_t n = Simd::Min(cell, M - i);
                        for (size_t k = 0; k < K; ++k)
                        {
                            for (size_t c = 0; c < n; ++c)
                                *(dst++) = src[c * K + k];
                        }
                        src += cell * K;
                    }
                }

                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY,
                    size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t cell, float* tmp, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (kernelX * kernelY != 1)
                    {
                        float* dst = tmp;
                        size_t channelSize = srcHeight * srcWidth;
                        if (dilationX * dilationY * strideX * strideY != 1)
                        {
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow * dilationY - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol * dilationX - padX;
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = src[srcRow * srcWidth + srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                    srcCol += strideX;
                                                }
                                            }
                                            else
                                            {
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                    *(dst++) = 0;
                                            }
                                            srcRow += strideY;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            const size_t bodySize = dstWidth - padX * 2;
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow, ++srcRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol - padX, dstCol = 0;
                                                const float* psrc = src + srcRow * srcWidth;
                                                for (; dstCol < padX; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                                memcpy(dst, psrc + srcCol, bodySize * 4);
                                                dst += bodySize;
                                                dstCol += bodySize;
                                                srcCol += bodySize;
                                                for (; dstCol < dstWidth; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                            }
                                            else
                                            {
                                                memset(dst, 0, dstWidth * 4);
                                                dst += dstWidth;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        src = tmp;
                    }
                    size_t F = svcntw();
                    if (cell == 2 * F)
                    {
                        const svbool_t body = svptrue_b32();
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    svst1_f32(body, dst + 0, svld1_f32(body, psrc + 0));
                                    svst1_f32(body, dst + F, svld1_f32(body, psrc + F));
                                    dst += 2 * F;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            for (size_t k = 0; k < K; ++k)
                            {
                                const float* psrc = src + k * N;
                                size_t c = 0;
                                for (; c < n; ++c)
                                    *(dst++) = *(psrc++);
                                for (; c < cell; ++c)
                                    *(dst++) = 0;
                            }
                            src += cell;
                        }
                    }
                }

                SIMD_INLINE void AddSum(const svbool_t& mask, const svfloat32_t& sum, float* dst)
                {
                    svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), sum));
                }

                SIMD_INLINE void AddSums2FRow(const svfloat32_t& s0, const svfloat32_t& s4, size_t nValid, float* dst)
                {
                    size_t F = svcntw();
                    const svbool_t body = svptrue_b32();
                    svbool_t mask0 = nValid >= F ? body : svwhilelt_b32(0 * F, nValid);
                    svbool_t mask1 = nValid > F ? (nValid >= 2 * F ? body : svwhilelt_b32(F, nValid)) : svpfalse_b();
                    AddSum(mask0, s0, dst + 0);
                    AddSum(mask1, s4, dst + F);
                }

                SIMD_INLINE void KernelMx2F(size_t N, size_t K, const float* a, const float* b, float* c, size_t nValid, size_t m)
                {
                    size_t F = svcntw();
                    const svbool_t body = svptrue_b32();
                    const svfloat32_t zero = svdup_n_f32(0.0f);
                    svfloat32_t s0 = zero, s1 = zero, s2 = zero, s3 = zero;
                    svfloat32_t s4 = zero, s5 = zero, s6 = zero, s7 = zero;
                    for (size_t k = 0; k < K; ++k)
                    {
                        svfloat32_t b0 = svld1_f32(body, b + 0);
                        svfloat32_t b1 = svld1_f32(body, b + F);
                        if (m > 0)
                        {
                            svfloat32_t a0 = svdup_n_f32(a[0]);
                            s0 = svmla_f32_m(body, s0, a0, b0);
                            s4 = svmla_f32_m(body, s4, a0, b1);
                        }
                        if (m > 1)
                        {
                            svfloat32_t a0 = svdup_n_f32(a[1]);
                            s1 = svmla_f32_m(body, s1, a0, b0);
                            s5 = svmla_f32_m(body, s5, a0, b1);
                        }
                        if (m > 2)
                        {
                            svfloat32_t a0 = svdup_n_f32(a[2]);
                            s2 = svmla_f32_m(body, s2, a0, b0);
                            s6 = svmla_f32_m(body, s6, a0, b1);
                        }
                        if (m > 3)
                        {
                            svfloat32_t a0 = svdup_n_f32(a[3]);
                            s3 = svmla_f32_m(body, s3, a0, b0);
                            s7 = svmla_f32_m(body, s7, a0, b1);
                        }
                        b += 2 * F;
                        a += m;
                    }
                    if (m > 0) AddSums2FRow(s0, s4, nValid, c + 0 * N);
                    if (m > 1) AddSums2FRow(s1, s5, nValid, c + 1 * N);
                    if (m > 2) AddSums2FRow(s2, s6, nValid, c + 2 * N);
                    if (m > 3) AddSums2FRow(s3, s7, nValid, c + 3 * N);
                }

                SIMD_INLINE void Kernel4x2F(size_t N, size_t K, const float* a, const float* b, float* c, size_t nValid)
                {
                    size_t F = svcntw();
                    const svbool_t body = svptrue_b32();
                    const svfloat32_t zero = svdup_n_f32(0.0f);
                    svfloat32_t s0 = zero, s1 = zero, s2 = zero, s3 = zero;
                    svfloat32_t s4 = zero, s5 = zero, s6 = zero, s7 = zero;
                    for (size_t k = 0; k < K; ++k)
                    {
                        svfloat32_t b0 = svld1_f32(body, b + 0);
                        svfloat32_t b1 = svld1_f32(body, b + F);
                        s0 = svmla_f32_m(body, s0, svdup_n_f32(a[0]), b0);
                        s4 = svmla_f32_m(body, s4, svdup_n_f32(a[0]), b1);
                        s1 = svmla_f32_m(body, s1, svdup_n_f32(a[1]), b0);
                        s5 = svmla_f32_m(body, s5, svdup_n_f32(a[1]), b1);
                        s2 = svmla_f32_m(body, s2, svdup_n_f32(a[2]), b0);
                        s6 = svmla_f32_m(body, s6, svdup_n_f32(a[2]), b1);
                        s3 = svmla_f32_m(body, s3, svdup_n_f32(a[3]), b0);
                        s7 = svmla_f32_m(body, s7, svdup_n_f32(a[3]), b1);
                        b += 2 * F;
                        a += 4;
                    }
                    AddSums2FRow(s0, s4, nValid, c + 0 * N);
                    AddSums2FRow(s1, s5, nValid, c + 1 * N);
                    AddSums2FRow(s2, s6, nValid, c + 2 * N);
                    AddSums2FRow(s3, s7, nValid, c + 3 * N);
                }

                void Execute4x2F(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t F = svcntw();
                    size_t cellB = 2 * F;
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t NB = Simd::AlignLo(N, cellB);
                    size_t nTail = N - NB;
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < NB; j += cellB)
                            Kernel4x2F(N, K, a + i * K, b + j * K, c + i * N + j, cellB);
                        if (NB < N)
                            Kernel4x2F(N, K, a + i * K, b + j * K, c + i * N + j, nTail);
                    }
                    if (M4 < M)
                    {
                        size_t j = 0;
                        for (; j < NB; j += cellB)
                            KernelMx2F(N, K, a + i * K, b + j * K, c + i * N + j, cellB, M - M4);
                        if (NB < N)
                            KernelMx2F(N, K, a + i * K, b + j * K, c + i * N + j, nTail, M - M4);
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c, size_t cellA, size_t cellB)
                {
                    size_t F = svcntw();
                    if (cellA == 4 && cellB == 2 * F)
                        Execute4x2F(M, N, K, a, b, c);
                }
            }

            namespace Ver2
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t padX, size_t padY, float* dst, size_t dstWidth, size_t dstHeight)
                {
                    for (size_t channel = 0; channel < srcDepth; ++channel)
                    {
                        const float* s = src;
                        float* d = dst;
                        memset(d, 0, padY * dstWidth * 4);
                        d += padY * dstWidth;
                        for (size_t row = padY; row < dstHeight - padY; ++row)
                        {
                            memset(d, 0, padX * 4);
                            memcpy(d + padX, s, srcWidth * 4);
                            memset(d + padX + srcWidth, 0, padX * 4);
                            d += dstWidth;
                            s += srcWidth;
                        }
                        memset(d, 0, padY * dstWidth * 4);
                        src += srcWidth * srcHeight;
                        dst += dstWidth * dstHeight;
                    }
                }

                template<size_t kernelX, size_t kernelY> void AddConvolution(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth);

                template<> void AddConvolution<2, 2>(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    size_t F = svcntw();
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    const svbool_t body = svptrue_b32();
                    for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                    {
                        for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * 4;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            svfloat32_t w0 = svdup_n_f32(pweight[0]);
                            svfloat32_t w1 = svdup_n_f32(pweight[1]);
                            svfloat32_t w2 = svdup_n_f32(pweight[2]);
                            svfloat32_t w3 = svdup_n_f32(pweight[3]);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    svfloat32_t _dst = svld1_f32(body, pdst + col);
                                    _dst = svadd_f32_x(body, _dst, Convolution2x2Forward(body, psrc + col, srcWidth, w0, w1, w2, w3));
                                    svst1_f32(body, pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    svbool_t tail = svwhilelt_b32(col, dstWidth);
                                    svfloat32_t _dst = svld1_f32(tail, pdst + col);
                                    _dst = svadd_f32_x(tail, _dst, Convolution2x2Forward(tail, psrc + col, srcWidth, w0, w1, w2, w3));
                                    svst1_f32(tail, pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                template<> void AddConvolution<3, 3>(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    size_t F = svcntw();
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    const svbool_t body = svptrue_b32();
                    for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                    {
                        for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * 9;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            svfloat32_t w0 = svdup_n_f32(pweight[0]);
                            svfloat32_t w1 = svdup_n_f32(pweight[1]);
                            svfloat32_t w2 = svdup_n_f32(pweight[2]);
                            svfloat32_t w3 = svdup_n_f32(pweight[3]);
                            svfloat32_t w4 = svdup_n_f32(pweight[4]);
                            svfloat32_t w5 = svdup_n_f32(pweight[5]);
                            svfloat32_t w6 = svdup_n_f32(pweight[6]);
                            svfloat32_t w7 = svdup_n_f32(pweight[7]);
                            svfloat32_t w8 = svdup_n_f32(pweight[8]);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    svfloat32_t _dst = svld1_f32(body, pdst + col);
                                    _dst = svadd_f32_x(body, _dst, Convolution3x3Forward(body, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8));
                                    svst1_f32(body, pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    svbool_t tail = svwhilelt_b32(col, dstWidth);
                                    svfloat32_t _dst = svld1_f32(tail, pdst + col);
                                    _dst = svadd_f32_x(tail, _dst, Convolution3x3Forward(tail, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8));
                                    svst1_f32(tail, pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                template<> void AddConvolution<4, 4>(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    size_t F = svcntw();
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    const svbool_t body = svptrue_b32();
                    for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                    {
                        for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * 16;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            svfloat32_t w0 = svdup_n_f32(pweight[0]);
                            svfloat32_t w1 = svdup_n_f32(pweight[1]);
                            svfloat32_t w2 = svdup_n_f32(pweight[2]);
                            svfloat32_t w3 = svdup_n_f32(pweight[3]);
                            svfloat32_t w4 = svdup_n_f32(pweight[4]);
                            svfloat32_t w5 = svdup_n_f32(pweight[5]);
                            svfloat32_t w6 = svdup_n_f32(pweight[6]);
                            svfloat32_t w7 = svdup_n_f32(pweight[7]);
                            svfloat32_t w8 = svdup_n_f32(pweight[8]);
                            svfloat32_t w9 = svdup_n_f32(pweight[9]);
                            svfloat32_t w10 = svdup_n_f32(pweight[10]);
                            svfloat32_t w11 = svdup_n_f32(pweight[11]);
                            svfloat32_t w12 = svdup_n_f32(pweight[12]);
                            svfloat32_t w13 = svdup_n_f32(pweight[13]);
                            svfloat32_t w14 = svdup_n_f32(pweight[14]);
                            svfloat32_t w15 = svdup_n_f32(pweight[15]);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    svfloat32_t _dst = svld1_f32(body, pdst + col);
                                    _dst = svadd_f32_x(body, _dst, Convolution4x4Forward(body, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15));
                                    svst1_f32(body, pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    svbool_t tail = svwhilelt_b32(col, dstWidth);
                                    svfloat32_t _dst = svld1_f32(tail, pdst + col);
                                    _dst = svadd_f32_x(tail, _dst, Convolution4x4Forward(tail, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15));
                                    svst1_f32(tail, pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                template<> void AddConvolution<5, 5>(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    size_t F = svcntw();
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    const svbool_t body = svptrue_b32();
                    for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                    {
                        for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * 25;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            svfloat32_t w0 = svdup_n_f32(pweight[0]);
                            svfloat32_t w1 = svdup_n_f32(pweight[1]);
                            svfloat32_t w2 = svdup_n_f32(pweight[2]);
                            svfloat32_t w3 = svdup_n_f32(pweight[3]);
                            svfloat32_t w4 = svdup_n_f32(pweight[4]);
                            svfloat32_t w5 = svdup_n_f32(pweight[5]);
                            svfloat32_t w6 = svdup_n_f32(pweight[6]);
                            svfloat32_t w7 = svdup_n_f32(pweight[7]);
                            svfloat32_t w8 = svdup_n_f32(pweight[8]);
                            svfloat32_t w9 = svdup_n_f32(pweight[9]);
                            svfloat32_t w10 = svdup_n_f32(pweight[10]);
                            svfloat32_t w11 = svdup_n_f32(pweight[11]);
                            svfloat32_t w12 = svdup_n_f32(pweight[12]);
                            svfloat32_t w13 = svdup_n_f32(pweight[13]);
                            svfloat32_t w14 = svdup_n_f32(pweight[14]);
                            svfloat32_t w15 = svdup_n_f32(pweight[15]);
                            svfloat32_t w16 = svdup_n_f32(pweight[16]);
                            svfloat32_t w17 = svdup_n_f32(pweight[17]);
                            svfloat32_t w18 = svdup_n_f32(pweight[18]);
                            svfloat32_t w19 = svdup_n_f32(pweight[19]);
                            svfloat32_t w20 = svdup_n_f32(pweight[20]);
                            svfloat32_t w21 = svdup_n_f32(pweight[21]);
                            svfloat32_t w22 = svdup_n_f32(pweight[22]);
                            svfloat32_t w23 = svdup_n_f32(pweight[23]);
                            svfloat32_t w24 = svdup_n_f32(pweight[24]);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    svfloat32_t _dst = svld1_f32(body, pdst + col);
                                    _dst = svadd_f32_x(body, _dst, Convolution5x5Forward(body, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14,
                                        w15, w16, w17, w18, w19, w20, w21, w22, w23, w24));
                                    svst1_f32(body, pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    svbool_t tail = svwhilelt_b32(col, dstWidth);
                                    svfloat32_t _dst = svld1_f32(tail, pdst + col);
                                    _dst = svadd_f32_x(tail, _dst, Convolution5x5Forward(tail, psrc + col, srcWidth, w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14,
                                        w15, w16, w17, w18, w19, w20, w21, w22, w23, w24));
                                    svst1_f32(tail, pdst + col, _dst);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                void Execute(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, size_t kernelX, size_t kernelY, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    assert(kernelX == kernelY);
                    if (kernelX == 2)
                        AddConvolution<2, 2>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 3)
                        AddConvolution<3, 3>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 4)
                        AddConvolution<4, 4>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 5)
                        AddConvolution<5, 5>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else
                        assert(0);
                }

                bool Preferable(size_t /*srcDepth*/, size_t kernelX, size_t kernelY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t /*dstDepth*/)
                {
                    if (kernelX == kernelY && kernelX >= 2 && kernelX <= 5 && strideX * strideY * dilationX * dilationY == 1)
                    {
                        if (dstWidth * dstHeight * kernelX * kernelY >= 8 * 8 * 5 * 5)
                            return true;
                    }
                    return false;
                }
            }

            struct Opt
            {
                enum Alg
                {
                    None,
                    Ver0,
                    Ver1,
                    Ver2,
                } alg;

                size_t sizeA;
                size_t sizeB;
                size_t sizeT;

                size_t cellA;
                size_t cellB;

                size_t M, N, K;
                size_t strideB;
                size_t paddedW;
                size_t paddedH;

                Opt(size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    alg = None;
                    sizeA = 0;
                    sizeB = 0;
                    sizeT = 0;
                    cellA = 1;
                    cellB = 1;

                    M = dstDepth;
                    N = dstHeight * dstWidth;
                    K = kernelX * kernelY * srcDepth;

                    if (dstWidth * dstHeight / kernelX <= 2000)
                        alg = Ver0;
                    else
                        alg = Ver1;
                    if (Ver2::Preferable(srcDepth, kernelX, kernelY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth))
                        alg = Ver2;

                    switch (alg)
                    {
                    case Ver0:
                        sizeB = N * K;
                        break;
                    case Ver1:
                    {
                        size_t F = svcntw();
                        cellA = 4;
                        cellB = 2 * F;
                        sizeA = M * K;
                        strideB = Simd::AlignHi(N, cellB);
                        sizeB = strideB * K;
                        if (kernelX * kernelY > 1)
                            sizeT = sizeB;
                        break;
                    }
                    case Ver2:
                        if (padX > 0 || padY > 0)
                        {
                            size_t F = svcntw();
                            paddedW = Simd::AlignHi(srcWidth + 2 * padX, F);
                            paddedH = srcHeight + 2 * padY;
                            sizeB = paddedW * paddedH * srcDepth;
                        }
                        else
                        {
                            paddedW = srcWidth;
                            paddedH = srcHeight;
                        }
                        break;
                    default:
                        assert(0);
                        break;
                    }
                }
            };

            struct Data
            {
                float* a;
                float* b;
                float* t;

                Data(size_t sizeA, size_t sizeB, size_t sizeT, void* externalData, size_t* externalSize)
                    : a(0)
                    , b(0)
                    , _data(0)
                {
                    sizeA = AlignHi(sizeA, svcntw());
                    sizeB = AlignHi(sizeB, svcntw());
                    sizeT = AlignHi(sizeT, svcntw());
                    size_t size = (sizeA + sizeB + sizeT) * sizeof(float);
                    if (size == 0)
                        return;
                    if (externalData != AlignHi(externalData, SIMD_ALIGN))
                        size += SIMD_ALIGN;
                    float* data = NULL;
                    if (externalData == NULL || externalSize == NULL || *externalSize < size)
                    {
                        _data = Simd::Allocate(size);
                        if (externalSize)
                            *externalSize = size;
                        data = (float*)_data;
                    }
                    else
                        data = (float*)AlignHi(externalData, SIMD_ALIGN);
                    if (sizeA)
                        a = data;
                    if (sizeB)
                        b = data + sizeA;
                    if (sizeT)
                        t = data + sizeA + sizeB;
                }

                ~Data()
                {
                    if (_data)
                        Simd::Free(_data);
                }

            private:
                void* _data;
            };
        }

        void NeuralConvolutionForward(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float* weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void* buffer, size_t* size, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            using namespace Ncf;

            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (!add)
                memset(dst, 0, dstWidth * dstHeight * dstDepth * sizeof(float));

            Opt opt(srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth);

            Data data(opt.sizeA, opt.sizeB, opt.sizeT, buffer, size);

            if (opt.sizeA)
            {
                switch (opt.alg)
                {
                case Opt::Ver1: Ver1::PrepareA(weight, opt.M, opt.K, opt.cellA, data.a);
                default:
                    break;
                }
            }
            else
                data.a = (float*)weight;

            if (opt.sizeB)
            {
                switch (opt.alg)
                {
                case Opt::Ver0: Ver0::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, data.b); break;
                case Opt::Ver1: Ver1::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, opt.cellB, data.t, data.b); break;
                case Opt::Ver2: Ver2::PrepareB(src, srcWidth, srcHeight, srcDepth, padX, padY, data.b, opt.paddedW, opt.paddedH); break;
                default: break;
                }
            }
            else
                data.b = (float*)src;

            switch (opt.alg)
            {
            case Opt::Ver0: Ver0::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst); break;
            case Opt::Ver1: Ver1::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst, opt.cellA, opt.cellB); break;
            case Opt::Ver2: Ver2::Execute(data.b, opt.paddedW, opt.paddedH, srcDepth, weight, kernelX, kernelY, dst, dstWidth, dstHeight, dstDepth); break;
            default: break;
            }
        }
    }
#endif
}

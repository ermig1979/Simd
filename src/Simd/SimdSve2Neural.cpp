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
#include "Simd/SimdPow.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        template <bool inversion> SIMD_INLINE svuint32_t Invert(const svbool_t& mask, const svuint32_t& value);

        template <> SIMD_INLINE svuint32_t Invert<true>(const svbool_t& mask, const svuint32_t& value)
        {
            return svsub_u32_x(mask, svdup_n_u32(0xFF), value);
        }

        template <> SIMD_INLINE svuint32_t Invert<false>(const svbool_t& mask, const svuint32_t& value)
        {
            return value;
        }

        template <bool inversion> SIMD_INLINE void Convert(const svbool_t& mask, const svuint32_t& src, const svfloat32_t& scale, float* dst)
        {
            svst1_f32(mask, dst, svmul_f32_x(mask, svcvt_f32_u32_x(mask, Invert<inversion>(mask, src)), scale));
        }

        template <bool inversion> SIMD_INLINE void Convert(const uint8_t* src, const svfloat32_t& scale, float* dst, size_t F,
            const svbool_t& mask0, const svbool_t& mask1, const svbool_t& mask2, const svbool_t& mask3)
        {
            Convert<inversion>(mask0, svld1ub_u32(mask0, src + 0 * F), scale, dst + 0 * F);
            Convert<inversion>(mask1, svld1ub_u32(mask1, src + 1 * F), scale, dst + 1 * F);
            Convert<inversion>(mask2, svld1ub_u32(mask2, src + 2 * F), scale, dst + 2 * F);
            Convert<inversion>(mask3, svld1ub_u32(mask3, src + 3 * F), scale, dst + 3 * F);
        }

        template <bool inversion> void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            size_t F = svcntw(), A = svcntb();
            const svbool_t body32 = svptrue_b32();
            const svfloat32_t scale = svdup_n_f32(1.0f / 255.0f);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col + A <= width; col += A)
                    Convert<inversion>(src + col, scale, dst + col, F, body32, body32, body32, body32);
                if (col < width)
                    Convert<inversion>(src + col, scale, dst + col, F,
                        svwhilelt_b32(col + 0 * F, width), svwhilelt_b32(col + 1 * F, width),
                        svwhilelt_b32(col + 2 * F, width), svwhilelt_b32(col + 3 * F, width));
                src += srcStride;
                dst += dstStride;
            }
        }

        void NeuralConvert(const uint8_t* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride, int inversion)
        {
            if (inversion)
                NeuralConvert<true>(src, srcStride, width, height, dst, dstStride);
            else
                NeuralConvert<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void ProductSum(const svbool_t& mask, const float* a, const float* b, svfloat32_t& sum)
        {
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, a), svld1_f32(mask, b));
        }

        void NeuralProductSum(const float* a, const float* b, size_t size, float* sum)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            svfloat32_t sum0 = svdup_n_f32(0.0f);
            svfloat32_t sum1 = svdup_n_f32(0.0f);
            svfloat32_t sum2 = svdup_n_f32(0.0f);
            svfloat32_t sum3 = svdup_n_f32(0.0f);

            for (; i + QF <= size; i += QF)
            {
                ProductSum(body, a + i + 0 * F, b + i + 0 * F, sum0);
                ProductSum(body, a + i + 1 * F, b + i + 1 * F, sum1);
                ProductSum(body, a + i + 2 * F, b + i + 2 * F, sum2);
                ProductSum(body, a + i + 3 * F, b + i + 3 * F, sum3);
            }
            for (; i + F <= size; i += F)
                ProductSum(body, a + i, b + i, sum0);
            if (i < size)
                ProductSum(svwhilelt_b32(i, size), a + i, b + i, sum0);

            sum0 = svadd_f32_x(body, sum0, sum1);
            sum2 = svadd_f32_x(body, sum2, sum3);
            *sum = svaddv_f32(body, svadd_f32_x(body, sum0, sum2));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void AddMultiplied(const svbool_t& mask, const svfloat32_t& value, const float* src, float* dst)
        {
            svst1_f32(mask, dst, svmla_f32_m(mask, svld1_f32(mask, dst), svld1_f32(mask, src), value));
        }

        void NeuralAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _value = svdup_n_f32(value[0]);

            for (; i + QF <= size; i += QF)
            {
                AddMultiplied(body, _value, src + i + 0 * F, dst + i + 0 * F);
                AddMultiplied(body, _value, src + i + 1 * F, dst + i + 1 * F);
                AddMultiplied(body, _value, src + i + 2 * F, dst + i + 2 * F);
                AddMultiplied(body, _value, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddMultiplied(body, _value, src + i, dst + i);
            if (i < size)
                AddMultiplied(svwhilelt_b32(i, size), _value, src + i, dst + i);
        }

        SIMD_INLINE void AddVector(const svbool_t& mask, const float* src, float* dst)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), svld1_f32(mask, src)));
        }

        void NeuralAddVector(const float* src, size_t size, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();

            for (; i + QF <= size; i += QF)
            {
                AddVector(body, src + i + 0 * F, dst + i + 0 * F);
                AddVector(body, src + i + 1 * F, dst + i + 1 * F);
                AddVector(body, src + i + 2 * F, dst + i + 2 * F);
                AddVector(body, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddVector(body, src + i, dst + i);
            if (i < size)
                AddVector(svwhilelt_b32(i, size), src + i, dst + i);
        }

        SIMD_INLINE void AddValue(const svbool_t& mask, const svfloat32_t& value, float* dst)
        {
            svst1_f32(mask, dst, svadd_f32_x(mask, svld1_f32(mask, dst), value));
        }

        void NeuralAddValue(const float* value, float* dst, size_t size)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _value = svdup_n_f32(value[0]);

            for (; i + QF <= size; i += QF)
            {
                AddValue(body, _value, dst + i + 0 * F);
                AddValue(body, _value, dst + i + 1 * F);
                AddValue(body, _value, dst + i + 2 * F);
                AddValue(body, _value, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                AddValue(body, _value, dst + i);
            if (i < size)
                AddValue(svwhilelt_b32(i, size), _value, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DerivativeSigmoid(const svbool_t& mask, const svfloat32_t& _1, const svfloat32_t& slope, const float* src, float* dst)
        {
            svfloat32_t _src = svld1_f32(mask, src);
            svfloat32_t _dst = svld1_f32(mask, dst);
            svst1_f32(mask, dst, svmul_f32_x(mask, svmul_f32_x(mask, _dst, slope), svmul_f32_x(mask, svsub_f32_x(mask, _1, _src), _src)));
        }

        void NeuralDerivativeSigmoid(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            const svfloat32_t _slope = svdup_n_f32(slope[0]);

            for (; i + QF <= size; i += QF)
            {
                DerivativeSigmoid(body, _1, _slope, src + i + 0 * F, dst + i + 0 * F);
                DerivativeSigmoid(body, _1, _slope, src + i + 1 * F, dst + i + 1 * F);
                DerivativeSigmoid(body, _1, _slope, src + i + 2 * F, dst + i + 2 * F);
                DerivativeSigmoid(body, _1, _slope, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                DerivativeSigmoid(body, _1, _slope, src + i, dst + i);
            if (i < size)
                DerivativeSigmoid(svwhilelt_b32(i, size), _1, _slope, src + i, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DerivativeTanh(const svbool_t& mask, const svfloat32_t& _1, const svfloat32_t& slope, const float* src, float* dst)
        {
            svfloat32_t _src = svld1_f32(mask, src);
            svfloat32_t _dst = svld1_f32(mask, dst);
            svst1_f32(mask, dst, svmul_f32_x(mask, svmul_f32_x(mask, _dst, slope), svsub_f32_x(mask, _1, svmul_f32_x(mask, _src, _src))));
        }

        void NeuralDerivativeTanh(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            const svfloat32_t _slope = svdup_n_f32(slope[0]);

            for (; i + QF <= size; i += QF)
            {
                DerivativeTanh(body, _1, _slope, src + i + 0 * F, dst + i + 0 * F);
                DerivativeTanh(body, _1, _slope, src + i + 1 * F, dst + i + 1 * F);
                DerivativeTanh(body, _1, _slope, src + i + 2 * F, dst + i + 2 * F);
                DerivativeTanh(body, _1, _slope, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                DerivativeTanh(body, _1, _slope, src + i, dst + i);
            if (i < size)
                DerivativeTanh(svwhilelt_b32(i, size), _1, _slope, src + i, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DerivativeRelu(const svbool_t& mask, const svfloat32_t& _1, const svfloat32_t& slope, const float* src, float* dst)
        {
            svfloat32_t _src = svld1_f32(mask, src);
            svfloat32_t _dst = svld1_f32(mask, dst);
            svbool_t positive = svcmpgt_n_f32(mask, _src, 0.0f);
            svst1_f32(mask, dst, svmul_f32_x(mask, svsel_f32(positive, _1, slope), _dst));
        }

        void NeuralDerivativeRelu(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            const svfloat32_t _slope = svdup_n_f32(slope[0]);

            for (; i + QF <= size; i += QF)
            {
                DerivativeRelu(body, _1, _slope, src + i + 0 * F, dst + i + 0 * F);
                DerivativeRelu(body, _1, _slope, src + i + 1 * F, dst + i + 1 * F);
                DerivativeRelu(body, _1, _slope, src + i + 2 * F, dst + i + 2 * F);
                DerivativeRelu(body, _1, _slope, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                DerivativeRelu(body, _1, _slope, src + i, dst + i);
            if (i < size)
                DerivativeRelu(svwhilelt_b32(i, size), _1, _slope, src + i, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void NeuralPow(const svbool_t& mask, const Pow& pow, const svfloat32_t& exponent, const float* src, float* dst)
        {
            svst1_f32(mask, dst, pow(mask, svld1_f32(mask, src), exponent));
        }

        void NeuralPow(const float* src, size_t size, const float* exponent, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _exponent = svdup_n_f32(exponent[0]);
            Pow pow;

            for (; i + QF <= size; i += QF)
            {
                NeuralPow(body, pow, _exponent, src + i + 0 * F, dst + i + 0 * F);
                NeuralPow(body, pow, _exponent, src + i + 1 * F, dst + i + 1 * F);
                NeuralPow(body, pow, _exponent, src + i + 2 * F, dst + i + 2 * F);
                NeuralPow(body, pow, _exponent, src + i + 3 * F, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                NeuralPow(body, pow, _exponent, src + i, dst + i);
            if (i < size)
                NeuralPow(svwhilelt_b32(i, size), pow, _exponent, src + i, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

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
    }
#endif
}

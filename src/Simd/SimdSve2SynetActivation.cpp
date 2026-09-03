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
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE void SynetElu32f(const float* src, const svbool_t& mask, svfloat32_t alpha, float* dst)
        {
            svst1_f32(mask, dst, Elu(mask, svld1_f32(mask, src), alpha));
        }

        void SynetElu32f(const float* src, size_t size, const float* alpha, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _alpha = svdup_n_f32(alpha[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetElu32f(src + i + 0 * F, body, _alpha, dst + i + 0 * F);
                SynetElu32f(src + i + 1 * F, body, _alpha, dst + i + 1 * F);
                SynetElu32f(src + i + 2 * F, body, _alpha, dst + i + 2 * F);
                SynetElu32f(src + i + 3 * F, body, _alpha, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetElu32f(src + i, body, _alpha, dst + i);
            if (i < size)
                SynetElu32f(src + i, svwhilelt_b32(i, size), _alpha, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t Erf(const svbool_t& mask, svfloat32_t x)
        {
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            svfloat32_t a = svmin_f32_x(mask, svabs_f32_x(mask, x), svdup_n_f32(9.0f));
            svfloat32_t p = svdup_n_f32(0.0000430638f);
            p = svmla_f32_x(mask, svdup_n_f32(0.0002765672f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0001520143f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0092705272f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0422820123f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0705230784f), a, p);
            p = svmla_f32_x(mask, _1, a, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            svfloat32_t r = svsub_f32_x(mask, _1, svdiv_f32_x(mask, _1, p));
            return svsel_f32(svcmplt_n_f32(mask, x, 0.0f), svneg_f32_x(mask, r), r);
        }

        SIMD_INLINE svfloat32_t Gelu(const svbool_t& mask, svfloat32_t x)
        {
            svfloat32_t t = svmul_n_f32_x(mask, x, float(M_SQRT1_2));
            return svmul_f32_x(mask, svmul_n_f32_x(mask, t, float(M_SQRT1_2)), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
        }

        SIMD_INLINE void SynetGelu32f(const float* src, const svbool_t& mask, float* dst)
        {
            svst1_f32(mask, dst, Gelu(mask, svld1_f32(mask, src)));
        }

        void SynetGelu32f(const float* src, size_t size, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            for (; i + QF <= size; i += QF)
            {
                SynetGelu32f(src + i + 0 * F, body, dst + i + 0 * F);
                SynetGelu32f(src + i + 1 * F, body, dst + i + 1 * F);
                SynetGelu32f(src + i + 2 * F, body, dst + i + 2 * F);
                SynetGelu32f(src + i + 3 * F, body, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetGelu32f(src + i, body, dst + i);
            if (i < size)
                SynetGelu32f(src + i, svwhilelt_b32(i, size), dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetHardSigmoid32f(const svbool_t& mask, svfloat32_t value, svfloat32_t scale, svfloat32_t shift)
        {
            return svmax_f32_x(mask, svmin_f32_x(mask, svmla_f32_x(mask, shift, value, scale), svdup_n_f32(1.0f)), svdup_n_f32(0.0f));
        }

        SIMD_INLINE void SynetHardSigmoid32f(const float* src, const svbool_t& mask, svfloat32_t scale, svfloat32_t shift, float* dst)
        {
            svst1_f32(mask, dst, SynetHardSigmoid32f(mask, svld1_f32(mask, src), scale, shift));
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _scale = svdup_n_f32(scale[0]);
            const svfloat32_t _shift = svdup_n_f32(shift[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetHardSigmoid32f(src + i + 0 * F, body, _scale, _shift, dst + i + 0 * F);
                SynetHardSigmoid32f(src + i + 1 * F, body, _scale, _shift, dst + i + 1 * F);
                SynetHardSigmoid32f(src + i + 2 * F, body, _scale, _shift, dst + i + 2 * F);
                SynetHardSigmoid32f(src + i + 3 * F, body, _scale, _shift, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetHardSigmoid32f(src + i, body, _scale, _shift, dst + i);
            if (i < size)
                SynetHardSigmoid32f(src + i, svwhilelt_b32(i, size), _scale, _shift, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetHswish32f(const svbool_t& mask, svfloat32_t value, svfloat32_t shift, svfloat32_t scale)
        {
            svfloat32_t upper = svmin_f32_x(mask, value, shift);
            svfloat32_t positive = svmax_f32_x(mask, svadd_f32_x(mask, upper, shift), svdup_n_f32(0.0f));
            return svmul_f32_x(mask, svmul_f32_x(mask, positive, scale), value);
        }

        SIMD_INLINE void SynetHswish32f(const float* src, const svbool_t& mask, svfloat32_t shift, svfloat32_t scale, float* dst)
        {
            svst1_f32(mask, dst, SynetHswish32f(mask, svld1_f32(mask, src), shift, scale));
        }

        void SynetHswish32f(const float* src, size_t size, const float* shift, const float* scale, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _shift = svdup_n_f32(shift[0]);
            const svfloat32_t _scale = svdup_n_f32(scale[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetHswish32f(src + i + 0 * F, body, _shift, _scale, dst + i + 0 * F);
                SynetHswish32f(src + i + 1 * F, body, _shift, _scale, dst + i + 1 * F);
                SynetHswish32f(src + i + 2 * F, body, _shift, _scale, dst + i + 2 * F);
                SynetHswish32f(src + i + 3 * F, body, _shift, _scale, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetHswish32f(src + i, body, _shift, _scale, dst + i);
            if (i < size)
                SynetHswish32f(src + i, svwhilelt_b32(i, size), _shift, _scale, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetMish32f(const svbool_t& mask, svfloat32_t value, svfloat32_t threshold)
        {
            return Mish(mask, value, threshold);
        }

        SIMD_INLINE void SynetMish32f(const float* src, const svbool_t& mask, svfloat32_t threshold, float* dst)
        {
            svst1_f32(mask, dst, SynetMish32f(mask, svld1_f32(mask, src), threshold));
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _threshold = svdup_n_f32(threshold[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetMish32f(src + i + 0 * F, body, _threshold, dst + i + 0 * F);
                SynetMish32f(src + i + 1 * F, body, _threshold, dst + i + 1 * F);
                SynetMish32f(src + i + 2 * F, body, _threshold, dst + i + 2 * F);
                SynetMish32f(src + i + 3 * F, body, _threshold, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetMish32f(src + i, body, _threshold, dst + i);
            if (i < size)
                SynetMish32f(src + i, svwhilelt_b32(i, size), _threshold, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetPreluLayerForward(const svbool_t& mask, svfloat32_t src, svfloat32_t slope)
        {
            svfloat32_t pos = svmax_n_f32_x(mask, src, 0.0f);
            svfloat32_t neg = svmin_n_f32_x(mask, src, 0.0f);
            return svmla_f32_x(mask, pos, slope, neg);
        }

        SIMD_INLINE void SynetPreluLayerForward(const float* src, const svbool_t& mask, svfloat32_t slope, float* dst)
        {
            svst1_f32(mask, dst, SynetPreluLayerForward(mask, svld1_f32(mask, src), slope));
        }

        SIMD_INLINE void SynetPreluLayerForward(const float* src, const svbool_t& mask, const float* slope, float* dst)
        {
            svst1_f32(mask, dst, SynetPreluLayerForward(mask, svld1_f32(mask, src), svld1_f32(mask, slope)));
        }

        void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                svfloat32_t _slope = svdup_n_f32(slope[c]);
                for (; s + QF <= spatial; s += QF)
                {
                    SynetPreluLayerForward(src + s + 0 * F, body, _slope, dst + s + 0 * F);
                    SynetPreluLayerForward(src + s + 1 * F, body, _slope, dst + s + 1 * F);
                    SynetPreluLayerForward(src + s + 2 * F, body, _slope, dst + s + 2 * F);
                    SynetPreluLayerForward(src + s + 3 * F, body, _slope, dst + s + 3 * F);
                }
                for (; s + F <= spatial; s += F)
                    SynetPreluLayerForward(src + s, body, _slope, dst + s);
                if (s < spatial)
                    SynetPreluLayerForward(src + s, svwhilelt_b32(s, spatial), _slope, dst + s);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c + QF <= channels; c += QF)
                {
                    SynetPreluLayerForward(src + c + 0 * F, body, slope + c + 0 * F, dst + c + 0 * F);
                    SynetPreluLayerForward(src + c + 1 * F, body, slope + c + 1 * F, dst + c + 1 * F);
                    SynetPreluLayerForward(src + c + 2 * F, body, slope + c + 2 * F, dst + c + 2 * F);
                    SynetPreluLayerForward(src + c + 3 * F, body, slope + c + 3 * F, dst + c + 3 * F);
                }
                for (; c + F <= channels; c += F)
                    SynetPreluLayerForward(src + c, body, slope + c, dst + c);
                if (c < channels)
                    SynetPreluLayerForward(src + c, svwhilelt_b32(c, channels), slope + c, dst + c);
                src += channels;
                dst += channels;
            }
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetPreluLayerForwardNchw(src, slope, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetPreluLayerForwardNhwc(src, slope, channels, spatial, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetRelu32f(const svbool_t& mask, svfloat32_t src, svfloat32_t slope)
        {
            svfloat32_t pos = svmax_n_f32_x(mask, src, 0.0f);
            svfloat32_t neg = svmin_n_f32_x(mask, src, 0.0f);
            return svmla_f32_x(mask, pos, slope, neg);
        }

        SIMD_INLINE void SynetRelu32f(const float* src, const svbool_t& mask, svfloat32_t slope, float* dst)
        {
            svst1_f32(mask, dst, SynetRelu32f(mask, svld1_f32(mask, src), slope));
        }

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _slope = svdup_n_f32(slope[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetRelu32f(src + i + 0 * F, body, _slope, dst + i + 0 * F);
                SynetRelu32f(src + i + 1 * F, body, _slope, dst + i + 1 * F);
                SynetRelu32f(src + i + 2 * F, body, _slope, dst + i + 2 * F);
                SynetRelu32f(src + i + 3 * F, body, _slope, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetRelu32f(src + i, body, _slope, dst + i);
            if (i < size)
                SynetRelu32f(src + i, svwhilelt_b32(i, size), _slope, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetRestrictRange32f(const float* src, const svbool_t& mask, svfloat32_t lower, svfloat32_t upper, float* dst)
        {
            svst1_f32(mask, dst, svmin_f32_x(mask, svmax_f32_x(mask, svld1_f32(mask, src), lower), upper));
        }

        void SynetRestrictRange32f(const float* src, size_t size, const float* lower, const float* upper, float* dst)
        {
            assert(lower[0] <= upper[0]);

            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _lower = svdup_n_f32(lower[0]);
            const svfloat32_t _upper = svdup_n_f32(upper[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetRestrictRange32f(src + i + 0 * F, body, _lower, _upper, dst + i + 0 * F);
                SynetRestrictRange32f(src + i + 1 * F, body, _lower, _upper, dst + i + 1 * F);
                SynetRestrictRange32f(src + i + 2 * F, body, _lower, _upper, dst + i + 2 * F);
                SynetRestrictRange32f(src + i + 3 * F, body, _lower, _upper, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetRestrictRange32f(src + i, body, _lower, _upper, dst + i);
            if (i < size)
                SynetRestrictRange32f(src + i, svwhilelt_b32(i, size), _lower, _upper, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetSigmoid32f(const svbool_t& mask, svfloat32_t value, svfloat32_t slope)
        {
            svfloat32_t exp = Exponent(mask, svmul_f32_x(mask, value, slope));
            return svdiv_f32_x(mask, svdup_n_f32(1.0f), svadd_n_f32_x(mask, exp, 1.0f));
        }

        SIMD_INLINE void SynetSigmoid32f(const float* src, const svbool_t& mask, svfloat32_t slope, float* dst)
        {
            svst1_f32(mask, dst, SynetSigmoid32f(mask, svld1_f32(mask, src), slope));
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _slope = svdup_n_f32(-slope[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetSigmoid32f(src + i + 0 * F, body, _slope, dst + i + 0 * F);
                SynetSigmoid32f(src + i + 1 * F, body, _slope, dst + i + 1 * F);
                SynetSigmoid32f(src + i + 2 * F, body, _slope, dst + i + 2 * F);
                SynetSigmoid32f(src + i + 3 * F, body, _slope, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetSigmoid32f(src + i, body, _slope, dst + i);
            if (i < size)
                SynetSigmoid32f(src + i, svwhilelt_b32(i, size), _slope, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetTanh32f(const svbool_t& mask, svfloat32_t value, svfloat32_t slope)
        {
            return Tanh(mask, svmul_f32_x(mask, value, slope));
        }

        SIMD_INLINE void SynetTanh32f(const float* src, const svbool_t& mask, svfloat32_t slope, float* dst)
        {
            svst1_f32(mask, dst, SynetTanh32f(mask, svld1_f32(mask, src), slope));
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _slope = svdup_n_f32(slope[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetTanh32f(src + i + 0 * F, body, _slope, dst + i + 0 * F);
                SynetTanh32f(src + i + 1 * F, body, _slope, dst + i + 1 * F);
                SynetTanh32f(src + i + 2 * F, body, _slope, dst + i + 2 * F);
                SynetTanh32f(src + i + 3 * F, body, _slope, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetTanh32f(src + i, body, _slope, dst + i);
            if (i < size)
                SynetTanh32f(src + i, svwhilelt_b32(i, size), _slope, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetSoftplus32f(const svbool_t& mask, svfloat32_t value, svfloat32_t beta, svfloat32_t threshold)
        {
            return Softplus(mask, value, beta, threshold);
        }

        SIMD_INLINE void SynetSoftplus32f(const float* src, const svbool_t& mask, svfloat32_t beta, svfloat32_t threshold, float* dst)
        {
            svst1_f32(mask, dst, SynetSoftplus32f(mask, svld1_f32(mask, src), beta, threshold));
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _beta = svdup_n_f32(beta[0]);
            const svfloat32_t _threshold = svdup_n_f32(threshold[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetSoftplus32f(src + i + 0 * F, body, _beta, _threshold, dst + i + 0 * F);
                SynetSoftplus32f(src + i + 1 * F, body, _beta, _threshold, dst + i + 1 * F);
                SynetSoftplus32f(src + i + 2 * F, body, _beta, _threshold, dst + i + 2 * F);
                SynetSoftplus32f(src + i + 3 * F, body, _beta, _threshold, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetSoftplus32f(src + i, body, _beta, _threshold, dst + i);
            if (i < size)
                SynetSoftplus32f(src + i, svwhilelt_b32(i, size), _beta, _threshold, dst + i);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t SynetSwish32f(const svbool_t& mask, svfloat32_t value, svfloat32_t slope)
        {
            return Swish(mask, value, slope);
        }

        SIMD_INLINE void SynetSwish32f(const float* src, const svbool_t& mask, svfloat32_t slope, float* dst)
        {
            svst1_f32(mask, dst, SynetSwish32f(mask, svld1_f32(mask, src), slope));
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            size_t F = svcntw(), QF = 4 * F, i = 0;
            const svbool_t body = svptrue_b32();
            const svfloat32_t _slope = svdup_n_f32(slope[0]);
            for (; i + QF <= size; i += QF)
            {
                SynetSwish32f(src + i + 0 * F, body, _slope, dst + i + 0 * F);
                SynetSwish32f(src + i + 1 * F, body, _slope, dst + i + 1 * F);
                SynetSwish32f(src + i + 2 * F, body, _slope, dst + i + 2 * F);
                SynetSwish32f(src + i + 3 * F, body, _slope, dst + i + 3 * F);
            }
            for (; i + F <= size; i += F)
                SynetSwish32f(src + i, body, _slope, dst + i);
            if (i < size)
                SynetSwish32f(src + i, svwhilelt_b32(i, size), _slope, dst + i);
        }
    }
#endif
}

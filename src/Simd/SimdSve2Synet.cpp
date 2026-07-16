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
#include "Simd/SimdSynet.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdPow.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t BFloat16ToFloat32(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, svld1uh_u32(mask, src), Base::Bf16::SHIFT));
        }

        SIMD_INLINE void AddBFloat16ToSum(const uint16_t* src, svfloat32_t& sum, const svbool_t& mask)
        {
            sum = svadd_f32_m(mask, sum, BFloat16ToFloat32(src, mask));
        }

        void SynetChannelSum16b(const uint16_t* src, size_t channels, size_t spatial, SimdTensorFormatType format, float* sum)
        {
            size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();

            if (format == SimdTensorFormatNhwc)
            {
                size_t c = 0;
                for (; c + QF <= channels; c += QF)
                {
                    svst1_f32(body, sum + c + 0 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 1 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 2 * F, svdup_n_f32(0.0f));
                    svst1_f32(body, sum + c + 3 * F, svdup_n_f32(0.0f));
                }
                for (; c + F <= channels; c += F)
                    svst1_f32(body, sum + c, svdup_n_f32(0.0f));
                if (c < channels)
                    svst1_f32(svwhilelt_b32(c, channels), sum + c, svdup_n_f32(0.0f));

                for (size_t s = 0; s < spatial; ++s)
                {
                    c = 0;
                    for (; c + QF <= channels; c += QF)
                    {
                        svfloat32_t sum0 = svld1_f32(body, sum + c + 0 * F);
                        svfloat32_t sum1 = svld1_f32(body, sum + c + 1 * F);
                        svfloat32_t sum2 = svld1_f32(body, sum + c + 2 * F);
                        svfloat32_t sum3 = svld1_f32(body, sum + c + 3 * F);
                        AddBFloat16ToSum(src + c + 0 * F, sum0, body);
                        AddBFloat16ToSum(src + c + 1 * F, sum1, body);
                        AddBFloat16ToSum(src + c + 2 * F, sum2, body);
                        AddBFloat16ToSum(src + c + 3 * F, sum3, body);
                        svst1_f32(body, sum + c + 0 * F, sum0);
                        svst1_f32(body, sum + c + 1 * F, sum1);
                        svst1_f32(body, sum + c + 2 * F, sum2);
                        svst1_f32(body, sum + c + 3 * F, sum3);
                    }
                    for (; c + F <= channels; c += F)
                    {
                        svfloat32_t _sum = svld1_f32(body, sum + c);
                        AddBFloat16ToSum(src + c, _sum, body);
                        svst1_f32(body, sum + c, _sum);
                    }
                    if (c < channels)
                    {
                        svbool_t tail = svwhilelt_b32(c, channels);
                        svfloat32_t _sum = svld1_f32(tail, sum + c);
                        AddBFloat16ToSum(src + c, _sum, tail);
                        svst1_f32(tail, sum + c, _sum);
                    }
                    src += channels;
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                    svfloat32_t sum2 = svdup_n_f32(0.0f), sum3 = svdup_n_f32(0.0f);
                    size_t s = 0;
                    for (; s + QF <= spatial; s += QF)
                    {
                        AddBFloat16ToSum(src + s + 0 * F, sum0, body);
                        AddBFloat16ToSum(src + s + 1 * F, sum1, body);
                        AddBFloat16ToSum(src + s + 2 * F, sum2, body);
                        AddBFloat16ToSum(src + s + 3 * F, sum3, body);
                    }
                    sum0 = svadd_f32_x(body, svadd_f32_x(body, sum0, sum1), svadd_f32_x(body, sum2, sum3));
                    for (; s + F <= spatial; s += F)
                        AddBFloat16ToSum(src + s, sum0, body);
                    if (s < spatial)
                        AddBFloat16ToSum(src + s, sum0, svwhilelt_b32(s, spatial));
                    sum[c] = svaddv_f32(body, sum0);
                    src += spatial;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> svfloat32_t SynetEltwiseLayerForward(const svfloat32_t& src0, const svfloat32_t& src1, const svbool_t& mask);

        template <> SIMD_INLINE svfloat32_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(const svfloat32_t& src0, const svfloat32_t& src1, const svbool_t& mask)
        {
            return svmul_f32_x(mask, src0, src1);
        }

        template <> SIMD_INLINE svfloat32_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(const svfloat32_t& src0, const svfloat32_t& src1, const svbool_t& mask)
        {
            return svmax_f32_x(mask, src0, src1);
        }

        template <> SIMD_INLINE svfloat32_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(const svfloat32_t& src0, const svfloat32_t& src1, const svbool_t& mask)
        {
            return svmin_f32_x(mask, src0, src1);
        }

        template <SimdSynetEltwiseOperationType type> SIMD_INLINE void SynetEltwiseLayerForward(const float* src0, const float* src1, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, SynetEltwiseLayerForward<type>(svld1_f32(mask, src0), svld1_f32(mask, src1), mask));
        }

        template <SimdSynetEltwiseOperationType type> void SynetEltwiseLayerForward(float const* const* src, size_t count, size_t size, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const float* src0 = src[0];
            const float* src1 = src[1];
            size_t j = 0;
            for (; j + QF <= size; j += QF)
            {
                SynetEltwiseLayerForward<type>(src0 + j + 0 * F, src1 + j + 0 * F, dst + j + 0 * F, body);
                SynetEltwiseLayerForward<type>(src0 + j + 1 * F, src1 + j + 1 * F, dst + j + 1 * F, body);
                SynetEltwiseLayerForward<type>(src0 + j + 2 * F, src1 + j + 2 * F, dst + j + 2 * F, body);
                SynetEltwiseLayerForward<type>(src0 + j + 3 * F, src1 + j + 3 * F, dst + j + 3 * F, body);
            }
            for (; j < size; j += F)
                SynetEltwiseLayerForward<type>(src0 + j, src1 + j, dst + j, svwhilelt_b32(j, size));
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                j = 0;
                for (; j + QF <= size; j += QF)
                {
                    SynetEltwiseLayerForward<type>(dst + j + 0 * F, srci + j + 0 * F, dst + j + 0 * F, body);
                    SynetEltwiseLayerForward<type>(dst + j + 1 * F, srci + j + 1 * F, dst + j + 1 * F, body);
                    SynetEltwiseLayerForward<type>(dst + j + 2 * F, srci + j + 2 * F, dst + j + 2 * F, body);
                    SynetEltwiseLayerForward<type>(dst + j + 3 * F, srci + j + 3 * F, dst + j + 3 * F, body);
                }
                for (; j < size; j += F)
                    SynetEltwiseLayerForward<type>(dst + j, srci + j, dst + j, svwhilelt_b32(j, size));
            }
        }

        SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src0, const svfloat32_t& weight0, const float* src1, const svfloat32_t& weight1, float* dst, const svbool_t& mask)
        {
            svfloat32_t sum = svmul_f32_x(mask, svld1_f32(mask, src0), weight0);
            svst1_f32(mask, dst, svmla_f32_x(mask, sum, svld1_f32(mask, src1), weight1));
        }

        SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src, const svfloat32_t& weight, float* dst, const svbool_t& mask)
        {
            svst1_f32(mask, dst, svmla_f32_x(mask, svld1_f32(mask, dst), svld1_f32(mask, src), weight));
        }

        void SynetEltwiseLayerForwardSum(float const* const* src, const float* weight, size_t count, size_t size, float* dst)
        {
            const size_t F = svcntw(), QF = 4 * F;
            const svbool_t body = svptrue_b32();
            const float* src0 = src[0];
            const float* src1 = src[1];
            svfloat32_t weight0 = svdup_n_f32(weight[0]);
            svfloat32_t weight1 = svdup_n_f32(weight[1]);
            size_t j = 0;
            for (; j + QF <= size; j += QF)
            {
                SynetEltwiseLayerForwardSum(src0 + j + 0 * F, weight0, src1 + j + 0 * F, weight1, dst + j + 0 * F, body);
                SynetEltwiseLayerForwardSum(src0 + j + 1 * F, weight0, src1 + j + 1 * F, weight1, dst + j + 1 * F, body);
                SynetEltwiseLayerForwardSum(src0 + j + 2 * F, weight0, src1 + j + 2 * F, weight1, dst + j + 2 * F, body);
                SynetEltwiseLayerForwardSum(src0 + j + 3 * F, weight0, src1 + j + 3 * F, weight1, dst + j + 3 * F, body);
            }
            for (; j < size; j += F)
                SynetEltwiseLayerForwardSum(src0 + j, weight0, src1 + j, weight1, dst + j, svwhilelt_b32(j, size));
            for (size_t i = 2; i < count; ++i)
            {
                const float* srci = src[i];
                svfloat32_t weighti = svdup_n_f32(weight[i]);
                j = 0;
                for (; j + QF <= size; j += QF)
                {
                    SynetEltwiseLayerForwardSum(srci + j + 0 * F, weighti, dst + j + 0 * F, body);
                    SynetEltwiseLayerForwardSum(srci + j + 1 * F, weighti, dst + j + 1 * F, body);
                    SynetEltwiseLayerForwardSum(srci + j + 2 * F, weighti, dst + j + 2 * F, body);
                    SynetEltwiseLayerForwardSum(srci + j + 3 * F, weighti, dst + j + 3 * F, body);
                }
                for (; j < size; j += F)
                    SynetEltwiseLayerForwardSum(srci + j, weighti, dst + j, svwhilelt_b32(j, size));
            }
        }

        void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst)
        {
            assert(count >= 2);
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetInnerProductLayerForward(const float* src, const float* weight, size_t offset, const svbool_t& mask, svfloat32_t& sum)
        {
            sum = svmla_f32_m(mask, sum, svld1_f32(mask, src + offset), svld1_f32(mask, weight + offset));
        }

        SIMD_INLINE void SynetInnerProductLayerForward(const float* src, const float* weight0, const float* weight1, size_t offset, const svbool_t& mask, svfloat32_t& sum0, svfloat32_t& sum1)
        {
            svfloat32_t _src = svld1_f32(mask, src + offset);
            sum0 = svmla_f32_m(mask, sum0, _src, svld1_f32(mask, weight0 + offset));
            sum1 = svmla_f32_m(mask, sum1, _src, svld1_f32(mask, weight1 + offset));
        }

        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            const size_t F = svcntw(), DF = 2 * F, QF = 4 * F;
            const svbool_t body = svptrue_b32();
            size_t i = 0, count2 = AlignLo(count, 2);
            for (; i < count2; i += 2)
            {
                size_t j = 0;
                const float* weight0 = weight + 0 * size;
                const float* weight1 = weight + 1 * size;
                svfloat32_t sum00 = svdup_n_f32(0.0f), sum01 = svdup_n_f32(0.0f);
                svfloat32_t sum10 = svdup_n_f32(0.0f), sum11 = svdup_n_f32(0.0f);
                for (; j + DF <= size; j += DF)
                {
                    SynetInnerProductLayerForward(src, weight0, weight1, j + 0 * F, body, sum00, sum01);
                    SynetInnerProductLayerForward(src, weight0, weight1, j + 1 * F, body, sum10, sum11);
                }
                sum00 = svadd_f32_x(body, sum00, sum10);
                sum01 = svadd_f32_x(body, sum01, sum11);
                if (j < size)
                    SynetInnerProductLayerForward(src, weight0, weight1, j, svwhilelt_b32(j, size), sum00, sum01);
                dst[i + 0] = svaddv_f32(body, sum00) + (bias ? bias[i + 0] : 0);
                dst[i + 1] = svaddv_f32(body, sum01) + (bias ? bias[i + 1] : 0);
                weight += 2 * size;
            }
            for (; i < count; ++i)
            {
                size_t j = 0;
                svfloat32_t sum0 = svdup_n_f32(0.0f), sum1 = svdup_n_f32(0.0f);
                svfloat32_t sum2 = svdup_n_f32(0.0f), sum3 = svdup_n_f32(0.0f);
                for (; j + QF <= size; j += QF)
                {
                    SynetInnerProductLayerForward(src, weight, j + 0 * F, body, sum0);
                    SynetInnerProductLayerForward(src, weight, j + 1 * F, body, sum1);
                    SynetInnerProductLayerForward(src, weight, j + 2 * F, body, sum2);
                    SynetInnerProductLayerForward(src, weight, j + 3 * F, body, sum3);
                }
                sum0 = svadd_f32_x(body, svadd_f32_x(body, sum0, sum1), svadd_f32_x(body, sum2, sum3));
                for (; j < size; j += F)
                    SynetInnerProductLayerForward(src, weight, j, svwhilelt_b32(j, size), sum0);
                dst[i] = svaddv_f32(body, sum0) + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE svfloat32_t Square(const svbool_t& mask, const float* src)
        {
            svfloat32_t _src = svld1_f32(mask, src);
            return svmul_f32_x(mask, _src, _src);
        }

        void SynetLrnLayerCrossChannelsNchw(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst)
        {
            const size_t F = svcntw();
            const svbool_t body = svptrue_b32();
            svfloat32_t k0 = svdup_n_f32(k[0]);
            svfloat32_t k1 = svdup_n_f32(k[1]);
            svfloat32_t k2 = svdup_n_f32(k[2]);
            Sve2::Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            for (size_t c = 0; c < half; ++c)
            {
                const float* pos = src + c * spatial;
                size_t s = 0;
                for (; s + F <= spatial; s += F)
                    svst1_f32(body, sum.data + s, svadd_f32_x(body, svld1_f32(body, sum.data + s), Square(body, pos + s)));
                if (s < spatial)
                {
                    svbool_t tail = svwhilelt_b32(s, spatial);
                    svst1_f32(tail, sum.data + s, svadd_f32_x(tail, svld1_f32(tail, sum.data + s), Square(tail, pos + s)));
                }
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float* pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float* neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                size_t s = 0;
                for (; s + F <= spatial; s += F)
                {
                    svfloat32_t _sum = svld1_f32(body, sum.data + s);
                    _sum = svsub_f32_x(body, svadd_f32_x(body, _sum, Square(body, pos + s)), Square(body, neg + s));
                    svst1_f32(body, sum.data + s, _sum);
                    svst1_f32(body, dst + s, svmul_f32_x(body, svld1_f32(body, src + s), pow(body, svmla_f32_x(body, k0, k1, _sum), k2)));
                }
                if (s < spatial)
                {
                    svbool_t tail = svwhilelt_b32(s, spatial);
                    svfloat32_t _sum = svld1_f32(tail, sum.data + s);
                    _sum = svsub_f32_x(tail, svadd_f32_x(tail, _sum, Square(tail, pos + s)), Square(tail, neg + s));
                    svst1_f32(tail, sum.data + s, _sum);
                    svst1_f32(tail, dst + s, svmul_f32_x(tail, svld1_f32(tail, src + s), pow(tail, svmla_f32_x(tail, k0, k1, _sum), k2)));
                }
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE float SynetLrnLayerCrossChannelsNhwc(const float* src, size_t half, size_t channels, size_t c, const float* k)
        {
            float sum = 0.0f;
            size_t beg = c > half ? c - half : 0;
            size_t end = Simd::Min(c + half + 1, channels);
            for (size_t i = beg; i < end; ++i)
                sum += Simd::Square(src[i]);
            return src[c] * Base::Pow(k[0] + k[1] * sum, k[2]);
        }

        void SynetLrnLayerCrossChannelsNhwc2h(const float* src, size_t channels, size_t spatial, const float* k, float* dst)
        {
            const size_t F = svcntw(), half = 2, end = channels - half;
            svfloat32_t k0 = svdup_n_f32(k[0]);
            svfloat32_t k1 = svdup_n_f32(k[1]);
            svfloat32_t k2 = svdup_n_f32(k[2]);
            Sve2::Pow pow;
            for (size_t s = 0; s < spatial; ++s)
            {
                for (size_t c = 0; c < half; ++c)
                    dst[c] = SynetLrnLayerCrossChannelsNhwc(src, half, channels, c, k);
                for (size_t c = half; c < end; c += F)
                {
                    svbool_t mask = svwhilelt_b32(c, end);
                    svfloat32_t sum = Square(mask, src + c - 2);
                    sum = svadd_f32_x(mask, sum, Square(mask, src + c - 1));
                    sum = svadd_f32_x(mask, sum, Square(mask, src + c + 0));
                    sum = svadd_f32_x(mask, sum, Square(mask, src + c + 1));
                    sum = svadd_f32_x(mask, sum, Square(mask, src + c + 2));
                    svst1_f32(mask, dst + c, svmul_f32_x(mask, svld1_f32(mask, src + c), pow(mask, svmla_f32_x(mask, k0, k1, sum), k2)));
                }
                for (size_t c = end; c < channels; ++c)
                    dst[c] = SynetLrnLayerCrossChannelsNhwc(src, half, channels, c, k);
                src += channels;
                dst += channels;
            }
        }

        void SynetLrnLayerCrossChannels(const float* src, size_t half, size_t channels, size_t spatial, const float* k, float* dst, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
                SynetLrnLayerCrossChannelsNchw(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNhwc)
            {
                if (half == 2 && channels > 2 * half)
                    SynetLrnLayerCrossChannelsNhwc2h(src, channels, spatial, k, dst);
                else
                    Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, format);
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type)
        {
            if (format == SimdTensorFormatNchw)
                Base::SynetShuffleLayerForward(src0, src1, channels0, channels1, spatial, dst0, dst1, format, type);
            else if (format == SimdTensorFormatNhwc)
            {
                const size_t F = svcntw(), channels = (channels0 + channels1) / 2;
                if (type == 0)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0, cs0 = 0, cs1 = 0;
                        for (; cs0 < channels0;)
                        {
                            size_t count = Simd::Min(F, (channels0 - cs0) / 2);
                            svbool_t mask = svwhilelt_b32((size_t)0, count);
                            svfloat32x2_t _src0 = svld2_f32(mask, src0 + cs0);
                            svst1_f32(mask, dst0 + cd, svget2_f32(_src0, 0));
                            svst1_f32(mask, dst1 + cd, svget2_f32(_src0, 1));
                            cs0 += 2 * count;
                            cd += count;
                        }
                        for (; cs1 < channels1;)
                        {
                            size_t count = Simd::Min(F, (channels1 - cs1) / 2);
                            svbool_t mask = svwhilelt_b32((size_t)0, count);
                            svfloat32x2_t _src1 = svld2_f32(mask, src1 + cs1);
                            svst1_f32(mask, dst0 + cd, svget2_f32(_src1, 0));
                            svst1_f32(mask, dst1 + cd, svget2_f32(_src1, 1));
                            cs1 += 2 * count;
                            cd += count;
                        }
                        src0 += channels0;
                        src1 += channels1;
                        dst0 += channels;
                        dst1 += channels;
                    }
                }
                else if (type == 1)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cs = 0, cd0 = 0, cd1 = 0;
                        for (; cd0 < channels0;)
                        {
                            size_t count = Simd::Min(F, (channels0 - cd0) / 2);
                            svbool_t mask = svwhilelt_b32((size_t)0, count);
                            svst2_f32(mask, dst0 + cd0, svcreate2_f32(svld1_f32(mask, src0 + cs), svld1_f32(mask, src1 + cs)));
                            cd0 += 2 * count;
                            cs += count;
                        }
                        for (; cd1 < channels1;)
                        {
                            size_t count = Simd::Min(F, (channels1 - cd1) / 2);
                            svbool_t mask = svwhilelt_b32((size_t)0, count);
                            svst2_f32(mask, dst1 + cd1, svcreate2_f32(svld1_f32(mask, src0 + cs), svld1_f32(mask, src1 + cs)));
                            cd1 += 2 * count;
                            cs += count;
                        }
                        src0 += channels;
                        src1 += channels;
                        dst0 += channels0;
                        dst1 += channels1;
                    }
                }
                else
                    assert(0);
            }
            else
                assert(0);
        }
    }
#endif
}

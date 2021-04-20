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
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Neon
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(float32x4_t bias, float * dst)
        {
            Store<align>(dst, vaddq_f32(Load<align>(dst), bias));
        }

        template <bool align> void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    float32x4_t _bias = vdupq_n_f32(bias[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetAddBias<align>(_bias, dst + s + F * 0);
                        SynetAddBias<align>(_bias, dst + s + F * 1);
                        SynetAddBias<align>(_bias, dst + s + F * 2);
                        SynetAddBias<align>(_bias, dst + s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetAddBias<align>(_bias, dst + s);
                }
                for (; s < spatial; ++s)
                    dst[s] += bias[c];
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetAddBiasNchw(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(spatial, F) && Aligned(dst))
                SynetAddBiasNchw<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(channels, F) && Aligned(bias) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetAddBias<align>(bias + c + F * 0, dst + c + F * 0);
                        SynetAddBias<align>(bias + c + F * 1, dst + c + F * 1);
                        SynetAddBias<align>(bias + c + F * 2, dst + c + F * 2);
                        SynetAddBias<align>(bias + c + F * 3, dst + c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetAddBias<align>(bias + c, dst + c);
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        SIMD_INLINE void SynetAddBiasNhwc(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetAddBiasNhwc<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNhwc<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNchw4c(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(dst));

            size_t spatial4 = AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; c += F)
            {
                float32x4_t _bias = Load<false>(bias + c);
                size_t s = 0;
                for (; s < spatial4; s += 4, dst += 4 * F)
                {
                    SynetAddBias<align>(_bias, dst + 0 * F);
                    SynetAddBias<align>(_bias, dst + 1 * F);
                    SynetAddBias<align>(_bias, dst + 2 * F);
                    SynetAddBias<align>(_bias, dst + 3 * F);
                }
                for (; s < spatial; ++s, dst += F)
                    SynetAddBias<align>(_bias, dst);
            }
        }

        SIMD_INLINE void SynetAddBiasNchw4c(const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(dst))
                SynetAddBiasNchw4c<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw4c<false>(bias, channels, spatial, dst);
        }

        void SynetAddBias(const float * bias, size_t channels, size_t spatial, float * dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                SynetAddBiasNchw4c(bias, channels, spatial, dst);
            else
                Base::SynetAddBias(bias, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template <SimdSynetEltwiseOperationType type> float32x4_t SynetEltwiseLayerForward(float32x4_t src0, float32x4_t src1);

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(float32x4_t src0, float32x4_t src1)
        {
            return vmulq_f32(src0, src1);
        }

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(float32x4_t src0, float32x4_t src1)
        {
            return vmaxq_f32(src0, src1);
        }

        template <> SIMD_INLINE float32x4_t SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(float32x4_t src0, float32x4_t src1)
        {
            return vminq_f32(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align> SIMD_INLINE void SynetEltwiseLayerForward(const float * src0, const float * src1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, SynetEltwiseLayerForward<type>(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 0);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 1);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 2);
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForward<type, align>(src0, src1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = Base::SynetEltwiseLayerForward<type>(src0[j], src1[j]);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 0);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 1);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 2);
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForward<type, align>(dst, srci, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetEltwiseLayerForward<type>(dst[j], srci[j]);
            }
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src0, const float32x4_t & weight0, const float * src1, const float32x4_t & weight1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(vmulq_f32(Load<align>(src0 + offset), weight0), Load<align>(src1 + offset), weight1));
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src, const float32x4_t & weight, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmlaq_f32(Load<align>(dst + offset), Load<align>(src + offset), weight));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            float32x4_t weight0 = vdupq_n_f32(weight[0]);
            float32x4_t weight1 = vdupq_n_f32(weight[1]);
            size_t j = 0;
            if (partial)
            {
                for (; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align>(src0, weight0, src1, weight1, dst, j);
            }
            for (; j < size; ++j)
                dst[j] = src0[j] * weight[0] + src1[j] * weight[1];
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                float32x4_t weighti = vdupq_n_f32(weight[i]);
                size_t j = 0;
                if (partial)
                {
                    for (; j < aligned; j += QF)
                    {
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 0);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 1);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 2);
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetEltwiseLayerForwardSum<align>(srci, weighti, dst, j);
                }
                for (; j < size; ++j)
                    dst[j] += srci[j] * weight[i];
            }
        }

        template <bool align> void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum<align>(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax, align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMin:
                SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin, align>(src, count, size, dst);
                break;
            default:
                assert(0);
            }
        }

        void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            assert(count >= 2);
            bool aligned = Aligned(dst) && Aligned(src[0]) && Aligned(src[1]);
            for (size_t i = 2; i < count; ++i)
                aligned = aligned && Aligned(src[i]);
            if (aligned)
                SynetEltwiseLayerForward<true>(src, weight, count, size, type, dst);
            else
                SynetEltwiseLayerForward<false>(src, weight, count, size, type, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetInnerProductLayerForward(const float * src, const float * weight, size_t offset, float32x4_t & sum)
        {
            float32x4_t s = Load<align>(src + offset);
            float32x4_t w = Load<align>(weight + offset);
            sum = vmlaq_f32(sum, s, w);
        }

        template <bool align> SIMD_INLINE void SynetInnerProductLayerForward(const float * src, const float * weight0, const float * weight1, size_t offset, float32x4_t * sum)
        {
            float32x4_t s = Load<align>(src + offset);
            float32x4_t w0 = Load<align>(weight0 + offset);
            float32x4_t w1 = Load<align>(weight1 + offset);
            sum[0] = vmlaq_f32(sum[0], s, w0);
            sum[1] = vmlaq_f32(sum[1], s, w1);
        }

        template<bool align> void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst));
            size_t count2 = AlignLo(count, 2);
            size_t sizeF = AlignLo(size, F);
            size_t sizeDF = AlignLo(size, DF);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < count2; i += 2)
            {
                size_t j = 0;
                float sum0 = 0, sum1 = 0;
                const float * weight0 = weight + 0 * size;
                const float * weight1 = weight + 1 * size;
                if (sizeF)
                {
                    float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                    if (sizeDF)
                    {
                        for (; j < sizeDF; j += DF)
                        {
                            SynetInnerProductLayerForward<align>(src, weight0, weight1, j + 0 * F, sums + 0);
                            SynetInnerProductLayerForward<align>(src, weight0, weight1, j + 1 * F, sums + 2);
                        }
                        sums[0] = vaddq_f32(sums[0], sums[2]);
                        sums[1] = vaddq_f32(sums[1], sums[3]);
                    }
                    for (; j < sizeF; j += F)
                        SynetInnerProductLayerForward<align>(src, weight0, weight1, j, sums);
                    sum0 = ExtractSum32f(sums[0]);
                    sum1 = ExtractSum32f(sums[1]);
                }
                for (; j < size; ++j)
                {
                    sum0 += src[j] * weight0[j];
                    sum1 += src[j] * weight1[j];
                }
                dst[i + 0] = sum0 + (bias ? bias[i + 0] : 0);
                dst[i + 1] = sum1 + (bias ? bias[i + 1] : 0);
                weight += 2*size;
            }
            for (; i < count; ++i)
            {
                size_t j = 0;
                float sum = 0;
                if (sizeF)
                {
                    float32x4_t sums[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
                    if (sizeQF)
                    {
                        for (; j < sizeQF; j += QF)
                        {
                            SynetInnerProductLayerForward<align>(src, weight, j + 0 * F, sums[0]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 1 * F, sums[1]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 2 * F, sums[2]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 3 * F, sums[3]);
                        }
                        sums[0] = vaddq_f32(vaddq_f32(sums[0], sums[1]), vaddq_f32(sums[2], sums[3]));
                    }
                    for (; j < sizeF; j += F)
                        SynetInnerProductLayerForward<align>(src, weight, j, sums[0]);
                    sum = ExtractSum32f(sums[0]);
                }
                for (; j < size; ++j)
                    sum += src[j] * weight[j];
                dst[i] = sum + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst))
                SynetInnerProductLayerForward<true>(src, weight, bias, count, size, dst);
            else
                SynetInnerProductLayerForward<false>(src, weight, bias, count, size, dst);
        }

        //---------------------------------------------------------------------

        template<int shift> SIMD_INLINE float32x4_t LoadAtEdge(const float * src)
        {
            static const int32_t mask[3 * F] = { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 };
            return And(Load<false>(src + shift), Load<false>((float*)mask + F + shift));
        }

        SIMD_INLINE float32x4_t NoseSquareSum(const float * src)
        {
            float32x4_t s0 = LoadAtEdge<-2>(src);
            float32x4_t s1 = LoadAtEdge<-1>(src);
            float32x4_t s2 = Load<false>(src);
            float32x4_t s3 = Load<false>(src + 1);
            float32x4_t s4 = Load<false>(src + 2);
            return vaddq_f32(vmlaq_f32(vmulq_f32(s0, s0), s1, s1), vmlaq_f32(vmlaq_f32(vmulq_f32(s2, s2), s3, s3), s4, s4));
        }

        SIMD_INLINE float32x4_t BodySquareSum(const float * src)
        {
            float32x4_t s0 = Load<false>(src - 2);
            float32x4_t s1 = Load<false>(src - 1);
            float32x4_t s2 = Load<false>(src);
            float32x4_t s3 = Load<false>(src + 1);
            float32x4_t s4 = Load<false>(src + 2);
            return vaddq_f32(vmlaq_f32(vmulq_f32(s0, s0), s1, s1), vmlaq_f32(vmlaq_f32(vmulq_f32(s2, s2), s3, s3), s4, s4));
        }

        SIMD_INLINE float32x4_t TailSquareSum(const float * src)
        {
            float32x4_t s0 = Load<false>(src - 2);
            float32x4_t s1 = Load<false>(src - 1);
            float32x4_t s2 = Load<false>(src);
            float32x4_t s3 = LoadAtEdge<1>(src);
            float32x4_t s4 = LoadAtEdge<2>(src);
            return vaddq_f32(vmlaq_f32(vmulq_f32(s0, s0), s1, s1), vmlaq_f32(vmlaq_f32(vmulq_f32(s2, s2), s3, s3), s4, s4));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float32x4_t k0 = vdupq_n_f32(k[0]);
            float32x4_t k1 = vdupq_n_f32(k[1]);
            float32x4_t k2 = vdupq_n_f32(k[2]);
            Neon::Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    float32x4_t _pos = Neon::Load<align>(pos + s);
                    Neon::Store<true>(sum.data + s, vmlaq_f32(Neon::Load<true>(sum.data + s), _pos, _pos));
                }
                for (; s < spatial; ++s)
                    sum[s] += Simd::Square(pos[s]);
            }
            for (size_t c = 0; c < channels; ++c)
            {
                const float * pos = (c < channels - half) ? src + half * spatial : zero.data;
                const float * neg = (c > half) ? src - (half + 1) * spatial : zero.data;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    float32x4_t _pos = Neon::Load<align>(pos + s);
                    float32x4_t _neg = Neon::Load<align>(neg + s);
                    float32x4_t _sum = Neon::Load<true>(sum.data + s);
                    _sum = vmlsq_f32(vmlaq_f32(_sum, _pos, _pos), _neg, _neg);
                    float32x4_t _src = Neon::Load<align>(src + s);
                    Neon::Store<true>(sum.data + s, _sum);
                    Neon::Store<align>(dst + s, vmulq_f32(_src, pow(vmlaq_f32(k0, k1, _sum), k2)));
                }
                for (; s < spatial; ++s)
                {
                    sum[s] += Simd::Square(pos[s]);
                    sum[s] -= Simd::Square(neg[s]);
                    dst[s] = src[s] * Base::Pow(k[0] + k[1] * sum[s], k[2]);
                }
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(spatial, F))
                SynetLrnLayerCrossChannelsNchw<true>(src, half, channels, spatial, k, dst);
            else
                SynetLrnLayerCrossChannelsNchw<false>(src, half, channels, spatial, k, dst);
        }

        template<bool align> void SynetLrnLayerCrossChannelsNhwc2h(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            float32x4_t k0 = vdupq_n_f32(k[0]);
            float32x4_t k1 = vdupq_n_f32(k[1]);
            float32x4_t k2 = vdupq_n_f32(k[2]);
            Neon::Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Neon::Store<align>(dst + 0, vmulq_f32(Neon::Load<align>(src + 0), pow(vmlaq_f32(k0, k1, NoseSquareSum(src + 0)), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Neon::Store<align>(dst + c, vmulq_f32(Neon::Load<align>(src + c), pow(vmlaq_f32(k0, k1, BodySquareSum(src + c)), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Neon::Store<false>(dst + c, vmulq_f32(Neon::Load<false>(src + c), pow(vmlaq_f32(k0, k1, BodySquareSum(src + c)), k2)));
                }
                size_t c = channels - F;
                Neon::Store<false>(dst + c, vmulq_f32(Neon::Load<false>(src + c), pow(vmlaq_f32(k0, k1, TailSquareSum(src + c)), k2)));
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetLrnLayerCrossChannelsNhwc(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            if (half == 2 && channels >= F + half)
            {
                if (Aligned(src) && Aligned(dst) && Aligned(channels, F))
                    SynetLrnLayerCrossChannelsNhwc2h<true>(src, half, channels, spatial, k, dst);
                else
                    SynetLrnLayerCrossChannelsNhwc2h<false>(src, half, channels, spatial, k, dst);
            }
            else
                Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, SimdTensorFormatNhwc);
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNchw)
                SynetLrnLayerCrossChannelsNchw(src, half, channels, spatial, k, dst);
            else if (format == SimdTensorFormatNhwc)
                SynetLrnLayerCrossChannelsNhwc(src, half, channels, spatial, k, dst);
            else
                Base::SynetLrnLayerCrossChannels(src, half, channels, spatial, k, dst, format);
        }

        //---------------------------------------------------------------------

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, Fmadd<nofma>(Load<align>(src + offset), Load<align>(scale + offset), Load<align>(bias + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmulq_f32(Load<align>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float32x4_t & scale, const float32x4_t & bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, Fmadd<nofma>(Load<align>(src + offset), scale, bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float32x4_t & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, vmulq_f32(Load<align>(src + offset), scale));
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        float32x4_t _scale = vdupq_n_f32(scale[c]);
                        float32x4_t _bias = vdupq_n_f32(bias[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 0);
                            SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 1);
                            SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 2);
                            SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 3);
                        }
                        for (; s < partial; s += F)
                            SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s);
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * scale[c] + bias[c];
                    src += spatial;
                    dst += spatial;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    size_t s = 0;
                    if (partial)
                    {
                        float32x4_t _scale = vdupq_n_f32(scale[c]);
                        for (; s < aligned; s += QF)
                        {
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                            SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                        }
                        for (; s < partial; s += F)
                            SynetScaleLayerForward<align>(src, _scale, dst, s);
                    }
                    for (; s < spatial; ++s)
                        dst[s] = src[s] * scale[c];
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template<bool nofma> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetScaleLayerForwardNchw<true, nofma>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNchw<false, nofma>(src, scale, bias, channels, spatial, dst);
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            if (bias)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    if (partial)
                    {
                        for (; c < aligned; c += QF)
                        {
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < partial; c += F)
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c);
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c] + bias[c];
                    src += channels;
                    dst += channels;
                }
            }
            else
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    if (partial)
                    {
                        for (; c < aligned; c += QF)
                        {
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 3);
                        }
                        for (; c < partial; c += F)
                            SynetScaleLayerForward<align>(src, scale, dst, c);
                    }
                    for (; c < channels; ++c)
                        dst[c] = src[c] * scale[c];
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc3(const float * src, const float * scale, const float * bias, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            if (bias)
            {
                size_t s = 0;
                if (spatialF3)
                {
                    float _scale[F * 3], _bias[F * 3];
                    for (size_t i = 0; i < F; ++i)
                        for (size_t c = 0; c < 3; ++c)
                            _scale[i * 3 + c] = scale[c], _bias[i * 3 + c] = bias[c];
                    float32x4_t _scale0 = Load<false>(_scale + 0 * F);
                    float32x4_t _scale1 = Load<false>(_scale + 1 * F);
                    float32x4_t _scale2 = Load<false>(_scale + 2 * F);
                    float32x4_t _bias0 = Load<false>(_bias + 0 * F);
                    float32x4_t _bias1 = Load<false>(_bias + 1 * F);
                    float32x4_t _bias2 = Load<false>(_bias + 2 * F);
                    for (; s < spatialF3; s += F * 3)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale0, _bias0, dst, s + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale1, _bias1, dst, s + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale2, _bias2, dst, s + F * 2);
                    }
                }
                for (; s < spatial3; s += 3)
                {
                    dst[s + 0] = src[s + 0] * scale[0] + bias[0];
                    dst[s + 1] = src[s + 1] * scale[1] + bias[1];
                    dst[s + 2] = src[s + 2] * scale[2] + bias[2];
                }
            }
            else
            {
                size_t s = 0;
                if (spatialF3)
                {
                    float _scale[F * 3];
                    for (size_t i = 0; i < F; ++i)
                        for (size_t c = 0; c < 3; ++c)
                            _scale[i * 3 + c] = scale[c];
                    float32x4_t _scale0 = Load<false>(_scale + 0 * F);
                    float32x4_t _scale1 = Load<false>(_scale + 1 * F);
                    float32x4_t _scale2 = Load<false>(_scale + 2 * F);
                    for (; s < spatialF3; s += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, dst, s + F * 2);
                    }
                }
                for (; s < spatial3; s += 3)
                {
                    dst[s + 0] = src[s + 0] * scale[0];
                    dst[s + 1] = src[s + 1] * scale[1];
                    dst[s + 2] = src[s + 2] * scale[2];
                }
            }
        }

        template<bool nofma> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (channels == 3)
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNhwc3<true, nofma>(src, scale, bias, spatial, dst);
                else
                    SynetScaleLayerForwardNhwc3<false, nofma>(src, scale, bias, spatial, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true, nofma>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNhwc<false, nofma>(src, scale, bias, channels, spatial, dst);
            }
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw4c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            if (bias)
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    float32x4_t _scale = Load<false>(scale + c);
                    float32x4_t _bias = Load<false>(bias + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 2);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    float32x4_t _scale = Load<false>(scale + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align>(src, _scale, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
        }

        template<bool nofma> void SynetScaleLayerForwardNchw4c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetScaleLayerForwardNchw4c<true, nofma>(src, scale, bias, channels, spatial, dst);
            else
                SynetScaleLayerForwardNchw4c<false, nofma>(src, scale, bias, channels, spatial, dst);
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            bool nofma = Base::FmaAvoid(compatibility);
            if (Base::NchwCompatible(channels, spatial, format))
            {
                if(nofma)
                    SynetScaleLayerForwardNchw<true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw<false>(src, scale, bias, channels, spatial, dst);
            }
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (nofma)
                    SynetScaleLayerForwardNhwc<true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNhwc<false>(src, scale, bias, channels, spatial, dst);
            }
            else if (format == SimdTensorFormatNchw4c)
            {
                if (nofma)
                    SynetScaleLayerForwardNchw4c<true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw4c<false>(src, scale, bias, channels, spatial, dst);
            }
            else
                Base::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
        }

        //---------------------------------------------------------------------

        void SynetShuffleLayerForward(const float* src0, const float* src1, size_t channels0, size_t channels1, size_t spatial, float* dst0, float* dst1, SimdTensorFormatType format, int type)
        {
            if (format == SimdTensorFormatNchw)
                Base::SynetShuffleLayerForward(src0, src1, channels0, channels1, spatial, dst0, dst1, format, type);
            else if (format == SimdTensorFormatNhwc)
            {
                size_t channels = (channels0 + channels1) / 2;
                size_t channels0DF = AlignLo(channels0, DF);
                size_t channels1DF = AlignLo(channels1, DF);
                if (type == 0)
                {
                    for (size_t s = 0; s < spatial; ++s)
                    {
                        size_t cd = 0, cs0 = 0, cs1 = 0;
                        for (; cs0 < channels0DF; cs0 += DF, cd += F)
                        {
                            float32x4x2_t _src0 = Load2<false>(src0 + cs0);
                            Store<false>(dst0 + cd, _src0.val[0]);
                            Store<false>(dst1 + cd, _src0.val[1]);
                        }
                        for (; cs0 < channels0; cs0 += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs0 + 0];
                            dst1[cd] = src0[cs0 + 1];
                        }
                        for (; cs1 < channels1DF; cs1 += DF, cd += F)
                        {
                            float32x4x2_t _src1 = Load2<false>(src1 + cs1);
                            Store<false>(dst0 + cd, _src1.val[0]);
                            Store<false>(dst1 + cd, _src1.val[1]);
                        }
                        for (; cs1 < channels1; cs1 += 2, cd += 1)
                        {
                            dst0[cd] = src1[cs1 + 0];
                            dst1[cd] = src1[cs1 + 1];
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
                        for (; cd0 < channels0DF; cd0 += DF, cs += F)
                        {
                            float32x4x2_t s;
                            s.val[0] = Load<false>(src0 + cs);
                            s.val[1] = Load<false>(src1 + cs);
                            Store2<false>(dst0 + cd0, s);
                        }
                        for (; cd0 < channels0; cd0 += 2, cs += 1)
                        {
                            dst0[cd0 + 0] = src0[cs];
                            dst0[cd0 + 1] = src1[cs];
                        }
                        for (; cd1 < channels1DF; cd1 += DF, cs += F)
                        {
                            float32x4x2_t s;
                            s.val[0] = Load<false>(src0 + cs);
                            s.val[1] = Load<false>(src1 + cs);
                            Store2<false>(dst1 + cd1, s);
                        }
                        for (; cd1 < channels1; cd1 += 2, cs += 1)
                        {
                            dst1[cd1 + 0] = src0[cs];
                            dst1[cd1 + 1] = src1[cs];
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

        //---------------------------------------------------------------------

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            Exp exp;
            if (inner == 1 && count == 2)
            {
                size_t aligned = Simd::AlignLo(outer, F);
                size_t o = 0;
                for (; o < aligned; o += F)
                {
                    float32x4x2_t s = Load2<false>(src);
                    float32x4_t max = vmaxq_f32(s.val[0], s.val[1]);
                    float32x4_t exp0 = exp.Exponent(vsubq_f32(s.val[0], max));
                    float32x4_t exp1 = exp.Exponent(vsubq_f32(s.val[1], max));
                    float32x4_t sum = vaddq_f32(exp0, exp1);
                    float32x4x2_t d;
                    d.val[0] = Div<1>(exp0, sum);
                    d.val[1] = Div<1>(exp1, sum);
                    Store2<false>(dst, d);
                    src += DF;
                    dst += DF;
                }
                for (; o < outer; ++o)
                {
                    float max = Simd::Max(src[0], src[1]);
                    float exp0 = ::exp(src[0] - max);
                    float exp1 = ::exp(src[1] - max);
                    float sum = exp0 + exp1;
                    dst[0] = exp0 / sum;
                    dst[1] = exp1 / sum;
                    src += 2;
                    dst += 2;
                }
            }
            else
            {
                size_t aligned = Simd::AlignLo(inner, F);
                Array32f tmp(inner * 2);
                const float * s;
                float * max = tmp.data, *sum = tmp.data + inner, *d;
                for (size_t o = 0; o < outer; ++o)
                {
                    memcpy(max, src, inner * sizeof(float));
                    s = src + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            Store<false>(max + i, vmaxq_f32(Load<false>(s + i), Load<false>(max + i)));
                        for (; i < inner; ++i)
                            max[i] = Simd::Max(max[i], s[i]);
                        s += inner;
                    }

                    s = src;
                    d = dst;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                        {
                            float32x4_t _d = exp.Exponent(vsubq_f32(Load<false>(s + i), Load<false>(max + i)));
                            Store<false>(d + i, _d);
                            Store<false>(sum + i, vaddq_f32(_d, Load<false>(sum + i)));
                        }
                        for (; i < inner; ++i)
                        {
                            d[i] = ::exp(s[i] - max[i]);
                            sum[i] += d[i];
                        }
                        s += inner;
                        d += inner;
                    }

                    d = dst;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += F)
                            Store<false>(d + i, Div<1>(Load<false>(d + i), Load<false>(sum + i)));
                        for (; i < inner; ++i)
                            d[i] /= sum[i];
                        d += inner;
                    }
                    src += count * inner;
                    dst += count * inner;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdSynetUnaryOperation32fType type> float32x4_t SynetUnaryOperation32f(float32x4_t value);

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(float32x4_t value)
        {
            return vabsq_f32(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(float32x4_t value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(float32x4_t value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(float32x4_t value)
        {
            return vnegq_f32(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(float32x4_t value)
        {
            return ReciprocalSqrt<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(float32x4_t value)
        {
            return Sqrt<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(float32x4_t value)
        {
            return Tanh<1>(value);
        }

        template<> SIMD_INLINE float32x4_t SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(float32x4_t value)
        {
            return vdupq_n_f32(0.0f);
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Neon::Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 0 * F)));
                Neon::Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 1 * F)));
                Neon::Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 2 * F)));
                Neon::Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Neon::Store<align>(dst + i, SynetUnaryOperation32f<type>(Neon::Load<align>(src + i)));
            for (; i < size; ++i)
                dst[i] = Base::SynetUnaryOperation32f<type>(src[i]);
        }

        template<bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            switch (type)
            {
            case SimdSynetUnaryOperation32fAbs: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fAbs, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fExp: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fExp, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fLog: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fLog, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fNeg: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fNeg, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fRsqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fRsqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fSqrt: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fSqrt, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fTanh: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fTanh, align>(src, size, dst); break;
            case SimdSynetUnaryOperation32fZero: SynetUnaryOperation32fLayerForward<SimdSynetUnaryOperation32fZero, align>(src, size, dst); break;
            default:
                assert(0);
            }
        }

        void SynetUnaryOperation32fLayerForward(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetUnaryOperation32fLayerForward<true>(src, size, type, dst);
            else
                SynetUnaryOperation32fLayerForward<false>(src, size, type, dst);
        }
    }
#endif// SIMD_NEON_ENABLE
}

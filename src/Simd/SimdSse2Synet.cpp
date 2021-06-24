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
#include "Simd/SimdArray.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdGather.h"

namespace Simd
{
#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float* bias, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(__m128 bias, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), bias));
        }

        template <bool align> void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
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
                    __m128 _bias = _mm_set1_ps(bias[c]);
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

        SIMD_INLINE void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(spatial, F) && Aligned(dst))
                SynetAddBiasNchw<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
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

        SIMD_INLINE void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetAddBiasNhwc<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNhwc<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNchw4c(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(dst));

            size_t spatial4 = AlignLo(spatial, 4);
            for (size_t c = 0; c < channels; c += F)
            {
                __m128 _bias = Load<false>(bias + c);
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

        SIMD_INLINE void SynetAddBiasNchw4c(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(dst))
                SynetAddBiasNchw4c<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw4c<false>(bias, channels, spatial, dst);
        }

        void SynetAddBias(const float* bias, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
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

        template <SimdSynetEltwiseOperationType type> __m128 SynetEltwiseLayerForward(__m128 src0, __m128 src1);

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(__m128 src0, __m128 src1)
        {
            return _mm_mul_ps(src0, src1);
        }

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(__m128 src0, __m128 src1)
        {
            return _mm_max_ps(src0, src1);
        }

        template <> SIMD_INLINE __m128 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(__m128 src0, __m128 src1)
        {
            return _mm_min_ps(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align> SIMD_INLINE void SynetEltwiseLayerForward(const float* src0, const float* src1, float* dst, size_t offset)
        {
            Store<align>(dst + offset, SynetEltwiseLayerForward<type>(Load<align>(src0 + offset), Load<align>(src1 + offset)));
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const* const* src, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float* src0 = src[0];
            const float* src1 = src[1];
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
                const float* srci = src[i];
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

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src0, const __m128& weight0, const float* src1, const __m128& weight1, float* dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src0 + offset), weight0), _mm_mul_ps(Load<align>(src1 + offset), weight1)));
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float* src, const __m128& weight, float* dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src + offset), weight), Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const* const* src, const float* weight, size_t count, size_t size, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float* src0 = src[0];
            const float* src1 = src[1];
            __m128 weight0 = _mm_set1_ps(weight[0]);
            __m128 weight1 = _mm_set1_ps(weight[1]);
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
                const float* srci = src[i];
                __m128 weighti = _mm_set1_ps(weight[i]);
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

        template <bool align> void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst)
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

        void SynetEltwiseLayerForward(float const* const* src, const float* weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float* dst)
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

        template <bool align> SIMD_INLINE void SynetInnerProductLayerForward(const float* src, const float* weight, size_t offset, __m128& sum)
        {
            __m128 s = Load<align>(src + offset);
            __m128 w = Load<align>(weight + offset);
            sum = _mm_add_ps(_mm_mul_ps(s, w), sum);
        }

        template<bool align> void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst));
            size_t partial = AlignLo(size, F);
            size_t aligned = AlignLo(size, QF);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                float sum = 0;
                if (partial)
                {
                    __m128 sums[4] = { _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps() };
                    if (aligned)
                    {
                        for (; j < aligned; j += QF)
                        {
                            SynetInnerProductLayerForward<align>(src, weight, j + 0 * F, sums[0]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 1 * F, sums[1]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 2 * F, sums[2]);
                            SynetInnerProductLayerForward<align>(src, weight, j + 3 * F, sums[3]);
                        }
                        sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                    }
                    for (; j < partial; j += F)
                        SynetInnerProductLayerForward<align>(src, weight, j, sums[0]);
                    sum = ExtractSum(sums[0]);
                }
                for (; j < size; ++j)
                    sum += src[j] * weight[j];
                dst[i] = sum + (bias ? bias[i] : 0);
                weight += size;
            }
        }

        void SynetInnerProductLayerForward(const float* src, const float* weight, const float* bias, size_t count, size_t size, float* dst)
        {
            if (Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst))
                SynetInnerProductLayerForward<true>(src, weight, bias, count, size, dst);
            else
                SynetInnerProductLayerForward<false>(src, weight, bias, count, size, dst);
        }

        //---------------------------------------------------------------------

        template<int shift> SIMD_INLINE __m128 LoadAtEdge(const float * src)
        {
            static const int32_t mask[3 * F] = { 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0 };
            return _mm_and_ps(_mm_loadu_ps(src + shift), _mm_loadu_ps((float*)mask + F + shift));
        }

        SIMD_INLINE __m128 NoseSquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Square(LoadAtEdge<-2>(src)), Square(LoadAtEdge<-1>(src))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src + 1)), Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 BodySquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Square(_mm_loadu_ps(src - 2)), Square(_mm_loadu_ps(src - 1))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src + 1)), Square(_mm_loadu_ps(src + 2)))));
        }

        SIMD_INLINE __m128 TailSquareSum(const float * src)
        {
            return _mm_add_ps(_mm_add_ps(Square(LoadAtEdge<2>(src)), Square(LoadAtEdge<1>(src))),
                _mm_add_ps(Square(_mm_loadu_ps(src)), _mm_add_ps(Square(_mm_loadu_ps(src - 1)), Square(_mm_loadu_ps(src - 2)))));
        }

        template<bool align> void SynetLrnLayerCrossChannelsNchw(const float * src, size_t half, size_t channels, size_t spatial, const float * k, float * dst)
        {
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Sse2::Pow pow;
            Array32f sum(spatial, true), zero(spatial, true);
            size_t aligned = AlignLo(spatial, F);
            for (size_t c = 0; c < half; ++c)
            {
                const float * pos = src + c * spatial;
                size_t s = 0;
                for (; s < aligned; s += F)
                {
                    __m128 _pos = Load<align>(pos + s);
                    Store<true>(sum.data + s, _mm_add_ps(Load<true>(sum.data + s), _mm_mul_ps(_pos, _pos)));
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
                    __m128 _pos = Load<align>(pos + s);
                    __m128 _neg = Load<align>(neg + s);
                    __m128 _sum = Load<true>(sum.data + s);
                    _sum = _mm_add_ps(_sum, _mm_sub_ps(_mm_mul_ps(_pos, _pos), _mm_mul_ps(_neg, _neg)));
                    __m128 _src = Load<align>(src + s);
                    Store<true>(sum.data + s, _sum);
                    Store<align>(dst + s, _mm_mul_ps(_src, pow(_mm_add_ps(k0, _mm_mul_ps(k1, _sum)), k2)));
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
            __m128 k0 = _mm_set1_ps(k[0]);
            __m128 k1 = _mm_set1_ps(k[1]);
            __m128 k2 = _mm_set1_ps(k[2]);
            Sse2::Pow pow;
            size_t aligned = AlignLo(channels - half, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                Store<align>(dst + 0, _mm_mul_ps(Load<align>(src + 0), pow(_mm_add_ps(k0, _mm_mul_ps(k1, NoseSquareSum(src + 0))), k2)));
                for (size_t c = F; c < aligned; c += F)
                    Store<align>(dst + c, _mm_mul_ps(Load<align>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                if (aligned != channels - half)
                {
                    size_t c = channels - half - F;
                    Store<false>(dst + c, _mm_mul_ps(Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, BodySquareSum(src + c))), k2)));
                }
                size_t c = channels - F;
                Store<false>(dst + c, _mm_mul_ps(Load<false>(src + c), pow(_mm_add_ps(k0, _mm_mul_ps(k1, TailSquareSum(src + c))), k2)));
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
                            __m128 s0 = _mm_loadu_ps(src0 + cs0 + 0);
                            __m128 s1 = _mm_loadu_ps(src0 + cs0 + F);
                            _mm_storeu_ps(dst0 + cd, _mm_shuffle_ps(s0, s1, 0x88));
                            _mm_storeu_ps(dst1 + cd, _mm_shuffle_ps(s0, s1, 0xDD));
                        }
                        for (; cs0 < channels0; cs0 += 2, cd += 1)
                        {
                            dst0[cd] = src0[cs0 + 0];
                            dst1[cd] = src0[cs0 + 1];
                        }
                        for (; cs1 < channels1DF; cs1 += DF, cd += F)
                        {
                            __m128 s0 = _mm_loadu_ps(src1 + cs1 + 0);
                            __m128 s1 = _mm_loadu_ps(src1 + cs1 + F);
                            _mm_storeu_ps(dst0 + cd, _mm_shuffle_ps(s0, s1, 0x88));
                            _mm_storeu_ps(dst1 + cd, _mm_shuffle_ps(s0, s1, 0xDD));
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
                            __m128 s0 = _mm_loadu_ps(src0 + cs);
                            __m128 s1 = _mm_loadu_ps(src1 + cs);
                            _mm_storeu_ps(dst0 + cd0 + 0, _mm_unpacklo_ps(s0, s1));
                            _mm_storeu_ps(dst0 + cd0 + F, _mm_unpackhi_ps(s0, s1));
                        }
                        for (; cd0 < channels0; cd0 += 2, cs += 1)
                        {
                            dst0[cd0 + 0] = src0[cs];
                            dst0[cd0 + 1] = src1[cs];
                        }
                        for (; cd1 < channels1DF; cd1 += DF, cs += F)
                        {
                            __m128 s0 = _mm_loadu_ps(src0 + cs);
                            __m128 s1 = _mm_loadu_ps(src1 + cs);
                            _mm_storeu_ps(dst1 + cd1 + 0, _mm_unpacklo_ps(s0, s1));
                            _mm_storeu_ps(dst1 + cd1 + F, _mm_unpackhi_ps(s0, s1));
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

        void SynetSoftmaxLayerForward21(const float* src, size_t outer, float* dst)
        {
            Sse2::Exp exp;
            size_t aligned = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < aligned; o += F)
            {
                __m128 s0 = _mm_loadu_ps(src + 0);
                __m128 s1 = _mm_loadu_ps(src + F);
                __m128 ss0 = _mm_shuffle_ps(s0, s1, 0x88);
                __m128 ss1 = _mm_shuffle_ps(s0, s1, 0xDD);
                __m128 max = _mm_max_ps(ss0, ss1);
                __m128 exp0 = exp.Exponent(_mm_sub_ps(ss0, max));
                __m128 exp1 = exp.Exponent(_mm_sub_ps(ss1, max));
                __m128 sum = _mm_add_ps(exp0, exp1);
                __m128 d0 = _mm_div_ps(exp0, sum);
                __m128 d1 = _mm_div_ps(exp1, sum);
                _mm_storeu_ps(dst + 0, _mm_unpacklo_ps(d0, d1));
                _mm_storeu_ps(dst + F, _mm_unpackhi_ps(d0, d1));
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

        SIMD_INLINE void SynetSoftmaxLayerForward31(const Sse2::Exp& exp, __m128 buf[3])
        {
            __m128 max = _mm_max_ps(buf[0], _mm_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm_sub_ps(buf[2], max));
            __m128 sum = _mm_add_ps(buf[0], _mm_add_ps(buf[1], buf[2]));
            buf[0] = _mm_div_ps(buf[0], sum);
            buf[1] = _mm_div_ps(buf[1], sum);
            buf[2] = _mm_div_ps(buf[2], sum);
        }

        void SynetSoftmaxLayerForward31(const float* src, size_t outer, float* dst)
        {
            Sse2::Exp exp;
            __m128 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                buf[0] = Gather<3>(src + 0);
                buf[1] = Gather<3>(src + 1);
                buf[2] = Gather<3>(src + 2);
                SynetSoftmaxLayerForward31(exp, buf);
                Scater<3>(dst + 0, buf[0]);
                Scater<3>(dst + 1, buf[1]);
                Scater<3>(dst + 2, buf[2]);
                src += 3*F;
                dst += 3*F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                buf[0] = Gather<3>(src + 0, tail);
                buf[1] = Gather<3>(src + 1, tail);
                buf[2] = Gather<3>(src + 2, tail);
                SynetSoftmaxLayerForward31(exp, buf);
                Scater<3>(dst + 0, buf[0], tail);
                Scater<3>(dst + 1, buf[1], tail);
                Scater<3>(dst + 2, buf[2], tail);
            }
        }

        void SynetSoftmaxLayerForward(const float * src, size_t outer, size_t count, size_t inner, float * dst)
        {
            if (count == 2 && inner == 1)
                SynetSoftmaxLayerForward21(src, outer, dst);
            else if(count == 3 && inner == 1)
                SynetSoftmaxLayerForward31(src, outer, dst);
            else
            {
                Sse2::Exp exp;
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
                            _mm_storeu_ps(max + i, _mm_max_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
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
                            __m128 _d = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(s + i), _mm_loadu_ps(max + i)));
                            _mm_storeu_ps(d + i, _d);
                            _mm_storeu_ps(sum + i, _mm_add_ps(_d, _mm_loadu_ps(sum + i)));
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
                            _mm_storeu_ps(d + i, _mm_div_ps(_mm_loadu_ps(d + i), _mm_loadu_ps(sum + i)));
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

        template<SimdSynetUnaryOperation32fType type> __m128 SynetUnaryOperation32f(__m128 value);

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fAbs>(__m128 value)
        {
            return _mm_andnot_ps(_mm_set1_ps(-0.0f), value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fExp>(__m128 value)
        {
            return Exponent(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fLog>(__m128 value)
        {
            return Logarithm(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fNeg>(__m128 value)
        {
            return _mm_sub_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fRsqrt>(__m128 value)
        {
            return _mm_rsqrt_ps(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fSqrt>(__m128 value)
        {
            return _mm_sqrt_ps(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fTanh>(__m128 value)
        {
            return Tanh(value);
        }

        template<> SIMD_INLINE __m128 SynetUnaryOperation32f<SimdSynetUnaryOperation32fZero>(__m128 value)
        {
            return _mm_setzero_ps();
        }

        template<SimdSynetUnaryOperation32fType type, bool align> void SynetUnaryOperation32fLayerForward(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 0 * F)));
                Store<align>(dst + i + 1 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 1 * F)));
                Store<align>(dst + i + 2 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 2 * F)));
                Store<align>(dst + i + 3 * F, SynetUnaryOperation32f<type>(Load<align>(src + i + 3 * F)));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, SynetUnaryOperation32f<type>(Load<align>(src + i)));
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
#endif// SIMD_SSE2_ENABLE
}

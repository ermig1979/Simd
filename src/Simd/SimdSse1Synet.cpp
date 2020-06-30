/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse1.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, float * dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(__m128 bias, float * dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), bias));
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
                __m128 _bias = Load<false>(bias + c);
                size_t s = 0;
                for (; s < spatial4; s += 4, dst += 4*F)
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

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src0, const __m128 & weight0, const float * src1, const __m128 & weight1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src0 + offset), weight0), _mm_mul_ps(Load<align>(src1 + offset), weight1)));
        }

        template <bool align> SIMD_INLINE void SynetEltwiseLayerForwardSum(const float * src, const __m128 & weight, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src + offset), weight), Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
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
                const float * srci = src[i];
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

        template <bool align> SIMD_INLINE void SynetInnerProductLayerForward(const float * src, const float * weight, size_t offset, __m128 & sum)
        {
            __m128 s = Load<align>(src + offset);
            __m128 w = Load<align>(weight + offset);
            sum = _mm_add_ps(_mm_mul_ps(s, w), sum);
        }

        template<bool align> void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
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

        void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(weight) && Aligned(size) && Aligned(dst))
                SynetInnerProductLayerForward<true>(src, weight, bias, count, size, dst);
            else
                SynetInnerProductLayerForward<false>(src, weight, bias, count, size, dst);
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
    }
#endif// SIMD_SSE_ENABLE
}

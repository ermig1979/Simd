/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"

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

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetAddBias<align>(bias + i + F * 0, dst + i + F * 0);
                            SynetAddBias<align>(bias + i + F * 1, dst + i + F * 1);
                            SynetAddBias<align>(bias + i + F * 2, dst + i + F * 2);
                            SynetAddBias<align>(bias + i + F * 3, dst + i + F * 3);
                        }
                        for (; i < partial; i += F)
                            SynetAddBias<align>(bias + i, dst + i);
                    }
                    for (; i < count; ++i)
                        dst[i] += bias[i];
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);     
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m128 _bias = _mm_set1_ps(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetAddBias<align>(_bias, dst + j + F * 0);
                            SynetAddBias<align>(_bias, dst + j + F * 1);
                            SynetAddBias<align>(_bias, dst + j + F * 2);
                            SynetAddBias<align>(_bias, dst + j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetAddBias<align>(_bias, dst + j);
                    }
                    for (; j < size; ++j)
                        dst[j] += bias[i];
                    dst += size;
                }
            }
        }

        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst))
                SynetAddBias<true>(bias, count, size, dst, trans);
            else
                SynetAddBias<false>(bias, count, size, dst, trans);
        }

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

        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, __m128 bias, __m128 scale, __m128 sign, float * dst)
        {
            __m128 x = _mm_add_ps(Load<align>(src), bias);
            Store<align>(dst, _mm_add_ps(_mm_mul_ps(_mm_sub_ps(x, _mm_andnot_ps(sign, x)), scale), _mm_max_ps(_mm_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(size) && Aligned(dst));
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __m128 sign = _mm_set1_ps(-0.0f);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                if (partial)
                {
                    __m128 _bias = _mm_set1_ps(bias[i]);
                    __m128 _scale = _mm_set1_ps(scale[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetFusedLayerForward0<align>(src + j + 0 * F, _bias, _scale, sign, dst + j + 0 * F);
                        SynetFusedLayerForward0<align>(src + j + 1 * F, _bias, _scale, sign, dst + j + 1 * F);
                        SynetFusedLayerForward0<align>(src + j + 2 * F, _bias, _scale, sign, dst + j + 2 * F);
                        SynetFusedLayerForward0<align>(src + j + 3 * F, _bias, _scale, sign, dst + j + 3 * F);
                    }
                    for (; j < partial; j += F)
                        SynetFusedLayerForward0<align>(src + j, _bias, _scale, sign, dst + j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetFusedLayerForward0(src[j] + bias[i], scale[i]);
                src += size;
                dst += size;
            }
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(size) && Aligned(dst))
                SynetFusedLayerForward0<true>(src, bias, scale, count, size, dst);
            else
                SynetFusedLayerForward0<false>(src, bias, scale, count, size, dst);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, __m128 bias0, __m128 scale1, __m128 bias1, float * dst)
        {
            __m128 x = _mm_add_ps(Load<align>(src), bias0);
            Store<align>(dst, _mm_add_ps(_mm_add_ps(_mm_mul_ps(_mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(_mm_setzero_ps(), x)), scale1), bias1), _mm_max_ps(_mm_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(size) && Aligned(dst));
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                if (partial)
                {
                    __m128 _bias0 = _mm_set1_ps(bias0[i]);
                    __m128 _scale1 = _mm_set1_ps(scale1[i]);
                    __m128 _bias1 = _mm_set1_ps(bias1[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetFusedLayerForward1<align>(src + j + 0 * F, _bias0, _scale1, _bias1, dst + j + 0 * F);
                        SynetFusedLayerForward1<align>(src + j + 1 * F, _bias0, _scale1, _bias1, dst + j + 1 * F);
                        SynetFusedLayerForward1<align>(src + j + 2 * F, _bias0, _scale1, _bias1, dst + j + 2 * F);
                        SynetFusedLayerForward1<align>(src + j + 3 * F, _bias0, _scale1, _bias1, dst + j + 3 * F);
                    }
                    for (; j < partial; j += F)
                        SynetFusedLayerForward1<align>(src + j, _bias0, _scale1, _bias1, dst + j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetFusedLayerForward1(src[j] + bias0[i], scale1[i], bias1[i]);
                src += size;
                dst += size;
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst)
        {
            if (Aligned(src) && Aligned(size) && Aligned(dst))
                SynetFusedLayerForward1<true>(src, bias0, scale1, bias1, count, size, dst);
            else
                SynetFusedLayerForward1<false>(src, bias0, scale1, bias1, count, size, dst);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, __m128 scale, __m128 bias, __m128 slope, float * dst)
        {
            __m128 x = _mm_add_ps(_mm_mul_ps(Load<align>(src), scale), bias);
            Store<align>(dst, _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), x), _mm_mul_ps(_mm_min_ps(_mm_setzero_ps(), x), slope)));
        }

        template <bool align> void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(size) && Aligned(dst));
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __m128 _slope = _mm_set1_ps(slope[0]);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                if (partial)
                {
                    __m128 _scale = _mm_set1_ps(scale[i]);
                    __m128 _bias = _mm_set1_ps(bias[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetFusedLayerForward2<align>(src + j + 0 * F, _scale, _bias, _slope, dst + j + 0 * F);
                        SynetFusedLayerForward2<align>(src + j + 1 * F, _scale, _bias, _slope, dst + j + 1 * F);
                        SynetFusedLayerForward2<align>(src + j + 2 * F, _scale, _bias, _slope, dst + j + 2 * F);
                        SynetFusedLayerForward2<align>(src + j + 3 * F, _scale, _bias, _slope, dst + j + 3 * F);
                    }
                    for (; j < partial; j += F)
                        SynetFusedLayerForward2<align>(src + j, _scale, _bias, _slope, dst + j);
                }
                for (; j < size; ++j)
                    dst[j] = Base::SynetFusedLayerForward2(src[j], scale[i], bias[i], slope[0]);
                src += size;
                dst += size;
            }
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst)
        {
            if (Aligned(src) && Aligned(size) && Aligned(dst))
                SynetFusedLayerForward2<true>(src, scale, bias, count, size, slope, dst);
            else
                SynetFusedLayerForward2<false>(src, scale, bias, count, size, slope, dst);
        }

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

        template <bool align> void SynetRestrictRange(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            __m128 _min = _mm_set1_ps(min);
            __m128 _max = _mm_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm_min_ps(_mm_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm_min_ps(_mm_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm_min_ps(_mm_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm_min_ps(_mm_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm_min_ps(_mm_max_ps(_min, Load<align>(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        void SynetRestrictRange(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange<false>(src, size, lower, upper, dst);
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)), Load<align>(bias + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m128 & scale, const __m128 & bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_add_ps(_mm_mul_ps(Load<align>(src + offset), scale), bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m128 & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm_mul_ps(Load<align>(src + offset), scale));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                if (bias)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        if (partial)
                        {
                            for (; i < aligned; i += QF)
                            {
                                SynetScaleLayerForward<align>(src, scale, bias, dst, i + F * 0);
                                SynetScaleLayerForward<align>(src, scale, bias, dst, i + F * 1);
                                SynetScaleLayerForward<align>(src, scale, bias, dst, i + F * 2);
                                SynetScaleLayerForward<align>(src, scale, bias, dst, i + F * 3);
                            }
                            for (; i < partial; i += F)
                                SynetScaleLayerForward<align>(src, scale, bias, dst, i);
                        }
                        for (; i < count; ++i)
                            dst[i] = src[i] * scale[i] + bias[i];
                        src += count;
                        dst += count;
                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        if (partial)
                        {
                            for (; i < aligned; i += QF)
                            {
                                SynetScaleLayerForward<align>(src, scale, dst, i + F * 0);
                                SynetScaleLayerForward<align>(src, scale, dst, i + F * 1);
                                SynetScaleLayerForward<align>(src, scale, dst, i + F * 2);
                                SynetScaleLayerForward<align>(src, scale, dst, i + F * 3);
                            }
                            for (; i < partial; i += F)
                                SynetScaleLayerForward<align>(src, scale, dst, i);
                        }
                        for (; i < count; ++i)
                            dst[i] = src[i]*scale[i];
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            __m128 _scale = _mm_set1_ps(scale[i]);
                            __m128 _bias = _mm_set1_ps(bias[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 0);
                                SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 1);
                                SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 2);
                                SynetScaleLayerForward<align>(src, _scale, _bias, dst, j + F * 3);
                            }
                            for (; j < partial; j += F)
                                SynetScaleLayerForward<align>(src, _scale, _bias, dst, j);
                        }
                        for (; j < size; ++j)
                            dst[j] = src[j] * scale[i] + bias[i];
                        src += size;
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        size_t j = 0;
                        if (partial)
                        {
                            __m128 _scale = _mm_set1_ps(scale[i]);
                            for (; j < aligned; j += QF)
                            {
                                SynetScaleLayerForward<align>(src, _scale, dst, j + F * 0);
                                SynetScaleLayerForward<align>(src, _scale, dst, j + F * 1);
                                SynetScaleLayerForward<align>(src, _scale, dst, j + F * 2);
                                SynetScaleLayerForward<align>(src, _scale, dst, j + F * 3);
                            }
                            for (; j < partial; j += F)
                                SynetScaleLayerForward<align>(src, _scale, dst, j);
                        }
                        for (; j < size; ++j)
                            dst[j] = src[j] * scale[i];
                        src += size;
                        dst += size;
                    }
                }
            }
        }

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetScaleLayerForward<true>(src, scale, bias, count, size, dst, trans);
            else
                SynetScaleLayerForward<false>(src, scale, bias, count, size, dst, trans);
        }
    }
#endif// SIMD_SSE_ENABLE
}

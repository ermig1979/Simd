/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdSse1.h"
#include "Simd/SimdAvx1.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(__m256 bias, float * dst)
        {
            Store<align>(dst, _mm256_add_ps(Load<align>(dst), bias));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
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
                        __m256 _bias = _mm256_set1_ps(bias[i]);
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
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst))
                SynetAddBias<true>(bias, count, size, dst, trans);
            else
                SynetAddBias<false>(bias, count, size, dst, trans);
        }

        template <SimdSynetEltwiseOperationType type> __m256 SynetEltwiseLayerForward(__m256 src0, __m256 src1);

        template <> SIMD_INLINE __m256 SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(__m256 src0, __m256 src1)
        {
            return _mm256_mul_ps(src0, src1);
        }

        template <> SIMD_INLINE __m256 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(__m256 src0, __m256 src1)
        {
            return _mm256_max_ps(src0, src1);
        }

        template <> SIMD_INLINE __m256 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(__m256 src0, __m256 src1)
        {
            return _mm256_min_ps(src0, src1);
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

        template <bool align> void SynetEltwiseLayerForwardSum(const float * src0, const __m256 & weight0, const float * src1, const __m256 & weight1, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src0 + offset), weight0), _mm256_mul_ps(Load<align>(src1 + offset), weight1)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(const float * src, const __m256 & weight, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), weight), Load<align>(dst + offset)));
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            const float * src0 = src[0];
            const float * src1 = src[1];
            __m256 weight0 = _mm256_set1_ps(weight[0]);
            __m256 weight1 = _mm256_set1_ps(weight[1]);
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
                __m256 weighti = _mm256_set1_ps(weight[i]);
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

        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias);
            __m256 _scale = Load<align>(scale + offset);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(x, _mm256_andnot_ps(sign, x)), _scale), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward0(const float * src, __m256 bias, __m256 scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(x, _mm256_andnot_ps(sign, x)), scale), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            __m256 sign = _mm256_set1_ps(-0.0f);
            if ((trans || size == 1) && count != 1)
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
                            SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, i + 0 * F);
                            SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, i + 1 * F);
                            SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, i + 2 * F);
                            SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward0<align>(src, bias, scale, sign, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward0(src[i] + bias[i], scale[i]);
                    src += count;
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
                        __m256 _bias = _mm256_set1_ps(bias[i]);
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, j + 0 * F);
                            SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, j + 1 * F);
                            SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, j + 2 * F);
                            SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward0<align>(src, _bias, _scale, sign, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward0(src[j] + bias[i], scale[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward0<true>(src, bias, scale, count, size, dst, trans);
            else
                SynetFusedLayerForward0<false>(src, bias, scale, count, size, dst, trans);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float *  bias1, float * dst, size_t offset)
        {
            __m256 _bias0 = Load<align>(bias0 + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias0);
            __m256 _scale1 = Load<align>(scale1 + offset);
            __m256 _bias1 = Load<align>(bias1 + offset);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(_mm256_setzero_ps(), x)), _scale1), _bias1), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward1(const float * src, __m256 bias0, __m256 scale1, __m256 bias1, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias0);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_setzero_ps(), _mm256_sub_ps(_mm256_setzero_ps(), x)), scale1), bias1), _mm256_max_ps(_mm256_setzero_ps(), x)));
        }

        template <bool align> void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
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
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, i + 0 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, i + 1 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, i + 2 * F);
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward1<align>(src, bias0, scale1, bias1, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward1(src[i] + bias0[i], scale1[i], bias1[i]);
                    src += count;
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
                        __m256 _bias0 = _mm256_set1_ps(bias0[i]);
                        __m256 _scale1 = _mm256_set1_ps(scale1[i]);
                        __m256 _bias1 = _mm256_set1_ps(bias1[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, j + 0 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, j + 1 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, j + 2 * F);
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward1<align>(src, _bias0, _scale1, _bias1, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward1(src[j] + bias0[i], scale1[i], bias1[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward1<true>(src, bias0, scale1, bias1, count, size, dst, trans);
            else
                SynetFusedLayerForward1<false>(src, bias0, scale1, bias1, count, size, dst, trans);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, __m256 slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _scale = Load<align>(scale + offset);
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(_mm256_mul_ps(_src, _scale), _bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), x), _mm256_mul_ps(_mm256_min_ps(_mm256_setzero_ps(), x), slope)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward2(const float * src, __m256 scale, __m256 bias, __m256 slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 x = _mm256_add_ps(_mm256_mul_ps(_src, scale), bias);
            Store<align>(dst + offset, _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), x), _mm256_mul_ps(_mm256_min_ps(_mm256_setzero_ps(), x), slope)));
        }

        template <bool align> void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            __m256 _slope = _mm256_set1_ps(slope[0]);
            if ((trans || size == 1) && count != 1)
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
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, i + 0 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, i + 1 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, i + 2 * F);
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward2<align>(src, scale, bias, _slope, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward2(src[i], scale[i], bias[i], slope[0]);
                    src += count;
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
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        __m256 _bias = _mm256_set1_ps(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, j + 0 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, j + 1 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, j + 2 * F);
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward2<align>(src, _scale, _bias, _slope, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward2(src[j], scale[i], bias[i], slope[0]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward2<true>(src, scale, bias, count, size, slope, dst, trans);
            else
                SynetFusedLayerForward2<false>(src, scale, bias, count, size, slope, dst, trans);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 _bias = Load<align>(bias + offset);
            __m256 x = _mm256_add_ps(Load<align>(src + offset), _bias);
            __m256 _scale = Load<align>(scale + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), x);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), x);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(_scale, neg)));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward3(const float * src, __m256 bias, __m256 scale, __m256 sign, float * dst, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), x);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), x);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(scale, neg)));
        }

        template <bool align> void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            __m256 sign = _mm256_set1_ps(-0.0f);
            if ((trans || size == 1) && count != 1)
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
                            SynetFusedLayerForward3<align>(src, bias, scale, sign, dst, i + 0 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, sign, dst, i + 1 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, sign, dst, i + 2 * F);
                            SynetFusedLayerForward3<align>(src, bias, scale, sign, dst, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward3<align>(src, bias, scale, sign, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetFusedLayerForward3(src[i] + bias[i], scale[i]);
                    src += count;
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
                        __m256 _bias = _mm256_set1_ps(bias[i]);
                        __m256 _scale = _mm256_set1_ps(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward3<align>(src, _bias, _scale, sign, dst, j + 0 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, sign, dst, j + 1 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, sign, dst, j + 2 * F);
                            SynetFusedLayerForward3<align>(src, _bias, _scale, sign, dst, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward3<align>(src, _bias, _scale, sign, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetFusedLayerForward3(src[j] + bias[i], scale[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward3(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward3<true>(src, bias, scale, count, size, dst, trans);
            else
                SynetFusedLayerForward3<false>(src, bias, scale, count, size, dst, trans);
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, const float * bias0, __m256 scale1, __m256 bias1, float * dst0, float * dst1, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), Load<align>(bias0 + offset));
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), x));
            Store<align>(dst1 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(bias1, _mm256_mul_ps(scale1, x))));
        }

        template <bool align> SIMD_INLINE void SynetFusedLayerForward4(const float * src, __m256 bias0, __m256 scale1, __m256 bias1, float * dst0, float * dst1, size_t offset)
        {
            __m256 x = _mm256_add_ps(Load<align>(src + offset), bias0);
            Store<align>(dst0 + offset, _mm256_max_ps(_mm256_setzero_ps(), x));
            Store<align>(dst1 + offset, _mm256_max_ps(_mm256_setzero_ps(), _mm256_add_ps(bias1, _mm256_mul_ps(scale1, x))));
        }

        template<bool align> void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) : Aligned(size)) && Aligned(src) && Aligned(dst));
            __m256 _scale1 = _mm256_set1_ps(scale1[0]);
            __m256 _bias1 = _mm256_set1_ps(bias1[0]);
            if ((trans || size == 1) && count != 1)
            {
                float * dst0 = dst, *dst1 = dst + count;
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    if (partial)
                    {
                        for (; i < aligned; i += QF)
                        {
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, i + 0 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, i + 1 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, i + 2 * F);
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, i + 3 * F);
                        }
                        for (; i < partial; i += F)
                            SynetFusedLayerForward4<align>(src, bias0, _scale1, _bias1, dst0, dst1, i);
                    }
                    for (; i < count; ++i)
                        Base::SynetFusedLayerForward4(src[i], bias0[i], scale1[0], bias1[0], dst0 + i, dst1 + i);
                    src += count;
                    dst0 += 2 * count;
                    dst1 += 2 * count;
                }
            }
            else
            {
                float * dst0 = dst, *dst1 = dst + count * size;
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    if (partial)
                    {
                        __m256 _bias0 = _mm256_set1_ps(bias0[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, j + 0 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, j + 1 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, j + 2 * F);
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, j + 3 * F);
                        }
                        for (; j < partial; j += F)
                            SynetFusedLayerForward4<align>(src, _bias0, _scale1, _bias1, dst0, dst1, j);
                    }
                    for (; j < size; ++j)
                        Base::SynetFusedLayerForward4(src[j], bias0[i], scale1[0], bias1[0], dst0 + j, dst1 + j);
                    src += size;
                    dst0 += size;
                    dst1 += size;
                }
            }
        }

        void SynetFusedLayerForward4(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(bias0) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward4<true>(src, bias0, scale1, bias1, count, size, dst, trans);
            else
                SynetFusedLayerForward4<false>(src, bias0, scale1, bias1, count, size, dst, trans);
        }

        SIMD_INLINE __m256 Tail(size_t tail)
        {
            const int32_t mask[DF] = { 0, 0, 0, 0, 0, 0, 0, 0 , -1, -1, -1, -1, -1, -1, -1, -1 };
            return _mm256_loadu_ps((float*)(mask + tail));
        }

        void SynetInnerProductLayerForward1(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K8 = K & (~7);
            size_t K32 = K & (~31);
            const float * W0 = W + 0 * K;
            __m256 d00, d01, d02, d03;
            __m256 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t k = 0;
            d00 = _mm256_setzero_ps();
            if (K32)
            {
                d01 = _mm256_setzero_ps();
                d02 = _mm256_setzero_ps();
                d03 = _mm256_setzero_ps();
                for (; k < K32; k += 32)
                {
                    s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm256_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
                    d01 = _mm256_add_ps(_mm256_mul_ps(s1, w1), d01);
                    s2 = _mm256_loadu_ps(S0 + k + 2 * F);
                    s3 = _mm256_loadu_ps(S0 + k + 3 * F);
                    w2 = _mm256_loadu_ps(W0 + k + 2 * F);
                    w3 = _mm256_loadu_ps(W0 + k + 3 * F);
                    d02 = _mm256_add_ps(_mm256_mul_ps(s2, w2), d02);
                    d03 = _mm256_add_ps(_mm256_mul_ps(s3, w3), d03);
                }
                d00 = _mm256_add_ps(_mm256_add_ps(d00, d01), _mm256_add_ps(d02, d03));
            }
            for (; k < K8; k += 8)
            {
                s0 = _mm256_loadu_ps(S0 + k);
                w0 = _mm256_loadu_ps(W0 + k);
                d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                s0 = _mm256_and_ps(tail, _mm256_loadu_ps(S0 + k));
                w0 = _mm256_loadu_ps(W0 + k);
                d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
            }
            D[0] = Avx::ExtractSum(d00) + B[0];
        }

        void SynetInnerProductLayerForward4(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K8 = K & (~7);
            size_t K16 = K & (~15);
            const float * W0 = W + 0 * K;
            const float * W1 = W + 1 * K;
            const float * W2 = W + 2 * K;
            const float * W3 = W + 3 * K;
            __m256 d00, d01, d10, d11, d20, d21, d30, d31;
            __m256 s0, s1, w0, w1;
            size_t k = 0;
            d00 = _mm256_setzero_ps();
            d10 = _mm256_setzero_ps();
            d20 = _mm256_setzero_ps();
            d30 = _mm256_setzero_ps();
            if (K16)
            {
                d01 = _mm256_setzero_ps();
                d11 = _mm256_setzero_ps();
                d21 = _mm256_setzero_ps();
                d31 = _mm256_setzero_ps();
                for (; k < K16; k += 16)
                {
                    s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm256_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
                    d01 = _mm256_add_ps(_mm256_mul_ps(s1, w1), d01);
                    w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W1 + k + 1 * F);
                    d10 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d10);
                    d11 = _mm256_add_ps(_mm256_mul_ps(s1, w1), d11);
                    w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W2 + k + 1 * F);
                    d20 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d20);
                    d21 = _mm256_add_ps(_mm256_mul_ps(s1, w1), d21);
                    w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                    w1 = _mm256_loadu_ps(W3 + k + 1 * F);
                    d30 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d30);
                    d31 = _mm256_add_ps(_mm256_mul_ps(s1, w1), d31);
                }
                d00 = _mm256_add_ps(d00, d01);
                d10 = _mm256_add_ps(d10, d11);
                d20 = _mm256_add_ps(d20, d21);
                d30 = _mm256_add_ps(d30, d31);
            }
            for (; k < K8; k += 8)
            {
                s0 = _mm256_loadu_ps(S0 + k + 0 * F);
                w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
                w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                d10 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d10);
                w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                d20 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d20);
                w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                d30 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d30);
            }
            if (K8 < K)
            {
                size_t k = K - 8;
                __m256 tail = Tail(K - K8);
                s0 = _mm256_and_ps(tail, _mm256_loadu_ps(S0 + k));
                w0 = _mm256_loadu_ps(W0 + k + 0 * F);
                d00 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d00);
                w0 = _mm256_loadu_ps(W1 + k + 0 * F);
                d10 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d10);
                w0 = _mm256_loadu_ps(W2 + k + 0 * F);
                d20 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d20);
                w0 = _mm256_loadu_ps(W3 + k + 0 * F);
                d30 = _mm256_add_ps(_mm256_mul_ps(s0, w0), d30);
            }
            _mm_storeu_ps(D, _mm_add_ps(Extract4Sums(d00, d10, d20, d30), _mm_loadu_ps(B)));
        }

        void SynetInnerProductLayerForward(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst)
        {
            float _bias[4] = { 0, 0, 0, 0 };
            size_t count4 = AlignLo(count, 4);
            size_t i = 0;
            for (; i < count4; i += 4)
                SynetInnerProductLayerForward4(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
            for (; i < count; ++i)
                SynetInnerProductLayerForward1(src, weight + i * size, (bias ? bias + i : _bias), size, dst + i);
        }

        SIMD_INLINE void PoolingMaxHwc1(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
        }

        SIMD_INLINE void PoolingMaxHwc2(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMaxHwc4(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            __m256 max2 = min;
            __m256 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm256_max_ps(max2, _mm256_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm256_max_ps(max3, _mm256_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
            _mm256_storeu_ps(dst + 2 * F, max2);
            _mm256_storeu_ps(dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMaxHwc8(const float * src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m256 & min, float * dst)
        {
            __m256 max0 = min;
            __m256 max1 = min;
            __m256 max2 = min;
            __m256 max3 = min;
            __m256 max4 = min;
            __m256 max5 = min;
            __m256 max6 = min;
            __m256 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm256_max_ps(max0, _mm256_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm256_max_ps(max1, _mm256_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm256_max_ps(max2, _mm256_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm256_max_ps(max3, _mm256_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm256_max_ps(max4, _mm256_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm256_max_ps(max5, _mm256_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm256_max_ps(max6, _mm256_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm256_max_ps(max7, _mm256_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst + 0 * F, max0);
            _mm256_storeu_ps(dst + 1 * F, max1);
            _mm256_storeu_ps(dst + 2 * F, max2);
            _mm256_storeu_ps(dst + 3 * F, max3);
            _mm256_storeu_ps(dst + 4 * F, max4);
            _mm256_storeu_ps(dst + 5 * F, max5);
            _mm256_storeu_ps(dst + 6 * F, max6);
            _mm256_storeu_ps(dst + 7 * F, max7);
        }

        void SynetPoolingForwardMax(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float * dst, size_t dstH, size_t dstW, SimdBool trans)
        {
            if (trans)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    __m256 min = _mm256_set1_ps(-FLT_MAX);
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                        hStart = Simd::Max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                            wStart = Simd::Max<ptrdiff_t>(0, wStart);
                            const float * ps = src + hStart * srcS + wStart * srcC;
                            size_t c = 0;
                            for (; c < srcCF8; c += 8 * F)
                                PoolingMaxHwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF4; c += 4 * F)
                                PoolingMaxHwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF2; c += 2 * F)
                                PoolingMaxHwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF1; c += 1 * F)
                                PoolingMaxHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            if (c < srcC)
                                PoolingMaxHwc1(ps + srcC - F, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + srcC - F);
                            dst += srcC;
                        }
                    }
                    return;
                }
            }
            else
            {
                if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0 && dstW >= F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        Avx::NeuralPooling2x2Max2x2(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
            }
            Sse::SynetPoolingForwardMax(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, trans);
        }

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float * src, const float * slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _slope = Load<align>(slope + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _src);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _src);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(_slope, neg)));
        }

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float * src, __m256 slope, float * dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _src);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _src);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(slope, neg)));
        }

        template <bool align> void SynetPreluLayerForward(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(slope) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
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
                            SynetPreluLayerForward<align>(src, slope, dst, i + F * 0);
                            SynetPreluLayerForward<align>(src, slope, dst, i + F * 1);
                            SynetPreluLayerForward<align>(src, slope, dst, i + F * 2);
                            SynetPreluLayerForward<align>(src, slope, dst, i + F * 3);
                        }
                        for (; i < partial; i += F)
                            SynetPreluLayerForward<align>(src, slope, dst, i);
                    }
                    for (; i < count; ++i)
                        dst[i] = Base::SynetPreluLayerForward(src[i], slope[i]);
                    src += count;
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
                        __m256 _slope = _mm256_set1_ps(slope[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetPreluLayerForward<align>(src, _slope, dst, j + F * 0);
                            SynetPreluLayerForward<align>(src, _slope, dst, j + F * 1);
                            SynetPreluLayerForward<align>(src, _slope, dst, j + F * 2);
                            SynetPreluLayerForward<align>(src, _slope, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetPreluLayerForward<align>(src, _slope, dst, j);
                    }
                    for (; j < size; ++j)
                        dst[j] = Base::SynetPreluLayerForward(src[j], slope[i]);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetPreluLayerForward(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(slope) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetPreluLayerForward<true>(src, slope, count, size, dst, trans);
            else
                SynetPreluLayerForward<false>(src, slope, count, size, dst, trans);
        }

        template <bool align> void SynetRestrictRange(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            __m256 _min = _mm256_set1_ps(min);
            __m256 _max = _mm256_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i)), _max));
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
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)), Load<align>(bias + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), Load<align>(scale + offset)));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_add_ps(_mm256_mul_ps(Load<align>(src + offset), scale), bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Store<align>(dst + offset, _mm256_mul_ps(Load<align>(src + offset), scale));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if ((trans || size == 1) && count != 1)
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
                            dst[i] = src[i] * scale[i];
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
                            __m256 _scale = _mm256_set1_ps(scale[i]);
                            __m256 _bias = _mm256_set1_ps(bias[i]);
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
                            __m256 _scale = _mm256_set1_ps(scale[i]);
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
            if (((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetScaleLayerForward<true>(src, scale, bias, count, size, dst, trans);
            else
                SynetScaleLayerForward<false>(src, scale, bias, count, size, dst, trans);
        }
    }
#endif// SIMD_AVX_ENABLE
}

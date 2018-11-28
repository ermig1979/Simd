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
#include "Simd/SimdPow.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const __m512 & bias, float * dst, __mmask16 tail = -1)
        {
            Store<align, mask>(dst, _mm512_add_ps((Load<align, mask>(dst, tail)), bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const float * bias, float * dst, __mmask16 tail = -1)
        {
            __m512 _bias = Load<align, mask>(bias, tail);
            __m512 _dst = Load<align, mask>(dst, tail);
            Store<align, mask>(dst, _mm512_add_ps(_dst, _bias), tail);
        }

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(bias) : Aligned(size)) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += QF)
                    {
                        SynetAddBias<align, false>(bias + i + F * 0, dst + i + F * 0);
                        SynetAddBias<align, false>(bias + i + F * 1, dst + i + F * 1);
                        SynetAddBias<align, false>(bias + i + F * 2, dst + i + F * 2);
                        SynetAddBias<align, false>(bias + i + F * 3, dst + i + F * 3);
                    }
                    for (; i < partial; i += F)
                        SynetAddBias<align, false>(bias + i, dst + i);
                    if (i < count)
                        SynetAddBias<align, true>(bias + i, dst + i, tail);
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    __m512 _bias = _mm512_set1_ps(bias[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetAddBias<align, false>(_bias, dst + j + F * 0);
                        SynetAddBias<align, false>(_bias, dst + j + F * 1);
                        SynetAddBias<align, false>(_bias, dst + j + F * 2);
                        SynetAddBias<align, false>(_bias, dst + j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetAddBias<align, false>(_bias, dst + j);
                    if(j < size)
                        SynetAddBias<align, true>(_bias, dst + j, tail);
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

        template <SimdSynetEltwiseOperationType type> __m512 SynetEltwiseLayerForward(__m512 src0, __m512 src1);

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationProduct>(__m512 src0, __m512 src1)
        {
            return _mm512_mul_ps(src0, src1);
        }

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMax>(__m512 src0, __m512 src1)
        {
            return _mm512_max_ps(src0, src1);
        }

        template <> SIMD_INLINE __m512 SynetEltwiseLayerForward<SimdSynetEltwiseOperationMin>(__m512 src0, __m512 src1)
        {
            return _mm512_min_ps(src0, src1);
        }

        template <SimdSynetEltwiseOperationType type, bool align, bool mask > SIMD_INLINE void SynetEltwiseLayerForward(const float * src0, const float * src1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, SynetEltwiseLayerForward<type>((Load<align, mask>(src0 + offset, tail)), (Load<align, mask>(src1 + offset, tail))), tail);
        }

        template <SimdSynetEltwiseOperationType type, bool align> void SynetEltwiseLayerForward(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 0);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 1);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 2);
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForward<type, align, false>(src0, src1, dst, j);
            if (j < size)
                SynetEltwiseLayerForward<type, align, true>(src0, src1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 0);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 1);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 2);
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForward<type, align, false>(dst, srci, dst, j);
                if (j < size)
                    SynetEltwiseLayerForward<type, align, true>(dst, srci, dst, j, tail);
            }
        }

        template <bool align, bool mask> void SynetEltwiseLayerForwardSum(const float * src0, const __m512 & weight0, const float * src1, const __m512 & weight1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_fmadd_ps((Load<align, mask>(src0 + offset, tail)), weight0, _mm512_mul_ps((Load<align, mask>(src1 + offset, tail)), weight1)), tail);
        }

        template <bool align, bool mask> void SynetEltwiseLayerForwardSum(const float * src, const __m512 & weight, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_fmadd_ps((Load<align, mask>(src + offset, tail)), weight, (Load<align, mask>(dst + offset, tail))), tail);
        }

        template <bool align> void SynetEltwiseLayerForwardSum(float const * const * src, const float * weight, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float * src0 = src[0];
            const float * src1 = src[1];
            __m512 weight0 = _mm512_set1_ps(weight[0]);
            __m512 weight1 = _mm512_set1_ps(weight[1]);
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 0);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 1);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 2);
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForwardSum<align, false>(src0, weight0, src1, weight1, dst, j);
            if (j < size)
                SynetEltwiseLayerForwardSum<align, true>(src0, weight0, src1, weight1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                __m512 weighti = _mm512_set1_ps(weight[i]);
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 0);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 1);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 2);
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardSum<align, false>(srci, weighti, dst, j);
                if (j < size)
                    SynetEltwiseLayerForwardSum<align, true>(srci, weighti, dst, j, tail);
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

        template <bool align, bool mask> SIMD_INLINE void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, __m512 sign, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _bias = Load<align, mask>(bias + offset, tail);
            __m512 x = _mm512_add_ps((Load<align, mask>(src + offset, tail)), _bias);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            Store<align, mask>(dst + offset, _mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(x, _mm512_andnot_ps(sign, x)), _scale), _mm512_max_ps(_mm512_setzero_ps(), x)), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetFusedLayerForward0(const float * src, __m512 bias, __m512 scale, __m512 sign, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 x = _mm512_add_ps((Load<align, mask>(src + offset, tail)), bias);
            Store<align, mask>(dst + offset, _mm512_add_ps(_mm512_mul_ps(_mm512_sub_ps(x, _mm512_andnot_ps(sign, x)), scale), _mm512_max_ps(_mm512_setzero_ps(), x)), tail);
        }

        template <bool align> void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            __m512 sign = _mm512_set1_ps(-0.0f);
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += QF)
                    {
                        SynetFusedLayerForward0<align, false>(src, bias, scale, sign, dst, i + 0 * F);
                        SynetFusedLayerForward0<align, false>(src, bias, scale, sign, dst, i + 1 * F);
                        SynetFusedLayerForward0<align, false>(src, bias, scale, sign, dst, i + 2 * F);
                        SynetFusedLayerForward0<align, false>(src, bias, scale, sign, dst, i + 3 * F);
                    }
                    for (; i < partial; i += F)
                        SynetFusedLayerForward0<align, false>(src, bias, scale, sign, dst, i);
                    if (i < count)
                        SynetFusedLayerForward0<align, true>(src, bias, scale, sign, dst, i, tail);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    __m512 _bias = _mm512_set1_ps(bias[i]);
                    __m512 _scale = _mm512_set1_ps(scale[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetFusedLayerForward0<align, false>(src, _bias, _scale, sign, dst, j + 0 * F);
                        SynetFusedLayerForward0<align, false>(src, _bias, _scale, sign, dst, j + 1 * F);
                        SynetFusedLayerForward0<align, false>(src, _bias, _scale, sign, dst, j + 2 * F);
                        SynetFusedLayerForward0<align, false>(src, _bias, _scale, sign, dst, j + 3 * F);
                    }
                    for (; j < partial; j += F)
                        SynetFusedLayerForward0<align, false>(src, _bias, _scale, sign, dst, j);
                    if(j < size)
                        SynetFusedLayerForward0<align, true>(src, _bias, _scale, sign, dst, j, tail);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward0(const float * src, const float * bias, const float * scale, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward0<true>(src, bias, scale, count, size, dst, trans);
            else
                SynetFusedLayerForward0<false>(src, bias, scale, count, size, dst, trans);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _bias0 = Load<align, mask>(bias0 + offset, tail);
            __m512 x = _mm512_add_ps((Load<align, mask>(src + offset, tail)), _bias0);
            __m512 _scale1 = Load<align, mask>(scale1 + offset, tail);
            __m512 _bias1 = Load<align, mask>(bias1 + offset, tail);
            Store<align, mask>(dst + offset, _mm512_add_ps(_mm512_fmadd_ps(_mm512_max_ps(_mm512_setzero_ps(), _mm512_sub_ps(_mm512_setzero_ps(), x)), _scale1, _bias1), _mm512_max_ps(_mm512_setzero_ps(), x)), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetFusedLayerForward1(const float * src, __m512 bias0, __m512 scale1, __m512 bias1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 x = _mm512_add_ps((Load<align, mask>(src + offset, tail)), bias0);
            Store<align, mask>(dst + offset, _mm512_add_ps(_mm512_fmadd_ps(_mm512_max_ps(_mm512_setzero_ps(), _mm512_sub_ps(_mm512_setzero_ps(), x)), scale1, bias1), _mm512_max_ps(_mm512_setzero_ps(), x)), tail);
        }

        template <bool align> void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += QF)
                    {
                        SynetFusedLayerForward1<align, false>(src, bias0, scale1, bias1, dst, i + 0 * F);
                        SynetFusedLayerForward1<align, false>(src, bias0, scale1, bias1, dst, i + 1 * F);
                        SynetFusedLayerForward1<align, false>(src, bias0, scale1, bias1, dst, i + 2 * F);
                        SynetFusedLayerForward1<align, false>(src, bias0, scale1, bias1, dst, i + 3 * F);
                    }
                    for (; i < partial; i += F)
                        SynetFusedLayerForward1<align, false>(src, bias0, scale1, bias1, dst, i);
                    if (i < count)
                        SynetFusedLayerForward1<align, true>(src, bias0, scale1, bias1, dst, i, tail);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    __m512 _bias0 = _mm512_set1_ps(bias0[i]);
                    __m512 _scale1 = _mm512_set1_ps(scale1[i]);
                    __m512 _bias1 = _mm512_set1_ps(bias1[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetFusedLayerForward1<align, false>(src, _bias0, _scale1, _bias1, dst, j + 0 * F);
                        SynetFusedLayerForward1<align, false>(src, _bias0, _scale1, _bias1, dst, j + 1 * F);
                        SynetFusedLayerForward1<align, false>(src, _bias0, _scale1, _bias1, dst, j + 2 * F);
                        SynetFusedLayerForward1<align, false>(src, _bias0, _scale1, _bias1, dst, j + 3 * F);
                    }
                    for (; j < partial; j += F)
                        SynetFusedLayerForward1<align, false>(src, _bias0, _scale1, _bias1, dst, j);
                    if(j < size)
                        SynetFusedLayerForward1<align, true>(src, _bias0, _scale1, _bias1, dst, j, tail);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetFusedLayerForward1(const float * src, const float * bias0, const float * scale1, const float * bias1, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count) && Aligned(bias0) && Aligned(scale1) && Aligned(bias1) : Aligned(size)) && Aligned(src) && Aligned(dst))
                SynetFusedLayerForward1<true>(src, bias0, scale1, bias1, count, size, dst, trans);
            else
                SynetFusedLayerForward1<false>(src, bias0, scale1, bias1, count, size, dst, trans);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetFusedLayerForward2(const float * src, __m512 scale, __m512 bias, __m512 slope, float * dst, __mmask16 tail = -1)
        {
            __m512 x = _mm512_add_ps(_mm512_mul_ps((Load<align, mask>(src, tail)), scale), bias);
            Store<align, mask>(dst, _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), x), _mm512_mul_ps(_mm512_min_ps(_mm512_setzero_ps(), x), slope)), tail);
        }

        template <bool align> void SynetFusedLayerForward2(const float * src, const float * scale, const float * bias, size_t count, size_t size, const float * slope, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(size) && Aligned(dst));
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            __m512 _slope = _mm512_set1_ps(slope[0]);
            for (size_t i = 0; i < count; ++i)
            {
                size_t j = 0;
                __m512 _scale = _mm512_set1_ps(scale[i]);
                __m512 _bias = _mm512_set1_ps(bias[i]);
                for (; j < aligned; j += QF)
                {
                    SynetFusedLayerForward2<align, false>(src + j + 0 * F, _scale, _bias, _slope, dst + j + 0 * F);
                    SynetFusedLayerForward2<align, false>(src + j + 1 * F, _scale, _bias, _slope, dst + j + 1 * F);
                    SynetFusedLayerForward2<align, false>(src + j + 2 * F, _scale, _bias, _slope, dst + j + 2 * F);
                    SynetFusedLayerForward2<align, false>(src + j + 3 * F, _scale, _bias, _slope, dst + j + 3 * F);
                }
                for (; j < partial; j += F)
                    SynetFusedLayerForward2<align, false>(src + j, _scale, _bias, _slope, dst + j);
                if( j < size)
                    SynetFusedLayerForward2<align, true>(src + j, _scale, _bias, _slope, dst + j, tail);
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

        void SynetInnerProductLayerForward1(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K16 = K & (~15);
            size_t K64 = K & (~63);
            const float * W0 = W + 0 * K;
            __m512 d00, d01, d02, d03;
            __m512 s0, s1, s2, s3, w0, w1, w2, w3;
            size_t k = 0;
            d00 = _mm512_setzero_ps();
            if (K64)
            {
                d01 = _mm512_setzero_ps();
                d02 = _mm512_setzero_ps();
                d03 = _mm512_setzero_ps();
                for (; k < K64; k += 64)
                {
                    s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm512_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm512_fmadd_ps(s0, w0, d00);
                    d01 = _mm512_fmadd_ps(s1, w1, d01);
                    s2 = _mm512_loadu_ps(S0 + k + 2 * F);
                    s3 = _mm512_loadu_ps(S0 + k + 3 * F);
                    w2 = _mm512_loadu_ps(W0 + k + 2 * F);
                    w3 = _mm512_loadu_ps(W0 + k + 3 * F);
                    d02 = _mm512_fmadd_ps(s2, w2, d02);
                    d03 = _mm512_fmadd_ps(s3, w3, d03);
                }
                d00 = _mm512_add_ps(_mm512_add_ps(d00, d01), _mm512_add_ps(d02, d03));
            }
            for (; k < K16; k += 16)
            {
                s0 = _mm512_loadu_ps(S0 + k);
                w0 = _mm512_loadu_ps(W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                s0 = _mm512_maskz_loadu_ps(tail, S0 + k);
                w0 = _mm512_maskz_loadu_ps(tail, W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
            }
            D[0] = Avx512f::ExtractSum(d00) + B[0];
        }

        void SynetInnerProductLayerForward4(const float * S0, const float * W, const float * B, size_t K, float * D)
        {
            size_t K16 = K & (~15);
            size_t K32 = K & (~31);
            const float * W0 = W + 0 * K;
            const float * W1 = W + 1 * K;
            const float * W2 = W + 2 * K;
            const float * W3 = W + 3 * K;
            __m512 d00, d01, d10, d11, d20, d21, d30, d31;
            __m512 s0, s1, w0, w1;
            size_t k = 0;
            d00 = _mm512_setzero_ps();
            d10 = _mm512_setzero_ps();
            d20 = _mm512_setzero_ps();
            d30 = _mm512_setzero_ps();
            if (K32)
            {
                d01 = _mm512_setzero_ps();
                d11 = _mm512_setzero_ps();
                d21 = _mm512_setzero_ps();
                d31 = _mm512_setzero_ps();
                for (; k < K16; k += 32)
                {
                    s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                    s1 = _mm512_loadu_ps(S0 + k + 1 * F);
                    w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W0 + k + 1 * F);
                    d00 = _mm512_fmadd_ps(s0, w0, d00);
                    d01 = _mm512_fmadd_ps(s1, w1, d01);
                    w0 = _mm512_loadu_ps(W1 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W1 + k + 1 * F);
                    d10 = _mm512_fmadd_ps(s0, w0, d10);
                    d11 = _mm512_fmadd_ps(s1, w1, d11);
                    w0 = _mm512_loadu_ps(W2 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W2 + k + 1 * F);
                    d20 = _mm512_fmadd_ps(s0, w0, d20);
                    d21 = _mm512_fmadd_ps(s1, w1, d21);
                    w0 = _mm512_loadu_ps(W3 + k + 0 * F);
                    w1 = _mm512_loadu_ps(W3 + k + 1 * F);
                    d30 = _mm512_fmadd_ps(s0, w0, d30);
                    d31 = _mm512_fmadd_ps(s1, w1, d31);
                }
                d00 = _mm512_add_ps(d00, d01);
                d10 = _mm512_add_ps(d10, d11);
                d20 = _mm512_add_ps(d20, d21);
                d30 = _mm512_add_ps(d30, d31);
            }
            for (; k < K16; k += 16)
            {
                s0 = _mm512_loadu_ps(S0 + k + 0 * F);
                w0 = _mm512_loadu_ps(W0 + k + 0 * F);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
                w0 = _mm512_loadu_ps(W1 + k + 0 * F);
                d10 = _mm512_fmadd_ps(s0, w0, d10);
                w0 = _mm512_loadu_ps(W2 + k + 0 * F);
                d20 = _mm512_fmadd_ps(s0, w0, d20);
                w0 = _mm512_loadu_ps(W3 + k + 0 * F);
                d30 = _mm512_fmadd_ps(s0, w0, d30);
            }
            if (k < K)
            {
                __mmask16 tail = __mmask16(-1) >> (16 + k - K);
                s0 = _mm512_maskz_loadu_ps(tail, S0 + k);
                w0 = _mm512_maskz_loadu_ps(tail, W0 + k);
                d00 = _mm512_fmadd_ps(s0, w0, d00);
                w0 = _mm512_maskz_loadu_ps(tail, W1 + k);
                d10 = _mm512_fmadd_ps(s0, w0, d10);
                w0 = _mm512_maskz_loadu_ps(tail, W2 + k);
                d20 = _mm512_fmadd_ps(s0, w0, d20);
                w0 = _mm512_maskz_loadu_ps(tail, W3 + k);
                d30 = _mm512_fmadd_ps(s0, w0, d30);
            }
            _mm_storeu_ps(D, _mm_add_ps(Avx512f::Extract4Sums(d00, d10, d20, d30), _mm_loadu_ps(B)));
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

        template <bool align> SIMD_INLINE void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            size_t aligned = AlignLo(size, F);
            __mmask16 tail = TailMask16(size - aligned);
            Array32f sum(size, true), zero(size, true);

            for (size_t i = 0; i < half; ++i)
            {
                const float * pos = src + i * size;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m512 _pos = Avx512f::Load<align>(pos + j);
                    Avx512f::Store<true>(sum.data + j, _mm512_fmadd_ps(_pos, _pos, Avx512f::Load<true>(sum.data + j)));
                }
                if (j < size)
                {
                    __m512 _pos = Avx512f::Load<align, true>(pos + j, tail);
                    __m512 _sum = Avx512f::Load<true, true>(sum.data + j, tail);
                    Avx512f::Store<true, true>(sum.data + j, _mm512_fmadd_ps(_pos, _pos, _sum), tail);
                }
            }

            __m512 k0 = _mm512_set1_ps(k[0]);
            __m512 k1 = _mm512_set1_ps(k[1]);
            __m512 k2 = _mm512_set1_ps(k[2]);
            Avx512f::Pow pow;
            for (size_t i = 0; i < count; ++i)
            {
                const float * pos = (i < count - half) ? src + half * size : zero.data;
                const float * neg = (i > half) ? src - (half + 1) * size : zero.data;
                size_t j = 0;
                for (; j < aligned; j += F)
                {
                    __m512 _pos = Avx512f::Load<align>(pos + j);
                    __m512 _neg = Avx512f::Load<align>(neg + j);
                    __m512 _sum = Avx512f::Load<true>(sum.data + j);
                    _sum = _mm512_fmadd_ps(_pos, _pos, _mm512_fnmadd_ps(_neg, _neg, _sum));
                    __m512 _src = Avx512f::Load<align>(src + j);
                    Avx512f::Store<true>(sum.data + j, _sum);
                    Avx512f::Store<align>(dst + j, _mm512_mul_ps(_src, pow(_mm512_fmadd_ps(k1, _sum, k0), k2)));
                }
                if (j < size)
                {
                    __m512 _pos = Avx512f::Load<align, true>(pos + j, tail);
                    __m512 _neg = Avx512f::Load<align, true>(neg + j, tail);
                    __m512 _sum = Avx512f::Load<true, true>(sum.data + j, tail);
                    _sum = _mm512_fmadd_ps(_pos, _pos, _mm512_fnmadd_ps(_neg, _neg, _sum));
                    __m512 _src = Avx512f::Load<align, true>(src + j, tail);
                    Avx512f::Store<true, true>(sum.data + j, _sum, tail);
                    Avx512f::Store<align, true>(dst + j, _mm512_mul_ps(_src, pow(_mm512_fmadd_ps(k1, _sum, k0), k2)), tail);
                }
                src += size;
                dst += size;
            }
        }

        void SynetLrnLayerCrossChannels(const float * src, size_t half, size_t count, size_t size, const float * k, float * dst)
        {
            if (Aligned(src) && Aligned(dst) && Aligned(size))
                SynetLrnLayerCrossChannels<true>(src, half, count, size, k, dst);
            else
                SynetLrnLayerCrossChannels<false>(src, half, count, size, k, dst);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetPreluLayerForward(const float * src, const float * slope, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _slope = Load<align, mask>(slope + offset, tail);
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            Store<align, mask>(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(_slope, neg)), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetPreluLayerForward(const float * src, __m512 slope, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            Store<align, mask>(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(slope, neg)), tail);
        }

        template <bool align> void SynetPreluLayerForward(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(slope) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += QF)
                    {
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 0);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 1);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 2);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 3);
                    }
                    for (; i < partial; i += F)
                        SynetPreluLayerForward<align, false>(src, slope, dst, i);
                    if(i < count)
                        SynetPreluLayerForward<align, true>(src, slope, dst, i, tail);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    __m512 _slope = _mm512_set1_ps(slope[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 0);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 1);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 2);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j);
                    if (i < count)
                        SynetPreluLayerForward<align, true>(src, _slope, dst, j, tail);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetPreluLayerForward(const float * src, const float * slope, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if ((trans || size == 1 ? Aligned(count) && Aligned(slope) : Aligned(size)) && Aligned(src) && Aligned(dst))
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
            __m512 _min = _mm512_set1_ps(min);
            __m512 _max = _mm512_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i)), _max));
            if(i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                Store<align, true>(dst + i, _mm512_min_ps(_mm512_max_ps(_min, (Load<align, true>(src + i, tail))), _max), tail);
            }
        }

        void SynetRestrictRange(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange<false>(src, size, lower, upper, dst);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            __m512 _bias = Load<align, mask>(bias + offset, tail);
            Store<align, mask>(dst + offset, _mm512_fmadd_ps(_src, _scale, _bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _scale = Load<align, mask>(scale + offset, tail);
            Store<align, mask>(dst + offset, _mm512_mul_ps(_src, _scale), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m512 & scale, const __m512 & bias, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            Store<align, mask>(dst + offset, _mm512_fmadd_ps(_src, scale, bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m512 & scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            Store<align, mask>(dst + offset, _mm512_mul_ps(_src, scale), tail);
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst, SimdBool trans)
        {
            if (align)
                assert((trans || size == 1 ? Aligned(count) && Aligned(scale) && Aligned(bias) : Aligned(size)) && Aligned(src) && Aligned(dst));
            if (trans || size == 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                if (bias)
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, scale, bias, dst, i + F * 0);
                            SynetScaleLayerForward<align, false>(src, scale, bias, dst, i + F * 1);
                            SynetScaleLayerForward<align, false>(src, scale, bias, dst, i + F * 2);
                            SynetScaleLayerForward<align, false>(src, scale, bias, dst, i + F * 3);
                        }
                        for (; i < partial; i += F)
                            SynetScaleLayerForward<align, false>(src, scale, bias, dst, i);
                        if (i < count)
                            SynetScaleLayerForward<align, true>(src, scale, bias, dst, i, tail);
                        src += count;
                        dst += count;
                    }
                }
                else
                {
                    for (size_t j = 0; j < size; ++j)
                    {
                        size_t i = 0;
                        for (; i < aligned; i += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, scale, dst, i + F * 0);
                            SynetScaleLayerForward<align, false>(src, scale, dst, i + F * 1);
                            SynetScaleLayerForward<align, false>(src, scale, dst, i + F * 2);
                            SynetScaleLayerForward<align, false>(src, scale, dst, i + F * 3);
                        }
                        for (; i < partial; i += F)
                            SynetScaleLayerForward<align, false>(src, scale,  dst, i);
                        if (i < count)
                            SynetScaleLayerForward<align, true>(src, scale, dst, i, tail);
                        src += count;
                        dst += count;
                    }
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                if (bias)
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        size_t j = 0;
                        __m512 _scale = _mm512_set1_ps(scale[i]);
                        __m512 _bias = _mm512_set1_ps(bias[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, _scale, _bias, dst, j + F * 0);
                            SynetScaleLayerForward<align, false>(src, _scale, _bias, dst, j + F * 1);
                            SynetScaleLayerForward<align, false>(src, _scale, _bias, dst, j + F * 2);
                            SynetScaleLayerForward<align, false>(src, _scale, _bias, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align, false>(src, _scale, _bias, dst, j);
                        if (j < size)
                            SynetScaleLayerForward<align, true>(src, _scale, _bias, dst, j, tail);
                        src += size;
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                    {
                        size_t j = 0;
                        __m512 _scale = _mm512_set1_ps(scale[i]);
                        for (; j < aligned; j += QF)
                        {
                            SynetScaleLayerForward<align, false>(src, _scale, dst, j + F * 0);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, j + F * 1);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, j + F * 2);
                            SynetScaleLayerForward<align, false>(src, _scale, dst, j + F * 3);
                        }
                        for (; j < partial; j += F)
                            SynetScaleLayerForward<align, false>(src, _scale, dst, j);
                        if (j < size)
                            SynetScaleLayerForward<align, true>(src, _scale, dst, j, tail);
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
#endif// SIMD_AVX512F_ENABLE
}

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

        template <bool align> SIMD_INLINE void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
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

        void SynetAddBias(const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(dst) && Aligned(size))
                SynetAddBias<true>(bias, count, size, dst);
            else
                SynetAddBias<false>(bias, count, size, dst);
        }

        template <bool align, bool mask> void SynetEltwiseLayerForwardProduct(const float * src0, const float * src1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_mul_ps((Load<align, mask>(src0 + offset, tail)), (Load<align, mask>(src1 + offset, tail))), tail);
        }

        template <bool align> void SynetEltwiseLayerForwardProduct(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForwardProduct<align, false>(src0, src1, dst, j + F * 0);
                SynetEltwiseLayerForwardProduct<align, false>(src0, src1, dst, j + F * 1);
                SynetEltwiseLayerForwardProduct<align, false>(src0, src1, dst, j + F * 2);
                SynetEltwiseLayerForwardProduct<align, false>(src0, src1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForwardProduct<align, false>(src0, src1, dst, j);
            if (j < size)
                SynetEltwiseLayerForwardProduct<align, true>(src0, src1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardProduct<align, false>(dst, srci, dst, j + F * 0);
                    SynetEltwiseLayerForwardProduct<align, false>(dst, srci, dst, j + F * 1);
                    SynetEltwiseLayerForwardProduct<align, false>(dst, srci, dst, j + F * 2);
                    SynetEltwiseLayerForwardProduct<align, false>(dst, srci, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardProduct<align, false>(dst, srci, dst, j);
                if (j < size)
                    SynetEltwiseLayerForwardProduct<align, true>(dst, srci, dst, j, tail);
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

        template <bool align, bool mask> void SynetEltwiseLayerForwardMax(const float * src0, const float * src1, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_max_ps((Load<align, mask>(src0 + offset, tail)), (Load<align, mask>(src1 + offset, tail))), tail);
        }

        template <bool align> void SynetEltwiseLayerForwardMax(float const * const * src, size_t count, size_t size, float * dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + partial - size);
            const float * src0 = src[0];
            const float * src1 = src[1];
            size_t j = 0;
            for (; j < aligned; j += QF)
            {
                SynetEltwiseLayerForwardMax<align, false>(src0, src1, dst, j + F * 0);
                SynetEltwiseLayerForwardMax<align, false>(src0, src1, dst, j + F * 1);
                SynetEltwiseLayerForwardMax<align, false>(src0, src1, dst, j + F * 2);
                SynetEltwiseLayerForwardMax<align, false>(src0, src1, dst, j + F * 3);
            }
            for (; j < partial; j += F)
                SynetEltwiseLayerForwardMax<align, false>(src0, src1, dst, j);
            if(j < size)
                SynetEltwiseLayerForwardMax<align, true>(src0, src1, dst, j, tail);
            for (size_t i = 2; i < count; ++i)
            {
                const float * srci = src[i];
                for (j = 0; j < aligned; j += QF)
                {
                    SynetEltwiseLayerForwardMax<align, false>(dst, srci, dst, j + F * 0);
                    SynetEltwiseLayerForwardMax<align, false>(dst, srci, dst, j + F * 1);
                    SynetEltwiseLayerForwardMax<align, false>(dst, srci, dst, j + F * 2);
                    SynetEltwiseLayerForwardMax<align, false>(dst, srci, dst, j + F * 3);
                }
                for (; j < partial; j += F)
                    SynetEltwiseLayerForwardMax<align, false>(dst, srci, dst, j);
                if (j < size)
                    SynetEltwiseLayerForwardMax<align, true>(dst, srci, dst, j, tail);
            }
        }

        template <bool align> void SynetEltwiseLayerForward(float const * const * src, const float * weight, size_t count, size_t size, SimdSynetEltwiseOperationType type, float * dst)
        {
            switch (type)
            {
            case SimdSynetEltwiseOperationProduct:
                SynetEltwiseLayerForwardProduct<align>(src, count, size, dst);
                break;
            case SimdSynetEltwiseOperationSum:
                SynetEltwiseLayerForwardSum<align>(src, weight, count, size, dst);
                break;
            case SimdSynetEltwiseOperationMax:
                SynetEltwiseLayerForwardMax<align>(src, count, size, dst);
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

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m512 & scale, const __m512 & bias, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_fmadd_ps((Load<align, mask>(src + offset, tail)), scale, bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m512 & scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Store<align, mask>(dst + offset, _mm512_mul_ps((Load<align, mask>(src + offset, tail)), scale), tail);
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
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

        void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst)
        {
            if (Aligned(dst) && Aligned(size))
                SynetScaleLayerForward<true>(src, scale, bias, count, size, dst);
            else
                SynetScaleLayerForward<false>(src, scale, bias, count, size, dst);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}

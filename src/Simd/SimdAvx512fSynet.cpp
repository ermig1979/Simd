/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

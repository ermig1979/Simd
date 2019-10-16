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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        template<bool align, bool mask> SIMD_INLINE void SynetElu32f(const float * src, const Avx512f::Exp & exp, __m512 alpha, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Avx512f::Store<align, mask>(dst + offset, exp.Elu(Avx512f::Load<align, mask>(src + offset, tail), alpha), tail);
        }

        template<bool align> void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            __m512 _alpha = _mm512_set1_ps(alpha[0]);
            Avx512f::Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<align, false>(src, exp, _alpha, dst, i);
            if(i < size)
                SynetElu32f<align, true>(src, exp, _alpha, dst, i, tail);

        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetElu32f<true>(src, size, alpha, dst);
            else
                SynetElu32f<false>(src, size, alpha, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetHswish32f(const float * src, __m512 shift, __m512 scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = _mm512_mul_ps(_mm512_mul_ps(_mm512_max_ps(_mm512_add_ps(_mm512_min_ps(_src, shift), shift), _mm512_setzero_ps()), scale), _src);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            __m512 _shift = _mm512_set1_ps(shift[0]);
            __m512 _scale = _mm512_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i);
            if (i < size)
                SynetHswish32f<align, true>(src, _shift, _scale, dst, i, tail);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}

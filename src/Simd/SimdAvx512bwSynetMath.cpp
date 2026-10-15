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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
#if defined(SIMD_SYNET_ENABLE)
    namespace Avx512bw
    {
        void SynetAddVectorMultipliedByValue(const float* src, size_t size, const float* value, float* dst)
        {
            size_t aligned = AlignLo(size, QF);
            size_t partial = AlignLo(size, F);
            size_t i = 0;
            __m512 _value = _mm512_set1_ps(*value);
            for (; i < aligned; i += QF)
            {
                _mm512_storeu_ps(dst + i + F * 0, _mm512_fmadd_ps(_value, _mm512_loadu_ps(src + i + F * 0), _mm512_loadu_ps(dst + i + F * 0)));
                _mm512_storeu_ps(dst + i + F * 1, _mm512_fmadd_ps(_value, _mm512_loadu_ps(src + i + F * 1), _mm512_loadu_ps(dst + i + F * 1)));
                _mm512_storeu_ps(dst + i + F * 2, _mm512_fmadd_ps(_value, _mm512_loadu_ps(src + i + F * 2), _mm512_loadu_ps(dst + i + F * 2)));
                _mm512_storeu_ps(dst + i + F * 3, _mm512_fmadd_ps(_value, _mm512_loadu_ps(src + i + F * 3), _mm512_loadu_ps(dst + i + F * 3)));
            }
            for (; i < partial; i += F)
                _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_value, _mm512_loadu_ps(src + i), _mm512_loadu_ps(dst + i)));
            if (i < size)
            {
                __mmask16 tailMask = __mmask16(-1) >> (F + i - size);
                _mm512_mask_storeu_ps(dst + i, tailMask, _mm512_fmadd_ps(_value, _mm512_maskz_loadu_ps(tailMask, src + i), _mm512_maskz_loadu_ps(tailMask, dst + i)));
            }
        }
    }
#endif
#endif
}

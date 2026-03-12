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
#ifndef __SimdSynetApply16b_h__
#define __SimdSynetApply16b_h__

#include "Simd/SimdSynetConvolution16bCommon.h"

namespace Simd
{
#ifdef SIMD_AMXBF16_ENABLE
    namespace AmxBf16
    {
        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M> static SIMD_INLINE void ApplyMx1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0, f1;
            if (M > 0) f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0), bias[0]), params, 0);
            if (M > 1) f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + F), bias[1]), params, 1);
            if (term == Term16bLast16b)
            {
                if (M > 1)
                {
                    _mm512_mask_storeu_epi16((uint16_t*)ptr, tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
                else
                {
                    _mm256_mask_storeu_epi16((uint16_t*)ptr, (__mmask16)tail, (__m256i)_mm512_cvtneps_pbh(f0));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
            }
            else if (term == Term16bLast32f)
            {
                if (M > 1)
                {
                    _mm512_storeu_ps((float*)(ptr + 0), f0);
                    if (flush) _mm_prefetch((const char*)(ptr + 0), _MM_HINT_NTA);
                    _mm512_mask_storeu_ps((float*)(ptr + A), (__mmask16)tail, f1);
                    if (flush) _mm_prefetch((const char*)(ptr + A), _MM_HINT_NTA);
                }
                else
                    if (M > 0)
                    {
                        _mm512_mask_storeu_ps((float*)(ptr + 0), (__mmask16)tail, f0);
                        if (flush) _mm_prefetch((const char*)(ptr + 0), _MM_HINT_NTA);
                    }
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int N> static SIMD_INLINE void ApplyMxN(uint8_t* ptr, int dP, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) ApplyMx1<term, type, flush, M>(ptr + 0 * dP, buf + 0 * DF, bias, params, tail);
            if (N > 1) ApplyMx1<term, type, flush, M>(ptr + 1 * dP, buf + 1 * DF, bias, params, tail);
            if (N > 2) ApplyMx1<term, type, flush, M>(ptr + 2 * dP, buf + 2 * DF, bias, params, tail);
            if (N > 3) ApplyMx1<term, type, flush, M>(ptr + 3 * dP, buf + 3 * DF, bias, params, tail);
            if (N > 4) ApplyMx1<term, type, flush, M>(ptr + 4 * dP, buf + 4 * DF, bias, params, tail);
            if (N > 5) ApplyMx1<term, type, flush, M>(ptr + 5 * dP, buf + 5 * DF, bias, params, tail);
            if (N > 6) ApplyMx1<term, type, flush, M>(ptr + 6 * dP, buf + 6 * DF, bias, params, tail);
            if (N > 7) ApplyMx1<term, type, flush, M>(ptr + 7 * dP, buf + 7 * DF, bias, params, tail);
        }
    }
#endif
}

#endif

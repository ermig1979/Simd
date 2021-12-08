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
#ifndef __SimdDeinterleave_h__
#define __SimdDeinterleave_h__

#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        template<int part> SIMD_INLINE __m128i Deinterleave8(__m128i ab0, __m128i ab1);

        template<> SIMD_INLINE __m128i Deinterleave8<0>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(_mm_and_si128(ab0, K16_00FF), _mm_and_si128(ab1, K16_00FF));
        }

        template<> SIMD_INLINE __m128i Deinterleave8<1>(__m128i ab0, __m128i ab1)
        {
            return _mm_packus_epi16(
                _mm_and_si128(_mm_srli_si128(ab0, 1), K16_00FF), 
                _mm_and_si128(_mm_srli_si128(ab1, 1), K16_00FF));
        }
    }
#endif// SIMD_SSE2_ENABLE
}

#endif//__SimdDeinterleave_h__

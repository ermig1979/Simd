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
#ifndef __SimdConvert_h__
#define __SimdConvert_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template<int part> SIMD_INLINE __m128d Fp32ToFp64(__m128 ps);

        template <> SIMD_INLINE __m128d Fp32ToFp64<0>(__m128 s)
        {
            return _mm_cvtps_pd(s);
        }

        template <> SIMD_INLINE __m128d Fp32ToFp64<1>(__m128 s)
        {
            return _mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(s), 8)));
        }

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128 Fp64ToFp32(__m128d lo, __m128d hi)
        {
            return _mm_shuffle_ps(_mm_cvtpd_ps(lo), _mm_cvtpd_ps(hi), 0x44);
        }
    }
#endif
}

#endif

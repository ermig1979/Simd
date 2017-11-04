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
#ifndef __SimdUpdate_h__
#define __SimdUpdate_h__

#include "Simd/SimdStore.h"

namespace Simd
{
    enum UpdateType
    {
        UpdateSet = 0,
        UpdateAdd = 1,
    };

    namespace Base
    {
        template <UpdateType update> SIMD_INLINE void Update(float  * p, float a)
        {
            *p = a;
        }

        template <> SIMD_INLINE void Update<UpdateAdd>(float  * p, float a)
        {
            *p += a;
        }
    }

#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        template <UpdateType update, bool align> SIMD_INLINE void Update(float  * p, __m128 a)
        {
            Store<align>(p, a);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, false>(float  * p, __m128 a)
        {
            Store<false>(p, _mm_add_ps(Load<false>(p), a));
        }

        template <> SIMD_INLINE void Update<UpdateAdd, true>(float  * p, __m128 a)
        {
            Store<true>(p, _mm_add_ps(Load<true>(p), a));
        }
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        template <UpdateType update, bool align> SIMD_INLINE void Update(float  * p, __m256 a)
        {
            Store<align>(p, a);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, false>(float  * p, __m256 a)
        {
            Store<false>(p, _mm256_add_ps(Load<false>(p), a));
        }

        template <> SIMD_INLINE void Update<UpdateAdd, true>(float  * p, __m256 a)
        {
            Store<true>(p, _mm256_add_ps(Load<true>(p), a));
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        template <UpdateType update, bool align, bool mask> SIMD_INLINE void Update(float  * p, __m512 a, __mmask16 m)
        {
            Store<align, mask>(p, a, m);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, false, false>(float  * p, __m512 a, __mmask16 m)
        {
            Store<false, false>(p, _mm512_add_ps((Load<false, false>(p, m)), a), m);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, false, true>(float  * p, __m512 a, __mmask16 m)
        {
            Store<false, true>(p, _mm512_add_ps((Load<false, true>(p, m)), a), m);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, true, false>(float  * p, __m512 a, __mmask16 m)
        {
            Store<true, false>(p, _mm512_add_ps((Load<true, false>(p, m)), a), m);
        }

        template <> SIMD_INLINE void Update<UpdateAdd, true, true>(float  * p, __m512 a, __mmask16 m)
        {
            Store<true, true>(p, _mm512_add_ps((Load<true, true>(p, m)), a), m);
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
#endif//__SimdUpdate_h__

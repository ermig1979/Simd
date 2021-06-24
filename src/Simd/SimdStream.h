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
#ifndef __SimdStream_h__
#define __SimdStream_h__

#include "Simd/SimdDefs.h"

namespace Simd
{
    const size_t STREAM_SIZE_MIN = 0x00100000;

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        template <bool align, bool stream> SIMD_INLINE void Stream(float* p, __m128 a);

        template <> SIMD_INLINE void Stream<false, false>(float* p, __m128 a)
        {
            _mm_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<false, true>(float* p, __m128 a)
        {
            _mm_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, false>(float* p, __m128 a)
        {
            _mm_store_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, true>(float* p, __m128 a)
        {
            _mm_stream_ps(p, a);
        }

        template <bool align, bool stream> SIMD_INLINE void Stream(__m128i  * p, __m128i a);

        template <> SIMD_INLINE void Stream<false, false>(__m128i   * p, __m128i a)
        {
            _mm_storeu_si128(p, a);
        }

        template <> SIMD_INLINE void Stream<false, true>(__m128i   * p, __m128i a)
        {
            _mm_storeu_si128(p, a);
        }

        template <> SIMD_INLINE void Stream<true, false>(__m128i   * p, __m128i a)
        {
            _mm_store_si128(p, a);
        }

        template <> SIMD_INLINE void Stream<true, true>(__m128i   * p, __m128i a)
        {
            _mm_stream_si128(p, a);
        }
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        template <bool align, bool stream> SIMD_INLINE void Stream(float  * p, __m256 a);

        template <> SIMD_INLINE void Stream<false, false>(float  * p, __m256 a)
        {
            _mm256_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<false, true>(float  * p, __m256 a)
        {
            _mm256_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, false>(float  * p, __m256 a)
        {
            _mm256_store_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, true>(float  * p, __m256 a)
        {
            _mm256_stream_ps(p, a);
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template <bool align, bool stream> SIMD_INLINE void Stream(__m256i  * p, __m256i a);

        template <> SIMD_INLINE void Stream<false, false>(__m256i  * p, __m256i a)
        {
            _mm256_storeu_si256(p, a);
        }

        template <> SIMD_INLINE void Stream<false, true>(__m256i  * p, __m256i a)
        {
            _mm256_storeu_si256(p, a);
        }

        template <> SIMD_INLINE void Stream<true, false>(__m256i  * p, __m256i a)
        {
            _mm256_store_si256(p, a);
        }

        template <> SIMD_INLINE void Stream<true, true>(__m256i  * p, __m256i a)
        {
            _mm256_stream_si256(p, a);
        }
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        template <bool align, bool stream> SIMD_INLINE void Stream(float  * p, __m512 a);

        template <> SIMD_INLINE void Stream<false, false>(float  * p, __m512 a)
        {
            _mm512_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<false, true>(float  * p, __m512 a)
        {
            _mm512_storeu_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, false>(float  * p, __m512 a)
        {
            _mm512_store_ps(p, a);
        }

        template <> SIMD_INLINE void Stream<true, true>(float  * p, __m512 a)
        {
#if defined(__clang__)
            _mm512_store_ps(p, a);
#else
            _mm512_stream_ps(p, a);
#endif
        }
    }
#endif//SIMD_AVX512F_ENABLE
}
#endif//__SimdStream_h__

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
        template <bool align, bool mask> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t offset, __m512 & sum, __mmask16 tail = -1)
        {
            __m512 _a = Load<align, mask>(a + offset, tail);
            __m512 _b = Load<align, mask>(b + offset, tail);
            __m512 _d = _mm512_sub_ps(_a, _b);
            sum = _mm512_fmadd_ps(_d, _d, sum);
        }

        template <bool align> void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t alignedSize = AlignLo(size, F);
            __mmask16 tailMask = TailMask16(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += QF)
                {
                    SquaredDifferenceSum32f<align, false>(a, b, i + 0 * F, sums[0]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 1 * F, sums[1]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 2 * F, sums[2]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 3 * F, sums[3]);
                }
                sums[0] = _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3]));
            }
            for (; i < alignedSize; i += F)
                SquaredDifferenceSum32f<align, false>(a, b, i, sums[0]);
#if defined (NDEBUG) && defined(_MSC_VER)
            *sum = ExtractSum(sums[0]);
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
#else
            if (i < size)
                SquaredDifferenceSum32f<align, true>(a, b, i, sums[0], tailMask);
            *sum = ExtractSum(sums[0]);
#endif
        }

        void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum32f<false>(a, b, size, sum);
        }

        template <bool align, bool mask> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t offset, __m512 & sum, __m512 & correction, __mmask16 tail = -1)
        {
            __m512 _a = Load<align, mask>(a + offset, tail);
            __m512 _b = Load<align, mask>(b + offset, tail);
            __m512 _d = _mm512_sub_ps(_a, _b);
            __m512 term = _mm512_fmsub_ps(_d, _d, correction);
            __m512 temp = _mm512_add_ps(sum, term);
            correction = _mm512_sub_ps(_mm512_sub_ps(temp, sum), term);
            sum = temp;
        }

        template <bool align> void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (align)
                assert(Aligned(a) && Aligned(b));

            size_t alignedSize = AlignLo(size, F);
            __mmask16 tailMask = TailMask16(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
            __m512 corrections[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += QF)
                {
                    SquaredDifferenceKahanSum32f<align, false>(a, b, i + 0 * F, sums[0], corrections[0]);
                    SquaredDifferenceKahanSum32f<align, false>(a, b, i + 1 * F, sums[1], corrections[1]);
                    SquaredDifferenceKahanSum32f<align, false>(a, b, i + 2 * F, sums[2], corrections[2]);
                    SquaredDifferenceKahanSum32f<align, false>(a, b, i + 3 * F, sums[3], corrections[3]);
                }
                sums[0] = _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3]));
            }
            for (; i < alignedSize; i += F)
                SquaredDifferenceKahanSum32f<align, false>(a, b, i, sums[0], corrections[0]);
#if defined (NDEBUG) && defined(_MSC_VER)
            *sum = ExtractSum(sums[0]);
            for (; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
#else
            if (i < size)
                SquaredDifferenceKahanSum32f<align, true>(a, b, i, sums[0], corrections[0], tailMask);
            *sum = ExtractSum(sums[0]);
#endif
        }

        void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if (Aligned(a) && Aligned(b))
                SquaredDifferenceKahanSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceKahanSum32f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}

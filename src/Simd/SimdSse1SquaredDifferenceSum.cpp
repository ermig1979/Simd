/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t offset, __m128 & sum)
        {
            __m128 _a = Load<align>(a + offset);
            __m128 _b = Load<align>(b + offset);
            __m128 _d = _mm_sub_ps(_a, _b);
            sum = _mm_add_ps(sum, _mm_mul_ps(_d, _d));
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 4);
            size_t fullAlignedSize = AlignLo(size, 16);
            size_t i = 0;
            if(partialAlignedSize)
            {
                __m128 sums[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += 16)
                    {
                        SquaredDifferenceSum32f<align>(a, b, i, sums[0]);
                        SquaredDifferenceSum32f<align>(a, b, i + 4, sums[1]);
                        SquaredDifferenceSum32f<align>(a, b, i + 8, sums[2]);
                        SquaredDifferenceSum32f<align>(a, b, i + 12, sums[3]);
                    }
                    sums[0] = _mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3]));
                }
                for(; i < partialAlignedSize; i += 4)
                    SquaredDifferenceSum32f<align>(a, b, i, sums[0]);
                *sum += ExtractSum(sums[0]);
            }
            for(; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
                SquaredDifferenceSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum32f<false>(a, b, size, sum);
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t offset, __m128 & sum, __m128 & correction)
        {
            __m128 _a = Load<align>(a + offset);
            __m128 _b = Load<align>(b + offset);
            __m128 _d = _mm_sub_ps(_a, _b);
            __m128 term = _mm_sub_ps(_mm_mul_ps(_d, _d), correction);
            __m128 temp = _mm_add_ps(sum, term);
            correction = _mm_sub_ps(_mm_sub_ps(temp, sum), term);
            sum = temp; 
        }

        template <bool align> SIMD_INLINE void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t partialAlignedSize = AlignLo(size, 4);
            size_t fullAlignedSize = AlignLo(size, 16);
            size_t i = 0;
            if(partialAlignedSize)
            {
                __m128 sums[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
                __m128 corrections[4] = {_mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps(), _mm_setzero_ps()};
                if(fullAlignedSize)
                {
                    for(; i < fullAlignedSize; i += 16)
                    {
                        SquaredDifferenceKahanSum32f<align>(a, b, i, sums[0], corrections[0]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 4, sums[1], corrections[1]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 8, sums[2], corrections[2]);
                        SquaredDifferenceKahanSum32f<align>(a, b, i + 12, sums[3], corrections[3]);
                    }
                }
                for(; i < partialAlignedSize; i += 4)
                    SquaredDifferenceKahanSum32f<align>(a, b, i, sums[0], corrections[0]);
                *sum += ExtractSum(_mm_add_ps(_mm_add_ps(sums[0], sums[1]), _mm_add_ps(sums[2], sums[3])));
            }
            for(; i < size; ++i)
                *sum += Simd::Square(a[i] - b[i]);
        }

        void SquaredDifferenceKahanSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
                SquaredDifferenceKahanSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceKahanSum32f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_SSE_ENABLE
}

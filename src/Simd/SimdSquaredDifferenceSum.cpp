/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#include "Simd/SimdEnable.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdInit.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdSquaredDifferenceSum.h"

namespace Simd
{
    namespace Base
    {
        int SquaredDifferenceSum32(const uchar *a, const uchar *b, size_t size)
        {
            int sum = 0;
            for(size_t i = 0; i < size; ++i)
            {
                sum += SquaredDifference(a[i], b[i]);
            }
            return sum;
        }
        
        int SquaredDifferenceSum32(const uchar *a, const uchar *b, const uchar *mask, size_t size)
        {
            int sum = 0;
            for(size_t i = 0; i < size; ++i)
            {
                if(mask[i])
                {
                    sum += SquaredDifference(a[i], b[i]);
                }
            }
            return sum;
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE __m128i SquaredDifference32(__m128i a, __m128i b)
        {
            const __m128i aLo = _mm_unpacklo_epi8(a, _mm_setzero_si128());
            const __m128i bLo = _mm_unpacklo_epi8(b, _mm_setzero_si128());
            const __m128i dLo = _mm_sub_epi16(aLo, bLo);

            const __m128i aHi = _mm_unpackhi_epi8(a, _mm_setzero_si128());
            const __m128i bHi = _mm_unpackhi_epi8(b, _mm_setzero_si128());
            const __m128i dHi = _mm_sub_epi16(aHi, bHi);

            return _mm_add_epi32(_mm_madd_epi16(dLo, dLo), _mm_madd_epi16(dHi, dHi));
        }

        int SquaredDifferenceSum32A(const uchar *a, const uchar *b, size_t size)
        {
            assert(Aligned(a) && Aligned(b) && Aligned(size));

            __m128i sum = _mm_setzero_si128();
            for(size_t i = 0; i < size; i += A)
            {
                const __m128i a_ = _mm_load_si128((__m128i*)(a + i));
                const __m128i b_ = _mm_load_si128((__m128i*)(b + i)); 
                sum = _mm_add_epi32(sum, SquaredDifference32(a_, b_));
            }
            return ExtractInt32Sum(sum);
        }
        
        int SquaredDifferenceSum32A(const uchar *a, const uchar *b, const uchar *mask, size_t size)
        {
            assert(Aligned(a) && Aligned(b) && Aligned(size));

            __m128i sum = _mm_setzero_si128();
            for(size_t i = 0; i < size; i += A)
            {
                const __m128i mask_ = _mm_load_si128((__m128i*)(mask + i));
                const __m128i a_ = _mm_and_si128(_mm_load_si128((__m128i*)(a + i)), mask_);
                const __m128i b_ = _mm_and_si128(_mm_load_si128((__m128i*)(b + i)), mask_); 
                sum = _mm_add_epi32(sum, SquaredDifference32(a_, b_));
            }
			return ExtractInt32Sum(sum);
        }
    }
#endif// SIMD_SSE2_ENABLE

    FlatSquaredDifferenceSum32Ptr FlatSquaredDifferenceSum32A = 
        SIMD_SSE2_INIT_FUNCTION_PTR(FlatSquaredDifferenceSum32Ptr, Sse2::SquaredDifferenceSum32A, Base::SquaredDifferenceSum32);

    FlatMaskedSquaredDifferenceSum32Ptr FlatMaskedSquaredDifferenceSum32A = 
        SIMD_SSE2_INIT_FUNCTION_PTR(FlatMaskedSquaredDifferenceSum32Ptr, Sse2::SquaredDifferenceSum32A, Base::SquaredDifferenceSum32);
}

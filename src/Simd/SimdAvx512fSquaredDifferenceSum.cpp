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

        template <bool align> SIMD_INLINE void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(align)
                assert(Aligned(a) && Aligned(b));

            *sum = 0;
            size_t alignedSize = AlignLo(size, F);
			__mmask16 tailMask = TailMask16(size - alignedSize);
            size_t fullAlignedSize = AlignLo(size, QF);
            size_t i = 0;
            __m512 sums[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
            if(fullAlignedSize)
            {
                for(; i < fullAlignedSize; i += QF)
                {
                    SquaredDifferenceSum32f<align, false>(a, b, i + 0 * F, sums[0]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 1 * F, sums[1]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 2 * F, sums[2]);
                    SquaredDifferenceSum32f<align, false>(a, b, i + 3 * F, sums[3]);
                }
                sums[0] = _mm512_add_ps(_mm512_add_ps(sums[0], sums[1]), _mm512_add_ps(sums[2], sums[3]));
            }
            for(; i < alignedSize; i += F)
                SquaredDifferenceSum32f<align, false>(a, b, i, sums[0]);  
			if(i < size)
				SquaredDifferenceSum32f<align, true>(a, b, i, sums[0], tailMask);
			*sum = ExtractSum(sums[0]);
        }

        void SquaredDifferenceSum32f(const float * a, const float * b, size_t size, float * sum)
        {
            if(Aligned(a) && Aligned(b))
                SquaredDifferenceSum32f<true>(a, b, size, sum);
            else
                SquaredDifferenceSum32f<false>(a, b, size, sum);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}

/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSse2.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void Reorder16bit(const uint8_t * src, uint8_t * dst)
        {
            __m128i _src = Load<align>((__m128i*)src);
            Store<align>((__m128i*)dst, _mm_or_si128(_mm_srli_epi16(_src, 8), _mm_slli_epi16(_src, 8)));
        }

        template <bool align> void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            assert(size >= A && size%2 == 0);

            size_t alignedSize = AlignLo(size, A);
            for(size_t i = 0; i < alignedSize; i += A)
                Reorder16bit<align>(src + i, dst + i);
            for(size_t i = alignedSize; i < size; i += 2)
                Base::Reorder16bit(src + i, dst + i);
        }

        void Reorder16bit(const uint8_t * src, size_t size, uint8_t * dst)
        {
            if(Aligned(src) && Aligned(dst))
                Reorder16bit<true>(src, size, dst);
            else
                Reorder16bit<false>(src, size, dst);
        }
    }
#endif// SIMD_SSE2_ENABLE
}
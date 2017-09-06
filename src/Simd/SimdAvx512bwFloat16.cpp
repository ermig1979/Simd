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
#include "Simd/SimdExtract.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<bool align, bool mask> SIMD_INLINE void Float32ToFloat16(const float * src, uint16_t * dst, const __mmask16 * srcTails, __mmask32 dstTail)
        {
			__m256i lo = _mm512_cvtps_ph((Avx512f::Load<align, mask>(src + 0, srcTails[0])), 0);
			__m256i hi = _mm512_cvtps_ph((Avx512f::Load<align, mask>(src + F, srcTails[1])), 0);
			Store<align, mask>(dst, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1), dstTail);
        }

		template<bool align> SIMD_INLINE void Float32ToFloat16x4(const float * src, uint16_t * dst)
		{
			Store<align>(dst + 0 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(Avx512f::Load<align>(src + 0 * F), 0)), _mm512_cvtps_ph(Avx512f::Load<align>(src + 1 * F), 0), 1));
			Store<align>(dst + 1 * HA, _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtps_ph(Avx512f::Load<align>(src + 2 * F), 0)), _mm512_cvtps_ph(Avx512f::Load<align>(src + 3 * F), 0), 1));
		}

        template <bool align> void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t fullAlignedSize = Simd::AlignLo(size, QF);
            size_t alignedSize = Simd::AlignLo(size, DF);
			__mmask16 srcTailMasks[2];
			for (size_t c = 0; c < 2; ++c)
				srcTailMasks[c] = TailMask16(size - alignedSize - F*c);
			__mmask32 dstTailMask = TailMask32(size - alignedSize);

            size_t i = 0;
			for (; i < alignedSize; i += QF)
				Float32ToFloat16x4<align>(src + i, dst + i);
			for (; i < alignedSize; i += DF)
                Float32ToFloat16<align, false>(src + i, dst + i, srcTailMasks, dstTailMask);
            if(i < size)
                Float32ToFloat16<align, true>(src + i, dst + i, srcTailMasks, dstTailMask);
        }

        void Float32ToFloat16(const float * src, size_t size, uint16_t * dst)
        {
            if (Aligned(src) && Aligned(dst))
                Float32ToFloat16<true>(src, size, dst);
            else
                Float32ToFloat16<false>(src, size, dst);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

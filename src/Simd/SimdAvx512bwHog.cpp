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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdEnable.h"
#include "Simd/SimdAllocator.hpp"

#include <vector>

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
	namespace Avx512bw
	{
		SIMD_INLINE void HogDeinterleave(const float * src, size_t count, float ** dst, size_t offset, size_t i)
		{
			src += i;
			__m512 a0 = Load<false>(src + 0x0 * count, src + 0x4 * count, src + 0x8 * count, src + 0xC * count);
			__m512 a1 = Load<false>(src + 0x1 * count, src + 0x5 * count, src + 0x9 * count, src + 0xD * count);
			__m512 a2 = Load<false>(src + 0x2 * count, src + 0x6 * count, src + 0xA * count, src + 0xE * count);
			__m512 a3 = Load<false>(src + 0x3 * count, src + 0x7 * count, src + 0xB * count, src + 0xF * count);
			__m512 b0 = _mm512_unpacklo_ps(a0, a2);
			__m512 b1 = _mm512_unpackhi_ps(a0, a2);
			__m512 b2 = _mm512_unpacklo_ps(a1, a3);
			__m512 b3 = _mm512_unpackhi_ps(a1, a3);
			Avx512f::Store<false>(dst[i + 0] + offset, _mm512_unpacklo_ps(b0, b2));
			Avx512f::Store<false>(dst[i + 1] + offset, _mm512_unpackhi_ps(b0, b2));
			Avx512f::Store<false>(dst[i + 2] + offset, _mm512_unpacklo_ps(b1, b3));
			Avx512f::Store<false>(dst[i + 3] + offset, _mm512_unpackhi_ps(b1, b3));
		}

        void HogDeinterleave(const float * src, size_t srcStride, size_t width, size_t height, size_t count, float ** dst, size_t dstStride)
        {
            assert(width >= F && count >= 4);

            size_t alignedCount = AlignLo(count, 4);
            size_t alignedWidth = AlignLo(width, F);

            for (size_t row = 0; row < height; ++row)
            {
                size_t rowOffset = row*dstStride;
                for (size_t col = 0; col < alignedWidth; col += F)
                {
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += 4)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - 4);
                }
                if (alignedWidth != width)
                {
                    size_t col = width - F;
                    const float * s = src + count*col;
                    size_t offset = rowOffset + col;
                    for (size_t i = 0; i < alignedCount; i += 4)
                        HogDeinterleave(s, count, dst, offset, i);
                    if (alignedCount != count)
                        HogDeinterleave(s, count, dst, offset, count - 4);
                }
                src += srcStride;
            }
        }
	}
#endif
}

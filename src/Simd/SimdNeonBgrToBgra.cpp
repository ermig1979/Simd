/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE  
	namespace Neon
	{
		const size_t A3 = A * 3;
		const size_t A4 = A * 4;

		template <bool align> SIMD_INLINE void BgrToBgra(const uint8_t * bgr, uint8_t * bgra, uint8x16x4_t & _bgra)
        {
            *(uint8x16x3_t*)&_bgra = Load3<align>(bgr);
			Store4<align>(bgra, _bgra);
        }

		template <bool align> void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
		{
			assert(width >= A);
			if (align)
				assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride));

			size_t alignedWidth = AlignLo(width, A);

			uint8x16x4_t _bgra;
			_bgra.val[3] = vdupq_n_u8(alpha);

			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0, colBgra = 0, colBgr = 0; col < alignedWidth; col += A, colBgra += A4, colBgr += A3)
					BgrToBgra<align>(bgr + colBgr, bgra + colBgra, _bgra);
				if (width != alignedWidth)
					BgrToBgra<false>(bgr + 3 * (width - A), bgra + 4 * (width - A), _bgra);
				bgr += bgrStride;
				bgra += bgraStride;
			}
		}

		void BgrToBgra(const uint8_t * bgr, size_t width, size_t height, size_t bgrStride, uint8_t * bgra, size_t bgraStride, uint8_t alpha)
		{
			if (Aligned(bgra) && Aligned(bgraStride) && Aligned(bgr) && Aligned(bgrStride))
				BgrToBgra<true>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
			else
				BgrToBgra<false>(bgr, width, height, bgrStride, bgra, bgraStride, alpha);
		}
	}
#endif// SIMD_NEON_ENABLE
}
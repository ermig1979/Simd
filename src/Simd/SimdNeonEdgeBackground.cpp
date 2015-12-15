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
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeSlow(const uint8_t * value, uint8_t * background, uint8x16_t mask)
        {
            const uint8x16_t _value = Load<align>(value);
            const uint8x16_t _background = Load<align>(background);
            const uint8x16_t inc = vandq_u8(mask, vcgtq_u8(_value, _background));
            Store<align>(background, vqaddq_u8(_background, inc));
        }

        template <bool align> void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
             uint8_t * background, size_t backgroundStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(value) && Aligned(valueStride));
                assert(Aligned(background) && Aligned(backgroundStride));
            }

            size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_01, A - width + alignedWidth);
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0; col < alignedWidth; col += A)
                    EdgeBackgroundGrowRangeSlow<align>(value + col, background + col, K8_01);
                if(alignedWidth != width)
                    EdgeBackgroundGrowRangeSlow<false>(value + width - A, background + width - A, tailMask);
                value += valueStride;
                background += backgroundStride;
            }
        }

        void EdgeBackgroundGrowRangeSlow(const uint8_t * value, size_t valueStride, size_t width, size_t height,
             uint8_t * background, size_t backgroundStride)
        {
            if(Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
                EdgeBackgroundGrowRangeSlow<true>(value, valueStride, width, height, background, backgroundStride);
            else
                EdgeBackgroundGrowRangeSlow<false>(value, valueStride, width, height, background, backgroundStride);
        }

		template <bool align> SIMD_INLINE void EdgeBackgroundGrowRangeFast(const uint8_t * value, uint8_t * background)
		{
			const uint8x16_t _value = Load<align>(value);
			const uint8x16_t _background = Load<align>(background);
			Store<align>(background, vmaxq_u8(_background, _value));
		}

		template <bool align> void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			uint8_t * background, size_t backgroundStride)
		{
			assert(width >= A);
			if (align)
			{
				assert(Aligned(value) && Aligned(valueStride));
				assert(Aligned(background) && Aligned(backgroundStride));
			}

			size_t alignedWidth = AlignLo(width, A);
			for (size_t row = 0; row < height; ++row)
			{
				for (size_t col = 0; col < alignedWidth; col += A)
					EdgeBackgroundGrowRangeFast<align>(value + col, background + col);
				if (alignedWidth != width)
					EdgeBackgroundGrowRangeFast<false>(value + width - A, background + width - A);
				value += valueStride;
				background += backgroundStride;
			}
		}

		void EdgeBackgroundGrowRangeFast(const uint8_t * value, size_t valueStride, size_t width, size_t height,
			uint8_t * background, size_t backgroundStride)
		{
			if (Aligned(value) && Aligned(valueStride) && Aligned(background) && Aligned(backgroundStride))
				EdgeBackgroundGrowRangeFast<true>(value, valueStride, width, height, background, backgroundStride);
			else
				EdgeBackgroundGrowRangeFast<false>(value, valueStride, width, height, background, backgroundStride);
		}
    }
#endif// SIMD_NEON_ENABLE
}
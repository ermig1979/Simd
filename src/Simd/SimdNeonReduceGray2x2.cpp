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

#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
		SIMD_INLINE uint8x16_t Average(const uint8x16_t & s0, const uint8x16_t & s1)
		{
			uint16x8_t s00 = (uint16x8_t)vandq_u8(s0, (uint8x16_t)K16_00FF);
			uint16x8_t s01 = (uint16x8_t)vandq_u8(vrev16q_u8(s0), (uint8x16_t)K16_00FF);
			uint16x8_t s10 = (uint16x8_t)vandq_u8(s1, (uint8x16_t)K16_00FF);
			uint16x8_t s11 = (uint16x8_t)vandq_u8(vrev16q_u8(s1), (uint8x16_t)K16_00FF);
			return (uint8x16_t)vshrq_n_u16(vaddq_u16(vaddq_u16(vaddq_u16(s00, s01), vaddq_u16(s10, s11)), K16_0002), 2);
		}

		template <bool align> SIMD_INLINE uint8x16_t Average(const uint8_t * src0, const uint8_t * src1)
		{
			uint8x16_t lo = Average(Load<align>(src0 + 0), Load<align>(src1 + 0));
			uint8x16_t hi = Average(Load<align>(src0 + A), Load<align>(src1 + A));
			return vuzpq_u8(lo, hi).val[0];
		}

        template <bool align> void ReduceGray2x2(
            const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
             uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight && srcWidth >= DA);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenWidth = AlignLo(srcWidth, 2);
            for(size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t * src0 = src;
                const uint8_t * src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
				for (; srcOffset < alignedWidth; srcOffset += DA, dstOffset += A)
					Store<align>(dst + dstOffset, Average<align>(src0 + srcOffset, src1 + srcOffset));
                if(alignedWidth != srcWidth)
                {
                    dstOffset = dstWidth - A - (evenWidth != srcWidth ? 1 : 0);
                    srcOffset = evenWidth - DA;
					Store<align>(dst + dstOffset, Average<align>(src0 + srcOffset, src1 + srcOffset));
					if (evenWidth != srcWidth)
                        dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                }
                src += 2*srcStride;
                dst += dstStride;
            }
        }

        void ReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
             uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dst)&& Aligned(dstStride))
                ReduceGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}
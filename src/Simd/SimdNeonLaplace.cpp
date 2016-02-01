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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
		const uint8x8_t K8X8_08 = SIMD_VEC_SET1_EPI16(0x0008);

        template<int part> SIMD_INLINE int16x8_t Laplace(uint8x16_t a[3][3])
        {
            return vsubq_s16((int16x8_t)vmull_u8(K8X8_08, Half<part>(a[1][1])), (int16x8_t)vaddq_u16(
				vaddq_u16(vaddl_u8(Half<part>(a[0][0]), Half<part>(a[0][1])), vaddl_u8(Half<part>(a[0][2]), Half<part>(a[1][0]))),
				vaddq_u16(vaddl_u8(Half<part>(a[1][2]), Half<part>(a[2][0])), vaddl_u8(Half<part>(a[2][1]), Half<part>(a[2][2])))));
        }

        template<bool align> SIMD_INLINE void Laplace(uint8x16_t a[3][3], int16_t * dst)
        {
            Store<align>(dst + 0, Laplace<0>(a));
            Store<align>(dst + HA, Laplace<1>(a));
        }

        template <bool align> void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if(align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t a[3][3];

            for(size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if(row == 0)
                    src0 = src1;
                if(row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                Laplace<align>(a, dst + 0);
                for(size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    Laplace<align>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                Laplace<false>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride%sizeof(int16_t) == 0);

            if(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Laplace<true>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
            else
                Laplace<false>(src, srcStride, width, height, (int16_t *)dst, dstStride/sizeof(int16_t));
        }
    }
#endif// SIMD_NEON_ENABLE
}
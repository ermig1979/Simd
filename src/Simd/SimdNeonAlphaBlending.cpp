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
#include "Simd/SimdStore.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
	namespace Neon
	{
		SIMD_INLINE uint16x8_t AlphaBlending(uint16x8_t src, uint16x8_t dst, uint16x8_t alpha)
		{
			return DivideI16By255(vaddq_u16(vmulq_u16(src, alpha), vmulq_u16(dst, vsubq_u16(K16_00FF, alpha))));
		}

		template <bool align> SIMD_INLINE void AlphaBlending(const uint8_t * src, uint8_t * dst, uint8x8x2_t alpha)
		{
			uint8x16_t _src = Load<align>(src);
			uint8x16_t _dst = Load<align>(dst);
			uint16x8_t lo = AlphaBlending(UnpackU8<0>(_src), UnpackU8<0>(_dst), vmovl_u8(alpha.val[0]));
			uint16x8_t hi = AlphaBlending(UnpackU8<1>(_src), UnpackU8<1>(_dst), vmovl_u8(alpha.val[1]));
			Store<align>(dst, PackU16(lo, hi));
		}

		template <bool align, size_t channelCount> struct AlphaBlender
		{
			void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha);
		};

		template <bool align> struct AlphaBlender<align, 1>
		{
			SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
			{
				uint8x8x2_t _alpha{ vget_low_u8(alpha), vget_high_u8(alpha)};
				AlphaBlending<align>(src, dst, _alpha);
			}
		};

		const uint8x8_t K8_TBL1_2_0 = SIMD_VEC_SETR_EPI16(0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3);
		const uint8x8_t K8_TBL1_2_1 = SIMD_VEC_SETR_EPI16(0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7);

        template <bool align> struct AlphaBlender<align, 2>
        {

            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
				uint8x8x2_t _alpha {vget_low_u8(alpha), vget_high_u8(alpha)};
				AlphaBlending<align>(src + 0, dst + 0, { vtbl1_u8(_alpha.val[0], K8_TBL1_2_0), vtbl1_u8(_alpha.val[0], K8_TBL1_2_1) });
				AlphaBlending<align>(src + A, dst + A, { vtbl1_u8(_alpha.val[1], K8_TBL1_2_0), vtbl1_u8(_alpha.val[1], K8_TBL1_2_1) });
            }
        };

		const uint8x8_t K8_TBL1_3_0 = SIMD_VEC_SETR_EPI16(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x2, 0x2);
		const uint8x8_t K8_TBL1_3_1 = SIMD_VEC_SETR_EPI16(0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5);
		const uint8x8_t K8_TBL1_3_2 = SIMD_VEC_SETR_EPI16(0x5, 0x5, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7);

		template <bool align> struct AlphaBlender<align, 3>
		{
			SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
			{
				uint8x8x2_t _alpha{ vget_low_u8(alpha), vget_high_u8(alpha) };
				AlphaBlending<align>(src + 0, dst + 0, { vtbl1_u8(_alpha.val[0], K8_TBL1_3_0), vtbl1_u8(_alpha.val[0], K8_TBL1_3_1) });
				AlphaBlending<align>(src + A, dst + A, { vtbl1_u8(_alpha.val[0], K8_TBL1_3_2), vtbl1_u8(_alpha.val[1], K8_TBL1_3_0) });
				AlphaBlending<align>(src + DA, dst + DA, { vtbl1_u8(_alpha.val[1], K8_TBL1_3_1), vtbl1_u8(_alpha.val[1], K8_TBL1_3_2) });
			}
		};

		const uint8x8_t K8_TBL1_4_0 = SIMD_VEC_SETR_EPI16(0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1);
		const uint8x8_t K8_TBL1_4_1 = SIMD_VEC_SETR_EPI16(0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3);
		const uint8x8_t K8_TBL1_4_2 = SIMD_VEC_SETR_EPI16(0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5);
		const uint8x8_t K8_TBL1_4_3 = SIMD_VEC_SETR_EPI16(0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);

        template <bool align> struct AlphaBlender<align, 4>
        {
            SIMD_INLINE void operator()(const uint8_t * src, uint8_t * dst, uint8x16_t alpha)
            {
				uint8x8x2_t _alpha{ vget_low_u8(alpha), vget_high_u8(alpha) };
				AlphaBlending<align>(src + 0, dst + 0, { vtbl1_u8(_alpha.val[0], K8_TBL1_4_0), vtbl1_u8(_alpha.val[0], K8_TBL1_4_1) });
				AlphaBlending<align>(src + A, dst + A, { vtbl1_u8(_alpha.val[0], K8_TBL1_4_2), vtbl1_u8(_alpha.val[0], K8_TBL1_4_3) });
				AlphaBlending<align>(src + DA, dst + DA, { vtbl1_u8(_alpha.val[1], K8_TBL1_4_0), vtbl1_u8(_alpha.val[1], K8_TBL1_4_1) });
				AlphaBlending<align>(src + A*3, dst + A*3, { vtbl1_u8(_alpha.val[1], K8_TBL1_4_2), vtbl1_u8(_alpha.val[1], K8_TBL1_4_3) });
			}
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, 
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
			uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t step = channelCount*A;
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += step)
                {
					uint8x16_t _alpha = Load<align>(alpha + col);
                    AlphaBlender<align, channelCount>()(src + offset, dst + offset, _alpha);
                }
                if(alignedWidth != width)
                {
					uint8x16_t _alpha = vandq_u8(Load<false>(alpha + width - A), tailMask);
                    AlphaBlender<false, channelCount>()(src + (width - A)*channelCount, dst + (width - A)*channelCount, _alpha);
                }
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }        
        }

        template <bool align> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch(channelCount)
            {
            case 1 : AlphaBlending<align, 1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2 : AlphaBlending<align, 2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
			case 3 : AlphaBlending<align, 3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
			case 4 : AlphaBlending<align, 4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
		{
			if (Aligned(src) && Aligned(srcStride) && Aligned(alpha) && Aligned(alphaStride) && Aligned(dst) && Aligned(dstStride))
				AlphaBlending<true>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
			else
				AlphaBlending<false>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
		}
    }
#endif// SIMD_NEON_ENABLE
}
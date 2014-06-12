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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        SIMD_INLINE v128_u16 Average16(const v128_u16 & s00, const v128_u16 & s01, const v128_u16 & s10, const v128_u16 & s11)
        {
            return vec_sr(vec_add(vec_add(vec_add(s00, s01), vec_add(s10, s11)), K16_0002), K16_0002); 
        }

        SIMD_INLINE v128_u8 Average8(const v128_u8 & s00, const v128_u8 & s01, const v128_u8 & s10, const v128_u8 & s11)
        {
            v128_u16 lo = Average16(
                vec_mule(s00, K8_01), vec_mulo(s00, K8_01), 
                vec_mule(s10, K8_01), vec_mulo(s10, K8_01)); 
            v128_u16 hi = Average16(
                vec_mule(s01, K8_01), vec_mulo(s01, K8_01), 
                vec_mule(s11, K8_01), vec_mulo(s11, K8_01)); 
            return vec_pack(lo, hi);
        }

        template<bool align> void ReduceGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1)/2 == dstWidth && (srcHeight + 1)/2 == dstHeight);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride) && Aligned(dstWidth));
            }

            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenWidth = AlignLo(srcWidth, 2);
            for(size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for(; srcOffset < alignedWidth; srcOffset += DA, dstOffset += A)
                {
                    Store<align>(dst + dstOffset, Average8(
                        Load<align>(src0 + srcOffset), Load<align>(src0 + srcOffset + A), 
                        Load<align>(src1 + srcOffset), Load<align>(src1 + srcOffset + A)));
                }
                if(alignedWidth != srcWidth)
                {
                    dstOffset = dstWidth - A - (evenWidth != srcWidth ? 1 : 0);
                    srcOffset = evenWidth - DA;
                    Store<align>(dst + dstOffset, Average8(
                        Load<align>(src0 + srcOffset), Load<align>(src0 + srcOffset + A), 
                        Load<align>(src1 + srcOffset), Load<align>(src1 + srcOffset + A)));
                    if(evenWidth != srcWidth)
                    {
                        dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                    }
                }
                src += 2*srcStride;
                dst += dstStride;
            }
        }

        void ReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, 
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if(Aligned(src) && Aligned(srcWidth) && Aligned(srcStride) && Aligned(dst) && Aligned(dstWidth) && Aligned(dstStride))
                ReduceGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}
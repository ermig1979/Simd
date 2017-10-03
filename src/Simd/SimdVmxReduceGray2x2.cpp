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

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        SIMD_INLINE v128_u16 Average16(const v128_u8 & s0, const v128_u8 & s1)
        {
            v128_u32 lo = vec_msum((v128_u8)UnpackLoU16((v128_u16)s0, (v128_u16)s1), K8_01, K32_00000000);
            v128_u32 hi = vec_msum((v128_u8)UnpackHiU16((v128_u16)s0, (v128_u16)s1), K8_01, K32_00000000);
            return vec_sr(vec_add(vec_pack(lo, hi), K16_0002), K16_0002);
        }

        template<bool align> SIMD_INLINE v128_u8 Average(const uint8_t * src0, const uint8_t * src1, size_t offset)
        {
            v128_u16 lo = Average16(Load<align>(src0 + offset + 0), Load<align>(src1 + offset + 0));
            v128_u16 hi = Average16(Load<align>(src0 + offset + A), Load<align>(src1 + offset + A));
            return vec_pack(lo, hi);
        }

        template<bool align> void ReduceGray2x2(const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            size_t fullAlignedWidth = AlignLo(srcWidth, QA);
            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenWidth = AlignLo(srcWidth, 2);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t *src0 = src;
                const uint8_t *src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);

                if (align)
                {
                    size_t srcOffset = 0, dstOffset = 0;
                    for (; srcOffset < fullAlignedWidth; srcOffset += QA, dstOffset += DA)
                    {
                        Store<align>(dst + dstOffset, Average<align>(src0, src1, srcOffset));
                        Store<align>(dst + dstOffset + A, Average<align>(src0, src1, srcOffset + DA));
                    }
                    for (; srcOffset < alignedWidth; srcOffset += DA, dstOffset += A)
                        Store<align>(dst + dstOffset, Average<align>(src0, src1, srcOffset));
                }
                else
                {
                    Storer<align> _dst(dst);
                    _dst.First(Average<align>(src0, src1, 0));
                    for (size_t srcOffset = DA; srcOffset < alignedWidth; srcOffset += DA)
                        _dst.Next(Average<align>(src0, src1, srcOffset));
                    Flush(_dst);
                }

                if (alignedWidth != srcWidth)
                {
                    Store<false>(dst + dstWidth - A - (evenWidth != srcWidth ? 1 : 0), Average<false>(src0, src1, evenWidth - DA));
                    if (evenWidth != srcWidth)
                        dst[dstWidth - 1] = Base::Average(src0[evenWidth], src1[evenWidth]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        void ReduceGray2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                ReduceGray2x2<true>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
            else
                ReduceGray2x2<false>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}

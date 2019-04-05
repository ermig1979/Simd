/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
        SIMD_INLINE uint8x8_t Average(const uint8x16_t & s0, const uint8x16_t & s1)
        {
            return vshrn_n_u16(vaddq_u16(vaddq_u16(vpaddlq_u8(s0), vpaddlq_u8(s1)), vdupq_n_u16(2)), 2);
        }

        template <bool align> SIMD_INLINE void ReduceGray2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            uint8x8x2_t _dst;
            _dst.val[0] = Average(Load<align>(src0 + 0), Load<align>(src1 + 0));
            _dst.val[1] = Average(Load<align>(src0 + A), Load<align>(src1 + A));
            Store<align>(dst, *(uint8x16_t*)&_dst);
        }

        template <bool align> SIMD_INLINE void ReduceGray2x2(const uint8_t * src0, const uint8_t * src1, size_t size, uint8_t * dst)
        {
            for (size_t i = 0; i < size; i += DA, src0 += DA, src1 += DA, dst += A)
                ReduceGray2x2<align>(src0, src1, dst);
        }

#if defined(SIMD_NEON_ASM_ENABLE) && 0
        template <> void ReduceGray2x2<true>(const uint8_t * src0, const uint8_t * src1, size_t size, uint8_t * dst)
        {
            asm(
                "mov r4, #2               \n"
                "vdup.u16  q4, r4         \n"
                "mov r5, %0               \n"
                "mov r6, %1               \n"
                "mov r4, %2               \n"
                "mov r7, %3               \n"
                ".loop:                   \n"
                "vld1.8 {q0}, [r5:128]!   \n"
                "vld1.8 {q2}, [r6:128]!   \n"
                "vpaddl.u8 q0, q0         \n"
                "vpadal.u8 q0, q2         \n"
                "vadd.u16  q0, q0, q4     \n"
                "vshrn.u16 d10, q0, #2    \n"
                "vld1.8 {q1}, [r5:128]!   \n"
                "vld1.8 {q3}, [r6:128]!   \n"
                "vpaddl.u8 q1, q1         \n"
                "vpadal.u8 q1, q3         \n"
                "vadd.u16  q1, q1, q4     \n"
                "vshrn.u16 d11, q1, #2    \n"
                "vst1.8 {q5}, [r7:128]!   \n"
                "subs r4, r4, #32         \n"
                "bne .loop                \n"

                :
            : "r"(src0), "r"(src1), "r" (size), "r"(dst)
                : "q0", "q1", "q2", "q3", "q4", "q5", "r4", "r5", "r6", "r7", "memory"
                );
        }
#endif

        template <bool align> void ReduceGray2x2(
            const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight && srcWidth >= DA);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            size_t alignedWidth = AlignLo(srcWidth, DA);
            size_t evenWidth = AlignLo(srcWidth, 2);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t * src0 = src;
                const uint8_t * src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                ReduceGray2x2<align>(src0, src1, alignedWidth, dst);
                if (alignedWidth != srcWidth)
                {
                    size_t dstOffset = dstWidth - A - (evenWidth != srcWidth ? 1 : 0);
                    size_t srcOffset = evenWidth - DA;
                    ReduceGray2x2<false>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset);
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
#endif// SIMD_NEON_ENABLE
}

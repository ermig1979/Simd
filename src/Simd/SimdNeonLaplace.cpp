/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"

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

        template <bool align, bool abs> SIMD_INLINE void Laplace(uint8x16_t a[3][3], int16_t * dst)
        {
            Store<align>(dst + 0, ConditionalAbs<abs>(Laplace<0>(a)));
            Store<align>(dst + 8, ConditionalAbs<abs>(Laplace<1>(a)));
        }

        template <bool align, bool abs> void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, int16_t * dst, size_t dstStride)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride, HA));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;
            uint8x16_t a[3][3];

            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + srcStride*(row - 1);
                src1 = src0 + srcStride;
                src2 = src1 + srcStride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                Laplace<align, abs>(a, dst + 0);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    Laplace<align, abs>(a, dst + col);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                Laplace<false, abs>(a, dst + width - A);

                dst += dstStride;
            }
        }

        void Laplace(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Laplace<true, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                Laplace<false, false>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        void LaplaceAbs(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(dstStride % sizeof(int16_t) == 0);

            if (Aligned(src) && Aligned(srcStride) && Aligned(dst) && Aligned(dstStride))
                Laplace<true, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
            else
                Laplace<false, true>(src, srcStride, width, height, (int16_t *)dst, dstStride / sizeof(int16_t));
        }

        SIMD_INLINE void LaplaceAbsSum(uint8x16_t a[3][3], uint32x4_t & sum)
        {
            sum = vaddq_u32(sum, vpaddlq_u16((uint16x8_t)ConditionalAbs<true>(Laplace<0>(a))));
            sum = vaddq_u32(sum, vpaddlq_u16((uint16x8_t)ConditionalAbs<true>(Laplace<1>(a))));
        }

        SIMD_INLINE void SetMask3(uint8x16_t a[3], uint8x16_t mask)
        {
            a[0] = vandq_u8(a[0], mask);
            a[1] = vandq_u8(a[1], mask);
            a[2] = vandq_u8(a[2], mask);
        }

        SIMD_INLINE void SetMask3x3(uint8x16_t a[3][3], uint8x16_t mask)
        {
            SetMask3(a[0], mask);
            SetMask3(a[1], mask);
            SetMask3(a[2], mask);
        }

        template <bool align> void LaplaceAbsSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width > A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = Simd::AlignHi(width, A) - A;
            const uint8_t *src0, *src1, *src2;

            uint8x16_t a[3][3];
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + bodyWidth);

            uint64x2_t fullSum = K64_0000000000000000;
            for (size_t row = 0; row < height; ++row)
            {
                src0 = src + stride*(row - 1);
                src1 = src0 + stride;
                src2 = src1 + stride;
                if (row == 0)
                    src0 = src1;
                if (row == height - 1)
                    src2 = src1;

                uint32x4_t rowSum = K32_00000000;

                LoadNose3<align, 1>(src0 + 0, a[0]);
                LoadNose3<align, 1>(src1 + 0, a[1]);
                LoadNose3<align, 1>(src2 + 0, a[2]);
                LaplaceAbsSum(a, rowSum);
                for (size_t col = A; col < bodyWidth; col += A)
                {
                    LoadBody3<align, 1>(src0 + col, a[0]);
                    LoadBody3<align, 1>(src1 + col, a[1]);
                    LoadBody3<align, 1>(src2 + col, a[2]);
                    LaplaceAbsSum(a, rowSum);
                }
                LoadTail3<false, 1>(src0 + width - A, a[0]);
                LoadTail3<false, 1>(src1 + width - A, a[1]);
                LoadTail3<false, 1>(src2 + width - A, a[2]);
                SetMask3x3(a, tailMask);
                LaplaceAbsSum(a, rowSum);

                fullSum = vaddq_u64(fullSum, vpaddlq_u32(rowSum));
            }
            *sum = ExtractSum64u(fullSum);
        }

        void LaplaceAbsSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(srcStride))
                LaplaceAbsSum<true>(src, srcStride, width, height, sum);
            else
                LaplaceAbsSum<false>(src, srcStride, width, height, sum);
        }
    }
#endif// SIMD_NEON_ENABLE
}

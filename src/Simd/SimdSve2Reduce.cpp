/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint16_t Average16(const svuint8_t& s0, const svuint8_t& s1)
        {
            const svbool_t mask = svptrue_b16();
            svuint16_t sum = svadd_u16_x(mask, svaddlb_u16(s0, svext_u8(s0, s0, 1)), svaddlb_u16(s1, svext_u8(s1, s1, 1)));
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, sum, 2), 2);
        }

        SIMD_INLINE svuint8_t Average8(const svuint8_t& s00, const svuint8_t& s01, const svuint8_t& s10, const svuint8_t& s11)
        {
            return svuzp1_u8(svreinterpret_u8_u16(Average16(s00, s10)), svreinterpret_u8_u16(Average16(s01, s11)));
        }

        SIMD_INLINE svuint8_t ShuffleRc2()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xFC);
            svuint8_t even = svlsl_n_u8_x(all, svand_n_u8_x(all, i, 1), 1);
            svuint8_t odd = svlsr_n_u8_x(all, svand_n_u8_x(all, i, 2), 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, even, odd));
        }

        SIMD_INLINE svuint8_t ShuffleRc4()
        {
            const svbool_t all = svptrue_b8();
            svuint8_t i = svindex_u8(0, 1);
            svuint8_t base = svand_n_u8_x(all, i, 0xF8);
            svuint8_t p = svand_n_u8_x(all, i, 7);
            svuint8_t lo = svlsl_n_u8_x(all, svand_n_u8_x(all, p, 1), 2);
            svuint8_t hi = svlsr_n_u8_x(all, p, 1);
            return svadd_u8_x(all, base, svadd_u8_x(all, lo, hi));
        }

        template <size_t channelCount> SIMD_INLINE svuint8_t Shuffle(const svuint8_t& src, const svuint8_t& index)
        {
            if (channelCount == 1)
                return src;
            else
                return svtbl_u8(src, index);
        }

        template <size_t channelCount> SIMD_INLINE void ReduceColor2x2(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, const svuint8_t& index)
        {
            const svbool_t all = svptrue_b8();
            const size_t A = svcntb();
            svuint8_t s00 = Shuffle<channelCount>(svld1_u8(all, src0 + 0 * A), index);
            svuint8_t s01 = Shuffle<channelCount>(svld1_u8(all, src0 + 1 * A), index);
            svuint8_t s10 = Shuffle<channelCount>(svld1_u8(all, src1 + 0 * A), index);
            svuint8_t s11 = Shuffle<channelCount>(svld1_u8(all, src1 + 1 * A), index);
            svst1_u8(all, dst, Average8(s00, s01, s10, s11));
        }

        template <size_t channelCount> void ReduceColor2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb(), DA = 2 * A;
            svuint8_t index = svdup_n_u8(0);
            if (channelCount == 2)
                index = ShuffleRc2();
            else if (channelCount == 4)
                index = ShuffleRc4();
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t evenSize = evenWidth * channelCount;
            const size_t alignedSize = AlignLo(evenSize, DA);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += DA, dstOffset += A)
                    ReduceColor2x2<channelCount>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, index);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - DA;
                    dstOffset = srcOffset / 2;
                    ReduceColor2x2<channelCount>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, index);
                }
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < channelCount; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        SIMD_INLINE size_t BgrPairSrc(size_t dst)
        {
            return (dst / 6) * 6 + ((dst % 6) % 2) * 3 + (dst % 6) / 2;
        }

        SIMD_INLINE void InitBgrPairIndex(size_t A,
            uint8_t idx01[SIMD_SVE2_VECTOR_SIZE_MAX],
            uint8_t idx0[SIMD_SVE2_VECTOR_SIZE_MAX],
            uint8_t idx12[SIMD_SVE2_VECTOR_SIZE_MAX],
            uint8_t idx22[SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            for (size_t i = 0; i < A; ++i)
            {
                size_t src0 = BgrPairSrc(i);
                assert(src0 < 2 * A);
                idx01[i] = (uint8_t)src0;

                size_t src1 = BgrPairSrc(A + i);
                size_t vec1 = src1 / A;
                size_t lane1 = src1 % A;
                idx0[i] = (vec1 == 0) ? (uint8_t)lane1 : (uint8_t)0xFF;
                if (vec1 == 1)
                    idx12[i] = (uint8_t)lane1;
                else if (vec1 == 2)
                    idx12[i] = (uint8_t)(A + lane1);
                else
                    idx12[i] = (uint8_t)0xFF;

                size_t src2 = BgrPairSrc(2 * A + i);
                assert(src2 >= A && (src2 - A) < 2 * A);
                idx22[i] = (uint8_t)(src2 - A);
            }
        }

        SIMD_INLINE void ReduceBgr2x2(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t A,
            const svuint8_t& idx01, const svuint8_t& idx0, const svuint8_t& idx12, const svuint8_t& idx22)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t a0 = svld1_u8(all, src0 + 0 * A);
            svuint8_t a1 = svld1_u8(all, src0 + 1 * A);
            svuint8_t a2 = svld1_u8(all, src0 + 2 * A);
            svuint8_t b0 = svld1_u8(all, src1 + 0 * A);
            svuint8_t b1 = svld1_u8(all, src1 + 1 * A);
            svuint8_t b2 = svld1_u8(all, src1 + 2 * A);

            svuint8x2_t a01 = svcreate2_u8(a0, a1);
            svuint8x2_t b01 = svcreate2_u8(b0, b1);
            svuint8_t p00 = svtbl2_u8(a01, idx01);
            svuint8_t p10 = svtbl2_u8(b01, idx01);

            svuint8x2_t a12 = svcreate2_u8(a1, a2);
            svuint8x2_t b12 = svcreate2_u8(b1, b2);
            svuint8_t p01 = svorr_u8_x(all, svtbl_u8(a0, idx0), svtbl2_u8(a12, idx12));
            svuint8_t p11 = svorr_u8_x(all, svtbl_u8(b0, idx0), svtbl2_u8(b12, idx12));
            svst1_u8(all, dst, Average8(p00, p01, p10, p11));

            svst1b_u16(svptrue_b16(), dst + A, Average16(svtbl2_u8(a12, idx22), svtbl2_u8(b12, idx22)));
        }

        void ReduceBgr2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb(), HA = A / 2;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            const svbool_t all = svptrue_b8();
            uint8_t idx01[SIMD_SVE2_VECTOR_SIZE_MAX];
            uint8_t idx0[SIMD_SVE2_VECTOR_SIZE_MAX];
            uint8_t idx12[SIMD_SVE2_VECTOR_SIZE_MAX];
            uint8_t idx22[SIMD_SVE2_VECTOR_SIZE_MAX];
            InitBgrPairIndex(A, idx01, idx0, idx12, idx22);
            const svuint8_t index01 = svld1_u8(all, idx01);
            const svuint8_t index0 = svld1_u8(all, idx0);
            const svuint8_t index12 = svld1_u8(all, idx12);
            const svuint8_t index22 = svld1_u8(all, idx22);
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t alignedWidth = AlignLo(srcWidth, A);
            const size_t evenSize = evenWidth * 3;
            const size_t alignedSize = alignedWidth * 3;
            const size_t srcStep = A * 3;
            const size_t dstStep = HA * 3;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += srcStep, dstOffset += dstStep)
                    ReduceBgr2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, index01, index0, index12, index22);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - srcStep;
                    dstOffset = srcOffset / 2;
                    ReduceBgr2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, index01, index0, index12, index22);
                }
                if (evenWidth != srcWidth)
                {
                    for (size_t c = 0; c < 3; ++c)
                        dst[evenSize / 2 + c] = Base::Average(src0[evenSize + c], src1[evenSize + c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        void ReduceColor2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t* dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);
            assert(srcWidth >= 2 * svcntb());

            switch (channelCount)
            {
            case 1: ReduceColor2x2<1>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 2: ReduceColor2x2<2>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceBgr2x2(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }
    }
#endif
}

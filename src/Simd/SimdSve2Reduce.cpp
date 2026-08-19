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

        SIMD_INLINE void InitBgrPairIndex(size_t A, uint8_t index0[3][SIMD_SVE2_VECTOR_SIZE_MAX], uint8_t index12[3][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            for (size_t part = 0; part < 3; ++part)
            {
                for (size_t i = 0; i < A; ++i)
                {
                    size_t j = part * A + i;
                    size_t src = (j / 6) * 6 + ((j % 6) % 2) * 3 + (j % 6) / 2;
                    size_t vec = src / A;
                    size_t lane = src % A;
                    index0[part][i] = (vec == 0) ? (uint8_t)lane : (uint8_t)0xFF;
                    if (vec == 1)
                        index12[part][i] = (uint8_t)lane;
                    else if (vec == 2)
                        index12[part][i] = (uint8_t)(A + lane);
                    else
                        index12[part][i] = (uint8_t)0xFF;
                }
            }
        }

        SIMD_INLINE svuint8_t PairBgr(const svuint8_t& v0, const svuint8_t& v1, const svuint8_t& v2, const svuint8_t& index0, const svuint8_t& index12)
        {
            const svbool_t all = svptrue_b8();
            return svorr_u8_x(all, svtbl_u8(v0, index0), svtbl2_u8(svcreate2_u8(v1, v2), index12));
        }

        SIMD_INLINE void ReduceBgr2x2(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t A,
            const svuint8_t& index00, const svuint8_t& index120,
            const svuint8_t& index01, const svuint8_t& index121,
            const svuint8_t& index02, const svuint8_t& index122)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t a0 = svld1_u8(all, src0 + 0 * A);
            svuint8_t a1 = svld1_u8(all, src0 + 1 * A);
            svuint8_t a2 = svld1_u8(all, src0 + 2 * A);
            svuint8_t a3 = svld1_u8(all, src0 + 3 * A);
            svuint8_t a4 = svld1_u8(all, src0 + 4 * A);
            svuint8_t a5 = svld1_u8(all, src0 + 5 * A);
            svuint8_t b0 = svld1_u8(all, src1 + 0 * A);
            svuint8_t b1 = svld1_u8(all, src1 + 1 * A);
            svuint8_t b2 = svld1_u8(all, src1 + 2 * A);
            svuint8_t b3 = svld1_u8(all, src1 + 3 * A);
            svuint8_t b4 = svld1_u8(all, src1 + 4 * A);
            svuint8_t b5 = svld1_u8(all, src1 + 5 * A);

            svuint8_t p00 = PairBgr(a0, a1, a2, index00, index120);
            svuint8_t p01 = PairBgr(a0, a1, a2, index01, index121);
            svuint8_t p10 = PairBgr(b0, b1, b2, index00, index120);
            svuint8_t p11 = PairBgr(b0, b1, b2, index01, index121);
            svst1_u8(all, dst + 0 * A, Average8(p00, p01, p10, p11));

            svuint8_t p02 = PairBgr(a0, a1, a2, index02, index122);
            svuint8_t p03 = PairBgr(a3, a4, a5, index00, index120);
            svuint8_t p12 = PairBgr(b0, b1, b2, index02, index122);
            svuint8_t p13 = PairBgr(b3, b4, b5, index00, index120);
            svst1_u8(all, dst + 1 * A, Average8(p02, p03, p12, p13));

            svuint8_t p04 = PairBgr(a3, a4, a5, index01, index121);
            svuint8_t p05 = PairBgr(a3, a4, a5, index02, index122);
            svuint8_t p14 = PairBgr(b3, b4, b5, index01, index121);
            svuint8_t p15 = PairBgr(b3, b4, b5, index02, index122);
            svst1_u8(all, dst + 2 * A, Average8(p04, p05, p14, p15));
        }

        void ReduceBgr2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb(), DA = 2 * A;
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            const svbool_t all = svptrue_b8();
            uint8_t index0[3][SIMD_SVE2_VECTOR_SIZE_MAX];
            uint8_t index12[3][SIMD_SVE2_VECTOR_SIZE_MAX];
            InitBgrPairIndex(A, index0, index12);
            const svuint8_t index00 = svld1_u8(all, index0[0]);
            const svuint8_t index120 = svld1_u8(all, index12[0]);
            const svuint8_t index01 = svld1_u8(all, index0[1]);
            const svuint8_t index121 = svld1_u8(all, index12[1]);
            const svuint8_t index02 = svld1_u8(all, index0[2]);
            const svuint8_t index122 = svld1_u8(all, index12[2]);
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t alignedWidth = AlignLo(srcWidth, DA);
            const size_t evenSize = evenWidth * 3;
            const size_t alignedSize = alignedWidth * 3;
            const size_t srcStep = DA * 3;
            const size_t dstStep = A * 3;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += srcStep, dstOffset += dstStep)
                    ReduceBgr2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A,
                        index00, index120, index01, index121, index02, index122);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - srcStep;
                    dstOffset = srcOffset / 2;
                    ReduceBgr2x2(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A,
                        index00, index120, index01, index121, index02, index122);
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

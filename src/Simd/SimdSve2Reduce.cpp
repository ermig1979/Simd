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
#include "Simd/SimdMath.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svuint16_t Average16(const svuint8_t& s0, const svuint8_t& s1, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint16_t sum0 = svaddlb_u16(s0, svext_u8(s0, s0, 1));
            svuint16_t sum1 = svaddlb_u16(s1, svext_u8(s1, s1, 1));
            return svlsr_n_u16_x(mask16, svadd_n_u16_x(mask16, svadd_u16_x(mask16, sum0, sum1), 2), 2);
        }

        SIMD_INLINE svuint8_t AverageRows(const svuint8_t& s0, const svuint8_t& s1, const svbool_t& mask8, const svbool_t& mask16)
        {
            svuint8_t even = svqxtnb_u16(Average16(s0, s1, mask8, mask16));
            return svuzp1_u8(even, even);
        }

        SIMD_INLINE bool InitReduceColor2x2Index(uint8_t index[2][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; i += 4)
            {
                index[0][i] = (uint8_t)i;
                index[0][i + 1] = (uint8_t)(i + 2);
                index[0][i + 2] = (uint8_t)(i + 1);
                index[0][i + 3] = (uint8_t)(i + 3);
            }
            for (size_t i = 0; i < A; i += 8)
            {
                index[1][i] = (uint8_t)i;
                index[1][i + 1] = (uint8_t)(i + 4);
                index[1][i + 2] = (uint8_t)(i + 1);
                index[1][i + 3] = (uint8_t)(i + 5);
                index[1][i + 4] = (uint8_t)(i + 2);
                index[1][i + 5] = (uint8_t)(i + 6);
                index[1][i + 6] = (uint8_t)(i + 3);
                index[1][i + 7] = (uint8_t)(i + 7);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t REDUCE_COLOR2X2_INDEX[2][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool REDUCE_COLOR2X2_INDEX_INITED = InitReduceColor2x2Index(REDUCE_COLOR2X2_INDEX);

        SIMD_INLINE bool InitBgrDeinterlaceIndex(uint8_t index[3][SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                index[0][i] = (uint8_t)(3 * i);
                index[1][i] = (uint8_t)(3 * i + 1);
                index[2][i] = (uint8_t)(3 * i + 2);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t BGR_DEINTERLACE_INDEX[3][SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool BGR_DEINTERLACE_INDEX_INITED = InitBgrDeinterlaceIndex(BGR_DEINTERLACE_INDEX);

        SIMD_INLINE bool InitReduceBgrPackIndex(uint8_t packIndex[SIMD_SVE2_VECTOR_SIZE_MAX], uint8_t channelIndex[SIMD_SVE2_VECTOR_SIZE_MAX])
        {
            size_t A = svlen(svuint8_t());
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            for (size_t i = 0; i < A; ++i)
            {
                packIndex[i] = (uint8_t)(2 * (i / 3));
                channelIndex[i] = (uint8_t)(i % 3);
            }
            return true;
        }

        SIMD_ALIGNED(SIMD_ALIGN) uint8_t REDUCE_BGR_PACK_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        SIMD_ALIGNED(SIMD_ALIGN) uint8_t REDUCE_BGR_CHANNEL_INDEX[SIMD_SVE2_VECTOR_SIZE_MAX];
        const bool REDUCE_BGR_PACK_INDEX_INITED = InitReduceBgrPackIndex(REDUCE_BGR_PACK_INDEX, REDUCE_BGR_CHANNEL_INDEX);

        SIMD_INLINE svuint8_t PackBgr(const svuint8_t& packIdx, const svuint8_t& channelIdx,
            const svuint8_t& b, const svuint8_t& g, const svuint8_t& r, const svbool_t& mask)
        {
            svuint8_t bp = svtbl_u8(b, packIdx);
            svuint8_t gp = svtbl_u8(g, packIdx);
            svuint8_t rp = svtbl_u8(r, packIdx);
            return svsel_u8(svcmpeq_n_u8(mask, channelIdx, 0), bp, svsel_u8(svcmpeq_n_u8(mask, channelIdx, 1), gp, rp));
        }

        SIMD_INLINE void ReduceBgr2x2Kernel(const uint8_t* src0, const uint8_t* src1, uint8_t* dst, size_t pixelCount,
            const svuint8_t& bIdx, const svuint8_t& gIdx, const svuint8_t& rIdx,
            const svuint8_t& packIdx, const svuint8_t& channelIdx)
        {
            const uint64_t loadSize = pixelCount * 3;
            const uint64_t outPixels = pixelCount / 2;
            const uint64_t outBytes = outPixels * 3;
            const svbool_t maskLoad = svwhilelt_b8(0ull, loadSize);
            const svbool_t maskPlane = svwhilelt_b8(0ull, pixelCount);
            const svbool_t mask16 = svwhilelt_b16(0ull, outPixels);
            const svbool_t maskOut = svwhilelt_b8(0ull, outBytes);
            svuint8_t inter0 = svld1_u8(maskLoad, src0);
            svuint8_t inter1 = svld1_u8(maskLoad, src1);
            svuint8_t b0 = svtbl_u8(inter0, bIdx);
            svuint8_t g0 = svtbl_u8(inter0, gIdx);
            svuint8_t r0 = svtbl_u8(inter0, rIdx);
            svuint8_t b1 = svtbl_u8(inter1, bIdx);
            svuint8_t g1 = svtbl_u8(inter1, gIdx);
            svuint8_t r1 = svtbl_u8(inter1, rIdx);
            svuint8_t db = AverageRows(b0, b1, maskPlane, mask16);
            svuint8_t dg = AverageRows(g0, g1, maskPlane, mask16);
            svuint8_t dr = AverageRows(r0, r1, maskPlane, mask16);
            svst1_u8(maskOut, dst, PackBgr(packIdx, channelIdx, db, dg, dr, maskOut));
        }

        template <size_t channelCount> SIMD_INLINE void ReduceColor2x2Kernel(const uint8_t* src0, const uint8_t* src1, uint8_t* dst,
            size_t A, size_t HA, const svuint8_t& shuffleIdx, const svbool_t& maskLo, const svbool_t& maskHi, const svbool_t& mask16);

        template <> SIMD_INLINE void ReduceColor2x2Kernel<2>(const uint8_t* src0, const uint8_t* src1, uint8_t* dst,
            size_t A, size_t HA, const svuint8_t& shuffleIdx, const svbool_t& maskLo, const svbool_t& maskHi, const svbool_t& mask16)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t s00 = svtbl_u8(svld1_u8(all, src0), shuffleIdx);
            svuint8_t s01 = svtbl_u8(svld1_u8(all, src0 + A), shuffleIdx);
            svuint8_t s10 = svtbl_u8(svld1_u8(all, src1), shuffleIdx);
            svuint8_t s11 = svtbl_u8(svld1_u8(all, src1 + A), shuffleIdx);
            svst1_u8(maskLo, dst, AverageRows(s00, s10, all, mask16));
            svst1_u8(maskHi, dst + HA, AverageRows(s01, s11, all, mask16));
        }

        template <> SIMD_INLINE void ReduceColor2x2Kernel<4>(const uint8_t* src0, const uint8_t* src1, uint8_t* dst,
            size_t A, size_t HA, const svuint8_t& shuffleIdx, const svbool_t& maskLo, const svbool_t& maskHi, const svbool_t& mask16)
        {
            const svbool_t all = svptrue_b8();
            svuint8_t s00 = svtbl_u8(svld1_u8(all, src0), shuffleIdx);
            svuint8_t s01 = svtbl_u8(svld1_u8(all, src0 + A), shuffleIdx);
            svuint8_t s10 = svtbl_u8(svld1_u8(all, src1), shuffleIdx);
            svuint8_t s11 = svtbl_u8(svld1_u8(all, src1 + A), shuffleIdx);
            svst1_u8(maskLo, dst, AverageRows(s00, s10, all, mask16));
            svst1_u8(maskHi, dst + HA, AverageRows(s01, s11, all, mask16));
        }

        template <size_t channelCount> void ReduceColor2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb(), DA = 2 * A, HA = svcnth();
            const svbool_t all = svptrue_b8();
            const svbool_t maskLo = all;
            const svbool_t maskHi = all;
            const svbool_t mask16 = svptrue_b16();
            const svuint8_t shuffleIdx = svld1_u8(all, REDUCE_COLOR2X2_INDEX[channelCount == 4 ? 1 : 0]);
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t evenSize = evenWidth * channelCount;
            const size_t alignedSize = AlignLo(evenSize, DA);
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t* src0 = src;
                const uint8_t* src1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                size_t srcOffset = 0, dstOffset = 0;
                for (; srcOffset < alignedSize; srcOffset += DA, dstOffset += A)
                    ReduceColor2x2Kernel<channelCount>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, HA, shuffleIdx, maskLo, maskHi, mask16);
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - DA;
                    dstOffset = srcOffset / 2;
                    ReduceColor2x2Kernel<channelCount>(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, HA, shuffleIdx, maskLo, maskHi, mask16);
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

        void ReduceBgr2x2(const uint8_t* src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb(), DA = 2 * A, HA = svcnth();
            assert(A <= SIMD_SVE2_VECTOR_SIZE_MAX);
            const svbool_t all = svptrue_b8();
            const svuint8_t bIdx = svld1_u8(all, BGR_DEINTERLACE_INDEX[0]);
            const svuint8_t gIdx = svld1_u8(all, BGR_DEINTERLACE_INDEX[1]);
            const svuint8_t rIdx = svld1_u8(all, BGR_DEINTERLACE_INDEX[2]);
            const svuint8_t packIdx = svld1_u8(all, REDUCE_BGR_PACK_INDEX);
            const svuint8_t channelIdx = svld1_u8(all, REDUCE_BGR_CHANNEL_INDEX);
            const size_t evenWidth = AlignLo(srcWidth, 2);
            const size_t alignedWidth = AlignLo(srcWidth, A);
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
                {
                    ReduceBgr2x2Kernel(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, bIdx, gIdx, rIdx, packIdx, channelIdx);
                    ReduceBgr2x2Kernel(src0 + srcOffset + 3 * A, src1 + srcOffset + 3 * A, dst + dstOffset + 3 * HA, A, bIdx, gIdx, rIdx, packIdx, channelIdx);
                }
                if (alignedSize != evenSize)
                {
                    srcOffset = evenSize - srcStep;
                    dstOffset = srcOffset / 2;
                    ReduceBgr2x2Kernel(src0 + srcOffset, src1 + srcOffset, dst + dstOffset, A, bIdx, gIdx, rIdx, packIdx, channelIdx);
                    ReduceBgr2x2Kernel(src0 + srcOffset + 3 * A, src1 + srcOffset + 3 * A, dst + dstOffset + 3 * HA, A, bIdx, gIdx, rIdx, packIdx, channelIdx);
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
            case 1: ReduceGray2x2(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride); break;
            case 2: ReduceColor2x2<2>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceBgr2x2(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }
    }
#endif
}

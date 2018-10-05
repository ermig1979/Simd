/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

namespace Simd
{
    namespace Base
    {
        template <size_t channelCount> void ReduceColor2x2(const uint8_t * src0, const uint8_t * src1, uint8_t * dst);

        template <> void ReduceColor2x2<1>(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            dst[0] = Average(src0[0], src0[1], src1[0], src1[1]);
        }

        template <> void ReduceColor2x2<2>(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            dst[0] = Average(src0[0], src0[2], src1[0], src1[2]);
            dst[1] = Average(src0[1], src0[3], src1[1], src1[3]);
        }

        template <> void ReduceColor2x2<3>(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            dst[0] = Average(src0[0], src0[3], src1[0], src1[3]);
            dst[1] = Average(src0[1], src0[4], src1[1], src1[4]);
            dst[2] = Average(src0[2], src0[5], src1[2], src1[5]);
        }


        template <> void ReduceColor2x2<4>(const uint8_t * src0, const uint8_t * src1, uint8_t * dst)
        {
            dst[0] = Average(src0[0], src0[4], src1[0], src1[4]);
            dst[1] = Average(src0[1], src0[5], src1[1], src1[5]);
            dst[2] = Average(src0[2], src0[6], src1[2], src1[6]);
            dst[3] = Average(src0[3], src0[7], src1[3], src1[7]);
        }

        template <size_t channelCount> void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride, uint8_t * dst, size_t dstStride)
        {
            size_t evenWidth = AlignLo(srcWidth, 2);
            size_t evenSize = evenWidth * channelCount;
            size_t srcStep = 2 * channelCount;
            for (size_t srcRow = 0; srcRow < srcHeight; srcRow += 2)
            {
                const uint8_t * s0 = src;
                const uint8_t * s1 = (srcRow == srcHeight - 1 ? src : src + srcStride);
                const uint8_t * end = src + evenSize;
                uint8_t * d = dst;
                for (; s0 < end; s0 += srcStep, s1 += srcStep, d += channelCount)
                    ReduceColor2x2<channelCount>(s0, s1, d);
                if (evenWidth != srcWidth)
                {
                    for(size_t c = 0; c < channelCount; ++c)
                        d[c] = Average(s0[c], s1[c]);
                }
                src += 2 * srcStride;
                dst += dstStride;
            }
        }

        void ReduceColor2x2(const uint8_t * src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t * dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert((srcWidth + 1) / 2 == dstWidth && (srcHeight + 1) / 2 == dstHeight);

            switch (channelCount)
            {
            case 1: ReduceColor2x2<1>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 2: ReduceColor2x2<2>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 3: ReduceColor2x2<3>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            case 4: ReduceColor2x2<4>(src, srcWidth, srcHeight, srcStride, dst, dstStride); break;
            default: assert(0);
            }
        }
    }
}

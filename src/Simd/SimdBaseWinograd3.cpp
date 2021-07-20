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
#include "Simd/SimdWinograd.h"

namespace Simd
{
#if defined(SIMD_SYNET_ENABLE)
    namespace Base
    {
        void WinogradKernel3x3Block2x2SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel3x3Block2x2SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 9, dst += 1)
                    Base::WinogradKernel3x3Block2x2SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput1(const float src[16], float * dst, size_t stride)
        {
            dst[0 * stride] = (src[0] - src[8]) - (src[2] - src[10]);
            dst[1 * stride] = (src[1] - src[9]) + (src[2] - src[10]);
            dst[2 * stride] = (src[2] - src[10]) - (src[1] - src[9]);
            dst[3 * stride] = (src[1] - src[9]) - (src[3] - src[11]);
            dst[4 * stride] = (src[4] + src[8]) - (src[6] + src[10]);
            dst[5 * stride] = (src[5] + src[9]) + (src[6] + src[10]);
            dst[6 * stride] = (src[6] + src[10]) - (src[5] + src[9]);
            dst[7 * stride] = (src[5] + src[9]) - (src[7] + src[11]);
            dst[8 * stride] = (src[8] - src[4]) - (src[10] - src[6]);
            dst[9 * stride] = (src[9] - src[5]) + (src[10] - src[6]);
            dst[10 * stride] = (src[10] - src[6]) - (src[9] - src[5]);
            dst[11 * stride] = (src[9] - src[5]) - (src[11] - src[7]);
            dst[12 * stride] = (src[4] - src[12]) - (src[6] - src[14]);
            dst[13 * stride] = (src[5] - src[13]) + (src[6] - src[14]);
            dst[14 * stride] = (src[6] - src[14]) - (src[5] - src[13]);
            dst[15 * stride] = (src[5] - src[13]) - (src[7] - src[15]);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[16];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[0 * srcStride + 3];
            tmp[4] = src[1 * srcStride + 0];
            tmp[5] = src[1 * srcStride + 1];
            tmp[6] = src[1 * srcStride + 2];
            tmp[7] = src[1 * srcStride + 3];
            tmp[8] = src[2 * srcStride + 0];
            tmp[9] = src[2 * srcStride + 1];
            tmp[10] = src[2 * srcStride + 2];
            tmp[11] = src[2 * srcStride + 3];
            tmp[12] = src[3 * srcStride + 0];
            tmp[13] = src[3 * srcStride + 1];
            tmp[14] = src[3 * srcStride + 2];
            tmp[15] = src[3 * srcStride + 3];
            WinogradKernel3x3Block2x2SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput1n(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[16] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 4 + col] = src[row * srcStride + col];
            WinogradKernel3x3Block2x2SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput1t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[16];
                tmp[0] = src[0 * srcS + 0 * srcC];
                tmp[1] = src[0 * srcS + 1 * srcC];
                tmp[2] = src[0 * srcS + 2 * srcC];
                tmp[3] = src[0 * srcS + 3 * srcC];
                tmp[4] = src[1 * srcS + 0 * srcC];
                tmp[5] = src[1 * srcS + 1 * srcC];
                tmp[6] = src[1 * srcS + 2 * srcC];
                tmp[7] = src[1 * srcS + 3 * srcC];
                tmp[8] = src[2 * srcS + 0 * srcC];
                tmp[9] = src[2 * srcS + 1 * srcC];
                tmp[10] = src[2 * srcS + 2 * srcC];
                tmp[11] = src[2 * srcS + 3 * srcC];
                tmp[12] = src[3 * srcS + 0 * srcC];
                tmp[13] = src[3 * srcS + 1 * srcC];
                tmp[14] = src[3 * srcS + 2 * srcC];
                tmp[15] = src[3 * srcS + 3 * srcC];
                WinogradKernel3x3Block2x2SetInput1(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetInput1t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[16] = { 0 };
                for (size_t row = rowB; row < rowE; ++row)
                    for (size_t col = colB; col < colE; ++col)
                        tmp[row * 4 + col] = src[row * srcS + col * srcC];
                WinogradKernel3x3Block2x2SetInput1(tmp, dst, dstStride);
            }
        }

        void WinogradKernel3x3Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            size_t noseW = Simd::Min<size_t>(4, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(4, dstHeight + 1);
            size_t start = pad ? 2 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                src -= (srcWidth + 1)*(trans ? srcChannels : 1);
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            if (trans)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        WinogradKernel3x3Block2x2SetInput1t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 2)
                        WinogradKernel3x3Block2x2SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstHeightFull; row += 2)
                {
                    if (pad)
                        WinogradKernel3x3Block2x2SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 4, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 2)
                        WinogradKernel3x3Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 4, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstHeight)
                {
                    if (pad)
                        WinogradKernel3x3Block2x2SetInput1t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 2)
                        WinogradKernel3x3Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 4, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    size_t row = 0, col = 0;
                    if (pad)
                    {
                        if (pad)
                            WinogradKernel3x3Block2x2SetInput1n(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 2)
                            WinogradKernel3x3Block2x2SetInput1n(src + col, srcWidth, 1, noseH, 0, 4, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetInput1n(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                    }
                    for (row = start; row < dstHeightFull; row += 2)
                    {
                        if (pad)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth, srcWidth, 0, 4, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 2)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, 4, 0, tailW, dst++, dstStride);
                    }
                    if (row < dstHeight)
                    {
                        if (pad)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 2)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 4, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                    }
                    src += srcWidth * srcHeight;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput1(const float * src, size_t stride, float dst[4])
        {
            float s[16];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];
            s[6] = src[6 * stride];
            s[7] = src[7 * stride];
            s[8] = src[8 * stride];
            s[9] = src[9 * stride];
            s[10] = src[10 * stride];
            s[11] = src[11 * stride];
            s[12] = src[12 * stride];
            s[13] = src[13 * stride];
            s[14] = src[14 * stride];
            s[15] = src[15 * stride];

            float tmp[8];
            tmp[0] = s[0] + s[1] + s[2];
            tmp[1] = s[1] - s[2] - s[3];
            tmp[2] = s[4] + s[5] + s[6];
            tmp[3] = s[5] - s[6] - s[7];
            tmp[4] = s[8] + s[9] + s[10];
            tmp[5] = s[9] - s[10] - s[11];
            tmp[6] = s[12] + s[13] + s[14];
            tmp[7] = s[13] - s[14] - s[15];

            dst[0] = tmp[0] + tmp[2] + tmp[4];
            dst[1] = tmp[1] + tmp[3] + tmp[5];
            dst[2] = tmp[2] - tmp[4] - tmp[6];
            dst[3] = tmp[3] - tmp[5] - tmp[7];
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[4];
            WinogradKernel3x3Block2x2SetOutput1(src, srcStride, tmp);
            dst[0 * dstStride + 0] = tmp[0];
            dst[0 * dstStride + 1] = tmp[1];
            dst[1 * dstStride + 0] = tmp[2];
            dst[1 * dstStride + 1] = tmp[3];
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[4];
            WinogradKernel3x3Block2x2SetOutput1(src, srcStride, tmp);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 2 + col];
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel3x3Block2x2SetOutput1(src, srcStride, tmp);
                dst[0 * dstS + 0 * dstC] = tmp[0];
                dst[0 * dstS + 1 * dstC] = tmp[1];
                dst[1 * dstS + 0 * dstC] = tmp[2];
                dst[1 * dstS + 1 * dstC] = tmp[3];
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block2x2SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel3x3Block2x2SetOutput1(src, srcStride, tmp);
                for (size_t row = 0; row < rowE; ++row)
                    for (size_t col = 0; col < colE; ++col)
                        dst[row*dstS + col*dstC] = tmp[row * 2 + col];
            }
        }

        void WinogradKernel3x3Block2x2SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        WinogradKernel3x3Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        WinogradKernel3x3Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    size_t row, col;
                    for (row = 0; row < dstHeightFull; row += 2)
                    {
                        for (col = 0; col < dstWidthFull; col += 2)
                            WinogradKernel3x3Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, 2, dstWidth - col);
                    }
                    if (row < dstHeight)
                    {
                        for (col = 0; col < dstWidthFull; col += 2)
                            WinogradKernel3x3Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 2);
                        if (col < dstWidth)
                            WinogradKernel3x3Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                    }
                    dst += dstHeight * dstWidth;
                }
            }
        }

        //-----------------------------------------------------------------------

        void WinogradKernel3x3Block3x3SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel3x3Block3x3SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 9, dst += 1)
                    Base::WinogradKernel3x3Block3x3SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput1(const float src[25], float * dst, size_t stride)
        {
            float tmp[25];
            tmp[0] = 2 * src[0] - src[5] - 2 * src[10] + src[15];
            tmp[1] = 2 * src[1] - src[6] - 2 * src[11] + src[16];
            tmp[2] = 2 * src[2] - src[7] - 2 * src[12] + src[17];
            tmp[3] = 2 * src[3] - src[8] - 2 * src[13] + src[18];
            tmp[4] = 2 * src[4] - src[9] - 2 * src[14] + src[19];
            tmp[5] = -2 * src[5] - src[10] + src[15];
            tmp[6] = -2 * src[6] - src[11] + src[16];
            tmp[7] = -2 * src[7] - src[12] + src[17];
            tmp[8] = -2 * src[8] - src[13] + src[18];
            tmp[9] = -2 * src[9] - src[14] + src[19];
            tmp[10] = 2 * src[5] - 3 * src[10] + src[15];
            tmp[11] = 2 * src[6] - 3 * src[11] + src[16];
            tmp[12] = 2 * src[7] - 3 * src[12] + src[17];
            tmp[13] = 2 * src[8] - 3 * src[13] + src[18];
            tmp[14] = 2 * src[9] - 3 * src[14] + src[19];
            tmp[15] = -src[5] + src[15];
            tmp[16] = -src[6] + src[16];
            tmp[17] = -src[7] + src[17];
            tmp[18] = -src[8] + src[18];
            tmp[19] = -src[9] + src[19];
            tmp[20] = 2 * src[5] - src[10] - 2 * src[15] + src[20];
            tmp[21] = 2 * src[6] - src[11] - 2 * src[16] + src[21];
            tmp[22] = 2 * src[7] - src[12] - 2 * src[17] + src[22];
            tmp[23] = 2 * src[8] - src[13] - 2 * src[18] + src[23];
            tmp[24] = 2 * src[9] - src[14] - 2 * src[19] + src[24];

            dst[0 * stride] = 2 * tmp[0] - tmp[1] - 2 * tmp[2] + tmp[3];
            dst[1 * stride] = -2 * tmp[1] - tmp[2] + tmp[3];
            dst[2 * stride] = 2 * tmp[1] - 3 * tmp[2] + tmp[3];
            dst[3 * stride] = - tmp[1] + tmp[3];
            dst[4 * stride] = 2 * tmp[1] - tmp[2] - 2 * tmp[3] + tmp[4];
            dst[5 * stride] = 2 * tmp[5] - tmp[6] - 2 * tmp[7] + tmp[8];
            dst[6 * stride] = -2 * tmp[6] - tmp[7] + tmp[8];
            dst[7 * stride] = 2 * tmp[6] - 3 * tmp[7] + tmp[8];
            dst[8 * stride] = -tmp[6] + tmp[8];
            dst[9 * stride] = 2 * tmp[6] - tmp[7] - 2 * tmp[8] + tmp[9];
            dst[10 * stride] = 2 * tmp[10] - tmp[11] - 2 * tmp[12] + tmp[13];
            dst[11 * stride] = -2 * tmp[11] - tmp[12] + tmp[13];
            dst[12 * stride] = 2 * tmp[11] - 3 * tmp[12] + tmp[13];
            dst[13 * stride] = -tmp[11] + tmp[13];
            dst[14 * stride] = 2 * tmp[11] - tmp[12] - 2 * tmp[13] + tmp[14];
            dst[15 * stride] = 2 * tmp[15] - tmp[16] - 2 * tmp[17] + tmp[18];
            dst[16 * stride] = -2 * tmp[16] - tmp[17] + tmp[18];
            dst[17 * stride] = 2 * tmp[16] - 3 * tmp[17] + tmp[18];
            dst[18 * stride] = -tmp[16] + tmp[18];
            dst[19 * stride] = 2 * tmp[16] - tmp[17] - 2 * tmp[18] + tmp[19];
            dst[20 * stride] = 2 * tmp[20] - tmp[21] - 2 * tmp[22] + tmp[23];
            dst[21 * stride] = -2 * tmp[21] - tmp[22] + tmp[23];
            dst[22 * stride] = 2 * tmp[21] - 3 * tmp[22] + tmp[23];
            dst[23 * stride] = -tmp[21] + tmp[23];
            dst[24 * stride] = 2 * tmp[21] - tmp[22] - 2 * tmp[23] + tmp[24];
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[25];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[0 * srcStride + 3];
            tmp[4] = src[0 * srcStride + 4];
            tmp[5] = src[1 * srcStride + 0];
            tmp[6] = src[1 * srcStride + 1];
            tmp[7] = src[1 * srcStride + 2];
            tmp[8] = src[1 * srcStride + 3];
            tmp[9] = src[1 * srcStride + 4];
            tmp[10] = src[2 * srcStride + 0];
            tmp[11] = src[2 * srcStride + 1];
            tmp[12] = src[2 * srcStride + 2];
            tmp[13] = src[2 * srcStride + 3];
            tmp[14] = src[2 * srcStride + 4];
            tmp[15] = src[3 * srcStride + 0];
            tmp[16] = src[3 * srcStride + 1];
            tmp[17] = src[3 * srcStride + 2];
            tmp[18] = src[3 * srcStride + 3];
            tmp[19] = src[3 * srcStride + 4];
            tmp[20] = src[4 * srcStride + 0];
            tmp[21] = src[4 * srcStride + 1];
            tmp[22] = src[4 * srcStride + 2];
            tmp[23] = src[4 * srcStride + 3];
            tmp[24] = src[4 * srcStride + 4];
            WinogradKernel3x3Block3x3SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput1n(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[5 * 5] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 5 + col] = src[row * srcStride + col];
            WinogradKernel3x3Block3x3SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput1t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[25];
                tmp[0] = src[0 * srcS + 0 * srcC];
                tmp[1] = src[0 * srcS + 1 * srcC];
                tmp[2] = src[0 * srcS + 2 * srcC];
                tmp[3] = src[0 * srcS + 3 * srcC];
                tmp[4] = src[0 * srcS + 4 * srcC];
                tmp[5] = src[1 * srcS + 0 * srcC];
                tmp[6] = src[1 * srcS + 1 * srcC];
                tmp[7] = src[1 * srcS + 2 * srcC];
                tmp[8] = src[1 * srcS + 3 * srcC];
                tmp[9] = src[1 * srcS + 4 * srcC];
                tmp[10] = src[2 * srcS + 0 * srcC];
                tmp[11] = src[2 * srcS + 1 * srcC];
                tmp[12] = src[2 * srcS + 2 * srcC];
                tmp[13] = src[2 * srcS + 3 * srcC];
                tmp[14] = src[2 * srcS + 4 * srcC];
                tmp[15] = src[3 * srcS + 0 * srcC];
                tmp[16] = src[3 * srcS + 1 * srcC];
                tmp[17] = src[3 * srcS + 2 * srcC];
                tmp[18] = src[3 * srcS + 3 * srcC];
                tmp[19] = src[3 * srcS + 4 * srcC];
                tmp[20] = src[4 * srcS + 0 * srcC];
                tmp[21] = src[4 * srcS + 1 * srcC];
                tmp[22] = src[4 * srcS + 2 * srcC];
                tmp[23] = src[4 * srcS + 3 * srcC];
                tmp[24] = src[4 * srcS + 4 * srcC];
                WinogradKernel3x3Block3x3SetInput1(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetInput1t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[25] = { 0 };
                for (size_t row = rowB; row < rowE; ++row)
                    for (size_t col = colB; col < colE; ++col)
                        tmp[row * 5 + col] = src[row * srcS + col * srcC];
                WinogradKernel3x3Block3x3SetInput1(tmp, dst, dstStride);
            }
        }

        void WinogradKernel3x3Block3x3SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padY == padH && padY == padW && (padY == 0 || padY == 1));
            SimdBool pad = padY > 0 ? SimdTrue : SimdFalse;
            size_t dstHeight = pad ? srcHeight : srcHeight - 2;
            size_t dstWidth = pad ? srcWidth : srcWidth - 2;
            size_t dstHeightFull = dstHeight / 3 * 3;
            size_t dstWidthFull = dstWidth / 3 * 3;
            size_t noseW = Simd::Min<size_t>(5, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(5, dstHeight + 1);
            size_t start = pad ? 3 : 0;
            if (pad)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 3;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 3;
                src -= (srcWidth + 1)*(trans ? srcChannels : 1);
            }
            size_t tailW = dstWidth - dstWidthFull + (pad ? 1 : 2);
            size_t tailH = dstHeight - dstHeightFull + (pad ? 1 : 2);
            if (trans)
            {
                size_t row = 0, col = 0;
                if (pad)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput1t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 3)
                        WinogradKernel3x3Block3x3SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = start; row < dstHeightFull; row += 3)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 5, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 3)
                        WinogradKernel3x3Block3x3SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 5, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstHeight)
                {
                    if (pad)
                        WinogradKernel3x3Block3x3SetInput1t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = start; col < dstWidthFull; col += 3)
                        WinogradKernel3x3Block3x3SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    size_t row = 0, col = 0;
                    if (pad)
                    {
                        if (pad)
                            WinogradKernel3x3Block3x3SetInput1n(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 3)
                            WinogradKernel3x3Block3x3SetInput1n(src + col, srcWidth, 1, noseH, 0, 5, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block3x3SetInput1n(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                    }
                    for (row = start; row < dstHeightFull; row += 3)
                    {
                        if (pad)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth, srcWidth, 0, 5, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 3)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth + col, srcWidth, 0, 5, 0, tailW, dst++, dstStride);
                    }
                    if (row < dstHeight)
                    {
                        if (pad)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                        for (col = start; col < dstWidthFull; col += 3)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 5, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel3x3Block3x3SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                    }
                    src += srcWidth * srcHeight;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput1(const float * src, size_t stride, float dst[9])
        {
            float s[25];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];
            s[6] = src[6 * stride];
            s[7] = src[7 * stride];
            s[8] = src[8 * stride];
            s[9] = src[9 * stride];
            s[10] = src[10 * stride];
            s[11] = src[11 * stride];
            s[12] = src[12 * stride];
            s[13] = src[13 * stride];
            s[14] = src[14 * stride];
            s[15] = src[15 * stride];
            s[16] = src[16 * stride];
            s[17] = src[17 * stride];
            s[18] = src[18 * stride];
            s[19] = src[19 * stride];
            s[20] = src[20 * stride];
            s[21] = src[21 * stride];
            s[22] = src[22 * stride];
            s[23] = src[23 * stride];
            s[24] = src[24 * stride];

            float t[15];
            t[0] = s[0] + s[5] + s[10] + s[15];
            t[1] = s[1] + s[6] + s[11] + s[16];
            t[2] = s[2] + s[7] + s[12] + s[17];
            t[3] = s[3] + s[8] + s[13] + s[18];
            t[4] = s[4] + s[9] + s[14] + s[19];
            t[5] = s[5] - s[10] + 2 * s[15];
            t[6] = s[6] - s[11] + 2 * s[16];
            t[7] = s[7] - s[12] + 2 * s[17];
            t[8] = s[8] - s[13] + 2 * s[18];
            t[9] = s[9] - s[14] + 2 * s[19];
            t[10] = s[5] + s[10] + 4 * s[15] + s[20];
            t[11] = s[6] + s[11] + 4 * s[16] + s[21];
            t[12] = s[7] + s[12] + 4 * s[17] + s[22];
            t[13] = s[8] + s[13] + 4 * s[18] + s[23];
            t[14] = s[9] + s[14] + 4 * s[19] + s[24];

            dst[0] = t[0] + t[1] + t[2] + t[3];
            dst[1] = t[1] - t[2] + 2 * t[3];
            dst[2] = t[1] + t[2] + 4 * t[3] + t[4];
            dst[3] = t[5] + t[6] + t[7] + t[8];
            dst[4] = t[6] - t[7] + 2 * t[8];
            dst[5] = t[6] + t[7] + 4 * t[8] + t[9];
            dst[6] = t[10] + t[11] + t[12] + t[13];
            dst[7] = t[11] - t[12] + 2 * t[13];
            dst[8] = t[11] + t[12] + 4 * t[13] + t[14];
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[9];
            WinogradKernel3x3Block3x3SetOutput1(src, srcStride, tmp);
            dst[0 * dstStride + 0] = tmp[0];
            dst[0 * dstStride + 1] = tmp[1];
            dst[0 * dstStride + 2] = tmp[2];
            dst[1 * dstStride + 0] = tmp[3];
            dst[1 * dstStride + 1] = tmp[4];
            dst[1 * dstStride + 2] = tmp[5];
            dst[2 * dstStride + 0] = tmp[6];
            dst[2 * dstStride + 1] = tmp[7];
            dst[2 * dstStride + 2] = tmp[8];
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[9];
            WinogradKernel3x3Block3x3SetOutput1(src, srcStride, tmp);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 3 + col];
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[9];
                WinogradKernel3x3Block3x3SetOutput1(src, srcStride, tmp);
                dst[0 * dstS + 0 * dstC] = tmp[0];
                dst[0 * dstS + 1 * dstC] = tmp[1];
                dst[0 * dstS + 2 * dstC] = tmp[2];
                dst[1 * dstS + 0 * dstC] = tmp[3];
                dst[1 * dstS + 1 * dstC] = tmp[4];
                dst[1 * dstS + 2 * dstC] = tmp[5];
                dst[2 * dstS + 0 * dstC] = tmp[6];
                dst[2 * dstS + 1 * dstC] = tmp[7];
                dst[2 * dstS + 2 * dstC] = tmp[8];
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block3x3SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[9];
                WinogradKernel3x3Block3x3SetOutput1(src, srcStride, tmp);
                for (size_t row = 0; row < rowE; ++row)
                    for (size_t col = 0; col < colE; ++col)
                        dst[row*dstS + col * dstC] = tmp[row * 3 + col];
            }
        }

        void WinogradKernel3x3Block3x3SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstHeightFull = dstHeight / 3 * 3;
            size_t dstWidthFull = dstWidth / 3 * 3;
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 3)
                {
                    for (col = 0; col < dstWidthFull; col += 3)
                        WinogradKernel3x3Block3x3SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 3, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 3)
                        WinogradKernel3x3Block3x3SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 3), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block3x3SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    size_t row, col;
                    for (row = 0; row < dstHeightFull; row += 3)
                    {
                        for (col = 0; col < dstWidthFull; col += 3)
                            WinogradKernel3x3Block3x3SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel3x3Block3x3SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, 3, dstWidth - col);
                    }
                    if (row < dstHeight)
                    {
                        for (col = 0; col < dstWidthFull; col += 3)
                            WinogradKernel3x3Block3x3SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 3);
                        if (col < dstWidth)
                            WinogradKernel3x3Block3x3SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                    }
                    dst += dstHeight * dstWidth;
                }
            }
        }

        //-----------------------------------------------------------------------

        void WinogradKernel3x3Block4x4SetFilter(const float * src, size_t size, float * dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel3x3Block4x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 9, dst += 1)
                    Base::WinogradKernel3x3Block4x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput1(const float src[36], float * dst, size_t stride)
        {
            float tmp[36];
            tmp[0] = 4 * src[0] - 5 * src[12] + src[24];
            tmp[1] = 4 * src[1] - 5 * src[13] + src[25];
            tmp[2] = 4 * src[2] - 5 * src[14] + src[26];
            tmp[3] = 4 * src[3] - 5 * src[15] + src[27];
            tmp[4] = 4 * src[4] - 5 * src[16] + src[28];
            tmp[5] = 4 * src[5] - 5 * src[17] + src[29];
            tmp[6] = -4 * src[6] - 4 * src[12] + src[18] + src[24];
            tmp[7] = -4 * src[7] - 4 * src[13] + src[19] + src[25];
            tmp[8] = -4 * src[8] - 4 * src[14] + src[20] + src[26];
            tmp[9] = -4 * src[9] - 4 * src[15] + src[21] + src[27];
            tmp[10] = -4 * src[10] - 4 * src[16] + src[22] + src[28];
            tmp[11] = -4 * src[11] - 4 * src[17] + src[23] + src[29];
            tmp[12] = 4 * src[6] - 4 * src[12] - src[18] + src[24];
            tmp[13] = 4 * src[7] - 4 * src[13] - src[19] + src[25];
            tmp[14] = 4 * src[8] - 4 * src[14] - src[20] + src[26];
            tmp[15] = 4 * src[9] - 4 * src[15] - src[21] + src[27];
            tmp[16] = 4 * src[10] - 4 * src[16] - src[22] + src[28];
            tmp[17] = 4 * src[11] - 4 * src[17] - src[23] + src[29];
            tmp[18] = -2 * src[6] - src[12] + 2 * src[18] + src[24];
            tmp[19] = -2 * src[7] - src[13] + 2 * src[19] + src[25];
            tmp[20] = -2 * src[8] - src[14] + 2 * src[20] + src[26];
            tmp[21] = -2 * src[9] - src[15] + 2 * src[21] + src[27];
            tmp[22] = -2 * src[10] - src[16] + 2 * src[22] + src[28];
            tmp[23] = -2 * src[11] - src[17] + 2 * src[23] + src[29];
            tmp[24] = 2 * src[6] - src[12] - 2 * src[18] + src[24];
            tmp[25] = 2 * src[7] - src[13] - 2 * src[19] + src[25];
            tmp[26] = 2 * src[8] - src[14] - 2 * src[20] + src[26];
            tmp[27] = 2 * src[9] - src[15] - 2 * src[21] + src[27];
            tmp[28] = 2 * src[10] - src[16] - 2 * src[22] + src[28];
            tmp[29] = 2 * src[11] - src[17] - 2 * src[23] + src[29];
            tmp[30] = 4 * src[6] - 5 * src[18] + src[30];
            tmp[31] = 4 * src[7] - 5 * src[19] + src[31];
            tmp[32] = 4 * src[8] - 5 * src[20] + src[32];
            tmp[33] = 4 * src[9] - 5 * src[21] + src[33];
            tmp[34] = 4 * src[10] - 5 * src[22] + src[34];
            tmp[35] = 4 * src[11] - 5 * src[23] + src[35];

            dst[0 * stride] = tmp[0] * 4 - tmp[2] * 5 + tmp[4];
            dst[1 * stride] = -tmp[1] * 4 - tmp[2] * 4 + tmp[3] + tmp[4];
            dst[2 * stride] = tmp[1] * 4 - tmp[2] * 4 - tmp[3] + tmp[4];
            dst[3 * stride] = -tmp[1] * 2 - tmp[2] + tmp[3] * 2 + tmp[4];
            dst[4 * stride] = tmp[1] * 2 - tmp[2] - tmp[3] * 2 + tmp[4];
            dst[5 * stride] = tmp[1] * 4 - tmp[3] * 5 + tmp[5];
            dst[6 * stride] = tmp[6] * 4 - tmp[8] * 5 + tmp[10];
            dst[7 * stride] = -tmp[7] * 4 - tmp[8] * 4 + tmp[9] + tmp[10];
            dst[8 * stride] = tmp[7] * 4 - tmp[8] * 4 - tmp[9] + tmp[10];
            dst[9 * stride] = -tmp[7] * 2 - tmp[8] + tmp[9] * 2 + tmp[10];
            dst[10 * stride] = tmp[7] * 2 - tmp[8] - tmp[9] * 2 + tmp[10];
            dst[11 * stride] = tmp[7] * 4 - tmp[9] * 5 + tmp[11];
            dst[12 * stride] = tmp[12] * 4 - tmp[14] * 5 + tmp[16];
            dst[13 * stride] = -tmp[13] * 4 - tmp[14] * 4 + tmp[15] + tmp[16];
            dst[14 * stride] = tmp[13] * 4 - tmp[14] * 4 - tmp[15] + tmp[16];
            dst[15 * stride] = -tmp[13] * 2 - tmp[14] + tmp[15] * 2 + tmp[16];
            dst[16 * stride] = tmp[13] * 2 - tmp[14] - tmp[15] * 2 + tmp[16];
            dst[17 * stride] = tmp[13] * 4 - tmp[15] * 5 + tmp[17];
            dst[18 * stride] = tmp[18] * 4 - tmp[20] * 5 + tmp[22];
            dst[19 * stride] = -tmp[19] * 4 - tmp[20] * 4 + tmp[21] + tmp[22];
            dst[20 * stride] = tmp[19] * 4 - tmp[20] * 4 - tmp[21] + tmp[22];
            dst[21 * stride] = -tmp[19] * 2 - tmp[20] + tmp[21] * 2 + tmp[22];
            dst[22 * stride] = tmp[19] * 2 - tmp[20] - tmp[21] * 2 + tmp[22];
            dst[23 * stride] = tmp[19] * 4 - tmp[21] * 5 + tmp[23];
            dst[24 * stride] = tmp[24] * 4 - tmp[26] * 5 + tmp[28];
            dst[25 * stride] = -tmp[25] * 4 - tmp[26] * 4 + tmp[27] + tmp[28];
            dst[26 * stride] = tmp[25] * 4 - tmp[26] * 4 - tmp[27] + tmp[28];
            dst[27 * stride] = -tmp[25] * 2 - tmp[26] + tmp[27] * 2 + tmp[28];
            dst[28 * stride] = tmp[25] * 2 - tmp[26] - tmp[27] * 2 + tmp[28];
            dst[29 * stride] = tmp[25] * 4 - tmp[27] * 5 + tmp[29];
            dst[30 * stride] = tmp[30] * 4 - tmp[32] * 5 + tmp[34];
            dst[31 * stride] = -tmp[31] * 4 - tmp[32] * 4 + tmp[33] + tmp[34];
            dst[32 * stride] = tmp[31] * 4 - tmp[32] * 4 - tmp[33] + tmp[34];
            dst[33 * stride] = -tmp[31] * 2 - tmp[32] + tmp[33] * 2 + tmp[34];
            dst[34 * stride] = tmp[31] * 2 - tmp[32] - tmp[33] * 2 + tmp[34];
            dst[35 * stride] = tmp[31] * 4 - tmp[33] * 5 + tmp[35];
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[36];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[0 * srcStride + 3];
            tmp[4] = src[0 * srcStride + 4];
            tmp[5] = src[0 * srcStride + 5];
            tmp[6] = src[1 * srcStride + 0];
            tmp[7] = src[1 * srcStride + 1];
            tmp[8] = src[1 * srcStride + 2];
            tmp[9] = src[1 * srcStride + 3];
            tmp[10] = src[1 * srcStride + 4];
            tmp[11] = src[1 * srcStride + 5];
            tmp[12] = src[2 * srcStride + 0];
            tmp[13] = src[2 * srcStride + 1];
            tmp[14] = src[2 * srcStride + 2];
            tmp[15] = src[2 * srcStride + 3];
            tmp[16] = src[2 * srcStride + 4];
            tmp[17] = src[2 * srcStride + 5];
            tmp[18] = src[3 * srcStride + 0];
            tmp[19] = src[3 * srcStride + 1];
            tmp[20] = src[3 * srcStride + 2];
            tmp[21] = src[3 * srcStride + 3];
            tmp[22] = src[3 * srcStride + 4];
            tmp[23] = src[3 * srcStride + 5];
            tmp[24] = src[4 * srcStride + 0];
            tmp[25] = src[4 * srcStride + 1];
            tmp[26] = src[4 * srcStride + 2];
            tmp[27] = src[4 * srcStride + 3];
            tmp[28] = src[4 * srcStride + 4];
            tmp[29] = src[4 * srcStride + 5];
            tmp[30] = src[5 * srcStride + 0];
            tmp[31] = src[5 * srcStride + 1];
            tmp[32] = src[5 * srcStride + 2];
            tmp[33] = src[5 * srcStride + 3];
            tmp[34] = src[5 * srcStride + 4];
            tmp[35] = src[5 * srcStride + 5];
            WinogradKernel3x3Block4x4SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput1n(const float * src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            float tmp[6 * 6] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 6 + col] = src[row * srcStride + col];
            WinogradKernel3x3Block4x4SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput1t(const float * src, size_t srcW, size_t srcC, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[36];
                tmp[0] = src[0 * srcS + 0 * srcC];
                tmp[1] = src[0 * srcS + 1 * srcC];
                tmp[2] = src[0 * srcS + 2 * srcC];
                tmp[3] = src[0 * srcS + 3 * srcC];
                tmp[4] = src[0 * srcS + 4 * srcC];
                tmp[5] = src[0 * srcS + 5 * srcC];
                tmp[6] = src[1 * srcS + 0 * srcC];
                tmp[7] = src[1 * srcS + 1 * srcC];
                tmp[8] = src[1 * srcS + 2 * srcC];
                tmp[9] = src[1 * srcS + 3 * srcC];
                tmp[10] = src[1 * srcS + 4 * srcC];
                tmp[11] = src[1 * srcS + 5 * srcC];
                tmp[12] = src[2 * srcS + 0 * srcC];
                tmp[13] = src[2 * srcS + 1 * srcC];
                tmp[14] = src[2 * srcS + 2 * srcC];
                tmp[15] = src[2 * srcS + 3 * srcC];
                tmp[16] = src[2 * srcS + 4 * srcC];
                tmp[17] = src[2 * srcS + 5 * srcC];
                tmp[18] = src[3 * srcS + 0 * srcC];
                tmp[19] = src[3 * srcS + 1 * srcC];
                tmp[20] = src[3 * srcS + 2 * srcC];
                tmp[21] = src[3 * srcS + 3 * srcC];
                tmp[22] = src[3 * srcS + 4 * srcC];
                tmp[23] = src[3 * srcS + 5 * srcC];
                tmp[24] = src[4 * srcS + 0 * srcC];
                tmp[25] = src[4 * srcS + 1 * srcC];
                tmp[26] = src[4 * srcS + 2 * srcC];
                tmp[27] = src[4 * srcS + 3 * srcC];
                tmp[28] = src[4 * srcS + 4 * srcC];
                tmp[29] = src[4 * srcS + 5 * srcC];
                tmp[30] = src[5 * srcS + 0 * srcC];
                tmp[31] = src[5 * srcS + 1 * srcC];
                tmp[32] = src[5 * srcS + 2 * srcC];
                tmp[33] = src[5 * srcS + 3 * srcC];
                tmp[34] = src[5 * srcS + 4 * srcC];
                tmp[35] = src[5 * srcS + 5 * srcC];
                WinogradKernel3x3Block4x4SetInput1(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetInput1t(const float * src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float * dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[36] = { 0 };
                for (size_t row = rowB; row < rowE; ++row)
                    for (size_t col = colB; col < colE; ++col)
                        tmp[row * 6 + col] = src[row * srcS + col * srcC];
                WinogradKernel3x3Block4x4SetInput1(tmp, dst, dstStride);
            }
        }

        void WinogradKernel3x3Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY + padH <= 2 && padX + padW <= 2);
            size_t dstH = srcHeight - 2 + padY + padH;
            size_t dstW = srcWidth - 2 + padX + padW;
            size_t dstH4 = dstH / 4 * 4;
            size_t dstW4 = dstW / 4 * 4;
            size_t noseW = Simd::Min<size_t>(6, srcWidth + padX);
            size_t noseH = Simd::Min<size_t>(6, srcHeight + padY);
            size_t startY = padY ? 4 : 0;
            size_t startX = padX ? 4 : 0;
            if (padH && dstH == dstH4)
                dstH4 -= 4;
            if(padY)
                src -= srcWidth * (trans ? srcChannels : 1);
            if (padW && dstW == dstW4)
                dstW4 -= 4;
            if(padX)
                src -= 1 * (trans ? srcChannels : 1);
            size_t tailW = dstW - dstW4 + (padW ? 1 : 2);
            size_t tailH = dstH - dstH4 + (padH ? 1 : 2);
            if (trans)
            {
                size_t row = 0, col = 0;
                if (padY)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput1t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 6, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = startY; row < dstH4; row += 4)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 6, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 6, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstH)
                {
                    if (padX)
                        WinogradKernel3x3Block4x4SetInput1t(src + row * srcWidth* srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel3x3Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 6, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel3x3Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    size_t row = 0, col = 0;
                    if (padY)
                    {
                        if (padX)
                            WinogradKernel3x3Block4x4SetInput1n(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstW4; col += 4)
                            WinogradKernel3x3Block4x4SetInput1n(src + col, srcWidth, 1, noseH, 0, 6, dst++, dstStride);
                        if (col < dstW)
                            WinogradKernel3x3Block4x4SetInput1n(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                    }
                    for (row = startY; row < dstH4; row += 4)
                    {
                        if (padX)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth, srcWidth, 0, 6, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstW4; col += 4)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                        if (col < dstW)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, 6, 0, tailW, dst++, dstStride);
                    }
                    if (row < dstH)
                    {
                        if (padX)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstW4; col += 4)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 6, dst++, dstStride);
                        if (col < dstW)
                            WinogradKernel3x3Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                    }
                    src += srcWidth * srcHeight;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput1(const float * src, size_t stride, float dst[16])
        {
            float s[36];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];
            s[6] = src[6 * stride];
            s[7] = src[7 * stride];
            s[8] = src[8 * stride];
            s[9] = src[9 * stride];
            s[10] = src[10 * stride];
            s[11] = src[11 * stride];
            s[12] = src[12 * stride];
            s[13] = src[13 * stride];
            s[14] = src[14 * stride];
            s[15] = src[15 * stride];
            s[16] = src[16 * stride];
            s[17] = src[17 * stride];
            s[18] = src[18 * stride];
            s[19] = src[19 * stride];
            s[20] = src[20 * stride];
            s[21] = src[21 * stride];
            s[22] = src[22 * stride];
            s[23] = src[23 * stride];
            s[24] = src[24 * stride];
            s[25] = src[25 * stride];
            s[26] = src[26 * stride];
            s[27] = src[27 * stride];
            s[28] = src[28 * stride];
            s[29] = src[29 * stride];
            s[30] = src[30 * stride];
            s[31] = src[31 * stride];
            s[32] = src[32 * stride];
            s[33] = src[33 * stride];
            s[34] = src[34 * stride];
            s[35] = src[35 * stride];

            float t[24];
            t[0] = s[0] + s[6] + s[12] + s[18] + s[24];
            t[1] = s[1] + s[7] + s[13] + s[19] + s[25];
            t[2] = s[2] + s[8] + s[14] + s[20] + s[26];
            t[3] = s[3] + s[9] + s[15] + s[21] + s[27];
            t[4] = s[4] + s[10] + s[16] + s[22] + s[28];
            t[5] = s[5] + s[11] + s[17] + s[23] + s[29];
            t[6] = s[6] - s[12] + 2 * s[18] - 2 * s[24];
            t[7] = s[7] - s[13] + 2 * s[19] - 2 * s[25];
            t[8] = s[8] - s[14] + 2 * s[20] - 2 * s[26];
            t[9] = s[9] - s[15] + 2 * s[21] - 2 * s[27];
            t[10] = s[10] - s[16] + 2 * s[22] - 2 * s[28];
            t[11] = s[11] - s[17] + 2 * s[23] - 2 * s[29];
            t[12] = s[6] + s[12] + 4 * s[18] + 4 * s[24];
            t[13] = s[7] + s[13] + 4 * s[19] + 4 * s[25];
            t[14] = s[8] + s[14] + 4 * s[20] + 4 * s[26];
            t[15] = s[9] + s[15] + 4 * s[21] + 4 * s[27];
            t[16] = s[10] + s[16] + 4 * s[22] + 4 * s[28];
            t[17] = s[11] + s[17] + 4 * s[23] + 4 * s[29];
            t[18] = s[6] - s[12] + 8 * s[18] - 8 * s[24] + s[30];
            t[19] = s[7] - s[13] + 8 * s[19] - 8 * s[25] + s[31];
            t[20] = s[8] - s[14] + 8 * s[20] - 8 * s[26] + s[32];
            t[21] = s[9] - s[15] + 8 * s[21] - 8 * s[27] + s[33];
            t[22] = s[10] - s[16] + 8 * s[22] - 8 * s[28] + s[34];
            t[23] = s[11] - s[17] + 8 * s[23] - 8 * s[29] + s[35];

            dst[0] = t[0] + t[1] + t[2] + t[3] + t[4];
            dst[1] = t[1] - t[2] + 2 * t[3] - 2 * t[4];
            dst[2] = t[1] + t[2] + 4 * t[3] + 4 * t[4];
            dst[3] = t[1] - t[2] + 8 * t[3] - 8 * t[4] + t[5];
            dst[4] = t[6] + t[7] + t[8] + t[9] + t[10];
            dst[5] = t[7] - t[8] + 2 * t[9] - 2 * t[10];
            dst[6] = t[7] + t[8] + 4 * t[9] + 4 * t[10];
            dst[7] = t[7] - t[8] + 8 * t[9] - 8 * t[10] + t[11];
            dst[8] = t[12] + t[13] + t[14] + t[15] + t[16];
            dst[9] = t[13] - t[14] + 2 * t[15] - 2 * t[16];
            dst[10] = t[13] + t[14] + 4 * t[15] + 4 * t[16];
            dst[11] = t[13] - t[14] + 8 * t[15] - 8 * t[16] + t[17];
            dst[12] = t[18] + t[19] + t[20] + t[21] + t[22];
            dst[13] = t[19] - t[20] + 2 * t[21] - 2 * t[22];
            dst[14] = t[19] + t[20] + 4 * t[21] + 4 * t[22];
            dst[15] = t[19] - t[20] + 8 * t[21] - 8 * t[22] + t[23];
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float tmp[16];
            WinogradKernel3x3Block4x4SetOutput1(src, srcStride, tmp);
            dst[0 * dstStride + 0] = tmp[0];
            dst[0 * dstStride + 1] = tmp[1];
            dst[0 * dstStride + 2] = tmp[2];
            dst[0 * dstStride + 3] = tmp[3];
            dst[1 * dstStride + 0] = tmp[4];
            dst[1 * dstStride + 1] = tmp[5];
            dst[1 * dstStride + 2] = tmp[6];
            dst[1 * dstStride + 3] = tmp[7];
            dst[2 * dstStride + 0] = tmp[8];
            dst[2 * dstStride + 1] = tmp[9];
            dst[2 * dstStride + 2] = tmp[10];
            dst[2 * dstStride + 3] = tmp[11];
            dst[3 * dstStride + 0] = tmp[12];
            dst[3 * dstStride + 1] = tmp[13];
            dst[3 * dstStride + 2] = tmp[14];
            dst[3 * dstStride + 3] = tmp[15];
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput1n(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[16];
            WinogradKernel3x3Block4x4SetOutput1(src, srcStride, tmp);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row*dstStride + col] = tmp[row * 4 + col];
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[16];
                WinogradKernel3x3Block4x4SetOutput1(src, srcStride, tmp);
                dst[0 * dstS + 0 * dstC] = tmp[0];
                dst[0 * dstS + 1 * dstC] = tmp[1];
                dst[0 * dstS + 2 * dstC] = tmp[2];
                dst[0 * dstS + 3 * dstC] = tmp[3];
                dst[1 * dstS + 0 * dstC] = tmp[4];
                dst[1 * dstS + 1 * dstC] = tmp[5];
                dst[1 * dstS + 2 * dstC] = tmp[6];
                dst[1 * dstS + 3 * dstC] = tmp[7];
                dst[2 * dstS + 0 * dstC] = tmp[8];
                dst[2 * dstS + 1 * dstC] = tmp[9];
                dst[2 * dstS + 2 * dstC] = tmp[10];
                dst[2 * dstS + 3 * dstC] = tmp[11];
                dst[3 * dstS + 0 * dstC] = tmp[12];
                dst[3 * dstS + 1 * dstC] = tmp[13];
                dst[3 * dstS + 2 * dstC] = tmp[14];
                dst[3 * dstS + 3 * dstC] = tmp[15];
            }
        }

        SIMD_INLINE void WinogradKernel3x3Block4x4SetOutput1t(const float * src, size_t srcStride, float * dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[16];
                WinogradKernel3x3Block4x4SetOutput1(src, srcStride, tmp);
                for (size_t row = 0; row < rowE; ++row)
                    for (size_t col = 0; col < colE; ++col)
                        dst[row*dstS + col * dstC] = tmp[row * 4 + col];
            }
        }

        void WinogradKernel3x3Block4x4SetOutput(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstHeightFull = dstHeight / 4 * 4;
            size_t dstWidthFull = dstWidth / 4 * 4;
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 4)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel3x3Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, 4, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel3x3Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, 4), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel3x3Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col)*dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    size_t row, col;
                    for (row = 0; row < dstHeightFull; row += 4)
                    {
                        for (col = 0; col < dstWidthFull; col += 4)
                            WinogradKernel3x3Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel3x3Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, 4, dstWidth - col);
                    }
                    if (row < dstHeight)
                    {
                        for (col = 0; col < dstWidthFull; col += 4)
                            WinogradKernel3x3Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 4);
                        if (col < dstWidth)
                            WinogradKernel3x3Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                    }
                    dst += dstHeight * dstWidth;
                }
            }
        }
    }
#endif
}

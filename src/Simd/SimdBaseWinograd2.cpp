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
        void WinogradKernel2x2Block2x2SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel2x2Block2x2SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 4, dst += 1)
                    Base::WinogradKernel2x2Block2x2SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput1(const float src[16], float* dst, size_t stride)
        {
            dst[0 * stride] = src[0] - src[1] - src[3] + src[4];
            dst[1 * stride] = src[1] - src[4];
            dst[2 * stride] = src[2] - src[1] + src[4] - src[5];
            dst[3 * stride] = src[3] - src[4];
            dst[4 * stride] = src[4];
            dst[5 * stride] = src[5] - src[4];
            dst[6 * stride] = src[4] - src[3] + src[6] - src[7];
            dst[7 * stride] = src[7] - src[4];
            dst[8 * stride] = src[4] - src[5] + src[8] - src[7];
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput1n(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            float tmp[9];
            tmp[0] = src[0 * srcStride + 0];
            tmp[1] = src[0 * srcStride + 1];
            tmp[2] = src[0 * srcStride + 2];
            tmp[3] = src[1 * srcStride + 0];
            tmp[4] = src[1 * srcStride + 1];
            tmp[5] = src[1 * srcStride + 2];
            tmp[6] = src[2 * srcStride + 0];
            tmp[7] = src[2 * srcStride + 1];
            tmp[8] = src[2 * srcStride + 2];

            WinogradKernel2x2Block2x2SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput1n(const float* src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            float tmp[9] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 3 + col] = src[row * srcStride + col];
            WinogradKernel2x2Block2x2SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput1t(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[9];
                tmp[0] = src[0 * srcS + 0 * srcC];
                tmp[1] = src[0 * srcS + 1 * srcC];
                tmp[2] = src[0 * srcS + 2 * srcC];
                tmp[3] = src[1 * srcS + 0 * srcC];
                tmp[4] = src[1 * srcS + 1 * srcC];
                tmp[5] = src[1 * srcS + 2 * srcC];
                tmp[6] = src[2 * srcS + 0 * srcC];
                tmp[7] = src[2 * srcS + 1 * srcC];
                tmp[8] = src[2 * srcS + 2 * srcC];
                WinogradKernel2x2Block2x2SetInput1(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetInput1t(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[9] = { 0 };
                for (size_t row = rowB; row < rowE; ++row)
                    for (size_t col = colB; col < colE; ++col)
                        tmp[row * 3 + col] = src[row * srcS + col * srcC];
                WinogradKernel2x2Block2x2SetInput1(tmp, dst, dstStride);
            }
        }

        void WinogradKernel2x2Block2x2SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padW == padH && (padY + padH == 0 || padY + padH == 1));
            size_t dstHeight = srcHeight - 1 + padY + padH;
            size_t dstWidth = srcWidth - 1 + padX + padW;
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            size_t noseW = Simd::Min<size_t>(3, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(3, dstHeight + 1);
            size_t startY = padY ? 2 : 0;
            size_t startX = padX ? 2 : 0;
            if (padY || padH)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 2;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 2;
                if(padY)
                    src -= (srcWidth + 1) * (trans ? srcChannels : 1);
            }
            size_t tailW = dstWidth - dstWidthFull + (padW ? 0 : 1);
            size_t tailH = dstHeight - dstHeightFull + (padH ? 0 : 1);
            if (trans)
            {
                size_t row = 0, col = 0;
                if (padY)
                {
                    if (padX)
                        WinogradKernel2x2Block2x2SetInput1t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 2)
                        WinogradKernel2x2Block2x2SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 3, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block2x2SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = startY; row < dstHeightFull; row += 2)
                {
                    if (padX)
                        WinogradKernel2x2Block2x2SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 3, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 2)
                        WinogradKernel2x2Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 3, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstHeight)
                {
                    if (padX)
                        WinogradKernel2x2Block2x2SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 2)
                        WinogradKernel2x2Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 3, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block2x2SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
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
                            WinogradKernel2x2Block2x2SetInput1n(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 2)
                            WinogradKernel2x2Block2x2SetInput1n(src + col, srcWidth, 1, noseH, 0, 3, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block2x2SetInput1n(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                    }
                    for (row = startY; row < dstHeightFull; row += 2)
                    {
                        if (padX)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth, srcWidth, 0, 3, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 2)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, 3, 0, tailW, dst++, dstStride);
                    }
                    if (row < dstHeight)
                    {
                        if (padX)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 2)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 3, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block2x2SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                    }
                    src += srcWidth * srcHeight;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput1(const float* src, size_t stride, float dst[4])
        {
            float s[9];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];
            s[6] = src[6 * stride];
            s[7] = src[7 * stride];
            s[8] = src[8 * stride];

            dst[0] = s[0] + s[1] + s[3] + s[4];
            dst[1] = s[1] + s[2] + s[4] + s[5];
            dst[2] = s[3] + s[4] + s[6] + s[7];
            dst[3] = s[4] + s[5] + s[7] + s[8];
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput1n(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            float tmp[4];
            WinogradKernel2x2Block2x2SetOutput1(src, srcStride, tmp);
            dst[0 * dstStride + 0] = tmp[0];
            dst[0 * dstStride + 1] = tmp[1];
            dst[1 * dstStride + 0] = tmp[2];
            dst[1 * dstStride + 1] = tmp[3];
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput1n(const float* src, size_t srcStride, float* dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[4];
            WinogradKernel2x2Block2x2SetOutput1(src, srcStride, tmp);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row * dstStride + col] = tmp[row * 2 + col];
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel2x2Block2x2SetOutput1(src, srcStride, tmp);
                dst[0 * dstS + 0 * dstC] = tmp[0];
                dst[0 * dstS + 1 * dstC] = tmp[1];
                dst[1 * dstS + 0 * dstC] = tmp[2];
                dst[1 * dstS + 1 * dstC] = tmp[3];
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block2x2SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel2x2Block2x2SetOutput1(src, srcStride, tmp);
                for (size_t row = 0; row < rowE; ++row)
                    for (size_t col = 0; col < colE; ++col)
                        dst[row * dstS + col * dstC] = tmp[row * 2 + col];
            }
        }

        void WinogradKernel2x2Block2x2SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstHeightFull = AlignLo(dstHeight, 2);
            size_t dstWidthFull = AlignLo(dstWidth, 2);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 2)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        WinogradKernel2x2Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 2, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 2)
                        WinogradKernel2x2Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 2), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block2x2SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
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
                            WinogradKernel2x2Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel2x2Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, 2, dstWidth - col);
                    }
                    if (row < dstHeight)
                    {
                        for (col = 0; col < dstWidthFull; col += 2)
                            WinogradKernel2x2Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 2);
                        if (col < dstWidth)
                            WinogradKernel2x2Block2x2SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                    }
                    dst += dstHeight * dstWidth;
                }
            }
        }

        //-----------------------------------------------------------------------

        void WinogradKernel2x2Block4x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel2x2Block4x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 4, dst += 1)
                    Base::WinogradKernel2x2Block4x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput1(const float src[25], float* dst, size_t stride)
        {
            float tmp[25];
            tmp[0] = 2 * src[0] - src[5] - 2 * src[10] + src[15];
            tmp[1] = 2 * src[1] - src[6] - 2 * src[11] + src[16];
            tmp[2] = 2 * src[2] - src[7] - 2 * src[12] + src[17];
            tmp[3] = 2 * src[3] - src[8] - 2 * src[13] + src[18];
            tmp[4] = 2 * src[4] - src[9] - 2 * src[14] + src[19];
            tmp[5] = src[15] - 2 * src[5] - src[10];
            tmp[6] = src[16] - 2 * src[6] - src[11];
            tmp[7] = src[17] - 2 * src[7] - src[12];
            tmp[8] = src[18] - 2 * src[8] - src[13];
            tmp[9] = src[19] - 2 * src[9] - src[14];
            tmp[10] = 2 * src[5] - 3 * src[10] + src[15];
            tmp[11] = 2 * src[6] - 3 * src[11] + src[16];
            tmp[12] = 2 * src[7] - 3 * src[12] + src[17];
            tmp[13] = 2 * src[8] - 3 * src[13] + src[18];
            tmp[14] = 2 * src[9] - 3 * src[14] + src[19];
            tmp[15] = src[15] - src[5];
            tmp[16] = src[16] - src[6];
            tmp[17] = src[17] - src[7];
            tmp[18] = src[18] - src[8];
            tmp[19] = src[19] - src[9];
            tmp[20] = 2 * src[5] - src[10] - 2 * src[15] + src[20];
            tmp[21] = 2 * src[6] - src[11] - 2 * src[16] + src[21];
            tmp[22] = 2 * src[7] - src[12] - 2 * src[17] + src[22];
            tmp[23] = 2 * src[8] - src[13] - 2 * src[18] + src[23];
            tmp[24] = 2 * src[9] - src[14] - 2 * src[19] + src[24];

            dst[0 * stride] = 2 * tmp[0] - tmp[1] - 2 * tmp[2] + tmp[3];
            dst[1 * stride] = tmp[3] - 2 * tmp[1] - tmp[2];
            dst[2 * stride] = 2 * tmp[1] - 3 * tmp[2] + tmp[3];
            dst[3 * stride] = tmp[3] - tmp[1];
            dst[4 * stride] = 2 * tmp[1] - tmp[2] - 2 * tmp[3] + tmp[4];
            dst[5 * stride] = 2 * tmp[5] - tmp[6] - 2 * tmp[7] + tmp[8];
            dst[6 * stride] = tmp[8] - 2 * tmp[6] - tmp[7];
            dst[7 * stride] = 2 * tmp[6] - 3 * tmp[7] + tmp[8];
            dst[8 * stride] = tmp[8] - tmp[6];
            dst[9 * stride] = 2 * tmp[6] - tmp[7] - 2 * tmp[8] + tmp[9];
            dst[10 * stride] = 2 * tmp[10] - tmp[11] - 2 * tmp[12] + tmp[13];
            dst[11 * stride] = tmp[13] - 2 * tmp[11] - tmp[12];
            dst[12 * stride] = 2 * tmp[11] - 3 * tmp[12] + tmp[13];
            dst[13 * stride] = tmp[13] - tmp[11];
            dst[14 * stride] = 2 * tmp[11] - tmp[12] - 2 * tmp[13] + tmp[14];
            dst[15 * stride] = 2 * tmp[15] - tmp[16] - 2 * tmp[17] + tmp[18];
            dst[16 * stride] = tmp[18] - 2 * tmp[16] - tmp[17];
            dst[17 * stride] = 2 * tmp[16] - 3 * tmp[17] + tmp[18];
            dst[18 * stride] = tmp[18] - tmp[16];
            dst[19 * stride] = 2 * tmp[16] - tmp[17] - 2 * tmp[18] + tmp[19];
            dst[20 * stride] = 2 * tmp[20] - tmp[21] - 2 * tmp[22] + tmp[23];
            dst[21 * stride] = tmp[23] - 2 * tmp[21] - tmp[22];
            dst[22 * stride] = 2 * tmp[21] - 3 * tmp[22] + tmp[23];
            dst[23 * stride] = tmp[23] - tmp[21];
            dst[24 * stride] = 2 * tmp[21] - tmp[22] - 2 * tmp[23] + tmp[24];
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput1n(const float* src, size_t srcStride, float* dst, size_t dstStride)
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
            WinogradKernel2x2Block4x4SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput1n(const float* src, size_t srcStride, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            float tmp[25] = { 0 };
            for (size_t row = rowB; row < rowE; ++row)
                for (size_t col = colB; col < colE; ++col)
                    tmp[row * 5 + col] = src[row * srcStride + col];
            WinogradKernel2x2Block4x4SetInput1(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput1t(const float* src, size_t srcW, size_t srcC, float* dst, size_t dstStride)
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
                WinogradKernel2x2Block4x4SetInput1(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetInput1t(const float* src, size_t srcW, size_t srcC, size_t rowB, size_t rowE, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            size_t srcS = srcW * srcC;
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[25] = { 0 };
                for (size_t row = rowB; row < rowE; ++row)
                    for (size_t col = colB; col < colE; ++col)
                        tmp[row * 5 + col] = src[row * srcS + col * srcC];
                WinogradKernel2x2Block4x4SetInput1(tmp, dst, dstStride);
            }
        }

        void WinogradKernel2x2Block4x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padY == padX && padW == padH && (padY + padH == 0 || padY + padH == 1));
            size_t dstHeight = srcHeight - 1 + padY + padH;
            size_t dstWidth = srcWidth - 1 + padX + padW;
            size_t dstHeightFull = AlignLo(dstHeight, 4);
            size_t dstWidthFull = AlignLo(dstWidth, 4);
            size_t noseW = Simd::Min<size_t>(5, dstWidth + 1);
            size_t noseH = Simd::Min<size_t>(5, dstHeight + 1);
            size_t startY = padY ? 4 : 0;
            size_t startX = padX ? 4 : 0;
            if (padY || padH)
            {
                if (dstHeight == dstHeightFull)
                    dstHeightFull -= 4;
                if (dstWidth == dstWidthFull)
                    dstWidthFull -= 4;
                if (padY)
                    src -= (srcWidth + 1) * (trans ? srcChannels : 1);
            }
            size_t tailW = dstWidth - dstWidthFull + (padW ? 0 : 1);
            size_t tailH = dstHeight - dstHeightFull + (padH ? 0 : 1);
            if (trans)
            {
                size_t row = 0, col = 0;
                if (padY)
                {
                    if (padX)
                        WinogradKernel2x2Block4x4SetInput1t(src, srcWidth, srcChannels, 1, noseH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 4)
                        WinogradKernel2x2Block4x4SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block4x4SetInput1t(src + col * srcChannels, srcWidth, srcChannels, 1, noseH, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                for (row = startY; row < dstHeightFull; row += 4)
                {
                    if (padX)
                        WinogradKernel2x2Block4x4SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, 5, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 4)
                        WinogradKernel2x2Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, 5, 0, tailW, dst, dstStride), dst += srcChannels;
                }
                if (row < dstHeight)
                {
                    if (padX)
                        WinogradKernel2x2Block4x4SetInput1t(src + row * srcWidth * srcChannels, srcWidth, srcChannels, 0, tailH, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstWidthFull; col += 4)
                        WinogradKernel2x2Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, 5, dst, dstStride), dst += srcChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block4x4SetInput1t(src + (row * srcWidth + col) * srcChannels, srcWidth, srcChannels, 0, tailH, 0, tailW, dst, dstStride), dst += srcChannels;
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
                            WinogradKernel2x2Block4x4SetInput1n(src, srcWidth, 1, noseH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 4)
                            WinogradKernel2x2Block4x4SetInput1n(src + col, srcWidth, 1, noseH, 0, 5, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block4x4SetInput1n(src + col, srcWidth, 1, noseH, 0, tailW, dst++, dstStride);
                    }
                    for (row = startY; row < dstHeightFull; row += 4)
                    {
                        if (padX)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth, srcWidth, 0, 5, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 4)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, 5, 0, tailW, dst++, dstStride);
                    }
                    if (row < dstHeight)
                    {
                        if (padX)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth, srcWidth, 0, tailH, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstWidthFull; col += 4)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, 5, dst++, dstStride);
                        if (col < dstWidth)
                            WinogradKernel2x2Block4x4SetInput1n(src + row * srcWidth + col, srcWidth, 0, tailH, 0, tailW, dst++, dstStride);
                    }
                    src += srcWidth * srcHeight;
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput1(const float* src, size_t stride, float dst[4])
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

            float t[20];
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
            t[10] = s[5] + s[10] + 4 * s[15];
            t[11] = s[6] + s[11] + 4 * s[16];
            t[12] = s[7] + s[12] + 4 * s[17];
            t[13] = s[8] + s[13] + 4 * s[18];
            t[14] = s[9] + s[14] + 4 * s[19];
            t[15] = s[5] - s[10] + 8 * s[15] + s[20];
            t[16] = s[6] - s[11] + 8 * s[16] + s[21];
            t[17] = s[7] - s[12] + 8 * s[17] + s[22];
            t[18] = s[8] - s[13] + 8 * s[18] + s[23];
            t[19] = s[9] - s[14] + 8 * s[19] + s[24];

            dst[0] = t[0] + t[1] + t[2] + t[3];
            dst[1] = t[1] - t[2] + 2 * t[3];
            dst[2] = t[1] + t[2] + 4 * t[3];
            dst[3] = t[1] - t[2] + 8 * t[3] + t[4];
            dst[4] = t[5] + t[6] + t[7] + t[8];
            dst[5] = t[6] - t[7] + 2 * t[8];
            dst[6] = t[6] + t[7] + 4 * t[8];
            dst[7] = t[6] - t[7] + 8 * t[8] + t[9];
            dst[8] = t[10] + t[11] + t[12] + t[13];
            dst[9] = t[11] - t[12] + 2 * t[13];
            dst[10] = t[11] + t[12] + 4 * t[13];
            dst[11] = t[11] - t[12] + 8 * t[13] + t[14];
            dst[12] = t[15] + t[16] + t[17] + t[18];
            dst[13] = t[16] - t[17] + 2 * t[18];
            dst[14] = t[16] + t[17] + 4 * t[18];
            dst[15] = t[16] - t[17] + 8 * t[18] + t[19];
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput1n(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            float tmp[16];
            WinogradKernel2x2Block4x4SetOutput1(src, srcStride, tmp);
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

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput1n(const float* src, size_t srcStride, float* dst, size_t dstStride, size_t rowE, size_t colE)
        {
            float tmp[16];
            WinogradKernel2x2Block4x4SetOutput1(src, srcStride, tmp);
            for (size_t row = 0; row < rowE; ++row)
                for (size_t col = 0; col < colE; ++col)
                    dst[row * dstStride + col] = tmp[row * 4 + col];
        }

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[16];
                WinogradKernel2x2Block4x4SetOutput1(src, srcStride, tmp);
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

        SIMD_INLINE void WinogradKernel2x2Block4x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstW, size_t dstC, size_t rowE, size_t colE)
        {
            size_t dstS = dstW * dstC;
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[16];
                WinogradKernel2x2Block4x4SetOutput1(src, srcStride, tmp);
                for (size_t row = 0; row < rowE; ++row)
                    for (size_t col = 0; col < colE; ++col)
                        dst[row * dstS + col * dstC] = tmp[row * 4 + col];
            }
        }

        void WinogradKernel2x2Block4x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstHeightFull = AlignLo(dstHeight, 4);
            size_t dstWidthFull = AlignLo(dstWidth, 4);
            if (trans)
            {
                size_t row, col;
                for (row = 0; row < dstHeightFull; row += 4)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel2x2Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, 4, dstWidth - col), src += dstChannels;
                }
                if (row < dstHeight)
                {
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel2x2Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, 4), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel2x2Block4x4SetOutput1t(src, srcStride, dst + (row * dstWidth + col) * dstChannels, dstWidth, dstChannels, dstHeight - row, dstWidth - col), src += dstChannels;
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
                            WinogradKernel2x2Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth);
                        if (col < dstWidth)
                            WinogradKernel2x2Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, 4, dstWidth - col);
                    }
                    if (row < dstHeight)
                    {
                        for (col = 0; col < dstWidthFull; col += 4)
                            WinogradKernel2x2Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, 4);
                        if (col < dstWidth)
                            WinogradKernel2x2Block4x4SetOutput1n(src++, srcStride, dst + row * dstWidth + col, dstWidth, dstHeight - row, dstWidth - col);
                    }
                    dst += dstHeight * dstWidth;
                }
            }
        }
    }
#endif
}

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
        void WinogradKernel1x3Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel1x3Block1x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 3, dst += 1)
                    Base::WinogradKernel1x3Block1x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput1n(const float src[6], float* dst, size_t stride)
        {
            dst[0 * stride] = src[0] * 4 - src[2] * 5 + src[4];
            dst[1 * stride] = -src[1] * 4 - src[2] * 4 + src[3] + src[4];
            dst[2 * stride] = src[1] * 4 - src[2] * 4 - src[3] + src[4];
            dst[3 * stride] = -src[1] * 2 - src[2] + src[3] * 2 + src[4];
            dst[4 * stride] = src[1] * 2 - src[2] - src[3] * 2 + src[4];
            dst[5 * stride] = src[1] * 4 - src[3] * 5 + src[5];
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput1n(const float* src, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            float tmp[6] = { 0 };
            for (size_t col = colB; col < colE; ++col)
                tmp[col] = src[col];
            WinogradKernel1x3Block1x4SetInput1n(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput1t(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[6];
                tmp[0] = src[0 * srcC];
                tmp[1] = src[1 * srcC];
                tmp[2] = src[2 * srcC];
                tmp[3] = src[3 * srcC];
                tmp[4] = src[4 * srcC];
                tmp[5] = src[5 * srcC];
                WinogradKernel1x3Block1x4SetInput1n(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetInput1t(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[6] = { 0 };
                for (size_t col = colB; col < colE; ++col)
                    tmp[col] = src[col * srcC];
                WinogradKernel1x3Block1x4SetInput1n(tmp, dst, dstStride);
            }
        }

        void WinogradKernel1x3Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 1));
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 2;
            size_t dstW4 = dstW / 4 * 4;
            size_t noseW = Simd::Min<size_t>(6, dstW + 1);
            size_t startX = padX ? 4 : 0;
            if (padX)
            {
                if (dstW == dstW4)
                    dstW4 -= 4;
                src -= 1 * (trans ? srcChannels : 1);
            }
            size_t tailW = dstW - dstW4 + (padX ? 1 : 2);
            if (trans)
            {
                for (size_t row = 0; row < dstH; row += 1)
                {
                    size_t col = 0;
                    if (padX)
                        WinogradKernel1x3Block1x4SetInput1t(src, srcChannels, 1, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel1x3Block1x4SetInput1t(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                    if (col < dstW)
                        WinogradKernel1x3Block1x4SetInput1t(src + col * srcChannels, srcChannels, 0, tailW, dst, dstStride), dst += srcChannels;
                    src += srcWidth * srcChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    for (size_t row = 0; row < dstH; row += 1)
                    {
                        size_t col = 0;
                        if (padX)
                            WinogradKernel1x3Block1x4SetInput1n(src, 1, noseW, dst++, dstStride);
                        for (col = startX; col < dstW4; col += 4)
                            WinogradKernel1x3Block1x4SetInput1n(src + col, dst++, dstStride);
                        if (col < dstW)
                            WinogradKernel1x3Block1x4SetInput1n(src + col, 0, tailW, dst++, dstStride);
                        src += srcWidth;
                    }
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput1n(const float* src, size_t stride, float * dst)
        {
            float s[6];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];

            dst[0] = s[0] + s[1] + s[2] + s[3] + s[4];
            dst[1] = s[1] - s[2] + 2 * s[3] - 2 * s[4];
            dst[2] = s[1] + s[2] + 4 * s[3] + 4 * s[4];
            dst[3] = s[1] - s[2] + 8 * s[3] - 8 * s[4] + s[5];
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput1n(const float* src, size_t srcStride, float* dst, size_t colE)
        {
            float tmp[4];
            WinogradKernel1x3Block1x4SetOutput1n(src, srcStride, tmp);
            for (size_t col = 0; col < colE; ++col)
                dst[col] = tmp[col];
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstC)
        {
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel1x3Block1x4SetOutput1n(src, srcStride, tmp);
                dst[0 * dstC] = tmp[0];
                dst[1 * dstC] = tmp[1];
                dst[2 * dstC] = tmp[2];
                dst[3 * dstC] = tmp[3];
            }
        }

        SIMD_INLINE void WinogradKernel1x3Block1x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE)
        {
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel1x3Block1x4SetOutput1n(src, srcStride, tmp);
                for (size_t col = 0; col < colE; ++col)
                    dst[col * dstC] = tmp[col];
            }
        }

        void WinogradKernel1x3Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstWidthFull = dstWidth / 4 * 4;
            if (trans)
            {
                for (size_t row = 0; row < dstHeight; row += 1)
                {
                    size_t col;
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel1x3Block1x4SetOutput1t(src, srcStride, dst + col * dstChannels, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel1x3Block1x4SetOutput1t(src, srcStride, dst + col * dstChannels, dstChannels, dstWidth - col), src += dstChannels;
                    dst += dstWidth * dstChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    for (size_t row = 0; row < dstHeight; row += 1)
                    {
                        size_t col;   
                        for (col = 0; col < dstWidthFull; col += 4)
                            WinogradKernel1x3Block1x4SetOutput1n(src++, srcStride, dst + col);
                        if (col < dstWidth)
                            WinogradKernel1x3Block1x4SetOutput1n(src++, srcStride, dst + col, dstWidth - col);
                        dst += dstWidth;
                    }
                }
            }
        }

        //-----------------------------------------------------------------------

        void WinogradKernel1x5Block1x4SetFilter(const float* src, size_t size, float* dst, SimdBool trans)
        {
            if (trans)
            {
                for (size_t i = 0; i < size; i += 1)
                    Base::WinogradKernel1x5Block1x4SetFilter1t(src + i, dst + i, size);
            }
            else
            {
                for (size_t i = 0; i < size; i += 1, src += 5, dst += 1)
                    Base::WinogradKernel1x5Block1x4SetFilter1n(src, dst, size);
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput1n(const float src[8], float* dst, size_t stride)
        {
            dst[0 * stride] = 36 * src[0] - 49 * src[2] + 14 * src[4] - src[6];
            dst[1 * stride] = 36 * (src[2] + src[1]) - 13 * (src[4] + src[3]) + src[6] + src[5];
            dst[2 * stride] = 36 * (src[2] - src[1]) - 13 * (src[4] - src[3]) + src[6] - src[5];
            dst[3 * stride] = 9 * (src[2] + 2 * src[1]) - 10 * (src[4] + 2 * src[3]) + src[6] + 2 * src[5];
            dst[4 * stride] = 9 * (src[2] - 2 * src[1]) - 10 * (src[4] - 2 * src[3]) + src[6] - 2 * src[5];
            dst[5 * stride] = 4 * (src[2] + 3 * src[1]) - 5 * (src[4] + 3 * src[3]) + src[6] + 3 * src[5];
            dst[6 * stride] = 4 * (src[2] - 3 * src[1]) - 5 * (src[4] - 3 * src[3]) + src[6] - 3 * src[5];
            dst[7 * stride] = -(36 * src[1] - 49 * src[3] + 14 * src[5] - src[7]);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput1n(const float* src, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            float tmp[8] = { 0 };
            for (size_t col = colB; col < colE; ++col)
                tmp[col] = src[col];
            WinogradKernel1x5Block1x4SetInput1n(tmp, dst, dstStride);
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput1t(const float* src, size_t srcC, float* dst, size_t dstStride)
        {
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[8];
                tmp[0] = src[0 * srcC];
                tmp[1] = src[1 * srcC];
                tmp[2] = src[2 * srcC];
                tmp[3] = src[3 * srcC];
                tmp[4] = src[4 * srcC];
                tmp[5] = src[5 * srcC];
                tmp[6] = src[6 * srcC];
                tmp[7] = src[7 * srcC];
                WinogradKernel1x5Block1x4SetInput1n(tmp, dst, dstStride);
            }
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetInput1t(const float* src, size_t srcC, size_t colB, size_t colE, float* dst, size_t dstStride)
        {
            for (size_t c = 0; c < srcC; ++c, src++, dst++)
            {
                float tmp[8] = { 0 };
                for (size_t col = colB; col < colE; ++col)
                    tmp[col] = src[col * srcC];
                WinogradKernel1x5Block1x4SetInput1n(tmp, dst, dstStride);
            }
        }

        void WinogradKernel1x5Block1x4SetInput(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
            size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans)
        {
            assert(padX == padW && padY == 0 && padH == 0 && (padX == 0 || padX == 2));
            size_t dstH = srcHeight;
            size_t dstW = padX ? srcWidth : srcWidth - 4;
            size_t dstW4 = dstW / 4 * 4;
            size_t noseW = Simd::Min<size_t>(8, dstW + 2);
            size_t startX = padX ? 4 : 0;
            if (padX)
            {
                if (dstW == dstW4 || dstW == dstW4 + 1)
                    dstW4 -= 4;
                src -= 2 * (trans ? srcChannels : 1);
            }
            size_t tailW = dstW - dstW4 + (padX ? 2 : 4);
            if (trans)
            {
                for (size_t row = 0; row < dstH; row += 1)
                {
                    size_t col = 0;
                    if (padX)
                        WinogradKernel1x5Block1x4SetInput1t(src, srcChannels, 2, noseW, dst, dstStride), dst += srcChannels;
                    for (col = startX; col < dstW4; col += 4)
                        WinogradKernel1x5Block1x4SetInput1t(src + col * srcChannels, srcChannels, dst, dstStride), dst += srcChannels;
                    for (size_t tail = tailW; col < dstW; col += 4, tail -= 4)
                        WinogradKernel1x5Block1x4SetInput1t(src + col * srcChannels, srcChannels, 0, tail, dst, dstStride), dst += srcChannels;
                    src += srcWidth * srcChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < srcChannels; ++c)
                {
                    for (size_t row = 0; row < dstH; row += 1)
                    {
                        size_t col = 0;
                        if (padX)
                            WinogradKernel1x5Block1x4SetInput1n(src, 2, noseW, dst++, dstStride);
                        for (col = startX; col < dstW4; col += 4)
                            WinogradKernel1x5Block1x4SetInput1n(src + col, dst++, dstStride);
                        for (size_t tail = tailW; col < dstW; col += 4, tail -= 4)
                            WinogradKernel1x5Block1x4SetInput1n(src + col, 0, tail, dst++, dstStride);
                        src += srcWidth;
                    }
                }
            }
        }

        //-----------------------------------------------------------------------

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput1n(const float* src, size_t stride, float* dst)
        {
            float s[8];
            s[0] = src[0 * stride];
            s[1] = src[1 * stride];
            s[2] = src[2 * stride];
            s[3] = src[3 * stride];
            s[4] = src[4 * stride];
            s[5] = src[5 * stride];
            s[6] = src[6 * stride];
            s[7] = src[7 * stride];

            dst[0] = s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6];
            dst[1] = s[1] - s[2] + 2 * s[3] - 2 * s[4] + 3 * s[5] - 3 * s[6];
            dst[2] = s[1] + s[2] + 4 * s[3] + 4 * s[4] + 9 * s[5] + 9 * s[6];
            dst[3] = s[1] - s[2] + 8 * s[3] - 8 * s[4] + 27 * s[5] - 27 * s[6] + s[7];
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput1n(const float* src, size_t srcStride, float* dst, size_t colE)
        {
            float tmp[4];
            WinogradKernel1x5Block1x4SetOutput1n(src, srcStride, tmp);
            for (size_t col = 0; col < colE; ++col)
                dst[col] = tmp[col];
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstC)
        {
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel1x5Block1x4SetOutput1n(src, srcStride, tmp);
                dst[0 * dstC] = tmp[0];
                dst[1 * dstC] = tmp[1];
                dst[2 * dstC] = tmp[2];
                dst[3 * dstC] = tmp[3];
            }
        }

        SIMD_INLINE void WinogradKernel1x5Block1x4SetOutput1t(const float* src, size_t srcStride, float* dst, size_t dstC, size_t colE)
        {
            for (size_t d = 0; d < dstC; ++d, src++, dst++)
            {
                float tmp[4];
                WinogradKernel1x5Block1x4SetOutput1n(src, srcStride, tmp);
                for (size_t col = 0; col < colE; ++col)
                    dst[col * dstC] = tmp[col];
            }
        }

        void WinogradKernel1x5Block1x4SetOutput(const float* src, size_t srcStride, float* dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans)
        {
            size_t dstWidthFull = dstWidth / 4 * 4;
            if (trans)
            {
                for (size_t row = 0; row < dstHeight; row += 1)
                {
                    size_t col;
                    for (col = 0; col < dstWidthFull; col += 4)
                        WinogradKernel1x5Block1x4SetOutput1t(src, srcStride, dst + col * dstChannels, dstChannels), src += dstChannels;
                    if (col < dstWidth)
                        WinogradKernel1x5Block1x4SetOutput1t(src, srcStride, dst + col * dstChannels, dstChannels, dstWidth - col), src += dstChannels;
                    dst += dstWidth * dstChannels;
                }
            }
            else
            {
                for (size_t c = 0; c < dstChannels; ++c)
                {
                    for (size_t row = 0; row < dstHeight; row += 1)
                    {
                        size_t col;
                        for (col = 0; col < dstWidthFull; col += 4)
                            WinogradKernel1x5Block1x4SetOutput1n(src++, srcStride, dst + col);
                        if (col < dstWidth)
                            WinogradKernel1x5Block1x4SetOutput1n(src++, srcStride, dst + col, dstWidth - col);
                        dst += dstWidth;
                    }
                }
            }
        }
    }
#endif
}

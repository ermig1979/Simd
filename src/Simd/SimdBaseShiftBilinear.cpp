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
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int Interpolate(int s[2][2], int k[2][2])
        {
            return (s[0][0] * k[0][0] + s[0][1] * k[0][1] +
                s[1][0] * k[1][0] + s[1][1] * k[1][1] + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
        }

        SIMD_INLINE int Interpolate(const unsigned char *src, size_t dx, size_t dy, int k[2][2])
        {
            return (src[0] * k[0][0] + src[dx] * k[0][1] +
                src[dy] * k[1][0] + src[dx + dy] * k[1][1] + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
        }

        SIMD_INLINE int Interpolate(const unsigned char *src, size_t dr, int k[2])
        {
            return (src[0] * k[0] + src[dr] * k[1] + LINEAR_ROUND_TERM) >> LINEAR_SHIFT;
        }

        void MixBorder(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t * bkg, size_t bkgStride, ptrdiff_t iDx, ptrdiff_t iDy, int fDx, int fDy, uint8_t * dst, size_t dstStride)
        {
            size_t bkgWidth = Abs(iDx) - (iDx < 0 && fDx ? 1 : 0);
            size_t bkgHeight = Abs(iDy) - (iDy < 0 && fDy ? 1 : 0);

            size_t mainWidth = width - bkgWidth - (fDx ? 1 : 0);
            size_t mainHeight = height - bkgHeight - (fDy ? 1 : 0);

            int k[2][2];
            k[0][0] = (FRACTION_RANGE - fDx)*(FRACTION_RANGE - fDy);
            k[0][1] = fDx*(FRACTION_RANGE - fDy);
            k[1][0] = (FRACTION_RANGE - fDx)*fDy;
            k[1][1] = fDx*fDy;

            if (fDx)
            {
                const uint8_t * ps[2][2];
                size_t xOffset = (iDx >= 0 ? width - 1 - iDx : -iDx - 1)*channelCount;
                size_t bkgOffset = (iDy > 0 ? 0 : -iDy)*bkgStride + xOffset;
                size_t dstOffset = (iDy > 0 ? 0 : -iDy)*dstStride + xOffset;

                if (iDx < 0)
                {
                    ps[0][0] = bkg + bkgOffset;
                    ps[0][1] = src + (iDy < 0 ? 0 : iDy)*srcStride;
                    ps[1][0] = bkg + bkgOffset;
                    ps[1][1] = src + ((iDy < 0 ? 0 : iDy) + (fDy ? 1 : 0))*srcStride;
                }
                else
                {
                    ps[0][0] = src + (iDy < 0 ? 0 : iDy)*srcStride + (width - 1)*channelCount;
                    ps[0][1] = bkg + bkgOffset;
                    ps[1][0] = src + ((iDy < 0 ? 0 : iDy) + (fDy ? 1 : 0))*srcStride + (width - 1)*channelCount;
                    ps[1][1] = bkg + bkgOffset;
                }

                for (size_t row = 0; row < mainHeight; ++row)
                {
                    for (size_t channel = 0; channel < channelCount; channel++)
                    {
                        int s[2][2];
                        s[0][0] = ps[0][0][channel];
                        s[0][1] = ps[0][1][channel];
                        s[1][0] = ps[1][0][channel];
                        s[1][1] = ps[1][1][channel];
                        dst[dstOffset + channel] = Interpolate(s, k);
                    }
                    ps[0][0] += srcStride;
                    ps[0][1] += bkgStride;
                    ps[1][0] += srcStride;
                    ps[1][1] += bkgStride;
                    dstOffset += dstStride;
                }
            }

            if (fDy)
            {
                const uint8_t * ps[2][2];
                size_t bkgOffset = (iDy >= 0 ? height - 1 - iDy : -iDy - 1)*bkgStride + (iDx > 0 ? 0 : -iDx)*channelCount;
                size_t dstOffset = (iDy >= 0 ? height - 1 - iDy : -iDy - 1)*dstStride + (iDx > 0 ? 0 : -iDx)*channelCount;

                if (iDy < 0)
                {
                    ps[0][0] = bkg + bkgOffset;
                    ps[0][1] = bkg + bkgOffset;
                    ps[1][0] = src + (iDx < 0 ? 0 : iDx)*channelCount;
                    ps[1][1] = src + ((iDx < 0 ? 0 : iDx) + (fDx ? 1 : 0))*channelCount;
                }
                else
                {
                    ps[0][0] = src + (height - 1)*srcStride + (iDx < 0 ? 0 : iDx)*channelCount;
                    ps[0][1] = src + (height - 1)*srcStride + ((iDx < 0 ? 0 : iDx) + (fDx ? 1 : 0))*channelCount;
                    ps[1][0] = bkg + bkgOffset;
                    ps[1][1] = bkg + bkgOffset;
                }

                for (size_t col = 0; col < mainWidth; ++col)
                {
                    for (size_t channel = 0; channel < channelCount; channel++)
                    {
                        int s[2][2];
                        s[0][0] = ps[0][0][channel];
                        s[0][1] = ps[0][1][channel];
                        s[1][0] = ps[1][0][channel];
                        s[1][1] = ps[1][1][channel];
                        dst[dstOffset + channel] = Interpolate(s, k);
                    }
                    ps[0][0] += channelCount;
                    ps[0][1] += channelCount;
                    ps[1][0] += channelCount;
                    ps[1][1] += channelCount;
                    dstOffset += channelCount;
                }
            }

            if (fDx && fDy)
            {
                const uint8_t * ps[2][2];
                size_t xOffset = (iDx >= 0 ? width - 1 - iDx : -iDx - 1)*channelCount;
                size_t bkgOffset = (iDy >= 0 ? height - 1 - iDy : -iDy - 1)*bkgStride + xOffset;
                size_t dstOffset = (iDy >= 0 ? height - 1 - iDy : -iDy - 1)*dstStride + xOffset;

                ps[0][0] = (iDx >= 0 && iDy >= 0) ? (src + (height - 1)*srcStride + (width - 1)*channelCount) : bkg + bkgOffset;
                ps[0][1] = (iDx < 0 && iDy >= 0) ? (src + (height - 1)*srcStride) : bkg + bkgOffset;
                ps[1][0] = (iDx >= 0 && iDy < 0) ? (src + (width - 1)*channelCount) : bkg + bkgOffset;
                ps[1][1] = (iDx < 0 && iDy < 0) ? (src) : bkg + bkgOffset;

                for (size_t channel = 0; channel < channelCount; channel++)
                {
                    int s[2][2];
                    s[0][0] = ps[0][0][channel];
                    s[0][1] = ps[0][1][channel];
                    s[1][0] = ps[1][0][channel];
                    s[1][1] = ps[1][1][channel];
                    dst[dstOffset + channel] = Interpolate(s, k);
                }
            }
        }

        void CommonShiftAction(
            const uint8_t * & src, size_t srcStride, size_t & width, size_t & height, size_t channelCount,
            const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * & dst, size_t dstStride,
            int & fDx, int & fDy)
        {
            assert(cropLeft <= cropRight && cropTop <= cropBottom && cropRight <= width && cropBottom <= height);
            assert(*shiftX < cropRight - cropLeft && *shiftY < cropBottom - cropTop);

            Base::CopyFrame(src, srcStride, width, height, channelCount, cropLeft, cropTop, cropRight, cropBottom, dst, dstStride);

            dst += dstStride*cropTop + cropLeft*channelCount;
            src += srcStride*cropTop + cropLeft*channelCount;
            bkg += bkgStride*cropTop + cropLeft*channelCount;
            width = cropRight - cropLeft;
            height = cropBottom - cropTop;

            ptrdiff_t iDx = (ptrdiff_t)floor(*shiftX + FRACTION_ROUND_TERM);
            ptrdiff_t iDy = (ptrdiff_t)floor(*shiftY + FRACTION_ROUND_TERM);
            fDx = (int)floor((*shiftX + FRACTION_ROUND_TERM - iDx)*FRACTION_RANGE);
            fDy = (int)floor((*shiftY + FRACTION_ROUND_TERM - iDy)*FRACTION_RANGE);

            ptrdiff_t left = (iDx < 0 ? (-iDx - (fDx ? 1 : 0)) : 0);
            ptrdiff_t top = (iDy < 0 ? (-iDy - (fDy ? 1 : 0)) : 0);
            ptrdiff_t right = (iDx < 0 ? width : width - iDx);
            ptrdiff_t bottom = (iDy < 0 ? height : height - iDy);

            Base::CopyFrame(bkg, bkgStride, width, height, channelCount, left, top, right, bottom, dst, dstStride);

            MixBorder(src, srcStride, width, height, channelCount, bkg, bkgStride, iDx, iDy, fDx, fDy, dst, dstStride);

            src += Simd::Max((ptrdiff_t)0, iDy)*srcStride + Simd::Max((ptrdiff_t)0, iDx)*channelCount;
            dst += Simd::Max((ptrdiff_t)0, -iDy)*dstStride + Simd::Max((ptrdiff_t)0, -iDx)*channelCount;

            width = width - Abs(iDx) + (iDx < 0 && fDx ? 1 : 0) - (fDx ? 1 : 0);
            height = height - Abs(iDy) + (iDy < 0 && fDy ? 1 : 0) - (fDy ? 1 : 0);
        }

        void ShiftBilinear(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            int fDx, int fDy, uint8_t * dst, size_t dstStride)
        {
            size_t size = width*channelCount;
            if (fDy)
            {
                if (fDx)
                {
                    int k[2][2];
                    k[0][0] = (FRACTION_RANGE - fDx)*(FRACTION_RANGE - fDy);
                    k[0][1] = fDx*(FRACTION_RANGE - fDy);
                    k[1][0] = (FRACTION_RANGE - fDx)*fDy;
                    k[1][1] = fDx*fDy;
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < size; col++)
                        {
                            dst[col] = Interpolate(src + col, channelCount, srcStride, k);
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    int k[2];
                    k[0] = FRACTION_RANGE - fDy;
                    k[1] = fDy;
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < size; col++)
                        {
                            dst[col] = Interpolate(src + col, srcStride, k);
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
            }
            else
            {
                if (fDx)
                {
                    int k[2];
                    k[0] = FRACTION_RANGE - fDx;
                    k[1] = fDx;
                    for (size_t row = 0; row < height; ++row)
                    {
                        for (size_t col = 0; col < size; col++)
                        {
                            dst[col] = Interpolate(src + col, channelCount, k);
                        }
                        src += srcStride;
                        dst += dstStride;
                    }
                }
                else
                {
                    for (size_t row = 0; row < height; ++row)
                    {
                        memcpy(dst, src, size);
                        src += srcStride;
                        dst += dstStride;
                    }
                }
            }
        }

        void ShiftBilinear(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t * bkg, size_t bkgStride, const double * shiftX, const double * shiftY,
            size_t cropLeft, size_t cropTop, size_t cropRight, size_t cropBottom, uint8_t * dst, size_t dstStride)
        {
            int fDx, fDy;
            CommonShiftAction(src, srcStride, width, height, channelCount, bkg, bkgStride, shiftX, shiftY,
                cropLeft, cropTop, cropRight, cropBottom, dst, dstStride, fDx, fDy);

            ShiftBilinear(src, srcStride, width, height, channelCount, fDx, fDy, dst, dstStride);
        }
    }
}


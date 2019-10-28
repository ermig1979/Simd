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
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE bool SecondProductEnough32bit(uint64_t width, uint64_t height, uint64_t value)
        {
            return 
                width * width * value < 0x100000000ULL &&
                width * height * value < 0x100000000ULL &&
                height * height * value < 0x100000000ULL;
            ;
        }

        SIMD_INLINE bool FirstRowSumEnough32bit(uint64_t width, uint64_t height, uint64_t value)
        {
            return
                width * width * value < 0x200000000ULL &&
                width * height * value < 0x100000000ULL;
        }

        SIMD_INLINE bool SecondRowSumEnough32bit(uint64_t width, uint64_t height, uint64_t value)
        {
            return
                width * width * width * value < 0x300000000ULL &&
                width * width * height * value < 0x200000000ULL &&
                width * height * height * value < 0x100000000ULL;
        }

        void GetMaskMomentsSmall(const uint8_t * mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t * n, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            for (uint32_t row = 0; row < height; ++row)
            {
                uint32_t rowN = 0;
                uint32_t rowX = 0;
                uint32_t rowY = 0;
                uint32_t rowXX = 0;
                uint32_t rowXY = 0;
                uint32_t rowYY = 0;
                for (uint32_t col = 0; col < width; ++col)
                {
                    if (mask[col] == index)
                    {
                        rowN += 1;
                        rowX += col;
                        rowY += row;
                        rowXX += col*col;
                        rowXY += col*row;
                        rowYY += row*row;
                    }
                }
                *n += rowN;
                *x += rowX;
                *y += rowY;
                *xx += rowXX;
                *xy += rowXY;
                *yy += rowYY;

                mask += stride;
            }
        }

        void GetMaskMomentsLarge(const uint8_t * mask, size_t stride, uint32_t width, uint32_t height, uint8_t index,
            uint64_t * n, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy)
        {
            for (uint32_t row = 0; row < height; ++row)
            {
                uint32_t rowN = 0;
                uint32_t rowX = 0;
                uint32_t rowY = 0;
                for (uint32_t col = 0; col < width; ++col)
                {
                    if (mask[col] == index)
                    {
                        rowN += 1;
                        rowX += col;
                        rowY += row;
                        *xx += uint64_t(col)*col;
                        *xy += uint64_t(col)*row;
                        *yy += uint64_t(row)*row;
                    }
                }
                *n += rowN;
                *x += rowX;
                *y += rowY;

                mask += stride;
            }
        }

        void GetObjectMomentsSmall(const uint8_t * src, size_t srcStride, uint32_t width, uint32_t height, const uint8_t * mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            for (uint32_t row = 0; row < height; ++row)
            {
                uint32_t rowN = 0;
                uint32_t rowS = 0;
                uint32_t rowSX = 0;
                uint32_t rowSY = 0;
                uint32_t rowSXX = 0;
                uint32_t rowSXY = 0;
                uint32_t rowSYY = 0;
                if (mask)
                {
                    for (uint32_t col = 0; col < width; ++col)
                    {
                        if (mask[col] == index)
                        {
                            uint32_t val = src[col];
                            rowN += 1;
                            rowS += val;
                            rowSX += val * col;
                            rowSY += val * row;
                            rowSXX += val * col * col;
                            rowSXY += val * col * row;
                            rowSYY += val * row * row;
                        }
                    }
                    mask += maskStride;
                }
                else
                {
                    for (uint32_t col = 0; col < width; ++col)
                    {
                        uint32_t val = src[col];
                        rowN += 1;
                        rowS += val;
                        rowSX += val * col;
                        rowSY += val * row;
                        rowSXX += val * col * col;
                        rowSXY += val * col * row;
                        rowSYY += val * row * row;
                    }
                }
                *n += rowN;
                *s += rowS;
                *sx += rowSX;
                *sy += rowSY;
                *sxx += rowSXX;
                *sxy += rowSXY;
                *syy += rowSYY;
                src += srcStride;
            }
        }

        void GetObjectMomentsMedium(const uint8_t* src, size_t srcStride, uint32_t width, uint32_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            for (uint32_t row = 0; row < height; ++row)
            {
                uint32_t rowN = 0;
                uint32_t rowS = 0;
                uint32_t rowSX = 0;
                uint32_t rowSY = 0;
                if (mask)
                {
                    for (uint32_t col = 0; col < width; ++col)
                    {
                        if (mask[col] == index)
                        {
                            uint32_t val = src[col];
                            rowN += 1;
                            rowS += val;
                            rowSX += val * col;
                            rowSY += val * row;
                            *sxx += uint64_t(val) * col * col;
                            *sxy += uint64_t(val) * col * row;
                            *syy += uint64_t(val) * row * row;
                        }
                    }
                    mask += maskStride;
                }
                else
                {
                    for (uint32_t col = 0; col < width; ++col)
                    {
                        uint32_t val = src[col];
                        rowN += 1;
                        rowS += val;
                        rowSX += val * col;
                        rowSY += val * row;
                        *sxx += uint64_t(val) * col * col;
                        *sxy += uint64_t(val) * col * row;
                        *syy += uint64_t(val) * row * row;
                    }
                }
                *n += rowN;
                *s += rowS;
                *sx += rowSX;
                *sy += rowSY;
                src += srcStride;
            }
        }

        void GetObjectMomentsLarge(const uint8_t* src, size_t srcStride, uint32_t width, uint32_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            for (uint32_t row = 0; row < height; ++row)
            {
                uint32_t rowN = 0;
                uint32_t rowS = 0;
                if (mask)
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        if (mask[col] == index)
                        {
                            uint32_t val = src[col];
                            rowN += 1;
                            rowS += val;
                            *sx += uint64_t(val) * col;
                            *sy += uint64_t(val) * row;
                            *sxx += uint64_t(val) * col * col;
                            *sxy += uint64_t(val) * col * row;
                            *syy += uint64_t(val) * row * row;
                        }
                    }
                    mask += maskStride;
                }
                else
                {
                    for (size_t col = 0; col < width; ++col)
                    {
                        uint32_t val = src[col];
                        rowN += 1;
                        rowS += val;
                        *sx += uint64_t(val) * col;
                        *sy += uint64_t(val) * row;
                        *sxx += uint64_t(val) * col * col;
                        *sxy += uint64_t(val) * col * row;
                        *syy += uint64_t(val) * row * row;
                    }
                }
                *n += rowN;
                *s += rowS;
                src += srcStride;
            }
        }

        void GetMoments(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t* area, uint64_t* x, uint64_t* y, uint64_t* xx, uint64_t* xy, uint64_t* yy)
        {
            *area = 0;
            *x = 0;
            *y = 0;
            *xx = 0;
            *xy = 0;
            *yy = 0;
            if (SecondRowSumEnough32bit(width, height, 1))
                GetMaskMomentsSmall(mask, stride, (uint32_t)width, (uint32_t)height, index, area, x, y, xx, xy, yy);
            else
                GetMaskMomentsLarge(mask, stride, (uint32_t)width, (uint32_t)height, index, area, x, y, xx, xy, yy);
        }

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            *n = 0;
            *s = 0;
            *sx = 0;
            *sy = 0;
            *sxx = 0;
            *sxy = 0;
            *syy = 0;
            if (src)
            {
                if (SecondRowSumEnough32bit(width, height, 255))
                    GetObjectMomentsSmall(src, srcStride, (uint32_t)width, (uint32_t)height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
                else if (FirstRowSumEnough32bit(width, height, 255))
                    GetObjectMomentsMedium(src, srcStride, (uint32_t)width, (uint32_t)height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
                else
                    GetObjectMomentsLarge(src, srcStride, (uint32_t)width, (uint32_t)height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
            }
            else if (mask)
            {
                if (SecondRowSumEnough32bit(width, height, 1))
                    GetMaskMomentsSmall(mask, maskStride, (uint32_t)width, (uint32_t)height, index, n, sx, sy, sxx, sxy, syy);
                else
                    GetMaskMomentsLarge(mask, maskStride, (uint32_t)width, (uint32_t)height, index, n, sx, sy, sxx, sxy, syy);
                *s = *n;
            }
            else
                assert(0);
        }
    }
}

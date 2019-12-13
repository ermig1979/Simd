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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void GetObjectMoments(uint32_t src, uint32_t col, uint32_t & n, uint32_t & s, uint32_t& sx, uint32_t& sxx)
        {
            n += 1;
            s += src;
            sx += src * col;
            sxx += src * col * col;
        }

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            assert(src || mask);

            *n = 0;
            *s = 0;
            *sx = 0;
            *sy = 0;
            *sxx = 0;
            *sxy = 0;
            *syy = 0; 

            const size_t B = 181;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    uint32_t colE = (uint32_t)Simd::Min(colB + B, width);
                    uint32_t _n = 0;
                    uint32_t _s = 0;
                    uint32_t _sx = 0;
                    uint32_t _sxx = 0;
                    if (mask == NULL)
                    {
                        for (uint32_t col = (uint32_t)colB; col < colE; ++col)
                            GetObjectMoments(src[col], col - (uint32_t)colB, _n, _s, _sx, _sxx);
                    }
                    else if (src == NULL)
                    {
                        for (uint32_t col = (uint32_t)colB; col < colE; ++col)
                            if(mask[col] == index)
                                GetObjectMoments(1, col - (uint32_t)colB, _n, _s, _sx, _sxx);
                    }
                    else
                    {
                        for (uint32_t col = (uint32_t)colB; col < colE; ++col)
                            if (mask[col] == index)
                                GetObjectMoments(src[col], col - (uint32_t)colB, _n, _s, _sx, _sxx);
                    }
                    uint64_t _y = row;
                    uint64_t _x = colB;

                    *n += _n;
                    *s += _s;

                    *sx += _sx + _s * _x;
                    *sy += _s * _y;

                    *sxx += _sxx + _sx * _x * 2 + _s * _x * _x;
                    *sxy += _sx * _y + _s * _x * _y;
                    *syy += _s * _y * _y;

                    colB = colE;
                }
                if (src)
                    src += srcStride;
                if (mask)
                    mask += maskStride;
            }
        }

        void GetMoments(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t* area, uint64_t* x, uint64_t* y, uint64_t* xx, uint64_t* xy, uint64_t* yy)
        {
            uint64_t stub;
            GetObjectMoments(NULL, 0, width, height, mask, stride, index, &stub, area, x, y, xx, xy, yy);
        }
    }
}

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
#include "Simd/SimdMemory.h"

namespace Simd
{
    namespace Base
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width, size_t height)
                {
                    _p = Allocate(2 * sizeof(int)*(2 * width + height));
                    ix = (int*)_p;
                    ax = ix + width;
                    iy = ax + width;
                    ay = iy + height;
                    pbx[0] = (int*)(ay + height);
                    pbx[1] = pbx[0] + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                int * ix;
                int * ax;
                int * iy;
                int * ay;
                int * pbx[2];
            private:
                void *_p;
            };
        }

        void EstimateAlphaIndex(size_t srcSize, size_t dstSize, int * indexes, int * alphas, size_t channelCount)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5)*scale - 0.5);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if (index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if (index > (ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                for (size_t c = 0; c < channelCount; c++)
                {
                    size_t offset = i*channelCount + c;
                    indexes[offset] = (int)(channelCount*index + c);
                    alphas[offset] = (int)(alpha * FRACTION_RANGE + 0.5);
                }
            }
        }

        void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            size_t dstRowSize = channelCount*dstWidth;

            Buffer buffer(dstRowSize, dstHeight);

            EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

            EstimateAlphaIndex(srcWidth, dstWidth, buffer.ix, buffer.ax, channelCount);

            ptrdiff_t previous = -2;

            for (size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
            {
                int fy = buffer.ay[yDst];
                ptrdiff_t sy = buffer.iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(buffer.pbx[0], buffer.pbx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    int* pb = buffer.pbx[k];
                    const uint8_t* ps = src + (sy + k)*srcStride;
                    for (size_t x = 0; x < dstRowSize; x++)
                    {
                        size_t sx = buffer.ix[x];
                        int fx = buffer.ax[x];
                        int t = ps[sx];
                        pb[x] = (t << LINEAR_SHIFT) + (ps[sx + channelCount] - t)*fx;
                    }
                }

                if (fy == 0)
                    for (size_t xDst = 0; xDst < dstRowSize; xDst++)
                        dst[xDst] = ((buffer.pbx[0][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                else if (fy == FRACTION_RANGE)
                    for (size_t xDst = 0; xDst < dstRowSize; xDst++)
                        dst[xDst] = ((buffer.pbx[1][xDst] << LINEAR_SHIFT) + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                else
                {
                    for (size_t xDst = 0; xDst < dstRowSize; xDst++)
                    {
                        int t = buffer.pbx[0][xDst];
                        dst[xDst] = ((t << LINEAR_SHIFT) + (buffer.pbx[1][xDst] - t)*fy + BILINEAR_ROUND_TERM) >> BILINEAR_SHIFT;
                    }
                }
            }
        }
    }
}


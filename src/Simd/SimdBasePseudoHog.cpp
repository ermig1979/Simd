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
#include "Simd/SimdArray.h"

namespace Simd
{
    namespace Base
    {
        template <size_t cell> class PseudoHogHistogramExtractor
        {
            static const size_t Q = 8;

            typedef Array<int> ArrayInt;

            ArrayInt _histogram[2];
            int _k0[cell];
            int _k1[cell];

            void Init(size_t width)
            {
                for (size_t i = 0; i < cell; ++i)
                {
                    _k0[i] = int(cell - i - 1) * 2 + 1;
                    _k1[i] = int(i) * 2 + 1;
                }
                _histogram[0].Resize(width / cell * Q, true);
                _histogram[1].Resize(width / cell * Q, true);
            }

        public:

            void Run(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * histogram, size_t histogramStride)
            {
                assert(cell == 8 || cell == 4);
                assert(width >= cell * 2 && height >= cell * 2);

                Init(width);

                src += (srcStride + 1)*cell / 2;
                height = (height / cell - 1)*cell;
                width = (width / cell - 1)*cell;

                ArrayInt * hist0 = _histogram + 0;
                ArrayInt * hist1 = _histogram + 1;

                float k = 1.0f / 256.0f;
                for (size_t row = 0; row < height; ++row)
                {
                    int * h0 = hist0->data;
                    int * h1 = hist1->data;
                    size_t iy = row&(cell - 1);
                    int ky0 = _k0[iy];
                    int ky1 = _k1[iy];

                    for (size_t col = 0; col < width;)
                    {
                        for (size_t ix = 0; ix < cell; ++ix, ++col)
                        {
                            int dy = src[col + srcStride] - src[col - srcStride];
                            int dx = src[col + 1] - src[col - 1];
                            int adx = Abs(dx);
                            int ady = Abs(dy);
                            int value = RestrictRange(Max(adx, ady) + (Min(adx, ady) + 1) / 2);

                            size_t index = (ady > adx ? 0 : 1);
                            index = (dy > 0 ? index : (Q / 2 - 1) - index);
                            index = (dx > 0 ? index : (Q - 1) - index);

                            h0[0 + index] += value*_k0[ix] * ky0;
                            h1[0 + index] += value*_k0[ix] * ky1;
                            h0[Q + index] += value*_k1[ix] * ky0;
                            h1[Q + index] += value*_k1[ix] * ky1;
                        }
                        h0 += Q;
                        h1 += Q;
                    }

                    if (iy == cell - 1)
                    {
                        for (size_t i = 0; i < hist0->size; ++i)
                            histogram[i] = float(hist0->data[i])*k;
                        hist0->Clear();
                        Swap(hist0, hist1);
                        histogram += histogramStride;
                    }
                    src += srcStride;
                }
            }
        };

        void PseudoHogExtractHistogram8x8x8(const uint8_t * src, size_t srcStride, size_t width, size_t height, float * histogram, size_t histogramStride)
        {
            PseudoHogHistogramExtractor<8> extractor;
            extractor.Run(src, srcStride, width, height, histogram, histogramStride);
        }
    }
}

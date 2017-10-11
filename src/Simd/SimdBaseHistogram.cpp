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
#include "Simd/SimdCompare.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int AbsSecondDerivative(const uint8_t * src, ptrdiff_t step)
        {
            return AbsDifferenceU8(Average(src[step], src[-step]), src[0]);
        }

        void AbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t * histogram)
        {
            assert(width > 2 * indent && height > 2 * indent && indent >= step);

            memset(histogram, 0, sizeof(uint32_t)*HISTOGRAM_SIZE);

            src += indent*(stride + 1);
            height -= 2 * indent;
            width -= 2 * indent;

            size_t rowStep = step*stride;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    const int sdX = AbsSecondDerivative(src + col, step);
                    const int sdY = AbsSecondDerivative(src + col, rowStep);
                    const int sd = MaxU8(sdY, sdX);
                    ++histogram[sd];
                }
                src += stride;
            }
        }

        void Histogram(const uint8_t * src, size_t width, size_t height, size_t stride, uint32_t * histogram)
        {
            uint32_t histograms[4][HISTOGRAM_SIZE];
            memset(histograms, 0, sizeof(uint32_t)*HISTOGRAM_SIZE * 4);
            size_t alignedWidth = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += 4)
                {
                    ++histograms[0][src[col + 0]];
                    ++histograms[1][src[col + 1]];
                    ++histograms[2][src[col + 2]];
                    ++histograms[3][src[col + 3]];
                }
                for (; col < width; ++col)
                    ++histograms[0][src[col + 0]];

                src += stride;
            }

            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                histogram[i] = histograms[0][i] + histograms[1][i] + histograms[2][i] + histograms[3][i];
        }

        void HistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
        {
            uint32_t histograms[4][HISTOGRAM_SIZE + 4];
            memset(histograms, 0, sizeof(uint32_t)*(HISTOGRAM_SIZE + 4) * 4);
            size_t alignedWidth = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += 4)
                {
                    ++histograms[0][(4 + src[col + 0])*(mask[col + 0] == index)];
                    ++histograms[1][(4 + src[col + 1])*(mask[col + 1] == index)];
                    ++histograms[2][(4 + src[col + 2])*(mask[col + 2] == index)];
                    ++histograms[3][(4 + src[col + 3])*(mask[col + 3] == index)];
                }
                for (; col < width; ++col)
                    ++histograms[0][(4 + src[col + 0])*(mask[col + 0] == index)];

                src += srcStride;
                mask += maskStride;
            }
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                histogram[i] = histograms[0][4 + i] + histograms[1][4 + i] + histograms[2][4 + i] + histograms[3][4 + i];
        }

        template <SimdCompareType compareType>
        void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint32_t * histogram)
        {
            uint32_t histograms[4][HISTOGRAM_SIZE + 4];
            memset(histograms, 0, sizeof(uint32_t)*(HISTOGRAM_SIZE + 4) * 4);
            size_t alignedWidth = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += 4)
                {
                    ++histograms[0][(4 + src[col + 0])*Compare8u<compareType>(mask[col + 0], value)];
                    ++histograms[1][(4 + src[col + 1])*Compare8u<compareType>(mask[col + 1], value)];
                    ++histograms[2][(4 + src[col + 2])*Compare8u<compareType>(mask[col + 2], value)];
                    ++histograms[3][(4 + src[col + 3])*Compare8u<compareType>(mask[col + 3], value)];
                }
                for (; col < width; ++col)
                    ++histograms[0][(4 + src[col + 0])*Compare8u<compareType>(mask[col + 0], value)];

                src += srcStride;
                mask += maskStride;
            }
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                histogram[i] = histograms[0][4 + i] + histograms[1][4 + i] + histograms[2][4 + i] + histograms[3][4 + i];
        }

        void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return HistogramConditional<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareNotEqual:
                return HistogramConditional<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreater:
                return HistogramConditional<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreaterOrEqual:
                return HistogramConditional<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesser:
                return HistogramConditional<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesserOrEqual:
                return HistogramConditional<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            default:
                assert(0);
            }
        }

        void NormalizedColors(const uint32_t * histogram, uint8_t * colors)
        {
            uint32_t integral[HISTOGRAM_SIZE], sum = 0, minCount = 0, minColor = 0;
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
            {
                if (sum == 0 && histogram[i] != 0)
                {
                    minCount = histogram[i];
                    minColor = (uint32_t)i;
                }
                sum += histogram[i];
                integral[i] = sum;
            }

            uint32_t norm = sum - minCount, term = (sum - minCount) / 2;
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                colors[i] = i < minColor ? 0 : (norm ? (255 * (integral[i] - minCount) + term) / norm : minColor);
        }

        void ChangeColors(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * colors, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += 4)
                {
                    dst[col + 0] = colors[src[col + 0]];
                    dst[col + 1] = colors[src[col + 1]];
                    dst[col + 2] = colors[src[col + 2]];
                    dst[col + 3] = colors[src[col + 3]];
                }
                for (; col < width; ++col)
                    dst[col] = colors[src[col]];

                src += srcStride;
                dst += dstStride;
            }
        }

        void NormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            uint32_t histogram[HISTOGRAM_SIZE];
            Histogram(src, width, height, srcStride, histogram);

            uint8_t colors[HISTOGRAM_SIZE];
            NormalizedColors(histogram, colors);

            ChangeColors(src, srcStride, width, height, colors, dst, dstStride);
        }
    }
}

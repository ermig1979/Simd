/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"

namespace Simd
{
    namespace Base
    {
        template <SimdCompareType compareType>
        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    dst[col] = Compare8u<compareType>(src[col], value) ? positive : negative;
                src += srcStride;
                dst += dstStride;
            }
        }

        void Binarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride, SimdCompareType compareType)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return Binarization<SimdCompareEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareNotEqual:
                return Binarization<SimdCompareNotEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreater:
                return Binarization<SimdCompareGreater>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return Binarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesser:
                return Binarization<SimdCompareLesser>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return Binarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        template<SimdCompareType compareType> SIMD_INLINE uint32_t GetSa(uint8_t src, uint8_t value)
        {
#ifdef SIMD_BIG_ENDIAN
            return Compare8u<compareType>(src, value) ? 0x00010001 : 0x00000001;
#else
            return Compare8u<compareType>(src, value) ? 0x00010001 : 0x00010000;
#endif
        }

        template <SimdCompareType compareType>
        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative, uint8_t * dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood && neighborhood < 0x80);

            size_t edge = neighborhood + 1;
            Array32u buffer(width + 2 * edge, true);
            uint32_t* sa = buffer.data + edge;

            union SaSum
            {
                uint32_t sum;
                uint16_t sa[2];
            };

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t * s = src + row*srcStride;
                for (size_t col = 0; col < width; ++col)
                {
                    sa[col] += GetSa<compareType>(s[col], value);
                }
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t * s = src + (row + neighborhood)*srcStride;
                    for (size_t col = 0; col < width; ++col)
                    {
                        sa[col] += GetSa<compareType>(s[col], value);
                    }
                }

                if (row > neighborhood)
                {
                    const uint8_t * s = src + (row - neighborhood - 1)*srcStride;
                    for (size_t col = 0; col < width; ++col)
                    {
                        sa[col] -= GetSa<compareType>(s[col], value);
                    }
                }

                SaSum saSum = { 0 };
                for (size_t col = 0; col < neighborhood; ++col)
                    saSum.sum += sa[col];
                for (size_t col = 0; col < width; ++col)
                {
                    saSum.sum += sa[col + neighborhood];
                    saSum.sum -= sa[col - neighborhood - 1];
                    dst[col] = (saSum.sa[0] * 0xFF > threshold*saSum.sa[1]) ? positive : negative;
                }
                dst += dstStride;
            }
        }

        void AveragingBinarization(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t value, size_t neighborhood, uint8_t threshold, uint8_t positive, uint8_t negative,
            uint8_t * dst, size_t dstStride, SimdCompareType compareType)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return AveragingBinarization<SimdCompareEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareNotEqual:
                return AveragingBinarization<SimdCompareNotEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreater:
                return AveragingBinarization<SimdCompareGreater>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return AveragingBinarization<SimdCompareGreaterOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesser:
                return AveragingBinarization<SimdCompareLesser>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return AveragingBinarization<SimdCompareLesserOrEqual>(src, srcStride, width, height, value, neighborhood, threshold, positive, negative, dst, dstStride);
            default:
                assert(0);
            }
        }

        void AveragingBinarizationV2(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t neighborhood, int32_t shift, uint8_t positive, uint8_t negative, uint8_t* dst, size_t dstStride)
        {
            assert(width > neighborhood && height > neighborhood);

            size_t edge = neighborhood + 1, size = width + 2 * edge;
            Array32i buffer(2*size, true);
            int32_t* rs = buffer.data + edge, *ra = rs + size;

            for (size_t row = 0; row < neighborhood; ++row)
            {
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < width; ++col)
                {
                    rs[col] += ps[col];
                    ra[col] += 1;
                }
            }

            for (size_t row = 0; row < height; ++row)
            {
                if (row < height - neighborhood)
                {
                    const uint8_t* ps = src + (row + neighborhood) * srcStride;
                    for (size_t col = 0; col < width; ++col)
                    {
                        rs[col] += ps[col];
                        ra[col] += 1;
                    }
                }

                if (row > neighborhood)
                {
                    const uint8_t* ps = src + (row - neighborhood - 1) * srcStride;
                    for (size_t col = 0; col < width; ++col)
                    {
                        rs[col] -= ps[col];
                        ra[col] -= 1;
                    }
                }

                int sum = 0, area = 0;
                for (size_t col = 0; col < neighborhood; ++col)
                {
                    sum += rs[col];
                    area += ra[col];
                }
                const uint8_t* ps = src + row * srcStride;
                for (size_t col = 0; col < width; ++col)
                {
                    sum += rs[col + neighborhood] - rs[col - neighborhood - 1];
                    area += ra[col + neighborhood] - ra[col - neighborhood - 1];
                    dst[col] = (ps[col] + shift)*area > sum ? positive : negative;
                }
                dst += dstStride;
            }
        }
    }
}

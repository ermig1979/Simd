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
        SIMD_INLINE int TextureBoostedSaturatedGradient(const uint8_t * src, ptrdiff_t step, int saturation, int boost)
        {
            return (saturation + RestrictRange((int)src[step] - (int)src[-step], -saturation, saturation))*boost;
        }

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(int(2)*saturation*boost <= 0xFF);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                dx[0] = 0;
                dy[0] = 0;
                for (size_t col = 1; col < width - 1; ++col)
                {
                    dy[col] = TextureBoostedSaturatedGradient(src + col, srcStride, saturation, boost);
                    dx[col] = TextureBoostedSaturatedGradient(src + col, 1, saturation, boost);
                }
                dx[width - 1] = 0;
                dy[width - 1] = 0;
                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedUv(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t boost, uint8_t * dst, size_t dstStride)
        {
            assert(boost < 128);

            int min = 128 - (128 / boost);
            int max = 255 - min;

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                {
                    dst[col] = (RestrictRange(src[col], min, max) - min)*boost;
                }
                src += srcStride;
                dst += dstStride;
            }
        }

        void TextureGetDifferenceSum(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * lo, size_t loStride, const uint8_t * hi, size_t hiStride, int64_t * sum)
        {
            *sum = 0;
            for (size_t row = 0; row < height; ++row)
            {
                int rowSum = 0;
                for (size_t col = 0; col < width; ++col)
                    rowSum += src[col] - Average(lo[col], hi[col]);
                *sum += rowSum;

                src += srcStride;
                lo += loStride;
                hi += hiStride;
            }
        }

        void TexturePerformCompensation(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            int shift, uint8_t * dst, size_t dstStride)
        {
            assert(shift > -0xFF && shift < 0xFF);

            if (shift == 0)
            {
                if (src != dst)
                    Base::Copy(src, srcStride, width, height, 1, dst, dstStride);
                return;
            }
            else if (shift > 0)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                        dst[col] = Min(src[col] + shift, 0xFF);
                    src += srcStride;
                    dst += dstStride;
                }
            }
            else if (shift < 0)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < width; ++col)
                        dst[col] = Max(src[col] + shift, 0);
                    src += srcStride;
                    dst += dstStride;
                }
            }
        }
    }
}

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
#include "Simd/SimdIntegral.h"

namespace Simd
{
    namespace Base
    {
        void Integral(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t * sum, size_t sumStride, uint8_t * sqsum, size_t sqsumStride, uint8_t * tilted, size_t tiltedStride,
            SimdPixelFormatType sumFormat, SimdPixelFormatType sqsumFormat)
        {
            assert(sumFormat == SimdPixelFormatInt32 && sumStride % sizeof(uint32_t) == 0);
            if (tilted)
                assert(tiltedStride % sizeof(uint32_t) == 0);

            if (sqsum)
            {
                if (tilted)
                {
                    switch (sqsumFormat)
                    {
                    case SimdPixelFormatInt32:
                        IntegralSumSqsumTilted<uint32_t, uint32_t>(src, srcStride, width, height,
                            (uint32_t*)sum, sumStride / sizeof(uint32_t), (uint32_t*)sqsum, sqsumStride / sizeof(uint32_t), (uint32_t*)tilted, tiltedStride / sizeof(uint32_t));
                        break;
                    case SimdPixelFormatDouble:
                        IntegralSumSqsumTilted<uint32_t, double>(src, srcStride, width, height,
                            (uint32_t*)sum, sumStride / sizeof(uint32_t), (double*)sqsum, sqsumStride / sizeof(double), (uint32_t*)tilted, tiltedStride / sizeof(uint32_t));
                        break;
                    default:
                        assert(0);
                    }
                }
                else
                {
                    switch (sqsumFormat)
                    {
                    case SimdPixelFormatInt32:
                        IntegralSumSqsum<uint32_t, uint32_t>(src, srcStride, width, height,
                            (uint32_t*)sum, sumStride / sizeof(uint32_t), (uint32_t*)sqsum, sqsumStride / sizeof(uint32_t));
                        break;
                    case SimdPixelFormatDouble:
                        IntegralSumSqsum<uint32_t, double>(src, srcStride, width, height,
                            (uint32_t*)sum, sumStride / sizeof(uint32_t), (double*)sqsum, sqsumStride / sizeof(double));
                        break;
                    default:
                        assert(0);
                    }
                }
            }
            else
            {
                if (tilted)
                {
                    IntegralSumTilted<uint32_t>(src, srcStride, width, height,
                        (uint32_t*)sum, sumStride / sizeof(uint32_t), (uint32_t*)tilted, tiltedStride / sizeof(uint32_t));
                }
                else
                {
                    IntegralSum<uint32_t>(src, srcStride, width, height, (uint32_t*)sum, sumStride / sizeof(uint32_t));
                }
            }
        }
    }
}

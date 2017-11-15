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
#include "Simd/SimdInit.h"
#include "Simd/SimdIntegral.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        const __m256i K8_SUM_MASK = SIMD_MM256_SETR_EPI8(
            0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x00,
            0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00);

        const __m256i K32_PACK_64_TO_32 = SIMD_MM256_SETR_EPI32(0, 2, 4, 6, 1, 3, 5, 7);

        void IntegralSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint32_t * sum, size_t sumStride)
        {
            memset(sum, 0, (width + 1) * sizeof(uint32_t));
            sum += sumStride + 1;
            size_t alignedWidth = AlignLo(width, 4);

            for (size_t row = 0; row < height; row++)
            {
                sum[-1] = 0;
                size_t col = 0;
                __m256i _rowSums = K_ZERO;
                for (; col < alignedWidth; col += 4)
                {
                    __m256i _src = _mm256_and_si256(_mm256_set1_epi32(*(uint32_t*)(src + col)), K8_SUM_MASK);
                    _rowSums = _mm256_add_epi32(_rowSums, _mm256_sad_epu8(_src, K_ZERO));
                    _mm_storeu_si128((__m128i*)(sum + col), _mm_add_epi32(_mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_rowSums, K32_PACK_64_TO_32)), _mm_loadu_si128((__m128i*)(sum + col - sumStride))));
                    _rowSums = _mm256_permute4x64_epi64(_rowSums, 0xFF);
                }
                uint32_t rowSum = sum[col - 1] - sum[col - sumStride - 1];
                for (; col < width; col++)
                {
                    rowSum += src[col];
                    sum[col] = rowSum + sum[col - sumStride];
                }
                src += srcStride;
                sum += sumStride;
            }
        }

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
                    Avx2::IntegralSum(src, srcStride, width, height, (uint32_t*)sum, sumStride / sizeof(uint32_t));
                }
            }
        }
    }
#endif//SIMD_AVX2_ENABLE
}

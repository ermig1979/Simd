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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class HogLiteFeatureFilter
        {
            template<bool align> SIMD_INLINE void ProductSum1x1(const float * src, const float * filter, __m256 & sum)
            {
                __m256 _src = Load<align>(src);
                __m256 _filter = Load<align>(filter);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(_src, _filter));
            } 

            template<bool align, size_t step> SIMD_INLINE void ProductSum1x4(const float * src, const float * filter, __m256 * sums)
            {
                __m256 _filter = Load<align>(filter);
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(Load<align>(src + 0 * step), _filter));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(Load<align>(src + 1 * step), _filter));
                sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(Load<align>(src + 2 * step), _filter));
                sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(Load<align>(src + 3 * step), _filter));
            }

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize*filterSize;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterSize; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(sums[0], sums[1]), _mm256_hadd_ps(sums[2], sums[3]));
                        _mm_storeu_ps(dst + dstCol, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * pSrc = src + dstRow*srcStride + dstCol*featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterSize; ++filterRow)
                        {
                            for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        dst[dstCol] = Avx::ExtractSum(sum);
                    }
                    dst += dstStride;
                }
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                if(featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterSize, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterSize, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterSize && srcHeight >= filterSize);

                size_t dstWidth = srcWidth - filterSize + 1;
                size_t dstHeight = srcHeight - filterSize + 1;

                if (Aligned(src) && Aligned(srcStride, F) && Aligned(filter))
                    Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterSize, dst, dstStride);
                else
                    Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterSize, dst, dstStride);
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterSize, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterSize, dst, dstStride);
        }
    }
#endif// SIMD_AVX_ENABLE
}



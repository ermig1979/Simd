/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdUpdate.h"

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

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                        {
                            size_t filterCol = 0;
                            for (; filterCol < filterStride; filterCol += F)
                                ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                            pSrc += srcStride;
                            pFilter += filterStride;
                        }
                        _mm_storeu_ps(dst + dstCol, Avx::Extract4Sums(sums));
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                        const float * pFilter = filter;
                        for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
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

            template <bool align, size_t featureSize> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                size_t filterStride = featureSize * filterWidth;
                size_t alignedDstWidth = AlignLo(dstWidth, 4);
                __m128 _min = _mm_set1_ps(-FLT_MAX);
                for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                {
                    size_t dstCol = 0;
                    for (; dstCol < alignedDstWidth; dstCol += 4)
                    {
                        __m128 _mask = _mm_castsi128_ps(_mm_loadu_si128((__m128i*)(mask + dstCol)));
                        if (Sse41::TestZ(_mask))
                            _mm_storeu_ps(dst + dstCol, _min);
                        else
                        {
                            __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                size_t filterCol = 0;
                                for (; filterCol < filterStride; filterCol += F)
                                    ProductSum1x4<align, featureSize>(pSrc + filterCol, pFilter + filterCol, sums);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            _mm_storeu_ps(dst + dstCol, _mm_blendv_ps(_min, Avx::Extract4Sums(sums), _mask));
                        }
                    }
                    for (; dstCol < dstWidth; ++dstCol)
                    {
                        if (mask[dstCol])
                        {
                            __m256 sum = _mm256_setzero_ps();
                            const float * pSrc = src + dstRow * srcStride + dstCol * featureSize;
                            const float * pFilter = filter;
                            for (size_t filterRow = 0; filterRow < filterHeight; ++filterRow)
                            {
                                for (size_t filterCol = 0; filterCol < filterStride; filterCol += F)
                                    ProductSum1x1<align>(pSrc + filterCol, pFilter + filterCol, sum);
                                pSrc += srcStride;
                                pFilter += filterStride;
                            }
                            dst[dstCol] = Avx::ExtractSum(sum);
                        }
                        else
                            dst[dstCol] = -FLT_MAX;
                    }
                    dst += dstStride;
                    mask += maskStride;
                }
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, dst, dstStride);
            }

            template <bool align> void Filter(const float * src, size_t srcStride, size_t dstWidth, size_t dstHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                if (featureSize == 16)
                    Filter<align, 16>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                else
                    Filter<align, 8>(src, srcStride, dstWidth, dstHeight, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= filterWidth && srcHeight >= filterHeight);

                size_t dstWidth = srcWidth - filterWidth + 1;
                size_t dstHeight = srcHeight - filterHeight + 1;

                if (mask)
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
                }
                else
                {
                    if (Aligned(src) && Aligned(srcStride) && Aligned(filter))
                        Filter<true>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                    else
                        Filter<false>(src, srcStride, dstWidth, dstHeight, featureSize, filter, filterWidth, filterHeight, dst, dstStride);
                }
            }
        };

        void HogLiteFilterFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * filter, size_t filterWidth, size_t filterHeight, const uint32_t * mask, size_t maskStride, float * dst, size_t dstStride)
        {
            HogLiteFeatureFilter featureFilter;
            featureFilter.Run(src, srcStride, srcWidth, srcHeight, featureSize, filter, filterWidth, filterHeight, mask, maskStride, dst, dstStride);
        }

        namespace HogLiteFeatureResizerDetail
        {
            template <int size> struct Feature
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst);
            };

            template <> struct Feature<8>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src0 + 0 * F), k[0][0]), _mm256_mul_ps(Load<align>(src0 + 1 * F), k[0][1])),
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src1 + 0 * F), k[1][0]), _mm256_mul_ps(Load<align>(src1 + 1 * F), k[1][1]))));
                }
            };

            template <> struct Feature<16>
            {
                template <bool align> static SIMD_INLINE void Interpolate(const float * src0, const float * src1, const __m256 k[2][2], float * dst)
                {
                    Store<align>(dst + 0 * F, _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src0 + 0 * F), k[0][0]), _mm256_mul_ps(Load<align>(src0 + 2 * F), k[0][1])),
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src1 + 0 * F), k[1][0]), _mm256_mul_ps(Load<align>(src1 + 2 * F), k[1][1]))));
                    Store<align>(dst + 1 * F, _mm256_add_ps(
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src0 + 1 * F), k[0][0]), _mm256_mul_ps(Load<align>(src0 + 3 * F), k[0][1])),
                        _mm256_add_ps(_mm256_mul_ps(Load<align>(src1 + 1 * F), k[1][0]), _mm256_mul_ps(Load<align>(src1 + 3 * F), k[1][1]))));
                }
            };
        }


        class HogLiteFeatureResizer
        {
            typedef Array<int> Ints;
            typedef Array<float> Floats;

            Ints _iy, _ix;
            Floats _ky, _kx;

            void InitIndexWeight(size_t srcSize, size_t dstSize, size_t dstStep, Ints & indexes, Floats & weights)
            {
                indexes.Resize(dstSize);
                weights.Resize(dstSize);

                float scale = float(srcSize) / float(dstSize);
                for (size_t i = 0; i < dstSize; ++i)
                {
                    float weight = (float)((i + 0.5f)*scale - 0.5f);
                    int index = (int)::floor(weight);
                    weight -= index;
                    if (index < 0)
                    {
                        index = 0;
                        weight = 0.0f;
                    }
                    if (index > (int)srcSize - 2)
                    {
                        index = (int)srcSize - 2;
                        weight = 1.0f;
                    }
                    indexes[i] = int(index*dstStep);
                    weights[i] = weight;
                }
            }

            template<bool align, size_t featureSize> void Resize(const float * src, size_t srcStride, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                __m256 _1 = _mm256_set1_ps(1.0f);
                for (size_t rowDst = 0; rowDst < dstHeight; ++rowDst)
                {
                    __m256 ky1 = _mm256_set1_ps(_ky[rowDst]);
                    __m256 ky0 = _mm256_sub_ps(_1, ky1);
                    const float * pSrc = src + _iy[rowDst];
                    float * pDst = dst + rowDst * dstStride;
                    for (size_t colDst = 0; colDst < dstWidth; ++colDst, pDst += featureSize)
                    {
                        __m256 kx1 = _mm256_set1_ps(_kx[colDst]);
                        __m256 kx0 = _mm256_sub_ps(_1, kx1);
                        __m256 k[2][2];
                        k[0][0] = _mm256_mul_ps(ky0, kx0);
                        k[0][1] = _mm256_mul_ps(ky0, kx1);
                        k[1][0] = _mm256_mul_ps(ky1, kx0);
                        k[1][1] = _mm256_mul_ps(ky1, kx1);
                        const float * pSrc0 = pSrc + _ix[colDst];
                        const float * pSrc1 = pSrc0 + srcStride;
                        HogLiteFeatureResizerDetail::Feature<featureSize>:: template Interpolate<align>(pSrc0, pSrc1, k, pDst);
                    }
                }
            }

            template<bool align> void Resize(const float * src, size_t srcStride, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                if (featureSize == 8)
                    Resize<align, 8>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<align, 16>(src, srcStride, dst, dstStride, dstWidth, dstHeight);
            }

        public:
            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
            {
                assert(featureSize == 8 || featureSize == 16);

                if (srcWidth == dstWidth && srcHeight == dstHeight)
                {
                    size_t size = sizeof(float)*srcWidth*featureSize;
                    for (size_t row = 0; row < dstHeight; ++row)
                        memcpy(dst + row * dstStride, src + row * srcStride, size);
                    return;
                }

                InitIndexWeight(srcWidth, dstWidth, featureSize, _ix, _kx);
                InitIndexWeight(srcHeight, dstHeight, srcStride, _iy, _ky);

                if (Aligned(src) && Aligned(dst))
                    Resize<true>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
                else
                    Resize<false>(src, srcStride, featureSize, dst, dstStride, dstWidth, dstHeight);
            }
        };

        void HogLiteResizeFeatures(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, float * dst, size_t dstStride, size_t dstWidth, size_t dstHeight)
        {
            HogLiteFeatureResizer featureResizer;
            featureResizer.Run(src, srcStride, srcWidth, srcHeight, featureSize, dst, dstStride, dstWidth, dstHeight);
        }

        template<bool align> void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                const float * s = src;
                float * d = dst;
                for (size_t col = 0; col < width; ++col)
                {
                    const float * p = pca;
                    for (size_t i = 0; i < 8; i += 4, p += 64)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        for (size_t j = 0; j < 16; j += F)
                        {
                            __m256 _s = Load<align>(s + j);
                            sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(_s, Load<align>(p + j + 00)));
                            sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(_s, Load<align>(p + j + 16)));
                            sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(_s, Load<align>(p + j + 32)));
                            sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(_s, Load<align>(p + j + 48)));
                        }
                        __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(sums[0], sums[1]), _mm256_hadd_ps(sums[2], sums[3]));
                        Sse2::Store<align>(d + i, _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1)));
                    }
                    s += 16;
                    d += 8;
                }
                src += srcStride;
                dst += dstStride;
            }

        }

        void HogLiteCompressFeatures(const float * src, size_t srcStride, size_t width, size_t height, const float * pca, float * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(pca) && Aligned(dst))
                HogLiteCompressFeatures<true>(src, srcStride, width, height, pca, dst, dstStride);
            else
                HogLiteCompressFeatures<false>(src, srcStride, width, height, pca, dst, dstStride);
        }

        class HogLiteSeparableFilter
        {
            size_t _dstWidth, _dstHeight, _dstStride;
            Array32f _buffer;
            Array256f _filter;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstStride = AlignHi(_dstWidth, F);
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstStride*srcHeight);
            }

            template<bool align> static SIMD_INLINE void FilterHx1(const float * src, const float * filter, __m256 & sum)
            {
                __m256 _src = Avx::Load<align>(src);
                __m256 _filter = Avx::Load<align>(filter);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(_src, _filter));
            }

            template<bool align, size_t step> static SIMD_INLINE void FilterHx4(const float * src, const float * filter, __m256 * sums)
            {
                __m256 _filter = Avx::Load<align>(filter);
                sums[0] = _mm256_add_ps(sums[0], _mm256_mul_ps(Avx::Load<align>(src + 0 * step), _filter));
                sums[1] = _mm256_add_ps(sums[1], _mm256_mul_ps(Avx::Load<align>(src + 1 * step), _filter));
                sums[2] = _mm256_add_ps(sums[2], _mm256_mul_ps(Avx::Load<align>(src + 2 * step), _filter));
                sums[3] = _mm256_add_ps(sums[3], _mm256_mul_ps(Avx::Load<align>(src + 3 * step), _filter));
            }

            template <bool align, size_t step> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        __m256 sums[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx4<align, step>(s + i, filter + i, sums);
                        Sse2::Store<true>(dst + col, Avx::Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        __m256 sum = _mm256_setzero_ps();
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx1<align>(s + i, filter + i, sum);
                        dst[col] = Avx::ExtractSum(sum);
                    }
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <bool align> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, size_t step, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (step == 16)
                    FilterH<align, 16>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterH<align, 8>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

            template <bool srcAlign, bool dstAlign, UpdateType update, bool masked> static SIMD_INLINE void FilterV(const float * src, size_t stride, const __m256 * filter, size_t size, float * dst, const __m256 & mask)
            {
                __m256 sum = _mm256_setzero_ps();
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(Load<srcAlign>(src), filter[i]));
                Update<update, dstAlign>(dst, Masked<masked && update != UpdateSet>(sum, mask));
            }

            template <UpdateType update, bool align> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = _mm256_set1_ps(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                __m256 tailMask = RightNotZero32f(width - alignedWidth);

                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += F)
                        FilterV<true, align, update, false>(src + col, srcStride, _filter.data, size, dst + col, tailMask);
                    if (alignedWidth != width)
                        FilterV<false, false, update, true>(src + width - F, srcStride, _filter.data, size, dst + width - F, tailMask);
                    src += srcStride;
                    dst += dstStride;
                }
            }

            template <UpdateType update> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                if (Aligned(dst) && Aligned(dstStride))
                    FilterV<update, true>(src, srcStride, width, height, filter, size, dst, dstStride);
                else
                    FilterV<update, false>(src, srcStride, width, height, filter, size, dst, dstStride);
            }

        public:

            void Run(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
            {
                assert(featureSize == 8 || featureSize == 16);
                assert(srcWidth >= hSize && srcHeight >= vSize);

                Init(srcWidth, srcHeight, hSize, vSize);

                if (Aligned(src) && Aligned(srcStride) && Aligned(hFilter))
                    FilterH<true>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);
                else
                    FilterH<false>(src, srcStride, _dstWidth, srcHeight, featureSize, hFilter, hSize*featureSize, _buffer.data, _dstStride);

                if (add)
                    FilterV<UpdateAdd>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
                else
                    FilterV<UpdateSet>(_buffer.data, _dstStride, _dstWidth, _dstHeight, vFilter, vSize, dst, dstStride);
            }
        };

        void HogLiteFilterSeparable(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, size_t featureSize, const float * hFilter, size_t hSize, const float * vFilter, size_t vSize, float * dst, size_t dstStride, int add)
        {
            HogLiteSeparableFilter filter;
            filter.Run(src, srcStride, srcWidth, srcHeight, featureSize, hFilter, hSize, vFilter, vSize, dst, dstStride, add);
        }
    }
#endif// SIMD_AVX_ENABLE
}



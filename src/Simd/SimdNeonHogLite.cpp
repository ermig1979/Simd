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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdUpdate.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
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
                        float32x4_t sums[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
                        for (size_t j = 0; j < 16; j += F)
                        {
                            float32x4_t _s = Load<align>(s + j);
                            sums[0] = vmlaq_f32(sums[0], _s, Load<align>(p + j + 00));
                            sums[1] = vmlaq_f32(sums[1], _s, Load<align>(p + j + 16));
                            sums[2] = vmlaq_f32(sums[2], _s, Load<align>(p + j + 32));
                            sums[3] = vmlaq_f32(sums[3], _s, Load<align>(p + j + 48));
                        }
                        Store<align>(d + i, Extract4Sums(sums));
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
            Array128f _filter;

            void Init(size_t srcWidth, size_t srcHeight, size_t hSize, size_t vSize)
            {
                _dstWidth = srcWidth - hSize + 1;
                _dstStride = AlignHi(_dstWidth, F);
                _dstHeight = srcHeight - vSize + 1;
                _buffer.Resize(_dstStride*srcHeight);
            }

            template<bool align> static SIMD_INLINE void FilterHx1(const float * src, const float * filter, float32x4_t & sum)
            {
                float32x4_t _src = Load<align>(src);
                float32x4_t _filter = Load<align>(filter);
                sum = vmlaq_f32(sum, _src, _filter);
            }

            template<bool align, size_t step> static SIMD_INLINE void FilterHx4(const float * src, const float * filter, float32x4_t * sums)
            {
                float32x4_t _filter = Load<align>(filter);
                sums[0] = vmlaq_f32(sums[0], Load<align>(src + 0 * step), _filter);
                sums[1] = vmlaq_f32(sums[1], Load<align>(src + 1 * step), _filter);
                sums[2] = vmlaq_f32(sums[2], Load<align>(src + 2 * step), _filter);
                sums[3] = vmlaq_f32(sums[3], Load<align>(src + 3 * step), _filter);
            }

            template <bool align, size_t step> void FilterH(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                size_t alignedWidth = AlignLo(width, 4);
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += 4)
                    {
                        float32x4_t sums[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx4<align, step>(s + i, filter + i, sums);
                        Store<true>(dst + col, Extract4Sums(sums));
                    }
                    for (; col < width; ++col)
                    {
                        float32x4_t sum = vdupq_n_f32(0);
                        const float * s = src + col * step;
                        for (size_t i = 0; i < size; i += F)
                            FilterHx1<align>(s + i, filter + i, sum);
                        dst[col] = ExtractSum32f(sum);
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

            template <bool srcAlign, bool dstAlign, UpdateType update, bool masked> static SIMD_INLINE void FilterV(const float * src, size_t stride, const float32x4_t * filter, size_t size, float * dst, const float32x4_t & mask)
            {
                float32x4_t sum = vdupq_n_f32(0);
                for (size_t i = 0; i < size; ++i, src += stride)
                    sum = vmlaq_f32(sum, Load<srcAlign>(src), filter[i]);
                Update<update, dstAlign>(dst, Masked<masked && update != UpdateSet>(sum, mask));
            }

            template <UpdateType update, bool align> void FilterV(const float * src, size_t srcStride, size_t width, size_t height, const float * filter, size_t size, float * dst, size_t dstStride)
            {
                _filter.Resize(size);
                for (size_t i = 0; i < size; ++i)
                    _filter[i] = vdupq_n_f32(filter[i]);

                size_t alignedWidth = AlignLo(width, F);
                float32x4_t tailMask = RightNotZero(width - alignedWidth);

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

        void HogLiteFindMax7x7(const float * a, size_t aStride, const float * b, size_t bStride, size_t height, float * pValue, size_t * pCol, size_t * pRow)
        {
            float32x4_t max = vdupq_n_f32(-FLT_MAX), val;
            uint32x4_t idx = vdupq_n_u32(0);
            uint32x4_t cur = K32_0123;
            for (size_t row = 0; row < height; ++row)
            {
                val = vaddq_f32(Load<false>(a + 0), Load<false>(b + 0));
                max = vmaxq_f32(max, val);
                idx = vbslq_u32(vceqq_f32(max, val), cur, idx);
                cur = vaddq_u32(cur, K32_00000003);
                val = vaddq_f32(Load<false>(a + 3), Load<false>(b + 3));
                max = vmaxq_f32(max, val);
                idx = vbslq_u32(vceqq_f32(max, val), cur, idx);
                cur = vaddq_u32(cur, K32_00000005);
                a += aStride;
                b += bStride;
            }

            uint32_t _idx[F];
            float _max[F];
            Store<false>(_max, max);
            Store<false>(_idx, idx);
            *pValue = -FLT_MAX;
            for (size_t i = 0; i < F; ++i)
            {
                if (_max[i] > *pValue)
                {
                    *pValue = _max[i];
                    *pCol = _idx[i]&7;
                    *pRow = _idx[i]/8;
                }
            }
        }

        SIMD_INLINE void Fill7x7(uint32_t * dst, size_t stride)
        {
            for (size_t row = 0; row < 7; ++row)
            {
                Store<false>(dst + 0, K32_FFFFFFFF);
                Store<false>(dst + 3, K32_FFFFFFFF);
                dst += stride;
            }
        }

        template <size_t scale> void HogLiteCreateMask7x7(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, uint32_t * dst, size_t dstStride)
        {
            size_t dstStartEnd = 7 - scale;
            size_t dstRowSize = (srcWidth*scale + 7 - scale) * sizeof(uint32_t);
            for (size_t dstRow = 0; dstRow < dstStartEnd; ++dstRow)
                memset(dst + dstRow * dstStride, 0, dstRowSize);

            size_t alignedSrcWidth = AlignLo(srcWidth, F);
            float32x4_t _threshold = vdupq_n_f32(*threshold);
            for (size_t srcRow = 0; srcRow < srcHeight; ++srcRow)
            {
                for (size_t dstRow = 0; dstRow < scale; ++dstRow)
                    memset(dst + (dstStartEnd + dstRow)*dstStride, 0, dstRowSize);

                size_t srcCol = 0;
                for (; srcCol < alignedSrcWidth; srcCol += F)
                {
                    uint32x4_t mask = vcgtq_f32(Load<false>(src + srcCol), _threshold);
                    uint32_t * pDst = dst + srcCol * scale;
                    if (vgetq_lane_u32(mask, 0))
                        Fill7x7(pDst + 0 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 1))
                        Fill7x7(pDst + 1 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 2))
                        Fill7x7(pDst + 2 * scale, dstStride);
                    if (vgetq_lane_u32(mask, 3))
                        Fill7x7(pDst + 3 * scale, dstStride);
                }
                for (; srcCol < srcWidth; ++srcCol)
                {
                    if (src[srcCol] > *threshold)
                        Fill7x7(dst + srcCol * scale, dstStride);
                }
                src += srcStride;
                dst += dstStride * scale;
            }
        }

        void HogLiteCreateMask(const float * src, size_t srcStride, size_t srcWidth, size_t srcHeight, const float * threshold, size_t scale, size_t size, uint32_t * dst, size_t dstStride)
        {
            if (scale == 1 && size == 7)
                HogLiteCreateMask7x7<1>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else if (scale == 2 && size == 7)
                HogLiteCreateMask7x7<2>(src, srcStride, srcWidth, srcHeight, threshold, dst, dstStride);
            else
                Base::HogLiteCreateMask(src, srcStride, srcWidth, srcHeight, threshold, scale, size, dst, dstStride);
        }
    }
#endif// SIMD_NEON_ENABLE
}

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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
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

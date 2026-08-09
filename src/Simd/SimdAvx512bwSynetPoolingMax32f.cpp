/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        void SynetPoolingMax32f2DNhwcSolid2x2(const float* src0, size_t srcC, float* dst0, size_t dstH, size_t dstW)
        {
            size_t dstH2 = AlignLo(dstH, 2);
            size_t dstR = srcC * dstW, srcR = dstR + srcC, dstRF = AlignLo(dstR, F);
            __mmask16 tailR = TailMask16(dstR - dstRF);
            __m512 min = _mm512_set1_ps(-FLT_MAX);
            size_t h = 0;
            for (; h < dstH2; h += 2)
            {
                const float* src1 = src0 + srcR;
                const float* src2 = src1 + srcR;
                float* dst1 = dst0 + dstR;
                size_t r = 0;
                for (size_t r = 0; r < dstRF; r += F)
                {
                    __m512 max0 = _mm512_max_ps(_mm512_loadu_ps(src0 + r), _mm512_loadu_ps(src0 + r + srcC));
                    __m512 max1 = _mm512_max_ps(_mm512_loadu_ps(src1 + r), _mm512_loadu_ps(src1 + r + srcC));
                    __m512 max2 = _mm512_max_ps(_mm512_loadu_ps(src2 + r), _mm512_loadu_ps(src2 + r + srcC));
                    _mm512_storeu_ps(dst0 + r, _mm512_max_ps(max0, max1));
                    _mm512_storeu_ps(dst1 + r, _mm512_max_ps(max1, max2));
                }
                if (tailR)
                {
                    __m512 max0 = _mm512_max_ps(_mm512_maskz_loadu_ps(tailR, src0 + r), _mm512_maskz_loadu_ps(tailR, src0 + r + srcC));
                    __m512 max1 = _mm512_max_ps(_mm512_maskz_loadu_ps(tailR, src1 + r), _mm512_maskz_loadu_ps(tailR, src1 + r + srcC));
                    __m512 max2 = _mm512_max_ps(_mm512_maskz_loadu_ps(tailR, src2 + r), _mm512_maskz_loadu_ps(tailR, src2 + r + srcC));
                    _mm512_mask_storeu_ps(dst0 + r, tailR, _mm512_max_ps(max0, max1));
                    _mm512_mask_storeu_ps(dst1 + r, tailR, _mm512_max_ps(max1, max2));
                }
                src0 += 2 * srcR;
                dst0 += 2 * dstR;
            }
            for (; h < dstH; ++h)
            {
                const float* src1 = src0 + srcR;
                size_t r = 0;
                for (size_t r = 0; r < dstRF; r += F)
                {
                    __m512 max0 = _mm512_max_ps(_mm512_loadu_ps(src0 + r), _mm512_loadu_ps(src0 + r + srcC));
                    __m512 max1 = _mm512_max_ps(_mm512_loadu_ps(src1 + r), _mm512_loadu_ps(src1 + r + srcC));
                    _mm512_storeu_ps(dst0 + r, _mm512_max_ps(max0, max1));
                }
                if (tailR)
                {
                    __m512 max0 = _mm512_max_ps(_mm512_maskz_loadu_ps(tailR, src0 + r), _mm512_maskz_loadu_ps(tailR, src0 + r + srcC));
                    __m512 max1 = _mm512_max_ps(_mm512_maskz_loadu_ps(tailR, src1 + r), _mm512_maskz_loadu_ps(tailR, src1 + r + srcC));
                    _mm512_mask_storeu_ps(dst0 + r, tailR, _mm512_max_ps(max0, max1));
                }
                src0 += srcR;
                dst0 += dstR;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void PoolingMax32f2DHwc1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst, __mmask16 tail = -1)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_maskz_loadu_ps(tail, src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm512_mask_storeu_ps(dst + 0 * F, tail, max0);
        }

        SIMD_INLINE void PoolingMax32f2DHwc2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
        }

        SIMD_INLINE void PoolingMax32f2DHwc4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
            _mm512_storeu_ps(dst + 2 * F, max2);
            _mm512_storeu_ps(dst + 3 * F, max3);
        }

        SIMD_INLINE void PoolingMax32f2DHwc8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, max0);
            _mm512_storeu_ps(dst + 1 * F, max1);
            _mm512_storeu_ps(dst + 2 * F, max2);
            _mm512_storeu_ps(dst + 3 * F, max3);
            _mm512_storeu_ps(dst + 4 * F, max4);
            _mm512_storeu_ps(dst + 5 * F, max5);
            _mm512_storeu_ps(dst + 6 * F, max6);
            _mm512_storeu_ps(dst + 7 * F, max7);
        }


        SIMD_INLINE __m512 SynetPoolingMax32fNchw_1x1Max3x1Body(const float* src)
        {
            return _mm512_max_ps(_mm512_max_ps(Load<false>(src - 1), Load<false>(src)), Load<false>(src + 1));
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x3Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Body(src - stride);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Body(src);
            __m512 src2 = SynetPoolingMax32fNchw_1x1Max3x1Body(src + stride);
            Store<false>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x2Body(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Body(src);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Body(src + stride);
            Store<false>(dst, _mm512_max_ps(src0, src1));
        }

        const __m512i K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_NOSE = SIMD_MM512_SETR_EPI32(0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_1x1Max3x1Nose(const float* src)
        {
            __m512 src1 = Load<false>(src);
            __m512 src0 = _mm512_permutexvar_ps(K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_NOSE, src1);
            __m512 src2 = Load<false>(src + 1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x3Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Nose(src - stride);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Nose(src);
            __m512 src2 = SynetPoolingMax32fNchw_1x1Max3x1Nose(src + stride);
            Store<false>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x2Nose(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Nose(src);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Nose(src + stride);
            Store<false>(dst, _mm512_max_ps(src0, src1));
        }

        const __m512i K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_TAIL = SIMD_MM512_SETR_EPI32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 15);

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_1x1Max3x1Tail(const float* src)
        {
            __m512 src0 = Load<false>(src - 1);
            __m512 src1 = Load<false>(src);
            __m512 src2 = _mm512_permutexvar_ps(K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_TAIL, src1);
            return _mm512_max_ps(_mm512_max_ps(src0, src1), src2);
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x3Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Tail(src - stride);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Tail(src);
            __m512 src2 = SynetPoolingMax32fNchw_1x1Max3x1Tail(src + stride);
            Store<false>(dst, _mm512_max_ps(_mm512_max_ps(src0, src1), src2));
        }

        SIMD_INLINE void SynetPoolingMax32fNchw_1x1Max3x2Tail(const float* src, size_t stride, float* dst)
        {
            __m512 src0 = SynetPoolingMax32fNchw_1x1Max3x1Tail(src);
            __m512 src1 = SynetPoolingMax32fNchw_1x1Max3x1Tail(src + stride);
            Store<false>(dst, _mm512_max_ps(src0, src1));
        }

        void SynetPoolingMax32fNchw_1x1Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            assert(width > F && height > 1);

            size_t alignedWidth = AlignHi(width, F) - F;
            height -= 1;

            SynetPoolingMax32fNchw_1x1Max3x2Nose(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                SynetPoolingMax32fNchw_1x1Max3x2Body(src + col, srcStride, dst + col);
            SynetPoolingMax32fNchw_1x1Max3x2Tail(src + width - F, srcStride, dst + width - F);

            for (size_t row = 1; row < height; ++row)
            {
                src += srcStride;
                dst += dstStride;
                SynetPoolingMax32fNchw_1x1Max3x3Nose(src, srcStride, dst);
                for (size_t col = F; col < alignedWidth; col += F)
                    SynetPoolingMax32fNchw_1x1Max3x3Body(src + col, srcStride, dst + col);
                SynetPoolingMax32fNchw_1x1Max3x3Tail(src + width - F, srcStride, dst + width - F);
            }

            dst += dstStride;
            SynetPoolingMax32fNchw_1x1Max3x2Nose(src, srcStride, dst);
            for (size_t col = F; col < alignedWidth; col += F)
                SynetPoolingMax32fNchw_1x1Max3x2Body(src + col, srcStride, dst + col);
            SynetPoolingMax32fNchw_1x1Max3x2Tail(src + width - F, srcStride, dst + width - F);
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_0 = SIMD_MM512_SETR_EPI32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
        const __m512i K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_1 = SIMD_MM512_SETR_EPI32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
        const __m512i K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_2 = SIMD_MM512_SETR_EPI32(2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 0);

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max2x2Body(const float* src, size_t stride)
        {
            __m512 lo = _mm512_max_ps(Load<false>(src + 0), Load<false>(src + stride + 0));
            __m512 hi = _mm512_max_ps(Load<false>(src + F), Load<false>(src + stride + F));
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max2Body(const float* src)
        {
            __m512 lo = Load<false>(src + 0);
            __m512 hi = Load<false>(src + F);
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        void SynetPoolingMax32fNchw_2x2Max2x2(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t alignedWidth = AlignLo(width, DF);
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<false>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max2x2Body(src + col, srcStride));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max2x2Body(src + col, srcStride));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = Simd::Max(src[widthEven], src[widthEven + srcStride]);
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += DF)
                    Store<false>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max2Body(src + col));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - DF;
                    Store<false>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max2Body(src + col));
                }
                if (width - widthEven)
                    dst[widthEven >> 1] = src[widthEven];
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max1x3(const float* src, size_t stride)
        {
            return _mm512_max_ps(_mm512_max_ps(Load<false>(src), Load<false>(src + stride)), Load<false>(src + 2 * stride));
        }

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max3x3Body(const float* src, size_t stride)
        {
            __m512 s0 = SynetPoolingMax32fNchw_2x2Max1x3(src + 0, stride);
            __m512 sf = SynetPoolingMax32fNchw_2x2Max1x3(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max1x2(const float* src, size_t stride)
        {
            return _mm512_max_ps(Load<false>(src), Load<false>(src + stride));
        }

        SIMD_INLINE __m512 SynetPoolingMax32fNchw_2x2Max3x2(const float* src, size_t stride)
        {
            __m512 s0 = SynetPoolingMax32fNchw_2x2Max1x2(src + 0, stride);
            __m512 sf = SynetPoolingMax32fNchw_2x2Max1x2(src + F, stride);
            __m512 p0 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_0, sf);
            __m512 p1 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_1, sf);
            __m512 p2 = _mm512_permutex2var_ps(s0, K32_SYNET_POOLING_MAX_32F_NCHW_PERMUTE_2_2, sf);
            return _mm512_max_ps(_mm512_max_ps(p0, p1), p2);
        }

        void SynetPoolingMax32fNchw_2x2Max3x3(const float* src, size_t srcStride, size_t width, size_t height, float* dst, size_t dstStride)
        {
            height -= 1;
            width -= 1;
            size_t heightEven = Simd::AlignLo(height, 2);
            size_t widthEven = Simd::AlignLo(width, 2);
            size_t step = DF - 2;
            size_t alignedWidth = width / step * step;
            for (size_t row = 0; row < heightEven; row += 2)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max3x3Body(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max3x3Body(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x3s(src + widthEven, srcStride, dst + (widthEven >> 1));
                src += 2 * srcStride;
                dst += dstStride;
            }
            if (height - heightEven)
            {
                for (size_t col = 0; col < alignedWidth; col += step)
                    Store<false, true>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max3x2(src + col, srcStride), __mmask16(0x7FFF));
                if (widthEven - alignedWidth)
                {
                    size_t col = widthEven - step;
                    Store<false, true>(dst + (col >> 1), SynetPoolingMax32fNchw_2x2Max3x2(src + col, srcStride), __mmask16(0x7FFF));
                }
                if (width - widthEven)
                    Sse41::Max2x2s(src + widthEven, srcStride, dst + (widthEven >> 1));
            }
        }

        //-----------------------------------------------------------------------------------------

        void SynetPoolingMax32f2D(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (padY == 0 && padX == 0 && strideX == 1 && strideY == 1 && dstH == srcH + 1 - kernelY && dstW == srcW + 1 - kernelX && kernelX == 2 && kernelY == 2 && 1)
                {
                    SynetPoolingMax32f2DNhwcSolid2x2(src, srcC, dst, dstH, dstW);
                }
                else
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    __m512 min = _mm512_set1_ps(-FLT_MAX);
                    __mmask16 tail = TailMask16(srcC - srcCF1);
                    for (size_t ph = 0; ph < dstH; ++ph)
                    {
                        size_t hStart = ph * strideY - padY;
                        size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                        hStart = Simd::Max<ptrdiff_t>(0, hStart);
                        for (size_t pw = 0; pw < dstW; ++pw)
                        {
                            size_t wStart = pw * strideX - padX;
                            size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                            wStart = Simd::Max<ptrdiff_t>(0, wStart);
                            const float* ps = src + hStart * srcS + wStart * srcC;
                            size_t c = 0;
                            for (; c < srcCF8; c += 8 * F)
                                PoolingMax32f2DHwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF4; c += 4 * F)
                                PoolingMax32f2DHwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF2; c += 2 * F)
                                PoolingMax32f2DHwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF1; c += 1 * F)
                                PoolingMax32f2DHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            if (c < srcC)
                                PoolingMax32f2DHwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c, tail);
                            dst += srcC;
                        }
                    }
                }
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (strideY == 1 && strideX == 1 && kernelY == 3 && kernelX == 3 && srcH == dstH && srcW == dstW && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        SynetPoolingMax32fNchw_1x1Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 2 && kernelX == 2 && padY == 0 && padX == 0 && dstW >= F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        SynetPoolingMax32fNchw_2x2Max2x2(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                if (strideY == 2 && strideX == 2 && kernelY == 3 && kernelX == 3 && padY == 0 && padX == 0 && dstW > F)
                {
                    for (size_t c = 0; c < srcC; ++c, src += srcH * srcW, dst += dstH * dstW)
                        SynetPoolingMax32fNchw_2x2Max3x3(src, srcW, srcW, srcH, dst, dstW);
                    return;
                }
                Avx2::SynetPoolingMax32f(src, srcC, srcH, srcW, 1, kernelY, kernelX, 1, strideY, strideX, 0, padY, padX, dst, srcC, dstH, dstW, format);
            }
            else
                assert(0);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m512 MaxNhwcCr2(__m512 lo, __m512 hi)
        {
            __m512 _lo = _mm512_shuffle_f32x4(lo, hi, 0x88);
            __m512 _hi = _mm512_shuffle_f32x4(lo, hi, 0xDD);
            return _mm512_max_ps(_mm512_shuffle_ps(_lo, _hi, 0x88), _mm512_shuffle_ps(_lo, _hi, 0xDD));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst, _mm512_castps512_ps256(MaxNhwcCr2(max0, _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst, MaxNhwcCr2(max0, max1));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(max0, max1));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(max2, max3));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr2_8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(max0, max1));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(max2, max3));
            _mm512_storeu_ps(dst + 2 * F, MaxNhwcCr2(max4, max5));
            _mm512_storeu_ps(dst + 3 * F, MaxNhwcCr2(max6, max7));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_1(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                }
                src += srcS;
            }
            _mm_storeu_ps(dst, _mm512_castps512_ps128(MaxNhwcCr2(MaxNhwcCr2(max0, _mm512_setzero_ps()), _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_2(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                }
                src += srcS;
            }
            _mm256_storeu_ps(dst, _mm512_castps512_ps256(MaxNhwcCr2(MaxNhwcCr2(max0, max1), _mm512_setzero_ps())));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_4(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(MaxNhwcCr2(max0, max1), MaxNhwcCr2(max2, max3)));
        }

        SIMD_INLINE void PoolingMax32f3DNhwcCr4_8(const float* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const __m512& min, float* dst)
        {
            __m512 max0 = min;
            __m512 max1 = min;
            __m512 max2 = min;
            __m512 max3 = min;
            __m512 max4 = min;
            __m512 max5 = min;
            __m512 max6 = min;
            __m512 max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    max0 = _mm512_max_ps(max0, _mm512_loadu_ps(src + w * srcC + 0 * F));
                    max1 = _mm512_max_ps(max1, _mm512_loadu_ps(src + w * srcC + 1 * F));
                    max2 = _mm512_max_ps(max2, _mm512_loadu_ps(src + w * srcC + 2 * F));
                    max3 = _mm512_max_ps(max3, _mm512_loadu_ps(src + w * srcC + 3 * F));
                    max4 = _mm512_max_ps(max4, _mm512_loadu_ps(src + w * srcC + 4 * F));
                    max5 = _mm512_max_ps(max5, _mm512_loadu_ps(src + w * srcC + 5 * F));
                    max6 = _mm512_max_ps(max6, _mm512_loadu_ps(src + w * srcC + 6 * F));
                    max7 = _mm512_max_ps(max7, _mm512_loadu_ps(src + w * srcC + 7 * F));
                }
                src += srcS;
            }
            _mm512_storeu_ps(dst + 0 * F, MaxNhwcCr2(MaxNhwcCr2(max0, max1), MaxNhwcCr2(max2, max3)));
            _mm512_storeu_ps(dst + 1 * F, MaxNhwcCr2(MaxNhwcCr2(max4, max5), MaxNhwcCr2(max6, max7)));
        }

        void SynetPoolingMax32f3D(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc && srcC >= F)
            {
                if (kernelC == 2 && strideC == 2 && padC == 0 && srcC == dstC * 2)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcC16 = AlignLo(srcC, 16);
                    size_t srcC32 = AlignLo(srcC, 32);
                    size_t srcC64 = AlignLo(srcC, 64);
                    size_t srcC128 = AlignLo(srcC, 128);
                    __m512 min = _mm512_set1_ps(-FLT_MAX);
                    for (size_t dh = 0; dh < dstH; ++dh)
                    {
                        size_t hBeg = dh * strideY - padY;
                        size_t hEnd = Simd::Min(hBeg + kernelY, srcH);
                        hBeg = Simd::Max<ptrdiff_t>(0, hBeg);
                        for (size_t dw = 0; dw < dstW; ++dw)
                        {
                            size_t wBeg = dw * strideX - padX;
                            size_t wEnd = Simd::Min(wBeg + kernelX, srcW);
                            wBeg = Simd::Max<ptrdiff_t>(0, wBeg);
                            const float* ps = src + hBeg * srcS + wBeg * srcC;
                            size_t c = 0, d = 0;
                            for (; c < srcC128; c += 128, d += 64)
                                PoolingMax32f3DNhwcCr2_8(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC64; c += 64, d += 32)
                                PoolingMax32f3DNhwcCr2_4(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC32; c += 32, d += 16)
                                PoolingMax32f3DNhwcCr2_2(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC16; c += 16, d += 8)
                                PoolingMax32f3DNhwcCr2_1(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            if (c < srcC)
                                PoolingMax32f3DNhwcCr2_1(ps + srcC - 16, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + dstC - 8);
                            dst += dstC;
                        }
                    }
                    return;
                }
                if (kernelC == 4 && strideC == 4 && padC == 0 && srcC == dstC * 4)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcC16 = AlignLo(srcC, 16);
                    size_t srcC32 = AlignLo(srcC, 32);
                    size_t srcC64 = AlignLo(srcC, 64);
                    size_t srcC128 = AlignLo(srcC, 128);
                    __m512 min = _mm512_set1_ps(-FLT_MAX);
                    for (size_t dh = 0; dh < dstH; ++dh)
                    {
                        size_t hBeg = dh * strideY - padY;
                        size_t hEnd = Simd::Min(hBeg + kernelY, srcH);
                        hBeg = Simd::Max<ptrdiff_t>(0, hBeg);
                        for (size_t dw = 0; dw < dstW; ++dw)
                        {
                            size_t wBeg = dw * strideX - padX;
                            size_t wEnd = Simd::Min(wBeg + kernelX, srcW);
                            wBeg = Simd::Max<ptrdiff_t>(0, wBeg);
                            const float* ps = src + hBeg * srcS + wBeg * srcC;
                            size_t c = 0, d = 0;
                            for (; c < srcC128; c += 128, d += 32)
                                PoolingMax32f3DNhwcCr4_8(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC64; c += 64, d += 16)
                                PoolingMax32f3DNhwcCr4_4(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC32; c += 32, d += 8)
                                PoolingMax32f3DNhwcCr4_2(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            for (; c < srcC16; c += 16, d += 4)
                                PoolingMax32f3DNhwcCr4_1(ps + c, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + d);
                            if (c < srcC)
                                PoolingMax32f3DNhwcCr4_1(ps + srcC - 16, srcS, srcC, hEnd - hBeg, wEnd - wBeg, min, dst + dstC - 4);
                            dst += dstC;
                        }
                    }
                    return;
                }
            }
            Avx2::SynetPoolingMax32f(src, srcC, srcH, srcW, kernelC, kernelY, kernelX,
                strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }

        //-----------------------------------------------------------------------------------------

        void SynetPoolingMax32f(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (kernelC == 1 && strideC == 1 && padC == 0 && srcC == dstC)
                SynetPoolingMax32f2D(src, srcC, srcH, srcW, kernelY, kernelX,
                    strideY, strideX, padY, padX, dst, dstH, dstW, format);
            else
                SynetPoolingMax32f3D(src, srcC, srcH, srcW, kernelC, kernelY, kernelX,
                    strideC, strideY, strideX, padC, padY, padX, dst, dstC, dstH, dstW, format);
        }
    }
#endif
}

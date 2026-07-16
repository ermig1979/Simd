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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE int32x4_t QuantizeSumLinear(int32x4_t sum, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero)
        {
            return vaddq_s32(Round(vmulq_f32(vcvtq_f32_s32(vaddq_s32(sum, bias)), norm)), zero);
        }

        SIMD_INLINE void QuantizeSumLinear4(int32x4_t sum, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero, uint8_t* dst)
        {
            int32x4_t d0 = QuantizeSumLinear(sum, bias, norm, zero);
            uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vdup_n_s16(0)));
            vst1_lane_u32((uint32_t*)dst, vreinterpret_u32_u8(u8), 0);
        }

        SIMD_INLINE void QuantizeSumLinear16(int32x4_t sum0, int32x4_t sum1, int32x4_t sum2, int32x4_t sum3,
            const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero, uint8_t* dst)
        {
            int32x4_t d0 = QuantizeSumLinear(sum0, bias, norm, zero);
            int32x4_t d1 = QuantizeSumLinear(sum1, bias, norm, zero);
            int32x4_t d2 = QuantizeSumLinear(sum2, bias, norm, zero);
            int32x4_t d3 = QuantizeSumLinear(sum3, bias, norm, zero);
            vst1q_u8(dst, vcombine_u8(
                vqmovun_s16(vcombine_s16(vqmovn_s32(d0), vqmovn_s32(d1))),
                vqmovun_s16(vcombine_s16(vqmovn_s32(d2), vqmovn_s32(d3)))));
        }

        SIMD_INLINE int32x4_t Load4uAs32s(const uint8_t* src)
        {
            uint8x8_t s8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src));
            return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(s8))));
        }

        SIMD_INLINE int32_t Sum8u(const uint8_t* src, size_t size)
        {
            size_t i = 0, sizeQA = AlignLo(size, QA), sizeA = AlignLo(size, A);
            uint32x4_t sums[4] = { vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0), vdupq_n_u32(0) };
            for (; i < sizeQA; i += QA)
            {
                sums[0] = vaddq_u32(sums[0], vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + i + 0 * A))));
                sums[1] = vaddq_u32(sums[1], vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + i + 1 * A))));
                sums[2] = vaddq_u32(sums[2], vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + i + 2 * A))));
                sums[3] = vaddq_u32(sums[3], vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + i + 3 * A))));
            }
            sums[0] = vaddq_u32(vaddq_u32(sums[0], sums[1]), vaddq_u32(sums[2], sums[3]));
            for (; i < sizeA; i += A)
                sums[0] = vaddq_u32(sums[0], vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + i))));
            uint32_t buf[4];
            vst1q_u32(buf, sums[0]);
            int32_t sum = (int32_t)(buf[0] + buf[1] + buf[2] + buf[3]);
            for (; i < size; ++i)
                sum += src[i];
            return sum;
        }

        SIMD_INLINE void QuantizedPoolingAverageNhwc(const uint8_t* src, size_t srcS, size_t srcC, size_t srcCF4, size_t srcCF16,
            size_t kH, size_t kW, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero, uint8_t* dst)
        {
            size_t c = 0;
            for (; c < srcCF16; c += A)
            {
                int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0), sum2 = vdupq_n_s32(0), sum3 = vdupq_n_s32(0);
                for (size_t h = 0; h < kH; ++h)
                {
                    for (size_t w = 0; w < kW; ++w)
                    {
                        const uint8_t* ps = src + h * srcS + w * srcC + c;
                        sum0 = vaddq_s32(sum0, Load4uAs32s(ps + 0 * F));
                        sum1 = vaddq_s32(sum1, Load4uAs32s(ps + 1 * F));
                        sum2 = vaddq_s32(sum2, Load4uAs32s(ps + 2 * F));
                        sum3 = vaddq_s32(sum3, Load4uAs32s(ps + 3 * F));
                    }
                }
                QuantizeSumLinear16(sum0, sum1, sum2, sum3, bias, norm, zero, dst + c);
            }
            for (; c < srcCF4; c += F)
            {
                int32x4_t sum = vdupq_n_s32(0);
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                    {
                        uint8x8_t s8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)(src + h * srcS + w * srcC + c)));
                        sum = vaddq_s32(sum, vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(s8)))));
                    }
                QuantizeSumLinear4(sum, bias, norm, zero, dst + c);
            }
            for (; c < srcC; ++c)
            {
                int32_t sum = 0;
                for (size_t h = 0; h < kH; ++h)
                    for (size_t w = 0; w < kW; ++w)
                        sum += src[h * srcS + w * srcC + c];
                dst[c] = (uint8_t)Base::QuantizeSumLinear(sum, vgetq_lane_s32(bias, 0), vgetq_lane_f32(norm, 0), vgetq_lane_s32(zero, 0), 0, 255);
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNhwc(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            int32x4_t _bias = vdupq_n_s32(bias), _zero = vdupq_n_s32(dstZero);
            float32x4_t _norm = vdupq_n_f32(srcScale[0] / (dstScale[0] * float(spatial)));
            size_t channels4 = AlignLo(channels, F), channels16 = AlignLo(channels, A);
            Array32u sum(channels);
            for (size_t b = 0; b < batch; ++b)
            {
                GetColSums(src, channels, channels, spatial, sum.data);
                size_t c = 0;
                uint32_t* sums = sum.data;
                for (; c < channels16; c += A, sums += A)
                    QuantizeSumLinear16(vreinterpretq_s32_u32(vld1q_u32(sums + 0 * F)), vreinterpretq_s32_u32(vld1q_u32(sums + 1 * F)),
                        vreinterpretq_s32_u32(vld1q_u32(sums + 2 * F)), vreinterpretq_s32_u32(vld1q_u32(sums + 3 * F)), _bias, _norm, _zero, dst + c);
                for (; c < channels4; c += F, sums += F)
                    QuantizeSumLinear4(vreinterpretq_s32_u32(vld1q_u32(sums)), _bias, _norm, _zero, dst + c);
                for (; c < channels; ++c)
                {
                    int32_t sum = 0;
                    for (size_t s = 0; s < spatial; ++s)
                        sum += src[s * channels + c];
                    dst[c] = (uint8_t)Base::QuantizeSumLinear(sum, bias, vgetq_lane_f32(_norm, 0), dstZero, 0, 255);
                }
                src += spatial * channels;
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageGlobalNchw(const uint8_t* src, int srcZero, const float* srcScale,
            size_t batch, size_t channels, size_t spatial, uint8_t* dst, const float* dstScale, int dstZero)
        {
            int32_t bias = -srcZero * int32_t(spatial);
            const int32x4_t _bias = vdupq_n_s32(bias), _zero = vdupq_n_s32(dstZero);
            const float32x4_t _norm = vdupq_n_f32(srcScale[0] / (dstScale[0] * float(spatial)));
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    uint8x8_t u8 = vqmovun_s16(vcombine_s16(vqmovn_s32(QuantizeSumLinear(vdupq_n_s32(Sum8u(src, spatial)), _bias, _norm, _zero)), vdup_n_s16(0)));
                    dst[c] = vget_lane_u8(u8, 0);
                    src += spatial;
                }
                dst += channels;
            }
        }

        SIMD_INLINE void QuantizedPoolingAverageNchw2x2(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW,
            uint8_t* dst, size_t dstH, size_t dstW, const int32x4_t& bias, const float32x4_t& norm, const int32x4_t& zero)
        {
            size_t dstWF = AlignLo(dstW, F);
            for (size_t b = 0; b < srcC; ++b)
            {
                for (size_t dy = 0; dy < dstH; ++dy)
                {
                    size_t dx = 0, sx = 0;
                    const uint8_t* src0 = src + dy * 2 * srcW;
                    const uint8_t* src1 = src0 + srcW;
                    for (; dx < dstWF; dx += F, sx += DF)
                    {
                        uint16x8_t s0 = vmovl_u8(vld1_u8(src0 + sx));
                        uint16x8_t s1 = vmovl_u8(vld1_u8(src1 + sx));
                        int32x4_t sum = vreinterpretq_s32_u32(vpaddlq_u16(vaddq_u16(s0, s1)));
                        QuantizeSumLinear4(sum, bias, norm, zero, dst + dx);
                    }
                    for (; dx < dstW; ++dx, sx += 2)
                    {
                        int32_t sum = src0[sx] + src0[sx + 1] + src1[sx] + src1[sx + 1];
                        dst[dx] = (uint8_t)Base::QuantizeSumLinear(sum, vgetq_lane_s32(bias, 0), vgetq_lane_f32(norm, 0), vgetq_lane_s32(zero, 0), 0, 255);
                    }
                    dst += dstW;
                }
                src += srcH * srcW;
            }
        }

        void SynetQuantizedPoolingAverage(const uint8_t* src, const float* srcScale, int srcZero, size_t batch, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, SimdBool excludePad,
            uint8_t* dst, const float* dstScale, int dstZero, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc && srcC >= F)
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNhwc(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                size_t srcS = srcW * srcC, srcCF4 = AlignLo(srcC, F), srcCF16 = AlignLo(srcC, A);
                int32x4_t zero = vdupq_n_s32(dstZero);
                int32_t bias = -srcZero * int32_t(kernelY * kernelX);
                float norm = srcScale[0] / (dstScale[0] * float(kernelY * kernelX));
                for (size_t b = 0; b < batch; ++b)
                {
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
                            const uint8_t* ps = src + hStart * srcS + wStart * srcC;
                            if (excludePad)
                            {
                                int area = int(hEnd - hStart) * int(wEnd - wStart);
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF4, srcCF16, hEnd - hStart, wEnd - wStart,
                                    vdupq_n_s32(-srcZero * area), vdupq_n_f32(srcScale[0] / (dstScale[0] * float(area))), zero, dst);
                            }
                            else
                                QuantizedPoolingAverageNhwc(ps, srcS, srcC, srcCF4, srcCF16, hEnd - hStart, wEnd - wStart,
                                    vdupq_n_s32(bias), vdupq_n_f32(norm), zero, dst);
                            dst += srcC;
                        }
                    }
                    src += srcC * srcH * srcW;
                }
                return;
            }
            else if (format == SimdTensorFormatNchw)
            {
                if (kernelY == srcH && kernelX == srcW && strideY == 1 && strideX == 1 && padY == 0 && padX == 0)
                {
                    QuantizedPoolingAverageGlobalNchw(src, srcZero, srcScale, batch, srcC, srcH * srcW, dst, dstScale, dstZero);
                    return;
                }
                if (kernelY == 2 && kernelX == 2 && strideY == 2 && strideX == 2 && padY == 0 && padX == 0 && dstH * 2 <= srcH && dstW * 2 <= srcW)
                {
                    QuantizedPoolingAverageNchw2x2(src, srcC * batch, srcH, srcW, dst, dstH, dstW, vdupq_n_s32(-srcZero * 4),
                        vdupq_n_f32(srcScale[0] / (dstScale[0] * 4.0f)), vdupq_n_s32(dstZero));
                    return;
                }
            }
            Base::SynetQuantizedPoolingAverage(src, srcScale, srcZero, batch, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX,
                padY, padX, excludePad, dst, dstScale, dstZero, dstH, dstW, format);
        }
    }
#endif
}

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
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t LoadBFloat16(const uint16_t* src, const svbool_t& mask)
        {
            return svreinterpret_f32_u32(svlsl_n_u32_x(mask, svld1uh_u32(mask, src), Base::Bf16::SHIFT));
        }

        SIMD_INLINE svuint32_t Float32ToBFloat16(svfloat32_t value, const svbool_t& mask)
        {
            svuint32_t bits = svreinterpret_u32_f32(value);
            svuint32_t round = svadd_n_u32_x(mask, svand_n_u32_x(mask, svlsr_n_u32_x(mask, bits, Base::Bf16::SHIFT), 1), Base::Bf16::ROUND);
            return svlsr_n_u32_x(mask, svadd_u32_x(mask, bits, round), Base::Bf16::SHIFT);
        }

        SIMD_INLINE void StoreBFloat16(svfloat32_t value, uint16_t* dst, const svbool_t& mask)
        {
            svst1h_u32(mask, dst, Float32ToBFloat16(value, mask));
        }

        SIMD_INLINE void PoolingMax16bNhwc1(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, uint16_t* dst, const svbool_t& mask)
        {
            svfloat32_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                    max0 = svmax_f32_x(mask, max0, LoadBFloat16(src + w * srcC, mask));
                src += srcS;
            }
            StoreBFloat16(max0, dst, mask);
        }

        SIMD_INLINE void PoolingMax16bNhwc2(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, uint16_t* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, LoadBFloat16(ps + 0 * F, mask));
                    max1 = svmax_f32_x(mask, max1, LoadBFloat16(ps + 1 * F, mask));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F, mask);
            StoreBFloat16(max1, dst + 1 * F, mask);
        }

        SIMD_INLINE void PoolingMax16bNhwc4(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, uint16_t* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            svfloat32_t max2 = min;
            svfloat32_t max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, LoadBFloat16(ps + 0 * F, mask));
                    max1 = svmax_f32_x(mask, max1, LoadBFloat16(ps + 1 * F, mask));
                    max2 = svmax_f32_x(mask, max2, LoadBFloat16(ps + 2 * F, mask));
                    max3 = svmax_f32_x(mask, max3, LoadBFloat16(ps + 3 * F, mask));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F, mask);
            StoreBFloat16(max1, dst + 1 * F, mask);
            StoreBFloat16(max2, dst + 2 * F, mask);
            StoreBFloat16(max3, dst + 3 * F, mask);
        }

        SIMD_INLINE void PoolingMax16bNhwc8(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const svfloat32_t& min, uint16_t* dst, const svbool_t& mask)
        {
            const size_t F = svcntw();
            svfloat32_t max0 = min;
            svfloat32_t max1 = min;
            svfloat32_t max2 = min;
            svfloat32_t max3 = min;
            svfloat32_t max4 = min;
            svfloat32_t max5 = min;
            svfloat32_t max6 = min;
            svfloat32_t max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = svmax_f32_x(mask, max0, LoadBFloat16(ps + 0 * F, mask));
                    max1 = svmax_f32_x(mask, max1, LoadBFloat16(ps + 1 * F, mask));
                    max2 = svmax_f32_x(mask, max2, LoadBFloat16(ps + 2 * F, mask));
                    max3 = svmax_f32_x(mask, max3, LoadBFloat16(ps + 3 * F, mask));
                    max4 = svmax_f32_x(mask, max4, LoadBFloat16(ps + 4 * F, mask));
                    max5 = svmax_f32_x(mask, max5, LoadBFloat16(ps + 5 * F, mask));
                    max6 = svmax_f32_x(mask, max6, LoadBFloat16(ps + 6 * F, mask));
                    max7 = svmax_f32_x(mask, max7, LoadBFloat16(ps + 7 * F, mask));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F, mask);
            StoreBFloat16(max1, dst + 1 * F, mask);
            StoreBFloat16(max2, dst + 2 * F, mask);
            StoreBFloat16(max3, dst + 3 * F, mask);
            StoreBFloat16(max4, dst + 4 * F, mask);
            StoreBFloat16(max5, dst + 5 * F, mask);
            StoreBFloat16(max6, dst + 6 * F, mask);
            StoreBFloat16(max7, dst + 7 * F, mask);
        }

        SIMD_INLINE void PoolingMax16bNhwc(const uint16_t* src, size_t srcS, size_t srcC, size_t srcCF1,
            size_t srcCF2, size_t srcCF4, size_t srcCF8, size_t kernelY, size_t kernelX, const svfloat32_t& min, uint16_t* dst, const svbool_t& body)
        {
            const size_t F = svcntw();
            size_t c = 0;
            for (; c < srcCF8; c += 8 * F)
                PoolingMax16bNhwc8(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF4; c += 4 * F)
                PoolingMax16bNhwc4(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF2; c += 2 * F)
                PoolingMax16bNhwc2(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            for (; c < srcCF1; c += 1 * F)
                PoolingMax16bNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, body);
            if (c < srcC)
                PoolingMax16bNhwc1(src + c, srcS, srcC, kernelY, kernelX, min, dst + c, svwhilelt_b32(c, srcC));
        }

        void SynetPoolingMax16b(const uint16_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint16_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            const size_t F = svcntw();
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    svbool_t body = svptrue_b32();
                    svfloat32_t min = svdup_n_f32(-FLT_MAX);
                    if (padX == 0 && padY == 0 && (dstW - 1) * strideX + kernelX == srcW && (dstH - 1) * strideY + kernelY == srcH)
                    {
                        size_t stepY = srcW * srcC * strideY, stepX = strideX * srcC;
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            const uint16_t* ps = src + ph * stepY;
                            for (size_t pw = 0; pw < dstW; ++pw, ps += stepX, dst += srcC)
                                PoolingMax16bNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kernelY, kernelX, min, dst, body);
                        }
                    }
                    else
                    {
                        for (size_t ph = 0; ph < dstH; ++ph)
                        {
                            size_t hStart = ph * strideY - padY;
                            size_t hEnd = Simd::Min(hStart + kernelY, srcH);
                            hStart = Simd::Max<ptrdiff_t>(0, hStart);
                            size_t kH = hEnd - hStart;
                            for (size_t pw = 0; pw < dstW; ++pw)
                            {
                                size_t wStart = pw * strideX - padX;
                                size_t wEnd = Simd::Min(wStart + kernelX, srcW);
                                wStart = Simd::Max<ptrdiff_t>(0, wStart);
                                size_t kW = wEnd - wStart;
                                const uint16_t* ps = src + hStart * srcS + wStart * srcC;
                                PoolingMax16bNhwc(ps, srcS, srcC, srcCF1, srcCF2, srcCF4, srcCF8, kH, kW, min, dst, body);
                                dst += srcC;
                            }
                        }
                    }
                    return;
                }
            }
            Base::SynetPoolingMax16b(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
        }
    }
#endif
}

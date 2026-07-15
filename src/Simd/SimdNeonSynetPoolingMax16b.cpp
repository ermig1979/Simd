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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE float32x4_t LoadBFloat16(const uint16_t* src)
        {
            return BFloat16ToFloat32(vmovl_u16(vld1_u16(src)));
        }

        SIMD_INLINE void StoreBFloat16(const float32x4_t& value, uint16_t* dst)
        {
            vst1_u16(dst, vmovn_u32(Float32ToBFloat16(value)));
        }

        SIMD_INLINE void PoolingMax16bNhwc1(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& min, uint16_t* dst)
        {
            float32x4_t max0 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                    max0 = vmaxq_f32(max0, LoadBFloat16(src + w * srcC));
                src += srcS;
            }
            StoreBFloat16(max0, dst);
        }

        SIMD_INLINE void PoolingMax16bNhwc2(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& min, uint16_t* dst)
        {
            float32x4_t max0 = min, max1 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = vmaxq_f32(max0, LoadBFloat16(ps + 0 * F));
                    max1 = vmaxq_f32(max1, LoadBFloat16(ps + 1 * F));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F);
            StoreBFloat16(max1, dst + 1 * F);
        }

        SIMD_INLINE void PoolingMax16bNhwc4(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& min, uint16_t* dst)
        {
            float32x4_t max0 = min, max1 = min;
            float32x4_t max2 = min, max3 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = vmaxq_f32(max0, LoadBFloat16(ps + 0 * F));
                    max1 = vmaxq_f32(max1, LoadBFloat16(ps + 1 * F));
                    max2 = vmaxq_f32(max2, LoadBFloat16(ps + 2 * F));
                    max3 = vmaxq_f32(max3, LoadBFloat16(ps + 3 * F));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F);
            StoreBFloat16(max1, dst + 1 * F);
            StoreBFloat16(max2, dst + 2 * F);
            StoreBFloat16(max3, dst + 3 * F);
        }

        SIMD_INLINE void PoolingMax16bNhwc8(const uint16_t* src, size_t srcS, size_t srcC, size_t kH, size_t kW, const float32x4_t& min, uint16_t* dst)
        {
            float32x4_t max0 = min, max1 = min;
            float32x4_t max2 = min, max3 = min;
            float32x4_t max4 = min, max5 = min;
            float32x4_t max6 = min, max7 = min;
            for (size_t h = 0; h < kH; ++h)
            {
                for (size_t w = 0; w < kW; ++w)
                {
                    const uint16_t* ps = src + w * srcC;
                    max0 = vmaxq_f32(max0, LoadBFloat16(ps + 0 * F));
                    max1 = vmaxq_f32(max1, LoadBFloat16(ps + 1 * F));
                    max2 = vmaxq_f32(max2, LoadBFloat16(ps + 2 * F));
                    max3 = vmaxq_f32(max3, LoadBFloat16(ps + 3 * F));
                    max4 = vmaxq_f32(max4, LoadBFloat16(ps + 4 * F));
                    max5 = vmaxq_f32(max5, LoadBFloat16(ps + 5 * F));
                    max6 = vmaxq_f32(max6, LoadBFloat16(ps + 6 * F));
                    max7 = vmaxq_f32(max7, LoadBFloat16(ps + 7 * F));
                }
                src += srcS;
            }
            StoreBFloat16(max0, dst + 0 * F);
            StoreBFloat16(max1, dst + 1 * F);
            StoreBFloat16(max2, dst + 2 * F);
            StoreBFloat16(max3, dst + 3 * F);
            StoreBFloat16(max4, dst + 4 * F);
            StoreBFloat16(max5, dst + 5 * F);
            StoreBFloat16(max6, dst + 6 * F);
            StoreBFloat16(max7, dst + 7 * F);
        }

        void SynetPoolingMax16b(const uint16_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint16_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format)
        {
            if (format == SimdTensorFormatNhwc)
            {
                if (srcC >= F)
                {
                    size_t srcS = srcW * srcC;
                    size_t srcCF1 = AlignLo(srcC, 1 * F);
                    size_t srcCF2 = AlignLo(srcC, 2 * F);
                    size_t srcCF4 = AlignLo(srcC, 4 * F);
                    size_t srcCF8 = AlignLo(srcC, 8 * F);
                    float32x4_t min = vdupq_n_f32(-FLT_MAX);
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
                            const uint16_t* ps = src + hStart * srcS + wStart * srcC;
                            size_t c = 0;
                            for (; c < srcCF8; c += 8 * F)
                                PoolingMax16bNhwc8(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF4; c += 4 * F)
                                PoolingMax16bNhwc4(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF2; c += 2 * F)
                                PoolingMax16bNhwc2(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            for (; c < srcCF1; c += 1 * F)
                                PoolingMax16bNhwc1(ps + c, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + c);
                            if (c < srcC)
                                PoolingMax16bNhwc1(ps + srcC - F, srcS, srcC, hEnd - hStart, wEnd - wStart, min, dst + srcC - F);
                            dst += srcC;
                        }
                    }
                }
                else
                    Base::SynetPoolingMax16b(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else if (format == SimdTensorFormatNchw)
            {
                Base::SynetPoolingMax16b(src, srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dst, dstH, dstW, format);
            }
            else
                assert(0);
        }
    }
#endif
}

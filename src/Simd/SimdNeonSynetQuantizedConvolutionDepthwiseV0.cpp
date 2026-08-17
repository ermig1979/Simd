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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizedDepthwise.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE void Cvt8uTo32i(uint8x16_t src, int32x4_t& d0, int32x4_t& d1, int32x4_t& d2, int32x4_t& d3)
        {
            uint16x8_t lo = vmovl_u8(vget_low_u8(src));
            uint16x8_t hi = vmovl_u8(vget_high_u8(src));
            d0 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(lo)));
            d1 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(lo)));
            d2 = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(hi)));
            d3 = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(hi)));
        }

        SIMD_INLINE void Cvt8iTo32i(int8x16_t src, int32x4_t& d0, int32x4_t& d1, int32x4_t& d2, int32x4_t& d3)
        {
            int16x8_t lo = vmovl_s8(vget_low_s8(src));
            int16x8_t hi = vmovl_s8(vget_high_s8(src));
            d0 = vmovl_s16(vget_low_s16(lo));
            d1 = vmovl_s16(vget_high_s16(lo));
            d2 = vmovl_s16(vget_low_s16(hi));
            d3 = vmovl_s16(vget_high_s16(hi));
        }

        SIMD_INLINE int32x4_t LoadAs32i(const uint8_t* src)
        {
            uint8x8_t u8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src));
            return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(u8))));
        }

        SIMD_INLINE int32x4_t LoadAs32i(const int8_t* src)
        {
            int8x8_t i8 = vreinterpret_s8_s32(vdup_n_s32(*(const int32_t*)src));
            return vmovl_s16(vget_low_s16(vmovl_s8(i8)));
        }

        SIMD_INLINE void Madd1(int32x4_t& i32, int32x4_t u8, int32x4_t i8)
        {
            i32 = vmlaq_s32(i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_Default(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            int32x4_t _srcZero = vdupq_n_s32(srcZero);
            int32x4_t _dstZero = vdupq_n_s32(dstZero);
            int32x4_t d00, d01, d02, d03, w00, w01, w02, w03, s00, s01, s02, s03;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeA = AlignLo(size, A);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeA; i += A)
                    {
                        d00 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        d02 = vdupq_n_s32(0);
                        d03 = vdupq_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                int8x16_t w0 = vld1q_s8(weight + (ky * p.kernelX + kx) * size + i);
                                Cvt8iTo32i(w0, w00, w01, w02, w03);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    uint8x16_t s0 = vld1q_u8(src + (sy * p.srcW + sx) * size + i);
                                    Cvt8uTo32i(s0, s00, s01, s02, s03);
                                    Madd1(d00, s00, w00);
                                    Madd1(d01, s01, w01);
                                    Madd1(d02, s02, w02);
                                    Madd1(d03, s03, w03);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w00);
                                    Madd1(d01, _srcZero, w01);
                                    Madd1(d02, _srcZero, w02);
                                    Madd1(d03, _srcZero, w03);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                        Save1<term>(dst, d02, bias, norm, _dstZero, i + F * 2);
                        Save1<term>(dst, d03, bias, norm, _dstZero, i + F * 3);
                    }
                    for (; i < size; i += F)
                    {
                        size_t ci = i >= sizeF ? size - F : i;
                        d00 = vdupq_n_s32(0);
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                int32x4_t w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + ci);
                                int32x4_t s0;
                                if (sy < p.srcH && sx < p.srcW)
                                    s0 = LoadAs32i(src + (sy * p.srcW + sx) * size + ci);
                                else
                                    s0 = _srcZero;
                                Madd1(d00, s0, w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, ci);
                    }
                    dst += p.dstC * a.srcE;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Edge(
            const uint8_t* src, const int32x4_t& srcZero, const ConvParam& p, const AlgParamV0& a, size_t dy, size_t dx,
            const int8_t* weight, const int32_t* bias, const float* norm, const int32x4_t& dstZero, uint8_t* dst)
        {
            int32x4_t d00, d01, d02, d03, w00, w01, w02, w03, s00, s01, s02, s03;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                d00 = vdupq_n_s32(0);
                d01 = vdupq_n_s32(0);
                d02 = vdupq_n_s32(0);
                d03 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        int8x16_t w0 = vld1q_s8(weight + (ky * p.kernelX + kx) * size + i);
                        Cvt8iTo32i(w0, w00, w01, w02, w03);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            uint8x16_t s0 = vld1q_u8(src + (sy * p.srcW + sx) * size + i);
                            Cvt8uTo32i(s0, s00, s01, s02, s03);
                            Madd1(d00, s00, w00);
                            Madd1(d01, s01, w01);
                            Madd1(d02, s02, w02);
                            Madd1(d03, s03, w03);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w00);
                            Madd1(d01, srcZero, w01);
                            Madd1(d02, srcZero, w02);
                            Madd1(d03, srcZero, w03);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, i + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, i + F * 3);
            }
            for (; i < size; i += F)
            {
                size_t ci = i >= sizeF ? size - F : i;
                d00 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        int32x4_t w0 = LoadAs32i(weight + (ky * 3 + kx) * size + ci);
                        int32x4_t s0;
                        if (sy < p.srcH && sx < p.srcW)
                            s0 = LoadAs32i(src + (sy * p.srcW + sx) * size + ci);
                        else
                            s0 = srcZero;
                        Madd1(d00, s0, w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ci);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main1(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const int32x4_t& dstZero, uint8_t* dst)
        {
            int32x4_t d00, d01, d02, d03, w00, w01, w02, w03, s00, s01, s02, s03;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = vdupq_n_s32(0);
                d01 = vdupq_n_s32(0);
                d02 = vdupq_n_s32(0);
                d03 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        int8x16_t w0 = vld1q_s8(pw + kx * srcC);
                        uint8x16_t s0 = vld1q_u8(ps + kx * srcC);
                        Cvt8iTo32i(w0, w00, w01, w02, w03);
                        Cvt8uTo32i(s0, s00, s01, s02, s03);
                        Madd1(d00, s00, w00);
                        Madd1(d01, s01, w01);
                        Madd1(d02, s02, w02);
                        Madd1(d03, s03, w03);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + ct;
                    const int8_t* pw = weight + ky * 3 * srcC + ct;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        int32x4_t w0 = LoadAs32i(pw + kx * srcC);
                        int32x4_t s0 = LoadAs32i(ps + kx * srcC);
                        Madd1(d00, s0, w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ct);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main2(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const int32x4_t& dstZero, uint8_t* dst)
        {
            int32x4_t d00, d01, d02, d03, d10, d11, d12, d13, w00, w01, w02, w03, s00, s01, s02, s03, s10, s11, s12, s13;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t srcX = srcC * p.strideX;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = vdupq_n_s32(0);
                d01 = vdupq_n_s32(0);
                d02 = vdupq_n_s32(0);
                d03 = vdupq_n_s32(0);
                d10 = vdupq_n_s32(0);
                d11 = vdupq_n_s32(0);
                d12 = vdupq_n_s32(0);
                d13 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        int8x16_t w0 = vld1q_s8(pw + kx * srcC);
                        uint8x16_t s0 = vld1q_u8(ps + kx * srcC);
                        uint8x16_t s1 = vld1q_u8(ps + kx * srcC + srcX);
                        Cvt8iTo32i(w0, w00, w01, w02, w03);
                        Cvt8uTo32i(s0, s00, s01, s02, s03);
                        Cvt8uTo32i(s1, s10, s11, s12, s13);
                        Madd1(d00, s00, w00);
                        Madd1(d10, s10, w00);
                        Madd1(d01, s01, w01);
                        Madd1(d11, s11, w01);
                        Madd1(d02, s02, w02);
                        Madd1(d12, s12, w02);
                        Madd1(d03, s03, w03);
                        Madd1(d13, s13, w03);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst + srcC, d11, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst + srcC, d12, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst + srcC, d13, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = vdupq_n_s32(0);
                d10 = vdupq_n_s32(0);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + ct;
                    const int8_t* pw = weight + ky * 3 * srcC + ct;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        int32x4_t w0 = LoadAs32i(pw + kx * srcC);
                        int32x4_t s0 = LoadAs32i(ps + kx * srcC);
                        int32x4_t s1 = LoadAs32i(ps + kx * srcC + srcX);
                        Madd1(d00, s0, w0);
                        Madd1(d10, s1, w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ct);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, ct);
            }
        }

        template<Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_3x3(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            int32x4_t _srcZero = vdupq_n_s32(srcZero);
            int32x4_t _dstZero = vdupq_n_s32(dstZero);
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstC = p.dstC * a.dstE;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW2; dx += 2)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Main2<term>(src + offset, p, a, weight, bias, norm, _dstZero, dst), dst += dstC * 2, offset += srcX * 2;
                for (; dx < dstW; dx += 1)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Main1<term>(src + offset, p, a, weight, bias, norm, _dstZero, dst), dst += dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    QuantizedConvolutionNhwcDepthwiseV0_3x3Edge<term>(src, _srcZero, p, a, dy, dx, weight, bias, norm, _dstZero, dst), dst += dstC;
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void SetV0(const ConvParam& p, SynetQuantizedConvolutionNhwcDepthwiseV0::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1))
                convolution = QuantizedConvolutionNhwcDepthwiseV0_3x3<term>;
            else
                convolution = QuantizedConvolutionNhwcDepthwiseV0_Default<term>;
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV0::SynetQuantizedConvolutionNhwcDepthwiseV0(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV0(p)
        {
            if (p.dstT == SimdTensorData8u)
                SetV0<Term8iLast8u>(p, _convolution);
        }
    }
#endif
}

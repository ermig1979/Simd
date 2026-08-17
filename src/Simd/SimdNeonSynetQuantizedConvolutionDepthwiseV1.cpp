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
        using AlgParamV1 = SynetQuantizedConvolutionNhwcDepthwiseV1::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE int32x4_t LoadAs32i(const uint8_t* src)
        {
            uint8x8_t u8 = vreinterpret_u8_u32(vdup_n_u32(*(const uint32_t*)src));
            return vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(u8))));
        }

        SIMD_INLINE void Madd1(int32x4_t& i32, int32x4_t u8, int32x4_t i8)
        {
            i32 = vmlaq_s32(i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcDepthwiseV1_Preprocess(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParamV1& a, size_t dyBeg, size_t dyEnd, int32_t* dst)
        {
            int32x4_t _zero = vdupq_n_s32(zero);
            size_t srcC = p.srcC, byMask = a.bufH - 1;
            size_t byPad = p.kernelY - 1, srcR = p.srcW * p.srcC, bufR = a.bufW * a.bufC;
            size_t byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.reorderType == 0)
            {
                size_t bxPad = p.padX * a.bufC, bwPad = p.padW * a.bufC;
                for (size_t by = byBeg; by < byEnd; ++by)
                {
                    int32_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    if (sy < p.srcH)
                    {
                        const uint8_t* ps = src + sy * srcR;
                        if (bxPad)
                        {
                            for (size_t i = 0; i < bxPad; i += F)
                                vst1q_s32(pd + i, _zero);
                            pd += bxPad;
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++)
                        {
                            size_t sc = 0;
                            for (; sc < srcC; sc += F)
                                vst1q_s32(pd + sc, LoadAs32i(ps + sc));
                            ps += p.srcC;
                            pd += a.bufC;
                        }
                        if (bwPad)
                        {
                            for (size_t i = 0; i < bwPad; i += F)
                                vst1q_s32(pd + i, _zero);
                            pd += bwPad;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bufR; i += F)
                            vst1q_s32(pd + i, _zero);
                    }
                }
            }
            else
            {
                size_t bW = a.bufW, bC = a.bufC, xPad = p.padX, wPad = p.padW;
                for (size_t by = byBeg; by < byEnd; ++by)
                {
                    int32_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    if (sy < p.srcH)
                    {
                        const uint8_t* ps = src + sy * srcR;
                        if (xPad)
                        {
                            for (size_t x = 0; x < xPad; x += 1, pd += a.F)
                                for (size_t c = 0; c < bC; c += a.F)
                                    vst1q_s32(pd + c * bW, _zero);
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++, pd += a.F)
                        {
                            for (size_t sc = 0; sc < bC; sc += F)
                                vst1q_s32(pd + sc * bW, LoadAs32i(ps + sc));
                            ps += p.srcC;
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 1, pd += a.F)
                                for (size_t c = 0; c < bC; c += a.F)
                                    vst1q_s32(pd + c * bW, _zero);
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bufR; i += F)
                            vst1q_s32(pd + i, _zero);
                    }
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV1_AnyR0(const int32_t* src, const ConvParam& p, const AlgParamV1& a,
            const int32_t* weight, const int32_t* bias, const float* norm, size_t dyBeg, size_t dyEnd, uint32_t zero, uint8_t* dst)
        {
            int32x4_t _zero = vdupq_n_s32(zero);
            int32x4_t d00, d01, d02, d03, d10, d11, d12, d13, w0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), srcCF4 = AlignLo(srcC, F * 4), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX;
            size_t byMask = a.bufH - 1, bufC = a.bufC, bufR = a.bufW * a.bufC, dstW2 = AlignLo(p.dstW, 2), dD = p.dstC * a.srcE, dX = sX * bufC;
            dst += dyBeg * p.dstW * p.dstC * a.srcE;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dstW2; dx += 2)
                {
                    const int32_t* ps00 = src + (dx + 0) * sX * bufC;
                    size_t sc = 0;
                    for (; sc < srcCF4; sc += F * 4)
                    {
                        d00 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        d02 = vdupq_n_s32(0);
                        d03 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        d11 = vdupq_n_s32(0);
                        d12 = vdupq_n_s32(0);
                        d13 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw + F * 0);
                                Madd1(d00, vld1q_s32(ps0 + F * 0), w0);
                                Madd1(d10, vld1q_s32(ps1 + F * 0), w0);
                                w0 = vld1q_s32(pw + F * 1);
                                Madd1(d01, vld1q_s32(ps0 + F * 1), w0);
                                Madd1(d11, vld1q_s32(ps1 + F * 1), w0);
                                w0 = vld1q_s32(pw + F * 2);
                                Madd1(d02, vld1q_s32(ps0 + F * 2), w0);
                                Madd1(d12, vld1q_s32(ps1 + F * 2), w0);
                                w0 = vld1q_s32(pw + F * 3);
                                Madd1(d03, vld1q_s32(ps0 + F * 3), w0);
                                Madd1(d13, vld1q_s32(ps1 + F * 3), w0);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                        Save2<term>(dst, dst + dD, d01, d11, bias, norm, _zero, sc + F * 1);
                        Save2<term>(dst, dst + dD, d02, d12, bias, norm, _zero, sc + F * 2);
                        Save2<term>(dst, dst + dD, d03, d13, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps0), w0);
                                Madd1(d10, vld1q_s32(ps1), w0);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps0), w0);
                                Madd1(d10, vld1q_s32(ps1), w0);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0, srcC - srcCF);
                    }
                    dst += 2 * dD;
                }
                for (; dx < p.dstW; ++dx)
                {
                    const int32_t* ps0 = src + dx * sX * bufC;
                    size_t sc = 0;
                    for (; sc < srcCF4; sc += F * 4)
                    {
                        d00 = vdupq_n_s32(0);
                        d01 = vdupq_n_s32(0);
                        d02 = vdupq_n_s32(0);
                        d03 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw + F * 0);
                                Madd1(d00, vld1q_s32(ps + F * 0), w0);
                                w0 = vld1q_s32(pw + F * 1);
                                Madd1(d01, vld1q_s32(ps + F * 1), w0);
                                w0 = vld1q_s32(pw + F * 2);
                                Madd1(d02, vld1q_s32(ps + F * 2), w0);
                                w0 = vld1q_s32(pw + F * 3);
                                Madd1(d03, vld1q_s32(ps + F * 3), w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc + F * 0);
                        Save1<term>(dst, d01, bias, norm, _zero, sc + F * 1);
                        Save1<term>(dst, d02, bias, norm, _zero, sc + F * 2);
                        Save1<term>(dst, d03, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps), w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps), w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc, srcC - srcCF);
                    }
                    dst += dD;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV1_AnyR1(const int32_t* src, const ConvParam& p, const AlgParamV1& a,
            const int32_t* weight, const int32_t* bias, const float* norm, size_t dyBeg, size_t dyEnd, uint32_t zero, uint8_t* dst)
        {
            float32x4_t _norm;
            int32x4_t _zero = vdupq_n_s32(zero), _bias;
            int32x4_t d00, d10, d20, d30, w0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * F, dW = kY * kX;
            size_t byMask = a.bufH - 1, bW = a.bufW, bufR = a.bufW * a.bufC, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = p.dstC * a.srcE;
            dst += dyBeg * p.dstW * dD;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcCF; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int32_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        d20 = vdupq_n_s32(0);
                        d30 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps + 0 * dX), w0);
                                Madd1(d10, vld1q_s32(ps + 1 * dX), w0);
                                Madd1(d20, vld1q_s32(ps + 2 * dX), w0);
                                Madd1(d30, vld1q_s32(ps + 3 * dX), w0);
                            }
                        }
                        Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                        Save1<term>(pd + 2 * dD, d20, _bias, _norm, _zero);
                        Save1<term>(pd + 3 * dD, d30, _bias, _norm, _zero);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = vdupq_n_s32(0);
                        d10 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps + 0 * dX), w0);
                                Madd1(d10, vld1q_s32(ps + 1 * dX), w0);
                            }
                        }
                        Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps), w0);
                            }
                        }
                        Save1<term>(pd, d00, _bias, _norm, _zero);
                        pd += dD;
                    }
                }
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int32_t* ps0 = src + sc * bW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    size_t dx = 0, tail = srcC - srcCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = vdupq_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = vld1q_s32(pw);
                                Madd1(d00, vld1q_s32(ps), w0);
                            }
                        }
                        Save1<term>(pd, d00, _bias, _norm, _zero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV1_3x3R1(const int32_t* src, const ConvParam& p, const AlgParamV1& a,
            const int32_t* weight, const int32_t* bias, const float* norm, size_t dyBeg, size_t dyEnd, uint32_t zero, uint8_t* dst)
        {
            float32x4_t _norm;
            int32x4_t _zero = vdupq_n_s32(zero), _bias;
            int32x4_t d00, d10, w0, w1, w2, w3, w4, w5, w6, w7, w8, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), sY = p.strideY, sX = p.strideX, dX = sX * F, dW = 9;
            size_t byMask = a.bufH - 1, bW = a.bufW, bufR = a.bufW * a.bufC, dstW2 = sX == 1 ? AlignLo(p.dstW, 2) : 0, dD = p.dstC * a.srcE;
            dst += dyBeg * p.dstW * dD;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int32_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int32_t* ps1 = src + ((sy + 1) & byMask) * bufR + sc * bW;
                    const int32_t* ps2 = src + ((sy + 2) & byMask) * bufR + sc * bW;
                    const int32_t* pw = weight + sc * dW;
                    _bias = vld1q_s32(bias + sc);
                    _norm = vld1q_f32(norm + sc);
                    w0 = vld1q_s32(pw + F * 0);
                    w1 = vld1q_s32(pw + F * 1);
                    w2 = vld1q_s32(pw + F * 2);
                    w3 = vld1q_s32(pw + F * 3);
                    w4 = vld1q_s32(pw + F * 4);
                    w5 = vld1q_s32(pw + F * 5);
                    w6 = vld1q_s32(pw + F * 6);
                    w7 = vld1q_s32(pw + F * 7);
                    w8 = vld1q_s32(pw + F * 8);
                    if (sc < srcCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += DF, ps1 += DF, ps2 += DF)
                        {
                            d00 = vdupq_n_s32(0);
                            d10 = vdupq_n_s32(0);

                            s0 = vld1q_s32(ps0 + F * 0);
                            Madd1(d00, s0, w0);
                            s0 = vld1q_s32(ps0 + F * 1);
                            Madd1(d00, s0, w1);
                            Madd1(d10, s0, w0);
                            s0 = vld1q_s32(ps0 + F * 2);
                            Madd1(d00, s0, w2);
                            Madd1(d10, s0, w1);
                            s0 = vld1q_s32(ps0 + F * 3);
                            Madd1(d10, s0, w2);

                            s0 = vld1q_s32(ps1 + F * 0);
                            Madd1(d00, s0, w3);
                            s0 = vld1q_s32(ps1 + F * 1);
                            Madd1(d00, s0, w4);
                            Madd1(d10, s0, w3);
                            s0 = vld1q_s32(ps1 + F * 2);
                            Madd1(d00, s0, w5);
                            Madd1(d10, s0, w4);
                            s0 = vld1q_s32(ps1 + F * 3);
                            Madd1(d10, s0, w5);

                            s0 = vld1q_s32(ps2 + F * 0);
                            Madd1(d00, s0, w6);
                            s0 = vld1q_s32(ps2 + F * 1);
                            Madd1(d00, s0, w7);
                            Madd1(d10, s0, w6);
                            s0 = vld1q_s32(ps2 + F * 2);
                            Madd1(d00, s0, w8);
                            Madd1(d10, s0, w7);
                            s0 = vld1q_s32(ps2 + F * 3);
                            Madd1(d10, s0, w8);

                            Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                            Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps1 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);

                            s0 = vld1q_s32(ps0 + F * 0);
                            Madd1(d00, s0, w0);
                            s0 = vld1q_s32(ps0 + F * 1);
                            Madd1(d00, s0, w1);
                            s0 = vld1q_s32(ps0 + F * 2);
                            Madd1(d00, s0, w2);
                            s0 = vld1q_s32(ps1 + F * 0);
                            Madd1(d00, s0, w3);
                            s0 = vld1q_s32(ps1 + F * 1);
                            Madd1(d00, s0, w4);
                            s0 = vld1q_s32(ps1 + F * 2);
                            Madd1(d00, s0, w5);
                            s0 = vld1q_s32(ps2 + F * 0);
                            Madd1(d00, s0, w6);
                            s0 = vld1q_s32(ps2 + F * 1);
                            Madd1(d00, s0, w7);
                            s0 = vld1q_s32(ps2 + F * 2);
                            Madd1(d00, s0, w8);

                            Save1<term>(pd, d00, _bias, _norm, _zero);
                            pd += dD;
                        }
                    }
                    else
                    {
                        size_t tail = srcC - srcCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps1 += dX, ps2 += dX)
                        {
                            d00 = vdupq_n_s32(0);

                            s0 = vld1q_s32(ps0 + F * 0);
                            Madd1(d00, s0, w0);
                            s0 = vld1q_s32(ps0 + F * 1);
                            Madd1(d00, s0, w1);
                            s0 = vld1q_s32(ps0 + F * 2);
                            Madd1(d00, s0, w2);
                            s0 = vld1q_s32(ps1 + F * 0);
                            Madd1(d00, s0, w3);
                            s0 = vld1q_s32(ps1 + F * 1);
                            Madd1(d00, s0, w4);
                            s0 = vld1q_s32(ps1 + F * 2);
                            Madd1(d00, s0, w5);
                            s0 = vld1q_s32(ps2 + F * 0);
                            Madd1(d00, s0, w6);
                            s0 = vld1q_s32(ps2 + F * 1);
                            Madd1(d00, s0, w7);
                            s0 = vld1q_s32(ps2 + F * 2);
                            Madd1(d00, s0, w8);

                            Save1<term>(pd, d00, _bias, _norm, _zero, tail);
                            pd += dD;
                        }
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void SetV1(const ConvParam& p, const AlgParamV1& a, SynetQuantizedConvolutionNhwcDepthwiseV1::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && a.reorderType == 1)
                convolution = QuantizedConvolutionNhwcDepthwiseV1_3x3R1<term>;
            else
            {
                if (a.reorderType == 0)
                    convolution = QuantizedConvolutionNhwcDepthwiseV1_AnyR0<term>;
                else if (a.reorderType == 1)
                    convolution = QuantizedConvolutionNhwcDepthwiseV1_AnyR1<term>;
                else
                    assert(0);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV1::SynetQuantizedConvolutionNhwcDepthwiseV1(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV1(p)
        {
            SetAlgParam(F);
            _preprocess = QuantizedConvolutionNhwcDepthwiseV1_Preprocess;
            if (p.dstT == SimdTensorData8u)
                SetV1<Term8iLast8u>(p, _alg, _convolution);
        }
    }
#endif
}

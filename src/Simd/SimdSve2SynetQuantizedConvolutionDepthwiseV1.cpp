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
#include "Simd/SimdSve2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        using AlgParamV1 = SynetQuantizedConvolutionNhwcDepthwiseV1::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t LoadAs32i(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE void Madd1(svint32_t& i32, const svint32_t& u8, const svint32_t& i8, const svbool_t& mask)
        {
            i32 = svmla_s32_x(mask, i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcDepthwiseV1_Preprocess(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParamV1& a, size_t dyBeg, size_t dyEnd, int32_t* dst)
        {
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            svint32_t _zero = svdup_n_s32(zero);
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
                                svst1_s32(body, pd + i, _zero);
                            pd += bxPad;
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++)
                        {
                            size_t sc = 0;
                            for (; sc < srcC; sc += F)
                            {
                                svbool_t mask = svwhilelt_b32(sc, srcC);
                                svst1_s32(body, pd + sc, LoadAs32i(ps + sc, mask));
                            }
                            ps += p.srcC;
                            pd += a.bufC;
                        }
                        if (bwPad)
                        {
                            for (size_t i = 0; i < bwPad; i += F)
                                svst1_s32(body, pd + i, _zero);
                            pd += bwPad;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bufR; i += F)
                            svst1_s32(body, pd + i, _zero);
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
                                    svst1_s32(body, pd + c * bW, _zero);
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++, pd += a.F)
                        {
                            for (size_t sc = 0; sc < bC; sc += F)
                            {
                                svbool_t mask = svwhilelt_b32(sc, srcC);
                                svst1_s32(body, pd + sc * bW, LoadAs32i(ps + sc, mask));
                            }
                            ps += p.srcC;
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 1, pd += a.F)
                                for (size_t c = 0; c < bC; c += a.F)
                                    svst1_s32(body, pd + c * bW, _zero);
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bufR; i += F)
                            svst1_s32(body, pd + i, _zero);
                    }
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV1_AnyR0(const int32_t* src, const ConvParam& p, const AlgParamV1& a,
            const int32_t* weight, const int32_t* bias, const float* norm, size_t dyBeg, size_t dyEnd, uint32_t zero, uint8_t* dst)
        {
            const size_t F = a.F, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svint32_t _zero = svdup_n_s32(zero);
            svint32_t d00, d01, d02, d03, d10, d11, d12, d13, w0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), srcCF4 = AlignLo(srcC, QF), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX;
            size_t byMask = a.bufH - 1, bufC = a.bufC, bufR = a.bufW * a.bufC, dstW2 = AlignLo(p.dstW, 2), dD = p.dstC * a.srcE, dX = sX * bufC;
            dst += dyBeg * p.dstW * p.dstC * a.srcE;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dstW2; dx += 2)
                {
                    const int32_t* ps00 = src + (dx + 0) * sX * bufC;
                    size_t sc = 0;
                    for (; sc < srcCF4; sc += QF)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        d02 = svdup_n_s32(0);
                        d03 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        d11 = svdup_n_s32(0);
                        d12 = svdup_n_s32(0);
                        d13 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw + F * 0);
                                Madd1(d00, svld1_s32(body, ps0 + F * 0), w0, body);
                                Madd1(d10, svld1_s32(body, ps1 + F * 0), w0, body);
                                w0 = svld1_s32(body, pw + F * 1);
                                Madd1(d01, svld1_s32(body, ps0 + F * 1), w0, body);
                                Madd1(d11, svld1_s32(body, ps1 + F * 1), w0, body);
                                w0 = svld1_s32(body, pw + F * 2);
                                Madd1(d02, svld1_s32(body, ps0 + F * 2), w0, body);
                                Madd1(d12, svld1_s32(body, ps1 + F * 2), w0, body);
                                w0 = svld1_s32(body, pw + F * 3);
                                Madd1(d03, svld1_s32(body, ps0 + F * 3), w0, body);
                                Madd1(d13, svld1_s32(body, ps1 + F * 3), w0, body);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                        Save2<term>(dst, dst + dD, d01, d11, bias, norm, _zero, sc + F * 1);
                        Save2<term>(dst, dst + dD, d02, d12, bias, norm, _zero, sc + F * 2);
                        Save2<term>(dst, dst + dD, d03, d13, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps0), w0, body);
                                Madd1(d10, svld1_s32(body, ps1), w0, body);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, *ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps0), w0, body);
                                Madd1(d10, svld1_s32(body, ps1), w0, body);
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
                    for (; sc < srcCF4; sc += QF)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        d02 = svdup_n_s32(0);
                        d03 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw + F * 0);
                                Madd1(d00, svld1_s32(body, ps + F * 0), w0, body);
                                w0 = svld1_s32(body, pw + F * 1);
                                Madd1(d01, svld1_s32(body, ps + F * 1), w0, body);
                                w0 = svld1_s32(body, pw + F * 2);
                                Madd1(d02, svld1_s32(body, ps + F * 2), w0, body);
                                w0 = svld1_s32(body, pw + F * 3);
                                Madd1(d03, svld1_s32(body, ps + F * 3), w0, body);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc + F * 0);
                        Save1<term>(dst, d01, bias, norm, _zero, sc + F * 1);
                        Save1<term>(dst, d02, bias, norm, _zero, sc + F * 2);
                        Save1<term>(dst, d03, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps), w0, body);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps), w0, body);
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
            const size_t F = a.F;
            const svbool_t body = svptrue_b32();
            svfloat32_t _norm;
            svint32_t _zero = svdup_n_s32(zero), _bias;
            svint32_t d00, d10, d20, d30, w0;
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
                    _bias = svld1_s32(body, bias + sc);
                    _norm = svld1_f32(body, norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        d20 = svdup_n_s32(0);
                        d30 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps + 0 * dX), w0, body);
                                Madd1(d10, svld1_s32(body, ps + 1 * dX), w0, body);
                                Madd1(d20, svld1_s32(body, ps + 2 * dX), w0, body);
                                Madd1(d30, svld1_s32(body, ps + 3 * dX), w0, body);
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
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps + 0 * dX), w0, body);
                                Madd1(d10, svld1_s32(body, ps + 1 * dX), w0, body);
                            }
                        }
                        Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps), w0, body);
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
                    _bias = svld1_s32(body, bias + sc);
                    _norm = svld1_f32(body, norm + sc);
                    size_t dx = 0, tail = srcC - srcCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = svld1_s32(body, pw);
                                Madd1(d00, svld1_s32(body, ps), w0, body);
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
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t _norm;
            svint32_t _zero = svdup_n_s32(zero), _bias;
            svint32_t d00, d10, w0, w1, w2, w3, w4, w5, w6, w7, w8, s0;
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
                    _bias = svld1_s32(body, bias + sc);
                    _norm = svld1_f32(body, norm + sc);
                    w0 = svld1_s32(body, pw + F * 0);
                    w1 = svld1_s32(body, pw + F * 1);
                    w2 = svld1_s32(body, pw + F * 2);
                    w3 = svld1_s32(body, pw + F * 3);
                    w4 = svld1_s32(body, pw + F * 4);
                    w5 = svld1_s32(body, pw + F * 5);
                    w6 = svld1_s32(body, pw + F * 6);
                    w7 = svld1_s32(body, pw + F * 7);
                    w8 = svld1_s32(body, pw + F * 8);
                    if (sc < srcCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += DF, ps1 += DF, ps2 += DF)
                        {
                            d00 = svdup_n_s32(0);
                            d10 = svdup_n_s32(0);

                            s0 = svld1_s32(body, ps0 + F * 0);
                            Madd1(d00, s0, w0, body);
                            s0 = svld1_s32(body, ps0 + F * 1);
                            Madd1(d00, s0, w1, body);
                            Madd1(d10, s0, w0, body);
                            s0 = svld1_s32(body, ps0 + F * 2);
                            Madd1(d00, s0, w2, body);
                            Madd1(d10, s0, w1, body);
                            s0 = svld1_s32(body, ps0 + F * 3);
                            Madd1(d10, s0, w2, body);

                            s0 = svld1_s32(body, ps1 + F * 0);
                            Madd1(d00, s0, w3, body);
                            s0 = svld1_s32(body, ps1 + F * 1);
                            Madd1(d00, s0, w4, body);
                            Madd1(d10, s0, w3, body);
                            s0 = svld1_s32(body, ps1 + F * 2);
                            Madd1(d00, s0, w5, body);
                            Madd1(d10, s0, w4, body);
                            s0 = svld1_s32(body, ps1 + F * 3);
                            Madd1(d10, s0, w5, body);

                            s0 = svld1_s32(body, ps2 + F * 0);
                            Madd1(d00, s0, w6, body);
                            s0 = svld1_s32(body, ps2 + F * 1);
                            Madd1(d00, s0, w7, body);
                            Madd1(d10, s0, w6, body);
                            s0 = svld1_s32(body, ps2 + F * 2);
                            Madd1(d00, s0, w8, body);
                            Madd1(d10, s0, w7, body);
                            s0 = svld1_s32(body, ps2 + F * 3);
                            Madd1(d10, s0, w8, body);

                            Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                            Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps1 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);

                            s0 = svld1_s32(body, ps0 + F * 0);
                            Madd1(d00, s0, w0, body);
                            s0 = svld1_s32(body, ps0 + F * 1);
                            Madd1(d00, s0, w1, body);
                            s0 = svld1_s32(body, ps0 + F * 2);
                            Madd1(d00, s0, w2, body);
                            s0 = svld1_s32(body, ps1 + F * 0);
                            Madd1(d00, s0, w3, body);
                            s0 = svld1_s32(body, ps1 + F * 1);
                            Madd1(d00, s0, w4, body);
                            s0 = svld1_s32(body, ps1 + F * 2);
                            Madd1(d00, s0, w5, body);
                            s0 = svld1_s32(body, ps2 + F * 0);
                            Madd1(d00, s0, w6, body);
                            s0 = svld1_s32(body, ps2 + F * 1);
                            Madd1(d00, s0, w7, body);
                            s0 = svld1_s32(body, ps2 + F * 2);
                            Madd1(d00, s0, w8, body);

                            Save1<term>(pd, d00, _bias, _norm, _zero);
                            pd += dD;
                        }
                    }
                    else
                    {
                        size_t tail = srcC - srcCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps1 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);

                            s0 = svld1_s32(body, ps0 + F * 0);
                            Madd1(d00, s0, w0, body);
                            s0 = svld1_s32(body, ps0 + F * 1);
                            Madd1(d00, s0, w1, body);
                            s0 = svld1_s32(body, ps0 + F * 2);
                            Madd1(d00, s0, w2, body);
                            s0 = svld1_s32(body, ps1 + F * 0);
                            Madd1(d00, s0, w3, body);
                            s0 = svld1_s32(body, ps1 + F * 1);
                            Madd1(d00, s0, w4, body);
                            s0 = svld1_s32(body, ps1 + F * 2);
                            Madd1(d00, s0, w5, body);
                            s0 = svld1_s32(body, ps2 + F * 0);
                            Madd1(d00, s0, w6, body);
                            s0 = svld1_s32(body, ps2 + F * 1);
                            Madd1(d00, s0, w7, body);
                            s0 = svld1_s32(body, ps2 + F * 2);
                            Madd1(d00, s0, w8, body);

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
            SetAlgParam(svcntw());
            _preprocess = QuantizedConvolutionNhwcDepthwiseV1_Preprocess;
            if (p.dstT == SimdTensorData8u)
                SetV1<Term8iLast8u>(p, _alg, _convolution);
        }
    }
#endif
}

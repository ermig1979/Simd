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
        using AlgParam = SynetQuantizedConvolutionNhwcDepthwiseV2::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE svint32_t LoadAs32i(const uint8_t* src, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svld1ub_u32(mask, src));
        }

        SIMD_INLINE svint32_t PackRows(const svint32_t& s0, const svint32_t& s1, const svbool_t& mask)
        {
            return svorr_s32_x(mask, s0, svlsl_n_s32_x(mask, s1, 16));
        }

        SIMD_INLINE svint32_t ShiftLeft16(const svint32_t& value, const svbool_t& mask)
        {
            return svlsl_n_s32_x(mask, value, 16);
        }

        SIMD_INLINE svint32_t ShiftRight16(const svint32_t& value, const svbool_t& mask)
        {
            return svreinterpret_s32_u32(svlsr_n_u32_x(mask, svreinterpret_u32_s32(value), 16));
        }

        SIMD_INLINE svint32_t LoadI16(const int16_t* src, const svbool_t& mask)
        {
            return svld1_s32(mask, (const int32_t*)src);
        }

        SIMD_INLINE void Madd2(svint32_t& i32, const svint32_t& u8, const svint32_t& i8)
        {
            svint16_t a = svreinterpret_s16_s32(u8);
            svint16_t b = svreinterpret_s16_s32(i8);
            i32 = svmlalb_s32(i32, a, b);
            i32 = svmlalt_s32(i32, a, b);
        }

        //------------------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcDepthwiseV2_Preprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int16_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svint32_t _zero = svdup_n_s32(int32_t(zero[0]) | (int32_t(zero[0]) << 16));
            size_t srcC = p.srcC, byMask = a.bufH - 1;
            size_t byPad = p.kernelY - 1, srcR = p.srcW * p.srcC, bufR = a.bufW * a.bufC;
            size_t byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.reorderType == 0)
            {
                size_t bxPad = p.padX * a.bufC * 2, bwPad = p.padW * a.bufC * 2;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    int16_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * srcR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * srcR : zero;
                    if (bxPad)
                    {
                        for (size_t i = 0; i < bxPad; i += DF)
                            svst1_s32(body, (int32_t*)(pd + i), _zero);
                        pd += bxPad;
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++)
                    {
                        size_t sc = 0;
                        for (; sc < srcC; sc += F, pd += DF)
                        {
                            svbool_t mask = svwhilelt_b32(sc, srcC);
                            svst1_s32(body, (int32_t*)pd, PackRows(LoadAs32i(ps0 + sc, mask), LoadAs32i(ps1 + sc, mask), body));
                        }
                        ps0 += p.srcC;
                        ps1 += p.srcC;
                    }
                    if (bwPad)
                    {
                        for (size_t i = 0; i < bwPad; i += DF)
                            svst1_s32(body, (int32_t*)(pd + i), _zero);
                        pd += bwPad;
                    }
                }
            }
            else
            {
                size_t bW = a.bufW * 2, bC = a.bufC, xPad = p.padX * 2, wPad = p.padW * 2;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    int16_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * srcR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * srcR : zero;
                    if (xPad)
                    {
                        for (size_t x = 0; x < xPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                svst1_s32(body, (int32_t*)(pd + c * bW), _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += DF)
                    {
                        for (size_t sc = 0; sc < bC; sc += F)
                        {
                            svbool_t mask = svwhilelt_b32(sc, srcC);
                            svst1_s32(body, (int32_t*)(pd + sc * bW), PackRows(LoadAs32i(ps0 + sc, mask), LoadAs32i(ps1 + sc, mask), body));
                        }
                        ps0 += p.srcC;
                        ps1 += p.srcC;
                    }
                    if (wPad)
                    {
                        for (size_t x = 0; x < wPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                svst1_s32(body, (int32_t*)(pd + c * bW), _zero);
                    }
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV2_AnyR1(const int16_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int16_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2;
            const svbool_t body = svptrue_b32();
            svfloat32_t _sNorm, _iScale, _params[2], _dNorm;
            svint32_t _dZero = svdup_n_s32(dZero), _sBias, _iLo, _iHi;
            svint32_t d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 2, bufR = a.bufR, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = p.dstC * a.srcE;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.sizeW, dyD = p.dstW * dD;
            dst += dyBeg * p.dstW * dD;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = svdup_n_s32(-iZero);
                _iHi = svdup_n_s32(255 - iZero);
                _iScale = svdup_n_f32(iScale);
                _dNorm = svdup_n_f32(dNorm);
                _params[0] = svdup_n_f32(params[0]);
                _params[1] = svdup_n_f32(params[1]);
            }
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcCF; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        d20 = svdup_n_s32(0);
                        d30 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        d11 = svdup_n_s32(0);
                        d21 = svdup_n_s32(0);
                        d31 = svdup_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0, body);
                                w1 = LoadI16(pw1, body);
                                s0 = LoadI16(ps + 0 * dX, body);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = LoadI16(ps + 1 * dX, body);
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                                s0 = LoadI16(ps + 2 * dX, body);
                                Madd2(d20, s0, w0);
                                Madd2(d21, s0, w1);
                                s0 = LoadI16(ps + 3 * dX, body);
                                Madd2(d30, s0, w0);
                                Madd2(d31, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd0 + 2 * dD, d20, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd0 + 3 * dD, d30, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 2 * dD, d21, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 3 * dD, d31, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd0 += 4 * dD;
                        pd1 += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        d11 = svdup_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0, body);
                                w1 = LoadI16(pw1, body);
                                s0 = LoadI16(ps + 0 * dX, body);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = LoadI16(ps + 1 * dX, body);
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0, body);
                                w1 = LoadI16(pw1, body);
                                s0 = LoadI16(ps + 0 * dX, body);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    size_t dx = 0, tail = srcC - srcCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        d01 = svdup_n_s32(0);
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = LoadI16(pw0, body);
                                w1 = LoadI16(pw1, body);
                                s0 = LoadI16(ps + 0 * dX, body);
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                dst += p.dstW * 2 * dD;
            }
            for (; dy < dyEnd; ++dy)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcCF; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        d20 = svdup_n_s32(0);
                        d30 = svdup_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw, body);
                                Madd2(d00, LoadI16(ps + 0 * dX, body), w0);
                                Madd2(d10, LoadI16(ps + 1 * dX, body), w0);
                                Madd2(d20, LoadI16(ps + 2 * dX, body), w0);
                                Madd2(d30, LoadI16(ps + 3 * dX, body), w0);
                            }
                        }
                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd + 2 * dD, d20, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd + 3 * dD, d30, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = svdup_n_s32(0);
                        d10 = svdup_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw, body);
                                Madd2(d00, LoadI16(ps + 0 * dX, body), w0);
                                Madd2(d10, LoadI16(ps + 1 * dX, body), w0);
                            }
                        }
                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw, body);
                                Madd2(d00, LoadI16(ps, body), w0);
                            }
                        }
                        Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                        pd += dD;
                    }
                }
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    size_t dx = 0, tail = srcC - srcCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = svdup_n_s32(0);
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = LoadI16(pw, body);
                                Madd2(d00, LoadI16(ps, body), w0);
                            }
                        }
                        Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV2_3x3R1(const int16_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int16_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            const size_t F = a.F, DF = F * 2, QF = F * 4;
            const svbool_t body = svptrue_b32();
            svfloat32_t _sNorm, _iScale, _params[2], _dNorm;
            svint32_t _dZero = svdup_n_s32(dZero), _sBias, _iLo, _iHi;
            svint32_t d00, d10, w03, w14, w25, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 2, bufR = a.bufW * a.bufC, dstW2 = sX == 1 ? AlignLo(p.dstW, 2) : 0, dD = p.dstC * a.srcE;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.sizeW, dyD = p.dstW * dD;
            dst += dyBeg * p.dstW * dD;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = svdup_n_s32(-iZero);
                _iHi = svdup_n_s32(255 - iZero);
                _iScale = svdup_n_f32(iScale);
                _dNorm = svdup_n_f32(dNorm);
                _params[0] = svdup_n_f32(params[0]);
                _params[1] = svdup_n_f32(params[1]);
            }
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                svint32_t d01, w36, w47, w58;
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bufR + sc * bW;
                    const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    w03 = LoadI16(pw0 + 0 * DF, body);
                    w14 = LoadI16(pw0 + 1 * DF, body);
                    w25 = LoadI16(pw0 + 2 * DF, body);
                    w36 = LoadI16(pw1 + 3 * DF, body);
                    w47 = LoadI16(pw1 + 4 * DF, body);
                    w58 = LoadI16(pw1 + 5 * DF, body);
                    if (sc < srcCF)
                    {
                        size_t dx = 0;
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);
                            d01 = svdup_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * DF, body);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, ShiftLeft16(w03, body));
                            s0 = LoadI16(ps0 + 1 * DF, body);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, ShiftLeft16(w14, body));
                            s0 = LoadI16(ps0 + 2 * DF, body);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, ShiftLeft16(w25, body));
                            s0 = LoadI16(ps2 + 0 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w36, body));
                            Madd2(d01, s0, w36);
                            s0 = LoadI16(ps2 + 1 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w47, body));
                            Madd2(d01, s0, w47);
                            s0 = LoadI16(ps2 + 2 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w58, body));
                            Madd2(d01, s0, w58);

                            Save1<term, type>(pd0, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                            Save1<term, type>(pd1, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                    else
                    {
                        size_t tail = srcC - srcCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);
                            d01 = svdup_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * DF, body);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, ShiftLeft16(w03, body));
                            s0 = LoadI16(ps0 + 1 * DF, body);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, ShiftLeft16(w14, body));
                            s0 = LoadI16(ps0 + 2 * DF, body);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, ShiftLeft16(w25, body));
                            s0 = LoadI16(ps2 + 0 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w36, body));
                            Madd2(d01, s0, w36);
                            s0 = LoadI16(ps2 + 1 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w47, body));
                            Madd2(d01, s0, w47);
                            s0 = LoadI16(ps2 + 2 * DF, body);
                            Madd2(d00, s0, ShiftRight16(w58, body));
                            Madd2(d01, s0, w58);

                            Save1<term, type>(pd0, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                            Save1<term, type>(pd1, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                }
                dst += p.dstW * dD * 2;
            }
            for (; dy < dyEnd; ++dy)
            {
                svint32_t w6, w7, w8;
                size_t sc = 0, sy = dy * sY;
                for (; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bufR + sc * bW;
                    const int16_t* pw = weight + sc * dW;
                    _sBias = svld1_s32(body, sBias + sc);
                    _sNorm = svld1_f32(body, sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = svld1_f32(body, params + sc);
                    w03 = LoadI16(pw + 0 * DF, body);
                    w14 = LoadI16(pw + 1 * DF, body);
                    w25 = LoadI16(pw + 2 * DF, body);
                    w6 = LoadI16(pw + 3 * DF, body);
                    w7 = LoadI16(pw + 4 * DF, body);
                    w8 = LoadI16(pw + 5 * DF, body);
                    if (sc < srcCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += QF, ps2 += QF)
                        {
                            d00 = svdup_n_s32(0);
                            d10 = svdup_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * DF, body);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * DF, body);
                            Madd2(d00, s0, w14);
                            Madd2(d10, s0, w03);
                            s0 = LoadI16(ps0 + 2 * DF, body);
                            Madd2(d00, s0, w25);
                            Madd2(d10, s0, w14);
                            s0 = LoadI16(ps0 + 3 * DF, body);
                            Madd2(d10, s0, w25);

                            s0 = LoadI16(ps2 + 0 * DF, body);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * DF, body);
                            Madd2(d00, s0, w7);
                            Madd2(d10, s0, w6);
                            s0 = LoadI16(ps2 + 2 * DF, body);
                            Madd2(d00, s0, w8);
                            Madd2(d10, s0, w7);
                            s0 = LoadI16(ps2 + 3 * DF, body);
                            Madd2(d10, s0, w8);

                            Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                            Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * DF, body);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * DF, body);
                            Madd2(d00, s0, w14);
                            s0 = LoadI16(ps0 + 2 * DF, body);
                            Madd2(d00, s0, w25);
                            s0 = LoadI16(ps2 + 0 * DF, body);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * DF, body);
                            Madd2(d00, s0, w7);
                            s0 = LoadI16(ps2 + 2 * DF, body);
                            Madd2(d00, s0, w8);

                            Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero);
                            pd += dD;
                        }
                    }
                    else
                    {
                        size_t tail = srcC - srcCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = svdup_n_s32(0);

                            s0 = LoadI16(ps0 + 0 * DF, body);
                            Madd2(d00, s0, w03);
                            s0 = LoadI16(ps0 + 1 * DF, body);
                            Madd2(d00, s0, w14);
                            s0 = LoadI16(ps0 + 2 * DF, body);
                            Madd2(d00, s0, w25);
                            s0 = LoadI16(ps2 + 0 * DF, body);
                            Madd2(d00, s0, w6);
                            s0 = LoadI16(ps2 + 1 * DF, body);
                            Madd2(d00, s0, w7);
                            s0 = LoadI16(ps2 + 2 * DF, body);
                            Madd2(d00, s0, w8);

                            Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                            pd += dD;
                        }
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void SetV2(const ConvParam& p, const AlgParam& a, SynetQuantizedConvolutionNhwcDepthwiseV2::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && a.reorderType == 1)
                convolution = QuantizedConvolutionNhwcDepthwiseV2_3x3R1<term, type>;
            else
            {
                if (a.reorderType == 1)
                    convolution = QuantizedConvolutionNhwcDepthwiseV2_AnyR1<term, type>;
                else
                    assert(0);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV2::SynetQuantizedConvolutionNhwcDepthwiseV2(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV2(p)
        {
            SetAlgParam(svcntw());
            _preprocess = QuantizedConvolutionNhwcDepthwiseV2_Preprocess;
            if (p.dstT == SimdTensorData8u)
            {
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity: SetV2<Term8iLast8u, SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
                case SimdConvolutionActivationRelu: SetV2<Term8iLast8u, SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: SetV2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationRestrictRange: SetV2<Term8iLast8u, SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
                case SimdConvolutionActivationPrelu: SetV2<Term8iLast8u, SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationElu: SetV2<Term8iLast8u, SimdConvolutionActivationElu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationHswish: SetV2<Term8iLast8u, SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationMish: SetV2<Term8iLast8u, SimdConvolutionActivationMish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationHardSigmoid: SetV2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
                case SimdConvolutionActivationSwish: SetV2<Term8iLast8u, SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationGelu: SetV2<Term8iLast8u, SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
                default:
                    assert(0);
                }
            }
            else
                assert(0);
        }
    }
#endif
}

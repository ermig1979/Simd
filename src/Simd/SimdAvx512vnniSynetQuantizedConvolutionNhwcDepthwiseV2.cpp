/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetQuantizedDepthwise.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512vnni
    {
        using AlgParam = SynetQuantizedConvolutionNhwcDepthwiseV2::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE void Madd2(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_dpbusd_epi32(i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV2_AnyR1(const int16_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int16_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            __m512 _sNorm, _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _sBias, _iLo, _iHi;
            __m512i d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 2, bufR = a.bufR, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = p.dstC * a.srcE;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.sizeW, dyD = p.dstW * dD;
            dst += dyBeg * p.dstW * dD;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d20 = _mm512_setzero_si512();
                        d30 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d11 = _mm512_setzero_si512();
                        d21 = _mm512_setzero_si512();
                        d31 = _mm512_setzero_si512();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 1 * dX));
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 2 * dX));
                                Madd2(d20, s0, w0);
                                Madd2(d21, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 3 * dX));
                                Madd2(d30, s0, w0);
                                Madd2(d31, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 2 * dD, d20, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 3 * dD, d30, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 2 * dD, d21, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 3 * dD, d31, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += 4 * dD;
                        pd1 += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d11 = _mm512_setzero_si512();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 1 * dX));
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                            }
                        }
                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
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
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d20 = _mm512_setzero_si512();
                        d30 = _mm512_setzero_si512();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd2(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd2(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                                Madd2(d20, _mm512_loadu_si512((__m512i*)(ps + 2 * dX)), w0);
                                Madd2(d30, _mm512_loadu_si512((__m512i*)(ps + 3 * dX)), w0);
                            }
                        }
                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 2 * dD, d20, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 3 * dD, d30, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd2(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd2(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                            }
                        }
                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd2(d00, _mm512_loadu_si512((__m512i*)ps), w0);
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
            __m512 _sNorm, _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _sBias, _iLo, _iHi;
            __m512i d00, d10, w03, w14, w25, w6, w7, w8, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 2, bufR = a.bufW * a.bufC, dstW2 = sX == 1 ? AlignLo(p.dstW, 2) : 0, dD = p.dstC * a.srcE;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.sizeW, dyD = p.dstW * dD;
            dst += dyBeg * p.dstW * dD;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                __m512i d01, d11, w0, w1, w2, w36, w47, w58;
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bufR + sc * bW;
                    const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w03 = _mm512_loadu_si512((__m512i*)pw0 + 0);
                    w14 = _mm512_loadu_si512((__m512i*)pw0 + 1);
                    w25 = _mm512_loadu_si512((__m512i*)pw0 + 2);
                    w6 = _mm512_loadu_si512((__m512i*)pw0 + 3);
                    w7 = _mm512_loadu_si512((__m512i*)pw0 + 4);
                    w8 = _mm512_loadu_si512((__m512i*)pw0 + 5);
                    w0 = _mm512_loadu_si512((__m512i*)pw1 + 0);
                    w1 = _mm512_loadu_si512((__m512i*)pw1 + 1);
                    w2 = _mm512_loadu_si512((__m512i*)pw1 + 2);
                    w36 = _mm512_loadu_si512((__m512i*)pw1 + 3);
                    w47 = _mm512_loadu_si512((__m512i*)pw1 + 4);
                    w58 = _mm512_loadu_si512((__m512i*)pw1 + 5);

                    size_t dx = 0;
#if 1
                    for (; dx < dstW2; dx += 2, ps0 += QF, ps2 += QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d11 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd2(d00, s0, w03);
                        Madd2(d01, s0, w0);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd2(d00, s0, w14);
                        Madd2(d10, s0, w03);
                        Madd2(d01, s0, w1);
                        Madd2(d11, s0, w0);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd2(d00, s0, w25);
                        Madd2(d10, s0, w14);
                        Madd2(d01, s0, w2);
                        Madd2(d11, s0, w1);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd2(d10, s0, w25);
                        Madd2(d11, s0, w2);

                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 0);
                        Madd2(d00, s0, w6);
                        Madd2(d01, s0, w36);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 1);
                        Madd2(d00, s0, w7);
                        Madd2(d10, s0, w6);
                        Madd2(d01, s0, w47);
                        Madd2(d11, s0, w36);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 2);
                        Madd2(d00, s0, w8);
                        Madd2(d10, s0, w7);
                        Madd2(d01, s0, w58);
                        Madd2(d11, s0, w47);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 3);
                        Madd2(d10, s0, w8);
                        Madd2(d11, s0, w58);

                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
#endif
                    for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd2(d00, s0, w03);
                        Madd2(d01, s0, w0);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd2(d00, s0, w14);
                        Madd2(d01, s0, w1);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd2(d00, s0, w25);
                        Madd2(d01, s0, w2);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 0);
                        Madd2(d00, s0, w6);
                        Madd2(d01, s0, w36);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 1);
                        Madd2(d00, s0, w7);
                        Madd2(d01, s0, w47);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 2);
                        Madd2(d00, s0, w8);
                        Madd2(d01, s0, w58);

                        Save1<term, type>(pd0, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                dst += p.dstW * dD * 2;
            }
            for (; dy < dyEnd; ++dy)
            {
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bufR + sc * bW;
                    const int16_t* pw = weight + sc * dW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w03 = _mm512_loadu_si512((__m512i*)pw + 0);
                    w14 = _mm512_loadu_si512((__m512i*)pw + 1);
                    w25 = _mm512_loadu_si512((__m512i*)pw + 2);
                    w6 = _mm512_loadu_si512((__m512i*)pw + 3);
                    w7 = _mm512_loadu_si512((__m512i*)pw + 4);
                    w8 = _mm512_loadu_si512((__m512i*)pw + 5);

                    size_t dx = 0;
                    for (; dx < dstW2; dx += 2, ps0 += QF, ps2 += QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd2(d00, s0, w03);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd2(d00, s0, w14);
                        Madd2(d10, s0, w03);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd2(d00, s0, w25);
                        Madd2(d10, s0, w14);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd2(d10, s0, w25);

                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 0);
                        Madd2(d00, s0, w6);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 1);
                        Madd2(d00, s0, w7);
                        Madd2(d10, s0, w6);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 2);
                        Madd2(d00, s0, w8);
                        Madd2(d10, s0, w7);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 3);
                        Madd2(d10, s0, w8);

                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                    {
                        d00 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd2(d00, s0, w03);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd2(d00, s0, w14);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd2(d00, s0, w25);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 0);
                        Madd2(d00, s0, w6);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 1);
                        Madd2(d00, s0, w7);
                        s0 = _mm512_loadu_si512((__m512i*)ps2 + 2);
                        Madd2(d00, s0, w8);

                        Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += dD;
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
            : Avx512bw::SynetQuantizedConvolutionNhwcDepthwiseV2(p)
        {
            SetAlgParam(F);
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

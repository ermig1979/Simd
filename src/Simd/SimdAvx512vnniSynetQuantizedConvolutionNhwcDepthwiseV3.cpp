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
        using AlgParam = SynetQuantizedConvolutionNhwcDepthwiseV3::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE void Madd4(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_dpbusd_epi32(i32, u8, i8);
        }

        //------------------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcDepthwiseV3_Preprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint8_t* dst)
        {
            __m512i _zero = _mm512_set1_epi8(zero[0]);
            size_t srcC = p.srcC, srcCF = Simd::AlignLo(p.srcC, a.F), byMask = a.bufH - 1;
            __mmask16 tail = TailMask16(srcC - srcCF);
            size_t byPad = p.kernelY - 1, srcR = p.srcW * p.srcC, bufR = a.bufW * a.bufC * 2;
            size_t byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.reorderType == 0)
            {
                size_t bxPad = p.padX * a.bufC * 4, bwPad = p.padW * a.bufC * 4;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    uint8_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * srcR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * srcR : zero;
                    const uint8_t* ps2 = (sy + 2) < p.srcH ? src + (sy + 2) * srcR : zero;
                    const uint8_t* ps3 = (sy + 3) < p.srcH ? src + (sy + 3) * srcR : zero;
                    if (bxPad)
                    {
                        for (size_t i = 0; i < bxPad; i += QF)
                            _mm512_storeu_si512((__m512i*)(pd + i), _zero);
                        pd += bxPad;
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++)
                    {
                        size_t sc = 0;
                        for (; sc < srcCF; sc += F, pd += QF)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps0 + sc)));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps1 + sc)));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps2 + sc)));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps3 + sc)));
                            _mm512_storeu_si512((__m512i*)pd, _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        for (; sc < srcC; sc += F, pd += QF)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps0 + sc));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps1 + sc));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps2 + sc));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps3 + sc));
                            _mm512_storeu_si512((__m512i*)pd, _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        ps0 += p.srcC;
                        ps1 += p.srcC;
                        ps2 += p.srcC;
                        ps3 += p.srcC;
                    }
                    if (bwPad)
                    {
                        for (size_t i = 0; i < bwPad; i += QF)
                            _mm512_storeu_si512((__m512i*)(pd + i), _zero);
                        pd += bwPad;
                    }
                }
            }
            else
            {
                size_t bW = a.bufW * 4, bC = a.bufC, xPad = p.padX * 4, wPad = p.padW * 4;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    uint8_t* pd = dst + (by & byMask) * bufR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * srcR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * srcR : zero;
                    const uint8_t* ps2 = (sy + 2) < p.srcH ? src + (sy + 2) * srcR : zero;
                    const uint8_t* ps3 = (sy + 3) < p.srcH ? src + (sy + 3) * srcR : zero;
                    if (xPad)
                    {
                        for (size_t x = 0; x < xPad; x += 4, pd += QF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm512_storeu_si512((__m512i*)(pd + c * bW), _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += QF)
                    {
                        size_t sc = 0;
                        for (; sc < srcCF; sc += F)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps0 + sc)));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps1 + sc)));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps2 + sc)));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps3 + sc)));
                            _mm512_storeu_si512((__m512i*)(pd + sc * bW), _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        for (; sc < srcC; sc += F)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps0 + sc));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps1 + sc));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps2 + sc));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps3 + sc));
                            _mm512_storeu_si512((__m512i*)(pd + sc * bW), _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        ps0 += p.srcC;
                        ps1 += p.srcC;
                        ps2 += p.srcC;
                        ps3 += p.srcC;
                    }
                    if (wPad)
                    {
                        for (size_t x = 0; x < wPad; x += 4, pd += QF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm512_storeu_si512((__m512i*)(pd + c * bW), _zero);
                    }
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV3_AnyR1(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int8_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            __m512 _sNorm, _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _sBias, _iLo, _iHi;
            __m512i d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * QF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 4, bufR = a.bufR * 2, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = p.dstC * a.srcE;
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
                    const uint8_t* ps0 = src + sc * bW;
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
                        const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw0 += QF, pw1 += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd4(d00, s0, w0);
                                Madd4(d01, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 1 * dX));
                                Madd4(d10, s0, w0);
                                Madd4(d11, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 2 * dX));
                                Madd4(d20, s0, w0);
                                Madd4(d21, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 3 * dX));
                                Madd4(d30, s0, w0);
                                Madd4(d31, s0, w1);
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
                        const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw0 += QF, pw1 += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd4(d00, s0, w0);
                                Madd4(d01, s0, w1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 1 * dX));
                                Madd4(d10, s0, w0);
                                Madd4(d11, s0, w1);
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
                        const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw0 += QF, pw1 += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd4(d00, s0, w0);
                                Madd4(d01, s0, w1);
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
                    const uint8_t* ps0 = src + sc * bW;
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
                        const int8_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd4(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                                Madd4(d20, _mm512_loadu_si512((__m512i*)(ps + 2 * dX)), w0);
                                Madd4(d30, _mm512_loadu_si512((__m512i*)(ps + 3 * dX)), w0);
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
                        const int8_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd4(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                            }
                        }
                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        const int8_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)ps), w0);
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

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV3_3x3R1(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int8_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            __m512 _sNorm, _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _sBias, _iLo, _iHi;
            __m512i d00, d10, w00, w10, w20, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), sY = p.strideY, sX = p.strideX, dX = sX * QF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 4, bufR = a.bufR * 2, dstW2 = sX == 1 ? AlignLo(p.dstW, 2) : 0, dD = p.dstC * a.srcE;
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
                __m512i d01, d11, w01, w11, w21;
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w00 = _mm512_loadu_si512((__m512i*)pw0 + 0);
                    w10 = _mm512_loadu_si512((__m512i*)pw0 + 1);
                    w20 = _mm512_loadu_si512((__m512i*)pw0 + 2);
                    w01 = _mm512_loadu_si512((__m512i*)pw1 + 0);
                    w11 = _mm512_loadu_si512((__m512i*)pw1 + 1);
                    w21 = _mm512_loadu_si512((__m512i*)pw1 + 2);

                    size_t dx = 0;
#if 1
                    for (; dx < dstW2; dx += 2, ps0 += 2 * QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d11 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        Madd4(d01, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d10, s0, w00);
                        Madd4(d01, s0, w11);
                        Madd4(d11, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d10, s0, w10);
                        Madd4(d01, s0, w21);
                        Madd4(d11, s0, w11);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d10, s0, w20);
                        Madd4(d11, s0, w21);

                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
#endif
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        Madd4(d01, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d01, s0, w11);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d01, s0, w21);

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
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const int8_t* pw = weight + sc * dW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w00 = _mm512_loadu_si512((__m512i*)pw + 0);
                    w10 = _mm512_loadu_si512((__m512i*)pw + 1);
                    w20 = _mm512_loadu_si512((__m512i*)pw + 2);

                    size_t dx = 0;
                    for (; dx < dstW2; dx += 2, ps0 += 2 * QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d10, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d10, s0, w10);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d10, s0, w20);

                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);

                        Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcDepthwiseV3_5x5R1(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd,
            const int8_t* weight, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, uint8_t* dst)
        {
            __m512 _sNorm, _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _sBias, _iLo, _iHi;
            __m512i d00, d10, w00, w10, w20, w30, w40, w50, w60, w70, w80, w90, s0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), sY = p.strideY, sX = p.strideX, dX = sX * QF, dW = a.stepW;
            size_t byMask = a.bufH - 1, bW = a.bufW * 4, bufR = a.bufR * 2, dstW2 = sX == 1 ? AlignLo(p.dstW, 2) : 0, dD = p.dstC * a.srcE;
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
                __m512i d01, d11, w01, w11, w21, w31, w41, w51, w61, w71, w81, w91;
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < srcC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const uint8_t* ps4 = src + ((sy + 4) & byMask) * bufR + sc * bW;
                    const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w00 = _mm512_loadu_si512((__m512i*)pw0 + 0);
                    w10 = _mm512_loadu_si512((__m512i*)pw0 + 1);
                    w20 = _mm512_loadu_si512((__m512i*)pw0 + 2);
                    w30 = _mm512_loadu_si512((__m512i*)pw0 + 3);
                    w40 = _mm512_loadu_si512((__m512i*)pw0 + 4);
                    w50 = _mm512_loadu_si512((__m512i*)pw0 + 5);
                    w60 = _mm512_loadu_si512((__m512i*)pw0 + 6);
                    w70 = _mm512_loadu_si512((__m512i*)pw0 + 7);
                    w80 = _mm512_loadu_si512((__m512i*)pw0 + 8);
                    w90 = _mm512_loadu_si512((__m512i*)pw0 + 9);
                    w01 = _mm512_loadu_si512((__m512i*)pw1 + 0);
                    w11 = _mm512_loadu_si512((__m512i*)pw1 + 1);
                    w21 = _mm512_loadu_si512((__m512i*)pw1 + 2);
                    w31 = _mm512_loadu_si512((__m512i*)pw1 + 3);
                    w41 = _mm512_loadu_si512((__m512i*)pw1 + 4);
                    w51 = _mm512_loadu_si512((__m512i*)pw1 + 5);
                    w61 = _mm512_loadu_si512((__m512i*)pw1 + 6);
                    w71 = _mm512_loadu_si512((__m512i*)pw1 + 7);
                    w81 = _mm512_loadu_si512((__m512i*)pw1 + 8);
                    w91 = _mm512_loadu_si512((__m512i*)pw1 + 9);

                    size_t dx = 0;
#if 1
                    for (; dx < dstW2; dx += 2, ps0 += 2 * QF, ps4 += 2 * QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d11 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        Madd4(d01, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d10, s0, w00);
                        Madd4(d01, s0, w11);
                        Madd4(d11, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d10, s0, w10);
                        Madd4(d01, s0, w21);
                        Madd4(d11, s0, w11);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d00, s0, w30);
                        Madd4(d10, s0, w20);
                        Madd4(d01, s0, w31);
                        Madd4(d11, s0, w21);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 4);
                        Madd4(d00, s0, w40);
                        Madd4(d10, s0, w30);
                        Madd4(d01, s0, w41);
                        Madd4(d11, s0, w31);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 5);
                        Madd4(d10, s0, w40);
                        Madd4(d11, s0, w41);

                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 0);
                        Madd4(d00, s0, w50);
                        Madd4(d01, s0, w51);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 1);
                        Madd4(d00, s0, w60);
                        Madd4(d10, s0, w50);
                        Madd4(d01, s0, w61);
                        Madd4(d11, s0, w51);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 2);
                        Madd4(d00, s0, w70);
                        Madd4(d10, s0, w60);
                        Madd4(d01, s0, w71);
                        Madd4(d11, s0, w61);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 3);
                        Madd4(d00, s0, w80);
                        Madd4(d10, s0, w70);
                        Madd4(d01, s0, w81);
                        Madd4(d11, s0, w71);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 4);
                        Madd4(d00, s0, w90);
                        Madd4(d10, s0, w80);
                        Madd4(d01, s0, w91);
                        Madd4(d11, s0, w81);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 5);
                        Madd4(d10, s0, w90);
                        Madd4(d11, s0, w91);

                        Save1<term, type>(pd0 + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd0 + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 0 * dD, d01, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd1 + 1 * dD, d11, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
#endif
                    for (; dx < p.dstW; ++dx, ps0 += dX, ps4 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        Madd4(d01, s0, w01);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d01, s0, w11);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d01, s0, w21);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d00, s0, w30);
                        Madd4(d01, s0, w31);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 4);
                        Madd4(d00, s0, w40);
                        Madd4(d01, s0, w41);

                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 0);
                        Madd4(d00, s0, w50);
                        Madd4(d01, s0, w51);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 1);
                        Madd4(d00, s0, w60);
                        Madd4(d01, s0, w61);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 2);
                        Madd4(d00, s0, w70);
                        Madd4(d01, s0, w71);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 3);
                        Madd4(d00, s0, w80);
                        Madd4(d01, s0, w81);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 4);
                        Madd4(d00, s0, w90);
                        Madd4(d01, s0, w91);

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
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bufR + sc * bW;
                    const uint8_t* ps4 = src + ((sy + 4) & byMask) * bufR + sc * bW;
                    const int8_t* pw0 = weight + sc * dW;
                    _sBias = _mm512_loadu_si512((__m512i*)(sBias + sc));
                    _sNorm = _mm512_loadu_ps(sNorm + sc);
                    if (type == SimdConvolutionActivationPrelu)
                        _params[0] = _mm512_loadu_ps(params + sc);
                    __mmask16 tail = TailMask16(srcC - sc);
                    w00 = _mm512_loadu_si512((__m512i*)pw0 + 0);
                    w10 = _mm512_loadu_si512((__m512i*)pw0 + 1);
                    w20 = _mm512_loadu_si512((__m512i*)pw0 + 2);
                    w30 = _mm512_loadu_si512((__m512i*)pw0 + 3);
                    w40 = _mm512_loadu_si512((__m512i*)pw0 + 4);
                    w50 = _mm512_loadu_si512((__m512i*)pw0 + 5);
                    w60 = _mm512_loadu_si512((__m512i*)pw0 + 6);
                    w70 = _mm512_loadu_si512((__m512i*)pw0 + 7);
                    w80 = _mm512_loadu_si512((__m512i*)pw0 + 8);
                    w90 = _mm512_loadu_si512((__m512i*)pw0 + 9);

                    size_t dx = 0;
                    for (; dx < dstW2; dx += 2, ps0 += 2 * QF, ps4 += 2 * QF)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        Madd4(d10, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        Madd4(d10, s0, w10);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d00, s0, w30);
                        Madd4(d10, s0, w20);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 4);
                        Madd4(d00, s0, w40);
                        Madd4(d10, s0, w30);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 5);
                        Madd4(d10, s0, w40);

                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 0);
                        Madd4(d00, s0, w50);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 1);
                        Madd4(d00, s0, w60);
                        Madd4(d10, s0, w50);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 2);
                        Madd4(d00, s0, w70);
                        Madd4(d10, s0, w60);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 3);
                        Madd4(d00, s0, w80);
                        Madd4(d10, s0, w70);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 4);
                        Madd4(d00, s0, w90);
                        Madd4(d10, s0, w80);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 5);
                        Madd4(d10, s0, w90);

                        Save1<term, type>(pd + 0 * dD, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        Save1<term, type>(pd + 1 * dD, d10, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX, ps4 += dX)
                    {
                        d00 = _mm512_setzero_si512();

                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 0);
                        Madd4(d00, s0, w00);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 1);
                        Madd4(d00, s0, w10);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 2);
                        Madd4(d00, s0, w20);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 3);
                        Madd4(d00, s0, w30);
                        s0 = _mm512_loadu_si512((__m512i*)ps0 + 4);
                        Madd4(d00, s0, w40);

                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 0);
                        Madd4(d00, s0, w50);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 1);
                        Madd4(d00, s0, w60);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 2);
                        Madd4(d00, s0, w70);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 3);
                        Madd4(d00, s0, w80);
                        s0 = _mm512_loadu_si512((__m512i*)ps4 + 4);
                        Madd4(d00, s0, w90);

                        Save1<term, type>(pd, d00, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type> void SetV3(const ConvParam& p, const AlgParam& a, SynetQuantizedConvolutionNhwcDepthwiseV3::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(5) && p.IsDilation(1) && a.reorderType == 1)
                convolution = QuantizedConvolutionNhwcDepthwiseV3_5x5R1<term, type>;
            else 
                if (p.IsKernel(3) && p.IsDilation(1) && a.reorderType == 1)
                convolution = QuantizedConvolutionNhwcDepthwiseV3_3x3R1<term, type>;
            else
            {
                if (a.reorderType == 1)
                    convolution = QuantizedConvolutionNhwcDepthwiseV3_AnyR1<term, type>;
                else
                    assert(0);
            }
        }

        //------------------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcDepthwiseV3::SynetQuantizedConvolutionNhwcDepthwiseV3(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV3(p)
        {
            SetAlgParam(F);
            _preprocess = QuantizedConvolutionNhwcDepthwiseV3_Preprocess;
            if (p.dstT == SimdTensorData8u)
            {
                switch (p.activation)
                {
                case SimdConvolutionActivationIdentity: SetV3<Term8iLast8u, SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
                case SimdConvolutionActivationRelu: SetV3<Term8iLast8u, SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationLeakyRelu: SetV3<Term8iLast8u, SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationRestrictRange: SetV3<Term8iLast8u, SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
                case SimdConvolutionActivationPrelu: SetV3<Term8iLast8u, SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationElu: SetV3<Term8iLast8u, SimdConvolutionActivationElu>(p, _alg, _convolution); break;
                case SimdConvolutionActivationHswish: SetV3<Term8iLast8u, SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationMish: SetV3<Term8iLast8u, SimdConvolutionActivationMish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationHardSigmoid: SetV3<Term8iLast8u, SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
                case SimdConvolutionActivationSwish: SetV3<Term8iLast8u, SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
                case SimdConvolutionActivationGelu: SetV3<Term8iLast8u, SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
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

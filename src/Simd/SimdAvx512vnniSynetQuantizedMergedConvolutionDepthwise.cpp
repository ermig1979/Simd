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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512vnni
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwisePreprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd, uint8_t* dst)
        {
            __m512i _zero = _mm512_set1_epi8(zero[0]);
            size_t byMask = a.dbH - 1, byPad = p.kernelY - 1, byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.dsB)
            {
                size_t syMask = a.dsH - 1, sC = a.dsH * p.srcW, sR = p.srcW * F;
                size_t bW = a.dbW * 4, bR = a.dbW * a.maC * 2, xPad = p.padX * 4, wPad = p.padW * 4;
                for (size_t c = 0; c < maC; c += F)
                {
                    for (size_t by = byBeg; by < byEnd; by += 2)
                    {
                        uint8_t* pd = dst + (by & byMask) * bR;
                        size_t sy = by - p.padY;
                        const uint8_t* ps0 = (sy + 0) < p.srcH ? src + ((sy + 0) & syMask) * sR : zero;
                        const uint8_t* ps1 = (sy + 1) < p.srcH ? src + ((sy + 1) & syMask) * sR : zero;
                        const uint8_t* ps2 = (sy + 2) < p.srcH ? src + ((sy + 2) & syMask) * sR : zero;
                        const uint8_t* ps3 = (sy + 3) < p.srcH ? src + ((sy + 3) & syMask) * sR : zero;
                        if (xPad)
                        {
                            for (size_t x = 0; x < xPad; x += 4, pd += QF)
                                _mm512_storeu_si512((__m512i*)pd, _zero);
                        }
                        for (size_t sx = 0; sx < sR; sx += F, pd += QF)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps0 + sx)));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps1 + sx)));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps2 + sx)));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps3 + sx)));
                            _mm512_storeu_si512((__m512i*)(pd), _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 4, pd += QF)
                                _mm512_storeu_si512((__m512i*)pd, _zero);
                        }
                    }
                    src += sC * F;
                    dst += bW * F;
                }
            }
            else
            {
                size_t sR = p.srcW * p.srcC, sC = p.srcC, maCF = Simd::AlignLo(maC, F);
                size_t bW = a.dbW * 4, bC = a.maC, xPad = p.padX * 4, wPad = p.padW * 4, bR = a.dbW * a.maC * 2;
                __mmask16 tail = TailMask16(maC - maCF);
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    uint8_t* pd = dst + (by & byMask) * bR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * sR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * sR : zero;
                    const uint8_t* ps2 = (sy + 2) < p.srcH ? src + (sy + 2) * sR : zero;
                    const uint8_t* ps3 = (sy + 3) < p.srcH ? src + (sy + 3) * sR : zero;
                    if (xPad)
                    {
                        for (size_t x = 0; x < xPad; x += 4, pd += QF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm512_storeu_si512((__m512i*)(pd + c * bW), _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += QF)
                    {
                        size_t sc = 0;
                        for (; sc < maCF; sc += F)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps0 + sc)));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps1 + sc)));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps2 + sc)));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(ps3 + sc)));
                            _mm512_storeu_si512((__m512i*)(pd + sc * bW), _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        if(sc < maC)
                        {
                            __m512i s0 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps0 + sc));
                            __m512i s1 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps1 + sc));
                            __m512i s2 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps2 + sc));
                            __m512i s3 = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, ps3 + sc));
                            _mm512_storeu_si512((__m512i*)(pd + sc * bW), _mm512_or_si512(_mm512_or_si512(s0, _mm512_slli_epi32(s1, 8)), _mm512_or_si512(_mm512_slli_epi32(s2, 16), _mm512_slli_epi32(s3, 24))));
                        }
                        ps0 += sC;
                        ps1 += sC;
                        ps2 += sC;
                        ps3 += sC;
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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void Madd4(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_dpbusd_epi32(i32, u8, i8);
        }

        SIMD_INLINE void Save1(uint8_t* dst, __m512i sum, const __m512i& bias, const __m512& norm, const __m512i& zero, __mmask16 tail)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        //------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwiseConvolutionAny(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            uint8_t* dst0 = dst;
            __m512 _norm;
            __m512i _zero = _mm512_set1_epi32(zero), _bias;
            __m512i d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t sC = maC, kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * QF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 4, bR = a.dbW * a.maC * 2, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if(a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const uint8_t* ps0 = src + sc * bW;
                    _bias = _mm512_loadu_si512((__m512i*)(bias + sc));
                    _norm = _mm512_loadu_ps(norm + sc);
                    __mmask16 tail = TailMask16(sC - sc);
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
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
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
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero, tail);
                        Save1(pd0 + 2 * dD, d20, _bias, _norm, _zero, tail);
                        Save1(pd0 + 3 * dD, d30, _bias, _norm, _zero, tail);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero, tail);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero, tail);
                        Save1(pd1 + 2 * dD, d21, _bias, _norm, _zero, tail);
                        Save1(pd1 + 3 * dD, d31, _bias, _norm, _zero, tail);
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
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
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
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero, tail);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero, tail);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero, tail);
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
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw0 += QF, pw1 += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw0);
                                w1 = _mm512_loadu_si512((__m512i*)pw1);
                                s0 = _mm512_loadu_si512((__m512i*)(ps + 0 * dX));
                                Madd4(d00, s0, w0);
                                Madd4(d01, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero, tail);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                dst += p.dstW * 2 * dD;
            }
            for (; dy < dyEnd; ++dy)
            {
                size_t sy = dy * sY;
                for (size_t sc = 0; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const uint8_t* ps0 = src + sc * bW;
                    _bias = _mm512_loadu_si512((__m512i*)(bias + sc));
                    _norm = _mm512_loadu_ps(norm + sc);
                    __mmask16 tail = TailMask16(sC - sc);
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
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd4(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                                Madd4(d20, _mm512_loadu_si512((__m512i*)(ps + 2 * dX)), w0);
                                Madd4(d30, _mm512_loadu_si512((__m512i*)(ps + 3 * dX)), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero, tail);
                        Save1(pd + 2 * dD, d20, _bias, _norm, _zero, tail);
                        Save1(pd + 3 * dD, d30, _bias, _norm, _zero, tail);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = _mm512_setzero_si512();
                        d10 = _mm512_setzero_si512();
                        const int8_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)(ps + 0 * dX)), w0);
                                Madd4(d10, _mm512_loadu_si512((__m512i*)(ps + 1 * dX)), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero, tail);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm512_setzero_si512();
                        const int8_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 4)
                        {
                            const uint8_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += QF, pw += QF)
                            {
                                w0 = _mm512_loadu_si512((__m512i*)pw);
                                Madd4(d00, _mm512_loadu_si512((__m512i*)ps), w0);
                            }
                        }
                        Save1(pd, d00, _bias, _norm, _zero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwiseConvolution3x3(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            __m512 _norm;
            __m512i _zero = _mm512_set1_epi32(zero), _bias;
            __m512i d00, d10, w00, w10, w20, s0;
            size_t sC = maC, sCF = AlignLo(sC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * QF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 4, bR = a.dbW * a.maC * 2, dstW2 = (sX == 1 ? AlignLo(p.dstW, 2) : 0), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if (a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                __m512i d01, d11, w01, w11, w21;
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int8_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _bias = _mm512_loadu_si512((__m512i*)(bias + sc));
                    _norm = _mm512_loadu_ps(norm + sc);
                    __mmask16 tail = TailMask16(sC - sc);
                    w00 = _mm512_loadu_si512((__m512i*)pw0 + 0);
                    w10 = _mm512_loadu_si512((__m512i*)pw0 + 1);
                    w20 = _mm512_loadu_si512((__m512i*)pw0 + 2);
                    w01 = _mm512_loadu_si512((__m512i*)pw1 + 0);
                    w11 = _mm512_loadu_si512((__m512i*)pw1 + 1);
                    w21 = _mm512_loadu_si512((__m512i*)pw1 + 2);
                    size_t dx = 0;
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

                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero, tail);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero, tail);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero, tail);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
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

                        Save1(pd0, d00, _bias, _norm, _zero, tail);
                        Save1(pd1, d01, _bias, _norm, _zero, tail);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                dst += p.dstW * dD * 2;
            }
            for (; dy < dyEnd; ++dy)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const uint8_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int8_t* pw = weight + sc * dW;
                    _bias = _mm512_loadu_si512((__m512i*)(bias + sc));
                    _norm = _mm512_loadu_ps(norm + sc);
                    __mmask16 tail = TailMask16(sC - sc);
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

                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero, tail);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero, tail);
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

                        Save1(pd, d00, _bias, _norm, _zero, tail);
                        pd += dD;
                    }
                }
                dst += p.dstW * dD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func)
        {
            func = QuantizedMergedConvolutionDepthwisePreprocess;
        }

        void SetDepthwiseConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwiseConvolutionPtr& func)
        {
            if(p.IsKernel(3))
                func = QuantizedMergedConvolutionDepthwiseConvolution3x3;
            else
                func = QuantizedMergedConvolutionDepthwiseConvolutionAny;
        }
    }
#endif
}

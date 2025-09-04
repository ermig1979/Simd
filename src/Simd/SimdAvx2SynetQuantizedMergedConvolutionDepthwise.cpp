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
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx2
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwisePreprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd, uint8_t* dst)
        {
            __m256i _zero = _mm256_set1_epi16(zero[0]);
            size_t byMask = a.dbH - 1, byPad = p.kernelY - 1, byBeg = dyBeg ? dyBeg * p.strideY + byPad : 0, byEnd = dyEnd * p.strideY + byPad;
            if (a.dsB)
            {
                size_t syMask = a.dsH - 1, sC = a.dsH * p.srcW, sR = p.srcW * F;
                size_t bW = a.dbW * 2, bR = a.dbW * a.maC, xPad = p.padX * 2, wPad = p.padW * 2;
                for (size_t c = 0; c < maC; c += F)
                {
                    for (size_t by = byBeg; by < byEnd; by += 2)
                    {
                        int16_t* pd = (int16_t*)dst + (by & byMask) * bR;
                        size_t sy = by - p.padY;
                        const uint8_t* ps0 = (sy + 0) < p.srcH ? src + ((sy + 0) & syMask) * sR : zero;
                        const uint8_t* ps1 = (sy + 1) < p.srcH ? src + ((sy + 1) & syMask) * sR : zero;
                        if (xPad)
                        {
                            for (size_t x = 0; x < xPad; x += 2, pd += DF)
                                _mm256_storeu_si256((__m256i*)pd, _zero);
                        }
                        for (size_t sx = 0; sx < sR; sx += F, pd += DF)
                        {
                            __m256i s0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps0 + sx)));
                            __m256i s1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps1 + sx)));
                            _mm256_storeu_si256((__m256i*)(pd), _mm256_or_si256(s0, _mm256_slli_epi32(s1, 16)));
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 2, pd += DF)
                                _mm256_storeu_si256((__m256i*)pd, _zero);
                        }
                    }
                    src += sC * F;
                    dst += bW * DF;
                }
            }
            else
            {
                size_t sR = p.srcW * p.srcC, sC = p.srcC;
                size_t bW = a.dbW * 2, bC = a.maC, xPad = p.padX * 2, wPad = p.padW * 2, bR = a.dbW * a.maC;
                for (size_t by = byBeg; by < byEnd; by += 2)
                {
                    int16_t* pd = (int16_t*)dst + (by & byMask) * bR;
                    size_t sy = by - p.padY;
                    const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * sR : zero;
                    const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * sR : zero;
                    if (xPad)
                    {
                        for (size_t x = 0; x < xPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm256_storeu_si256((__m256i*)(pd + c * bW), _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += DF)
                    {
                        for (size_t sc = 0; sc < maC; sc += F)
                        {
                            __m256i s0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps0 + sc)));
                            __m256i s1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps1 + sc)));
                            _mm256_storeu_si256((__m256i*)(pd + sc * bW), _mm256_or_si256(s0, _mm256_slli_epi32(s1, 16)));
                        }
                        ps0 += sC;
                        ps1 += sC;
                    }
                    if (wPad)
                    {
                        for (size_t x = 0; x < wPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm256_storeu_si256((__m256i*)(pd + c * bW), _zero);
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void Madd2(__m256i& i32, __m256i u8, __m256i i8)
        {
            i32 = _mm256_add_epi32(i32, _mm256_madd_epi16(u8, i8));
        }

        SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero);
        }

        SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const __m256i& bias, const __m256& norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<Term8iLast8u>::template Save<0>(dst, (int32_t*)NULL, sum, &bias, &norm, zero, tail);
        }

        //------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwiseConvolutionAny(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight8, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            const int16_t* src = (int16_t*)src8, *weight = (int16_t*)weight8;
            __m256 _norm;
            __m256i _zero = _mm256_set1_epi32(zero), _bias;
            __m256i d00, d10, d20, d30, d01, d11, d21, d31, w0, w1, s0;
            size_t sC = maC, sCF = AlignLo(sC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 2, bR = a.dbW * a.maC, dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if(a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                size_t sc = 0, sy = dy * sY;
                for (; sc < sCF; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        d20 = _mm256_setzero_si256();
                        d30 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        d11 = _mm256_setzero_si256();
                        d21 = _mm256_setzero_si256();
                        d31 = _mm256_setzero_si256();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw0);
                                w1 = _mm256_loadu_si256((__m256i*)pw1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 1 * dX));
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 2 * dX));
                                Madd2(d20, s0, w0);
                                Madd2(d21, s0, w1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 3 * dX));
                                Madd2(d30, s0, w0);
                                Madd2(d31, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd0 + 2 * dD, d20, _bias, _norm, _zero);
                        Save1(pd0 + 3 * dD, d30, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero);
                        Save1(pd1 + 2 * dD, d21, _bias, _norm, _zero);
                        Save1(pd1 + 3 * dD, d31, _bias, _norm, _zero);
                        pd0 += 4 * dD;
                        pd1 += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        d11 = _mm256_setzero_si256();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw0);
                                w1 = _mm256_loadu_si256((__m256i*)pw1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 1 * dX));
                                Madd2(d10, s0, w0);
                                Madd2(d11, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd0 + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        Save1(pd1 + 1 * dD, d11, _bias, _norm, _zero);
                        pd0 += 2 * dD;
                        pd1 += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw0);
                                w1 = _mm256_loadu_si256((__m256i*)pw1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
                            }
                        }
                        Save1(pd0 + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd1 + 0 * dD, d01, _bias, _norm, _zero);
                        pd0 += dD;
                        pd1 += dD;
                    }
                }
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0, tail = sC - sCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw0 += DF, pw1 += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw0);
                                w1 = _mm256_loadu_si256((__m256i*)pw1);
                                s0 = _mm256_loadu_si256((__m256i*)(ps + 0 * dX));
                                Madd2(d00, s0, w0);
                                Madd2(d01, s0, w1);
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
                size_t sc = 0, sy = dy * sY;
                for (; sc < sCF; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        d20 = _mm256_setzero_si256();
                        d30 = _mm256_setzero_si256();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd2(d00, _mm256_loadu_si256((__m256i*)(ps + 0 * dX)), w0);
                                Madd2(d10, _mm256_loadu_si256((__m256i*)(ps + 1 * dX)), w0);
                                Madd2(d20, _mm256_loadu_si256((__m256i*)(ps + 2 * dX)), w0);
                                Madd2(d30, _mm256_loadu_si256((__m256i*)(ps + 3 * dX)), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                        Save1(pd + 2 * dD, d20, _bias, _norm, _zero);
                        Save1(pd + 3 * dD, d30, _bias, _norm, _zero);
                        pd += 4 * dD;
                    }
                    for (; dx < dstW2; dx += 2, ps0 += 2 * dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd2(d00, _mm256_loadu_si256((__m256i*)(ps + 0 * dX)), w0);
                                Madd2(d10, _mm256_loadu_si256((__m256i*)(ps + 1 * dX)), w0);
                            }
                        }
                        Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd2(d00, _mm256_loadu_si256((__m256i*)ps), w0);
                            }
                        }
                        Save1(pd, d00, _bias, _norm, _zero);
                        pd += dD;
                    }
                }
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + sc * bW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0, tail = sC - sCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        const int16_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ky += 2)
                        {
                            const int16_t* ps = ps0 + ((sy + ky) & byMask) * bR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += DF, pw += DF)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd2(d00, _mm256_loadu_si256((__m256i*)ps), w0);
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

        void QuantizedMergedConvolutionDepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd,
            const int8_t* weight8, const int32_t* bias, const float* norm, int32_t zero, uint8_t* dst)
        {
            const int16_t* src = (int16_t*)src8, * weight = (int16_t*)weight8;
            __m256 _norm;
            __m256i _zero = _mm256_set1_epi32(zero), _bias;
            __m256i d00, d10, w03, w14, w25, s0;
            size_t sC = maC, sCF = AlignLo(sC, F), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX, dX = sX * DF, dW = a.dwStep;
            size_t byMask = a.dbH - 1, bW = a.dbW * 2, bR = a.dbW * a.maC, dstW2 = (sX == 1 ? AlignLo(p.dstW, 2) : 0), dD = a.ddB ? a.maC : p.dstC;
            size_t dyEnd2 = dyBeg + (sY == 1 ? AlignLo(dyEnd - dyBeg, 2) : 0), sizeW = a.dwSize, dyD = p.dstW * dD;
            if (a.ddB)
                dst += (dyBeg % a.ddStep) * p.dstW * dD;
            else
                dst += dyBeg * p.dstW * dD;
            size_t dy = dyBeg;
            for (; dy < dyEnd2; dy += 2)
            {
                __m256i d01, w36, w47, w58;
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd0 = dst + sc, * pd1 = pd0 + dyD;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bR + sc * bW;
                    const int16_t* pw0 = weight + sc * dW, * pw1 = pw0 + sizeW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    w03 = _mm256_loadu_si256((__m256i*)pw0 + 0);
                    w14 = _mm256_loadu_si256((__m256i*)pw0 + 1);
                    w25 = _mm256_loadu_si256((__m256i*)pw0 + 2);
                    w36 = _mm256_loadu_si256((__m256i*)pw1 + 3);
                    w47 = _mm256_loadu_si256((__m256i*)pw1 + 4);
                    w58 = _mm256_loadu_si256((__m256i*)pw1 + 5);
                    if (sc < sCF)
                    {
                        size_t dx = 0;
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = _mm256_setzero_si256();
                            d01 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, _mm256_slli_epi32(w03, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, _mm256_slli_epi32(w14, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, _mm256_slli_epi32(w25, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd2(d00, s0, _mm256_srli_epi32(w36, 16));
                            Madd2(d01, s0, w36);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd2(d00, s0, _mm256_srli_epi32(w47, 16));
                            Madd2(d01, s0, w47);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd2(d00, s0, _mm256_srli_epi32(w58, 16));
                            Madd2(d01, s0, w58);

                            Save1(pd0, d00, _bias, _norm, _zero);
                            Save1(pd1, d01, _bias, _norm, _zero);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                    else
                    {
                        size_t tail = sC - sCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = _mm256_setzero_si256();
                            d01 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd2(d00, s0, w03);
                            Madd2(d01, s0, _mm256_slli_epi32(w03, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd2(d00, s0, w14);
                            Madd2(d01, s0, _mm256_slli_epi32(w14, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd2(d00, s0, w25);
                            Madd2(d01, s0, _mm256_slli_epi32(w25, 16));
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd2(d00, s0, _mm256_srli_epi32(w36, 16));
                            Madd2(d01, s0, w36);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd2(d00, s0, _mm256_srli_epi32(w47, 16));
                            Madd2(d01, s0, w47);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd2(d00, s0, _mm256_srli_epi32(w58, 16));
                            Madd2(d01, s0, w58);

                            Save1(pd0, d00, _bias, _norm, _zero, tail);
                            Save1(pd1, d01, _bias, _norm, _zero, tail);
                            pd0 += dD;
                            pd1 += dD;
                        }
                    }
                }
                dst += p.dstW * dD * 2;
            }
            for (; dy < dyEnd; ++dy)
            {
                __m256i w6, w7, w8;
                size_t sc = 0, sy = dy * sY;
                for (; sc < sC; sc += F)
                {
                    uint8_t* pd = dst + sc;
                    const int16_t* ps0 = src + ((sy + 0) & byMask) * bR + sc * bW;
                    const int16_t* ps2 = src + ((sy + 2) & byMask) * bR + sc * bW;
                    const int16_t* pw = weight + sc * dW;
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    w03 = _mm256_loadu_si256((__m256i*)pw + 0);
                    w14 = _mm256_loadu_si256((__m256i*)pw + 1);
                    w25 = _mm256_loadu_si256((__m256i*)pw + 2);
                    w6 = _mm256_loadu_si256((__m256i*)pw + 3);
                    w7 = _mm256_loadu_si256((__m256i*)pw + 4);
                    w8 = _mm256_loadu_si256((__m256i*)pw + 5);
                    if (sc < sCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += QF, ps2 += QF)
                        {
                            d00 = _mm256_setzero_si256();
                            d10 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd2(d00, s0, w03);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd2(d00, s0, w14);
                            Madd2(d10, s0, w03);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd2(d00, s0, w25);
                            Madd2(d10, s0, w14);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 3);
                            Madd2(d10, s0, w25);

                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd2(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd2(d00, s0, w7);
                            Madd2(d10, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd2(d00, s0, w8);
                            Madd2(d10, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 3);
                            Madd2(d10, s0, w8);

                            Save1(pd + 0 * dD, d00, _bias, _norm, _zero);
                            Save1(pd + 1 * dD, d10, _bias, _norm, _zero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd2(d00, s0, w03);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd2(d00, s0, w14);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd2(d00, s0, w25);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd2(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd2(d00, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd2(d00, s0, w8);

                            Save1(pd, d00, _bias, _norm, _zero);
                            pd += dD;
                        }
                    }
                    else
                    {
                        size_t tail = sC - sCF;
                        for (size_t dx = 0; dx < p.dstW; ++dx, ps0 += dX, ps2 += dX)
                        {
                            d00 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd2(d00, s0, w03);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd2(d00, s0, w14);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd2(d00, s0, w25);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd2(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd2(d00, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd2(d00, s0, w8);

                            Save1(pd, d00, _bias, _norm, _zero, tail);
                            pd += dD;
                        }
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

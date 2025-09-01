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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionDepthwisePreprocess(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t maC, size_t dyBeg, size_t dyEnd, uint8_t* dst)
        {
            __m128i _zero = _mm_set1_epi16(zero[0]);
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
                        const uint8_t* ps0 = (sy + 0) < p.srcH ? src + (sy + 0) * sR : zero;
                        const uint8_t* ps1 = (sy + 1) < p.srcH ? src + (sy + 1) * sR : zero;
                        if (xPad)
                        {
                            for (size_t x = 0; x < xPad; x += 2, pd += DF)
                                _mm_storeu_si128((__m128i*)pd, _zero);
                        }
                        for (size_t sx = 0; sx < sR; sx += F, pd += DF)
                        {
                            __m128i s0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(ps0 + sx)));
                            __m128i s1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(ps1 + sx)));
                            _mm_storeu_si128((__m128i*)(pd), _mm_or_si128(s0, _mm_slli_epi32(s1, 16)));
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 2, pd += DF)
                                _mm_storeu_si128((__m128i*)pd, _zero);
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
                                _mm_storeu_si128((__m128i*)(pd + c * bW), _zero);
                    }
                    for (size_t sx = 0; sx < p.srcW; sx++, pd += DF)
                    {
                        for (size_t sc = 0; sc < bC; sc += F)
                        {
                            __m128i s0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(ps0 + sc)));
                            __m128i s1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(ps1 + sc)));
                            _mm_storeu_si128((__m128i*)(pd + sc * bW), _mm_or_si128(s0, _mm_slli_epi32(s1, 16)));
                        }
                        ps0 += sC;
                        ps1 += sC;
                    }
                    if (wPad)
                    {
                        for (size_t x = 0; x < wPad; x += 2, pd += DF)
                            for (size_t c = 0; c < bC; c += F)
                                _mm_storeu_si128((__m128i*)(pd + c * bW), _zero);
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void SetDepthwisePreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::DepthwisePreprocessPtr& func)
        {
            func = QuantizedMergedConvolutionDepthwisePreprocess;
        }
    }
#endif
}

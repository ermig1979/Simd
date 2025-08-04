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
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx512bw
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i LoadAs32i(const uint8_t* src, __mmask16 tail = -1)
        {
            return _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, src));
        }

        SIMD_INLINE __m512i LoadAs32i(const int8_t* src, __mmask16 tail = -1)
        {
            return _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(tail, src));
        }

        SIMD_INLINE void Madd1(__m512i& i32, __m512i u8, __m512i i8)
        {
            i32 = _mm512_add_epi32(i32, _mm512_madd_epi16(u8, i8));
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m512i sum, const int32_t* bias, const float* norm, const __m512i& zero, size_t offset, __mmask16 tail = -1)
        {
            __m512i _bias = _mm512_maskz_loadu_epi32(tail, bias + offset);
            __m512 _norm = _mm512_maskz_loadu_ps(tail, norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero, tail);
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_Default(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m512i _srcZero = _mm512_set1_epi32(srcZero);
            __m512i _dstZero = _mm512_set1_epi32(dstZero);
            __m512i d00, d01, d02, d03, w0, w1, w2, w3;
            size_t size = p.group, sizeF = AlignLo(size, F), sizeF2 = AlignLo(size, F * 2), sizeF4 = AlignLo(size, F * 4);
            __mmask16 tail = TailMask16(size - sizeF);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeF4; i += F * 4)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        d02 = _mm512_setzero_si512();
                        d03 = _mm512_setzero_si512();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F);
                                w1 = LoadAs32i(weight + ow + 1 * F);
                                w2 = LoadAs32i(weight + ow + 2 * F);
                                w3 = LoadAs32i(weight + ow + 3 * F);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                                    Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
                                    Madd1(d02, LoadAs32i(src + os + 2 * F), w2);
                                    Madd1(d03, LoadAs32i(src + os + 3 * F), w3);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0);
                                    Madd1(d01, _srcZero, w1);
                                    Madd1(d02, _srcZero, w2);
                                    Madd1(d03, _srcZero, w3);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                        Save1<term>(dst, d02, bias, norm, _dstZero, i + F * 2);
                        Save1<term>(dst, d03, bias, norm, _dstZero, i + F * 3);
                    }
                    for (; i < sizeF2; i += F * 2)
                    {
                        d00 = _mm512_setzero_si512();
                        d01 = _mm512_setzero_si512();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F);
                                w1 = LoadAs32i(weight + ow + 1 * F);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                                    Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0);
                                    Madd1(d01, _srcZero, w1);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                    }
                    for (; i < sizeF; i += F)
                    {
                        d00 = _mm512_setzero_si512();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                    }
                    for (; i < size; i += F)
                    {
                        d00 = _mm512_setzero_si512();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w0 = LoadAs32i(weight + ow + 0 * F, tail);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    Madd1(d00, LoadAs32i(src + os + 0 * F, tail), w0);
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, w0);
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0, tail);
                    }
                    dst += p.dstC * a.srcE;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Edge(
            const uint8_t* src, const __m512i& srcZero, const ConvParam& p, const AlgParamV0& a, size_t dy, size_t dx,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m512i& dstZero, uint8_t* dst)
        {
            __m512i d00, d01, d02, d03, w0, w1, w2, w3;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F), sizeDF = AlignLo(size, DF), sizeA = AlignLo(size, A);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                d02 = _mm512_setzero_si512();
                d03 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F);
                        w1 = LoadAs32i(weight + ow + 1 * F);
                        w2 = LoadAs32i(weight + ow + 2 * F);
                        w3 = LoadAs32i(weight + ow + 3 * F);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                            Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
                            Madd1(d02, LoadAs32i(src + os + 2 * F), w2);
                            Madd1(d03, LoadAs32i(src + os + 3 * F), w3);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0);
                            Madd1(d01, srcZero, w1);
                            Madd1(d02, srcZero, w2);
                            Madd1(d03, srcZero, w3);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, i + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, i + F * 3);
            }
            for (; i < sizeDF; i += DF)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F);
                        w1 = LoadAs32i(weight + ow + 1 * F);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                            Madd1(d01, LoadAs32i(src + os + 1 * F), w1);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0);
                            Madd1(d01, srcZero, w1);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
            }
            for (; i < sizeF; i += F)
            {
                d00 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F), w0);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
            }
            for (; i < size; i += F)
            {
                d00 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * p.kernelX + kx) * size + i;
                        w0 = LoadAs32i(weight + ow + 0 * F, tail);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            Madd1(d00, LoadAs32i(src + os + 0 * F, tail), w0);
                        }
                        else
                        {
                            Madd1(d00, srcZero, w0);
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0, tail);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main1(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m512i& dstZero, uint8_t* dst)
        {
            __m512i d00, d01, d02, d03;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCA = AlignLo(srcC, A);
            __mmask16 tail = TailMask16(srcC - srcCF);
            size_t srcS = srcC * p.srcW;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                d02 = _mm512_setzero_si512();
                d03 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        Madd1(d00, LoadAs32i(ps + 0 * F), LoadAs32i(pw + 0 * F));
                        Madd1(d01, LoadAs32i(ps + 1 * F), LoadAs32i(pw + 1 * F));
                        Madd1(d02, LoadAs32i(ps + 2 * F), LoadAs32i(pw + 2 * F));
                        Madd1(d03, LoadAs32i(ps + 3 * F), LoadAs32i(pw + 3 * F));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcCDF; c += DF)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        Madd1(d00, LoadAs32i(ps + 0 * F), LoadAs32i(pw + 0 * F));
                        Madd1(d01, LoadAs32i(ps + 1 * F), LoadAs32i(pw + 1 * F));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcC; c += F)
            {
                d00 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        Madd1(d00, LoadAs32i(ps + 0 * F, tail), LoadAs32i(pw + 0 * F, tail));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c, tail);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main2(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m512i& dstZero, uint8_t* dst)
        {
            __m512i d00, d01, d02, d03, d10, d11, d12, d13, w0;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCA = AlignLo(srcC, A);
            __mmask16 tail = TailMask16(srcC - srcCF);
            size_t srcS = srcC * p.srcW;
            size_t srcX = srcC * p.strideX;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                d02 = _mm512_setzero_si512();
                d03 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                d11 = _mm512_setzero_si512();
                d12 = _mm512_setzero_si512();
                d13 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
                        w0 = LoadAs32i(pw + 1 * F);
                        Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
                        Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
                        w0 = LoadAs32i(pw + 2 * F);
                        Madd1(d02, LoadAs32i(ps + 2 * F + 0 * srcX), w0);
                        Madd1(d12, LoadAs32i(ps + 2 * F + 1 * srcX), w0);
                        w0 = LoadAs32i(pw + 3 * F);
                        Madd1(d03, LoadAs32i(ps + 3 * F + 0 * srcX), w0);
                        Madd1(d13, LoadAs32i(ps + 3 * F + 1 * srcX), w0);
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
            for (; c < srcCDF; c += DF)
            {
                d00 = _mm512_setzero_si512();
                d01 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                d11 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
                        w0 = LoadAs32i(pw + 1 * F);
                        Madd1(d01, LoadAs32i(ps + 1 * F + 0 * srcX), w0);
                        Madd1(d11, LoadAs32i(ps + 1 * F + 1 * srcX), w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst + srcC, d11, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcCF; c += F)
            {
                d00 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX), w0);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX), w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c);
            }
            for (; c < srcC; c += F)
            {
                d00 = _mm512_setzero_si512();
                d10 = _mm512_setzero_si512();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, ps += srcC, pw += srcC)
                    {
                        w0 = LoadAs32i(pw + 0 * F, tail);
                        Madd1(d00, LoadAs32i(ps + 0 * F + 0 * srcX, tail), w0);
                        Madd1(d10, LoadAs32i(ps + 0 * F + 1 * srcX, tail), w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c, tail);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c, tail);
            }
        }

        template<Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_3x3(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m512i _srcZero = _mm512_set1_epi32(srcZero);
            __m512i _dstZero = _mm512_set1_epi32(dstZero);
            size_t srcS = p.srcC * p.srcW;
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
            : Avx2::SynetQuantizedConvolutionNhwcDepthwiseV0(p)
        {
            if (p.dstT == SimdTensorData8u)
                SetV0<Term8iLast8u>(p, _convolution);
            //else
            //    SetV0<Term8iLast32f>(p, _convolution);
        }
    }
#endif
}

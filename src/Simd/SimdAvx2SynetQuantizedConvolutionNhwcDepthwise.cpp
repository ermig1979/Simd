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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx2
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;
        using AlgParamV1 = SynetQuantizedConvolutionNhwcDepthwiseV1::AlgParam;

        //------------------------------------------------------------------------------------------------

        template<int part> SIMD_INLINE __m256i Cvt8uTo32i(__m128i src)
        {
            return _mm256_cvtepu8_epi32(_mm_srli_si128(src, part * 8));
        }

        template<int part> SIMD_INLINE __m256i Cvt8iTo32i(__m128i src)
        {
            return _mm256_cvtepi8_epi32(_mm_srli_si128(src, part * 8));
        }

        SIMD_INLINE __m256i LoadAs32i(const uint8_t* src)
        {
            return _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src)));
        }

        SIMD_INLINE __m256i LoadAs32i(const int8_t* src)
        {
            return _mm256_cvtepi8_epi32(_mm_loadl_epi64((__m128i*)(src)));
        }

        SIMD_INLINE void Madd1(__m256i& i32, __m256i u8, __m256i i8)
        {
            i32 = _mm256_add_epi32(i32, _mm256_madd_epi16(u8, i8));
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_Default(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m256i _srcZero = _mm256_set1_epi32(srcZero);
            __m256i _dstZero = _mm256_set1_epi32(dstZero);
            __m128i w01, w23, s01, s23;
            __m256i d00, d01, d02, d03, w0, s0;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F), sizeDF = AlignLo(size, DF), sizeA = AlignLo(size, A);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeA; i += A)
                    {
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        d02 = _mm256_setzero_si256();
                        d03 = _mm256_setzero_si256();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w01 = _mm_loadu_si128((__m128i*)(weight + ow) + 0);
                                w23 = _mm_loadu_si128((__m128i*)(weight + ow) + 1);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    s01 = _mm_loadu_si128((__m128i*)(src + os) + 0);
                                    Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                                    Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                                    s23 = _mm_loadu_si128((__m128i*)(src + os) + 1);
                                    Madd1(d02, Cvt8uTo32i<0>(s23), Cvt8iTo32i<0>(w23));
                                    Madd1(d03, Cvt8uTo32i<1>(s23), Cvt8iTo32i<1>(w23));
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, Cvt8iTo32i<0>(w01));
                                    Madd1(d01, _srcZero, Cvt8iTo32i<1>(w01));
                                    Madd1(d02, _srcZero, Cvt8iTo32i<0>(w23));
                                    Madd1(d03, _srcZero, Cvt8iTo32i<1>(w23));
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                        Save1<term>(dst, d02, bias, norm, _dstZero, i + F * 2);
                        Save1<term>(dst, d03, bias, norm, _dstZero, i + F * 3);
                    }
                    for (; i < sizeDF; i += DF)
                    {
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                size_t ow = (ky * p.kernelX + kx) * size + i;
                                w01 = _mm_loadu_si128((__m128i*)(weight + ow) + 0);
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    size_t os = (sy * p.srcW + sx) * size + i;
                                    s01 = _mm_loadu_si128((__m128i*)(src + os) + 0);
                                    Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                                    Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, Cvt8iTo32i<0>(w01));
                                    Madd1(d01, _srcZero, Cvt8iTo32i<1>(w01));
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                    }
                    for (; i < size; i += F)
                    {
                        size_t ci = i >= sizeF ? size - F : i;
                        d00 = _mm256_setzero_si256();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                w0 = LoadAs32i(weight + (ky * p.kernelX + kx) * size + ci);
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
            const uint8_t* src, const __m256i& srcZero, const ConvParam& p, const AlgParamV0& a, size_t dy, size_t dx,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m256i& dstZero, uint8_t* dst)
        {
            __m128i w01, w23, s01, s23;
            __m256i d00, d01, d02, d03, w0, s0;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F), sizeDF = AlignLo(size, DF), sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                d02 = _mm256_setzero_si256();
                d03 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * 3 + kx) * size + i;
                        w01 = _mm_loadu_si128((__m128i*)(weight + ow) + 0);
                        w23 = _mm_loadu_si128((__m128i*)(weight + ow) + 1);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            s01 = _mm_loadu_si128((__m128i*)(src + os) + 0);
                            Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                            Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                            s23 = _mm_loadu_si128((__m128i*)(src + os) + 1);
                            Madd1(d02, Cvt8uTo32i<0>(s23), Cvt8iTo32i<0>(w23));
                            Madd1(d03, Cvt8uTo32i<1>(s23), Cvt8iTo32i<1>(w23));
                        }
                        else
                        {
                            Madd1(d00, srcZero, Cvt8iTo32i<0>(w01));
                            Madd1(d01, srcZero, Cvt8iTo32i<1>(w01));
                            Madd1(d02, srcZero, Cvt8iTo32i<0>(w23));
                            Madd1(d03, srcZero, Cvt8iTo32i<1>(w23));
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
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        size_t ow = (ky * 3 + kx) * size + i;
                        w01 = _mm_loadu_si128((__m128i*)(weight + ow) + 0);
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            size_t os = (sy * p.srcW + sx) * size + i;
                            s01 = _mm_loadu_si128((__m128i*)(src + os) + 0);
                            Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                            Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                        }
                        else
                        {
                            Madd1(d00, srcZero, Cvt8iTo32i<0>(w01));
                            Madd1(d01, srcZero, Cvt8iTo32i<1>(w01));
                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
            }
            for (; i < size; i += F)
            {
                size_t ci = i >= sizeF ? size - F : i;
                d00 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        w0 = LoadAs32i(weight + (ky * 3 + kx) * size + ci);
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
            const int8_t* weight, const int32_t* bias, const float* norm, const __m256i& dstZero, uint8_t* dst)
        {
            __m128i w01, w23, s01, s23;
            __m256i d00, d01, d02, d03, w0, s0;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                d02 = _mm256_setzero_si256();
                d03 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, pw += srcC, ps += srcC)
                    {
                        w01 = _mm_loadu_si128((__m128i*)pw + 0);
                        s01 = _mm_loadu_si128((__m128i*)ps + 0);
                        Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                        Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                        w23 = _mm_loadu_si128((__m128i*)pw + 1);
                        s23 = _mm_loadu_si128((__m128i*)ps + 1);
                        Madd1(d02, Cvt8uTo32i<0>(s23), Cvt8iTo32i<0>(w23));
                        Madd1(d03, Cvt8uTo32i<1>(s23), Cvt8iTo32i<1>(w23));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcCDF; c += DF)
            {
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, pw += srcC, ps += srcC)
                    {
                        w01 = _mm_loadu_si128((__m128i*)pw + 0);
                        s01 = _mm_loadu_si128((__m128i*)ps + 0);
                        Madd1(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                        Madd1(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + ct;
                    const int8_t* pw = weight + ky * 3 * srcC + ct;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        w0 = LoadAs32i(pw + kx * srcC);
                        s0 = LoadAs32i(ps + kx * srcC);
                        Madd1(d00, s0, w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ct);
            }
        }

        template<Term8iType term> SIMD_INLINE void QuantizedConvolutionNhwcDepthwiseV0_3x3Main2(
            const uint8_t* src, const ConvParam& p, const AlgParamV0& a,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m256i& dstZero, uint8_t* dst)
        {
            __m128i w0, s0, s1;
            __m256i d00, d01, d02, d03, d10, d11, d12, d13, w00;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F), srcCDF = AlignLo(srcC, DF), srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t srcX = srcC * p.strideX;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                d02 = _mm256_setzero_si256();
                d03 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256();
                d11 = _mm256_setzero_si256();
                d12 = _mm256_setzero_si256();
                d13 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, pw += srcC, ps += srcC)
                    {
                        w0 = _mm_loadu_si128((__m128i*)pw + 0);
                        s0 = _mm_loadu_si128((__m128i*)ps + 0);
                        s1 = _mm_loadu_si128((__m128i*)(ps + srcX) + 0);
                        w00 = Cvt8iTo32i<0>(w0);
                        Madd1(d00, Cvt8uTo32i<0>(s0), w00);
                        Madd1(d10, Cvt8uTo32i<0>(s1), w00);
                        w00 = Cvt8iTo32i<1>(w0);
                        Madd1(d01, Cvt8uTo32i<1>(s0), w00);
                        Madd1(d11, Cvt8uTo32i<1>(s1), w00);
                        w0 = _mm_loadu_si128((__m128i*)pw + 1);
                        s0 = _mm_loadu_si128((__m128i*)ps + 1);
                        s1 = _mm_loadu_si128((__m128i*)(ps + srcX) + 1);
                        w00 = Cvt8iTo32i<0>(w0);
                        Madd1(d02, Cvt8uTo32i<0>(s0), w00);
                        Madd1(d12, Cvt8uTo32i<0>(s1), w00);
                        w00 = Cvt8iTo32i<1>(w0);
                        Madd1(d03, Cvt8uTo32i<1>(s0), w00);
                        Madd1(d13, Cvt8uTo32i<1>(s1), w00);
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
                d00 = _mm256_setzero_si256();
                d01 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256();
                d11 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx, pw += srcC, ps += srcC)
                    {
                        w0 = _mm_loadu_si128((__m128i*)pw + 0);
                        s0 = _mm_loadu_si128((__m128i*)ps + 0);
                        s1 = _mm_loadu_si128((__m128i*)(ps + srcX) + 0);
                        w00 = Cvt8iTo32i<0>(w0);
                        Madd1(d00, Cvt8uTo32i<0>(s0), w00);
                        Madd1(d10, Cvt8uTo32i<0>(s1), w00);
                        w00 = Cvt8iTo32i<1>(w0);
                        Madd1(d01, Cvt8uTo32i<1>(s0), w00);
                        Madd1(d11, Cvt8uTo32i<1>(s1), w00);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst + srcC, d11, bias, norm, dstZero, c + F * 1);
            }
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = _mm256_setzero_si256();
                d10 = _mm256_setzero_si256();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + ct;
                    const int8_t* pw = weight + ky * 3 * srcC + ct;
                    for (size_t kx = 0; kx < 3; ++kx, pw += srcC, ps += srcC)
                    {
                        w00 = LoadAs32i(pw);
                        Madd1(d00, LoadAs32i(ps), w00);
                        Madd1(d10, LoadAs32i(ps + srcX), w00);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ct);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, ct);
            }
        }

        template<Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_3x3(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m256i _srcZero = _mm256_set1_epi32(srcZero);
            __m256i _dstZero = _mm256_set1_epi32(dstZero);
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
            : Sse41::SynetQuantizedConvolutionNhwcDepthwiseV0(p)
        {
            if (p.dstT == SimdTensorData8u)
                SetV0<Term8iLast8u>(p, _convolution);
            //else
            //    SetV0<Term8iLast32f>(p, _convolution);
        }

        //------------------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcDepthwiseV1_Preprocess(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParamV1& a, size_t dyBeg, size_t dyEnd, int32_t* dst)
        {
            __m256i _zero = _mm256_set1_epi32(zero);
            size_t srcC = p.srcC, srcCF = Simd::AlignLo(p.srcC, a.F), byMask = a.bufH - 1;
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
                                _mm256_storeu_si256((__m256i*)(pd + i), _zero);
                            pd += bxPad;
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++)
                        {
                            size_t sc = 0;
                            for (; sc < srcC; sc += F)
                                _mm256_storeu_si256((__m256i*)(pd + sc), _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps + sc))));
                            ps += p.srcC;
                            pd += a.bufC;
                        }
                        if (bwPad)
                        {
                            for (size_t i = 0; i < bwPad; i += F)
                                _mm256_storeu_si256((__m256i*)(pd + i), _zero);
                            pd += bwPad;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < bufR; i += F)
                            _mm256_storeu_si256((__m256i*)(pd + i), _zero);
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
                                    _mm256_storeu_si256((__m256i*)(pd + c * bW), _zero);
                        }
                        for (size_t sx = 0; sx < p.srcW; sx++, pd += a.F)
                        {
                            for (size_t sc = 0; sc < bC; sc += F)
                                _mm256_storeu_si256((__m256i*)(pd + sc * bW), _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(ps + sc))));
                            ps += p.srcC;
                        }
                        if (wPad)
                        {
                            for (size_t x = 0; x < wPad; x += 1, pd += a.F)
                                for (size_t c = 0; c < bC; c += a.F)
                                    _mm256_storeu_si256((__m256i*)(pd + c * bW), _zero);
                        }
                    }
                    else
                    {

                        for (size_t i = 0; i < bufR; i += F)
                            _mm256_storeu_si256((__m256i*)(pd + i), _zero);
                    }
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV1_AnyR0(const int32_t* src, const ConvParam& p, const AlgParamV1& a,
            const int32_t* weight, const int32_t* bias, const float* norm, size_t dyBeg, size_t dyEnd, uint32_t zero, uint8_t* dst)
        {
            __m256i _zero = _mm256_set1_epi32(zero);
            __m256i d00, d01, d02, d03, d10, d11, d12, d13, w0;
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), srcCF4 = AlignLo(srcC, F * 4), kY = p.kernelY, kX = p.kernelX, sY = p.strideY, sX = p.strideX;
            size_t byMask = a.bufH - 1, bufC = a.bufC, bufR = a.bufW * a.bufC, dstW2 = AlignLo(p.dstW, 2), dD = p.dstC * a.srcE, dX = sX * bufC;
            dst += dyBeg * p.dstW * p.dstC * a.srcE;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                size_t dx = 0;
                for (; dx < dstW2; dx += 2)
                {
                    const int32_t* ps00 = src + (dx + 0) * sX * bufC;
                    uint8_t* dst0 = dst, * dst1 = dst + dD;
                    size_t sc = 0;
                    for (; sc < srcCF4; sc += F * 4)
                    {
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        d02 = _mm256_setzero_si256();
                        d03 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        d11 = _mm256_setzero_si256();
                        d12 = _mm256_setzero_si256();
                        d13 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, * ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw + 0);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps0 + 0), w0);
                                Madd1(d10, _mm256_loadu_si256((__m256i*)ps1 + 0), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 1);
                                Madd1(d01, _mm256_loadu_si256((__m256i*)ps0 + 1), w0);
                                Madd1(d11, _mm256_loadu_si256((__m256i*)ps1 + 1), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 2);
                                Madd1(d02, _mm256_loadu_si256((__m256i*)ps0 + 2), w0);
                                Madd1(d12, _mm256_loadu_si256((__m256i*)ps1 + 2), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 3);
                                Madd1(d03, _mm256_loadu_si256((__m256i*)ps0 + 3), w0);
                                Madd1(d13, _mm256_loadu_si256((__m256i*)ps1 + 3), w0);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                        Save2<term>(dst, dst + dD, d01, d11, bias, norm, _zero, sc + F * 1);
                        Save2<term>(dst, dst + dD, d02, d12, bias, norm, _zero, sc + F * 2);
                        Save2<term>(dst, dst + dD, d03, d13, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, * ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw + 0);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps0 + 0), w0);
                                Madd1(d10, _mm256_loadu_si256((__m256i*)ps1 + 0), w0);
                            }
                        }
                        Save2<term>(dst, dst + dD, d00, d10, bias, norm, _zero, sc + F * 0);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps0 = ps00 + (sy & byMask) * bufR + sc, * ps1 = ps0 + dX;
                            for (size_t kx = 0; kx < kX; ++kx, ps0 += bufC, ps1 += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw + 0);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps0 + 0), w0);
                                Madd1(d10, _mm256_loadu_si256((__m256i*)ps1 + 0), w0);
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
                        d00 = _mm256_setzero_si256();
                        d01 = _mm256_setzero_si256();
                        d02 = _mm256_setzero_si256();
                        d03 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw + 0);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps + 0), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 1);
                                Madd1(d01, _mm256_loadu_si256((__m256i*)ps + 1), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 2);
                                Madd1(d02, _mm256_loadu_si256((__m256i*)ps + 2), w0);
                                w0 = _mm256_loadu_si256((__m256i*)pw + 3);
                                Madd1(d03, _mm256_loadu_si256((__m256i*)ps + 3), w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc + F * 0);
                        Save1<term>(dst, d01, bias, norm, _zero, sc + F * 1);
                        Save1<term>(dst, d02, bias, norm, _zero, sc + F * 2);
                        Save1<term>(dst, d03, bias, norm, _zero, sc + F * 3);
                    }
                    for (; sc < srcCF; sc += F)
                    {
                        d00 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps), w0);
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _zero, sc);
                    }
                    for (; sc < srcC; sc += F)
                    {
                        d00 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = dy * sY + ky;
                            const int32_t* ps = ps0 + (sy & byMask) * bufR + sc;
                            for (size_t kx = 0; kx < kX; ++kx, ps += bufC, pw += bufC)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps), w0);
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
            __m256 _norm;
            __m256i _zero = _mm256_set1_epi32(zero), _bias;
            __m256i d00, d10, d20, d30, w0;
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
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, ps0 += 4 * dX)
                    {
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        d20 = _mm256_setzero_si256();
                        d30 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)(ps + 0 * dX)), w0);
                                Madd1(d10, _mm256_loadu_si256((__m256i*)(ps + 1 * dX)), w0);
                                Madd1(d20, _mm256_loadu_si256((__m256i*)(ps + 2 * dX)), w0);
                                Madd1(d30, _mm256_loadu_si256((__m256i*)(ps + 3 * dX)), w0);
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
                        d00 = _mm256_setzero_si256();
                        d10 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)(ps + 0 * dX)), w0);
                                Madd1(d10, _mm256_loadu_si256((__m256i*)(ps + 1 * dX)), w0);
                            }
                        }
                        Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                        Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                        pd += 2 * dD;
                    }
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps), w0);
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
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    size_t dx = 0, tail = srcC - srcCF;
                    for (; dx < p.dstW; ++dx, ps0 += dX)
                    {
                        d00 = _mm256_setzero_si256();
                        const int32_t* pw = weight + sc * dW;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            const int32_t* ps = ps0 + ((sy + ky) & byMask) * bufR;
                            for (size_t kx = 0; kx < kX; ++kx, ps += F, pw += F)
                            {
                                w0 = _mm256_loadu_si256((__m256i*)pw);
                                Madd1(d00, _mm256_loadu_si256((__m256i*)ps), w0);
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
            __m256 _norm;
            __m256i _zero = _mm256_set1_epi32(zero), _bias;
            __m256i d00, d10, w0, w1, w2, w3, w4, w5, w6, w7, w8, s0;
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
                    _bias = _mm256_loadu_si256((__m256i*)(bias + sc));
                    _norm = _mm256_loadu_ps(norm + sc);
                    w0 = _mm256_loadu_si256((__m256i*)pw + 0);
                    w1 = _mm256_loadu_si256((__m256i*)pw + 1);
                    w2 = _mm256_loadu_si256((__m256i*)pw + 2);
                    w3 = _mm256_loadu_si256((__m256i*)pw + 3);
                    w4 = _mm256_loadu_si256((__m256i*)pw + 4);
                    w5 = _mm256_loadu_si256((__m256i*)pw + 5);
                    w6 = _mm256_loadu_si256((__m256i*)pw + 6);
                    w7 = _mm256_loadu_si256((__m256i*)pw + 7);
                    w8 = _mm256_loadu_si256((__m256i*)pw + 8);
                    if (sc < srcCF)
                    {
                        size_t dx = 0;
                        for (; dx < dstW2; dx += 2, ps0 += DF, ps1 += DF, ps2 += DF)
                        {
                            d00 = _mm256_setzero_si256();
                            d10 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd1(d00, s0, w0);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd1(d00, s0, w1);
                            Madd1(d10, s0, w0);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd1(d00, s0, w2);
                            Madd1(d10, s0, w1);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 3);
                            Madd1(d10, s0, w2);

                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 0);
                            Madd1(d00, s0, w3);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 1);
                            Madd1(d00, s0, w4);
                            Madd1(d10, s0, w3);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 2);
                            Madd1(d00, s0, w5);
                            Madd1(d10, s0, w4);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 3);
                            Madd1(d10, s0, w5);

                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd1(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd1(d00, s0, w7);
                            Madd1(d10, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
                            Madd1(d00, s0, w8);
                            Madd1(d10, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 3);
                            Madd1(d10, s0, w8);

                            Save1<term>(pd + 0 * dD, d00, _bias, _norm, _zero);
                            Save1<term>(pd + 1 * dD, d10, _bias, _norm, _zero);
                            pd += 2 * dD;
                        }
                        for (; dx < p.dstW; ++dx, ps0 += dX, ps1 += dX, ps2 += dX)
                        {
                            d00 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd1(d00, s0, w0);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd1(d00, s0, w1);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd1(d00, s0, w2);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 0);
                            Madd1(d00, s0, w3);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 1);
                            Madd1(d00, s0, w4);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 2);
                            Madd1(d00, s0, w5);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd1(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd1(d00, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
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
                            d00 = _mm256_setzero_si256();

                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 0);
                            Madd1(d00, s0, w0);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 1);
                            Madd1(d00, s0, w1);
                            s0 = _mm256_loadu_si256((__m256i*)ps0 + 2);
                            Madd1(d00, s0, w2);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 0);
                            Madd1(d00, s0, w3);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 1);
                            Madd1(d00, s0, w4);
                            s0 = _mm256_loadu_si256((__m256i*)ps1 + 2);
                            Madd1(d00, s0, w5);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 0);
                            Madd1(d00, s0, w6);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 1);
                            Madd1(d00, s0, w7);
                            s0 = _mm256_loadu_si256((__m256i*)ps2 + 2);
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
            : Sse41::SynetQuantizedConvolutionNhwcDepthwiseV1(p)
        {
            SetAlgParam(F);
            _preprocess = QuantizedConvolutionNhwcDepthwiseV1_Preprocess;
            if (p.dstT == SimdTensorData8u)
                SetV1<Term8iLast8u>(p, _alg, _convolution);
            //else
            //    SetV0<Term8iLast32f>(p, _alg, _convolution);
        }
    }
#endif
}

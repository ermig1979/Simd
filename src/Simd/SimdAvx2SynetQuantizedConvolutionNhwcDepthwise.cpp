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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Avx2
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;

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

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m256i sum, const int32_t* bias, const float* norm, const __m256i& zero, size_t offset)
        {
            __m256i _bias = _mm256_loadu_si256((__m256i*)(bias + offset));
            __m256 _norm = _mm256_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero);
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

        template <Term8iType term> void SetV0(const ConvParam& p, SynetQuantizedConvolutionNhwcDepthwiseV0::ConvolutionPtr& convolution)
        {
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
    }
#endif
}

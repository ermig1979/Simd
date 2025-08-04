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
            __m512i d00, d01, d02, d03, w0, w1, w2, w3, s0;
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

        template <Term8iType term> void SetV0(const ConvParam& p, SynetQuantizedConvolutionNhwcDepthwiseV0::ConvolutionPtr& convolution)
        {
            //if (p.IsKernel(3) && p.IsDilation(1))
            //    convolution = QuantizedConvolutionNhwcDepthwiseV0_3x3<term>;
            //else
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

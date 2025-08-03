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
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sse41
    {
        using AlgParamV0 = SynetQuantizedConvolutionNhwcDepthwiseV0::AlgParam;

        //------------------------------------------------------------------------------------------------

        template<int part> SIMD_INLINE __m128i Cvt8uTo32i(__m128i src)
        {
            return _mm_cvtepu8_epi32(_mm_srli_si128(src, part * 4));
        }

        template<int part> SIMD_INLINE __m128i Cvt8iTo32i(__m128i src)
        {
            return _mm_cvtepi8_epi32(_mm_srli_si128(src, part * 4));
        }

        SIMD_INLINE __m128i LoadAs32i(const uint8_t* src)
        {
            return _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)src));
        }

        SIMD_INLINE __m128i LoadAs32i(const int8_t* src)
        {
            return _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(int32_t*)src));
        }

        SIMD_INLINE void Madd1(__m128i& i32, __m128i u8, __m128i i8)
        {
            i32 = _mm_add_epi32(i32, _mm_madd_epi16(u8, i8));
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> SIMD_INLINE void Save1(uint8_t* dst, __m128i sum, const int32_t* bias, const float* norm, const __m128i& zero, size_t offset)
        {
            __m128i _bias = _mm_loadu_si128((__m128i*)(bias + offset));
            __m128 _norm = _mm_loadu_ps(norm + offset);
            QuntizedTerm8i<term>::template Save<0>(dst + offset, (int32_t*)NULL, sum, &_bias, &_norm, zero);
        }

        //------------------------------------------------------------------------------------------------

        template <Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_Default(const uint8_t* src, uint32_t srcZero, 
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m128i _srcZero = _mm_set1_epi32(srcZero);
            __m128i _dstZero = _mm_set1_epi32(dstZero);
            __m128i d00, d01, d02, d03, w0, s0;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeA = AlignLo(size, A);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeA; i += A)
                    {
                        d00 = _mm_setzero_si128();
                        d01 = _mm_setzero_si128();
                        d02 = _mm_setzero_si128();
                        d03 = _mm_setzero_si128();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            for (size_t kx = 0; kx < p.kernelX; ++kx)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                w0 = _mm_loadu_si128((__m128i*)(weight + (ky * p.kernelX + kx) * size + i));
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    s0 = _mm_loadu_si128((__m128i*)(src + (sy * p.srcW + sx) * size + i));
                                    Madd1(d00, Cvt8uTo32i<0>(s0), Cvt8iTo32i<0>(w0));
                                    Madd1(d01, Cvt8uTo32i<1>(s0), Cvt8iTo32i<1>(w0));
                                    Madd1(d02, Cvt8uTo32i<2>(s0), Cvt8iTo32i<2>(w0));
                                    Madd1(d03, Cvt8uTo32i<3>(s0), Cvt8iTo32i<3>(w0));
                                }
                                else
                                {
                                    Madd1(d00, _srcZero, Cvt8iTo32i<0>(w0));
                                    Madd1(d01, _srcZero, Cvt8iTo32i<1>(w0));
                                    Madd1(d02, _srcZero, Cvt8iTo32i<2>(w0));
                                    Madd1(d03, _srcZero, Cvt8iTo32i<3>(w0));
                                }
                            }
                        }
                        Save1<term>(dst, d00, bias, norm, _dstZero, i + F * 0);
                        Save1<term>(dst, d01, bias, norm, _dstZero, i + F * 1);
                        Save1<term>(dst, d02, bias, norm, _dstZero, i + F * 2);
                        Save1<term>(dst, d03, bias, norm, _dstZero, i + F * 3);
                    }
                    for (; i < size; i += F)
                    {
                        size_t ci = i >= sizeF ? size - F : i;
                        d00 = _mm_setzero_si128();
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
            const uint8_t* src, const __m128i &srcZero, const ConvParam& p, const AlgParamV0& a, size_t dy, size_t dx,
            const int8_t* weight, const int32_t* bias, const float* norm, const __m128i& dstZero, uint8_t* dst)
        {
            __m128i d00, d01, d02, d03, w0, s0;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeA = AlignLo(size, A);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                d00 = _mm_setzero_si128();
                d01 = _mm_setzero_si128();
                d02 = _mm_setzero_si128();
                d03 = _mm_setzero_si128();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        w0 = _mm_loadu_si128((__m128i*)(weight + (ky * p.kernelX + kx) * size + i));
                        if (sy < p.srcH && sx < p.srcW)
                        {
                            s0 = _mm_loadu_si128((__m128i*)(src + (sy * p.srcW + sx) * size + i));
                            Madd1(d00, Cvt8uTo32i<0>(s0), Cvt8iTo32i<0>(w0));
                            Madd1(d01, Cvt8uTo32i<1>(s0), Cvt8iTo32i<1>(w0));
                            Madd1(d02, Cvt8uTo32i<2>(s0), Cvt8iTo32i<2>(w0));
                            Madd1(d03, Cvt8uTo32i<3>(s0), Cvt8iTo32i<3>(w0));
                        }
                        else
                        {
                            Madd1(d00, srcZero, Cvt8iTo32i<0>(w0));
                            Madd1(d01, srcZero, Cvt8iTo32i<1>(w0));
                            Madd1(d02, srcZero, Cvt8iTo32i<2>(w0));
                            Madd1(d03, srcZero, Cvt8iTo32i<3>(w0));

                        }
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, i + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, i + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, i + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, i + F * 3);
            }
            for (; i < size; i += F)
            {
                size_t ci = i >= sizeF ? size - F : i;
                d00 = _mm_setzero_si128();
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
            const int8_t* weight, const int32_t* bias, const float* norm, const __m128i& dstZero, uint8_t* dst)
        {
            __m128i d00, d01, d02, d03, w0, s0;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm_setzero_si128();
                d01 = _mm_setzero_si128();
                d02 = _mm_setzero_si128();
                d03 = _mm_setzero_si128();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        w0 = _mm_loadu_si128((__m128i*)(pw + kx * srcC));
                        s0 = _mm_loadu_si128((__m128i*)(ps + kx * srcC));
                        Madd1(d00, Cvt8uTo32i<0>(s0), Cvt8iTo32i<0>(w0));
                        Madd1(d01, Cvt8uTo32i<1>(s0), Cvt8iTo32i<1>(w0));
                        Madd1(d02, Cvt8uTo32i<2>(s0), Cvt8iTo32i<2>(w0));
                        Madd1(d03, Cvt8uTo32i<3>(s0), Cvt8iTo32i<3>(w0));
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, c + F * 0);
                Save1<term>(dst, d01, bias, norm, dstZero, c + F * 1);
                Save1<term>(dst, d02, bias, norm, dstZero, c + F * 2);
                Save1<term>(dst, d03, bias, norm, dstZero, c + F * 3);
            }
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = _mm_setzero_si128();
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
            const int8_t* weight, const int32_t* bias, const float* norm, const __m128i& dstZero, uint8_t* dst)
        {
            __m128i d00, d01, d02, d03, d10, d11, d12, d13, w0, w00, s0, s1;
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t srcCA = AlignLo(srcC, A);
            size_t srcS = srcC * p.srcW;
            size_t srcX = srcC * p.strideX;
            size_t c = 0;
            for (; c < srcCA; c += A)
            {
                d00 = _mm_setzero_si128();
                d01 = _mm_setzero_si128();
                d02 = _mm_setzero_si128();
                d03 = _mm_setzero_si128();
                d10 = _mm_setzero_si128();
                d11 = _mm_setzero_si128();
                d12 = _mm_setzero_si128();
                d13 = _mm_setzero_si128();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + c;
                    const int8_t* pw = weight + ky * 3 * srcC + c;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        w0 = _mm_loadu_si128((__m128i*)(pw + kx * srcC));
                        s0 = _mm_loadu_si128((__m128i*)(ps + kx * srcC));
                        s1 = _mm_loadu_si128((__m128i*)(ps + kx * srcC + srcX));
                        w00 = Cvt8iTo32i<0>(w0);
                        Madd1(d00, Cvt8uTo32i<0>(s0), w00);
                        Madd1(d10, Cvt8uTo32i<0>(s1), w00);
                        w00 = Cvt8iTo32i<1>(w0);
                        Madd1(d01, Cvt8uTo32i<1>(s0), w00);
                        Madd1(d11, Cvt8uTo32i<1>(s1), w00);
                        w00 = Cvt8iTo32i<2>(w0);
                        Madd1(d02, Cvt8uTo32i<2>(s0), w00);
                        Madd1(d12, Cvt8uTo32i<2>(s1), w00);
                        w00 = Cvt8iTo32i<3>(w0);
                        Madd1(d03, Cvt8uTo32i<3>(s0), w00);
                        Madd1(d13, Cvt8uTo32i<3>(s1), w00);
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
            for (; c < srcC; c += F)
            {
                size_t ct = c >= srcCF ? srcC - F : c;
                d00 = _mm_setzero_si128();
                d10 = _mm_setzero_si128();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const uint8_t* ps = src + ky * srcS + ct;
                    const int8_t* pw = weight + ky * 3 * srcC + ct;
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        w0 = LoadAs32i(pw + kx * srcC);
                        s0 = LoadAs32i(ps + kx * srcC);
                        s1 = LoadAs32i(ps + kx * srcC + srcX);
                        Madd1(d00, s0, w0);
                        Madd1(d10, s1, w0);
                    }
                }
                Save1<term>(dst, d00, bias, norm, dstZero, ct);
                Save1<term>(dst + srcC, d10, bias, norm, dstZero, ct);
            }
        }

        template<Term8iType term> void QuantizedConvolutionNhwcDepthwiseV0_3x3(const uint8_t* src, uint32_t srcZero,
            const ConvParam& p, const AlgParamV0& a, const int8_t* weight, const int32_t* bias, const float* norm, uint32_t dstZero, uint8_t* dst)
        {
            __m128i _srcZero = _mm_set1_epi32(srcZero);
            __m128i _dstZero = _mm_set1_epi32(dstZero);
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
            : Base::SynetQuantizedConvolutionNhwcDepthwiseV0(p)
        {
            if(p.dstT == SimdTensorData8u)
                SetV0<Term8iLast8u>(p, _convolution);
            //else
            //    SetV0<Term8iLast32f>(p, _convolution);
        }
    }
#endif
}

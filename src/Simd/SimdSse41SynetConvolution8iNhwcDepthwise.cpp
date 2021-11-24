/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) && defined(SIMD_INT8_DEBUG_ENABLE) 
    namespace Sse41
    {
        using AlgParam = SynetConvolution8iNhwcDepthwise::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDepthwise::ConvolutionPtr;

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

        template <Term8iType term, SimdConvolutionActivationType activation> void ConvolutionNhwcDepthwiseDefault(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm, 
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m128i zero = _mm_set1_epi32(a.zero);
            __m128i upper = _mm_set1_epi32(a.upper);
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
                                    Madd1(d00, zero, Cvt8iTo32i<0>(w0));
                                    Madd1(d01, zero, Cvt8iTo32i<1>(w0));
                                    Madd1(d02, zero, Cvt8iTo32i<2>(w0));
                                    Madd1(d03, zero, Cvt8iTo32i<3>(w0));

                                }
                            }
                        }
                        Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
                        Save<term, activation>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
                        Save<term, activation>(dst, d02, norm, bias, params, scale, shift, upper, i + F * 2);
                        Save<term, activation>(dst, d03, norm, bias, params, scale, shift, upper, i + F * 3);
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
                                    s0 = zero;
                                Madd1(d00, s0, w0);
                            }
                        }
                        Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, ci);
                    }
                    dst += p.dstC * a.size;
                }
            }
        }

        template<Term8iType term, SimdConvolutionActivationType activation> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Edge(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, size_t dy, size_t dx, const int8_t* weight, 
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m128i zero = _mm_set1_epi32(a.zero);
            __m128i upper = _mm_set1_epi32(a.upper);
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
                            Madd1(d00, zero, Cvt8iTo32i<0>(w0));
                            Madd1(d01, zero, Cvt8iTo32i<1>(w0));
                            Madd1(d02, zero, Cvt8iTo32i<2>(w0));
                            Madd1(d03, zero, Cvt8iTo32i<3>(w0));

                        }
                    }
                }
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
                Save<term, activation>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
                Save<term, activation>(dst, d02, norm, bias, params, scale, shift, upper, i + F * 2);
                Save<term, activation>(dst, d03, norm, bias, params, scale, shift, upper, i + F * 3);
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
                            s0 = zero;
                        Madd1(d00, s0, w0);
                    }
                }
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, ci);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType activation> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Main1(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm, 
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m128i d00, d01, d02, d03, w0, s0;
            __m128i upper = _mm_set1_epi32(a.upper);
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
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
                Save<term, activation>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
                Save<term, activation>(dst, d02, norm, bias, params, scale, shift, upper, c + F * 2);
                Save<term, activation>(dst, d03, norm, bias, params, scale, shift, upper, c + F * 3);
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
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, ct);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType activation> SIMD_INLINE void ConvolutionNhwcDepthwise3x3Main2(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm,
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m128i d00, d01, d02, d03, d10, d11, d12, d13, w0, w00, s0, s1;
            __m128i upper = _mm_set1_epi32(a.upper);
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
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, c + F * 0);
                Save<term, activation>(dst, d01, norm, bias, params, scale, shift, upper, c + F * 1);
                Save<term, activation>(dst, d02, norm, bias, params, scale, shift, upper, c + F * 2);
                Save<term, activation>(dst, d03, norm, bias, params, scale, shift, upper, c + F * 3);
                Save<term, activation>(dst + srcC, d10, norm, bias, params, scale, shift, upper, c + F * 0);
                Save<term, activation>(dst + srcC, d11, norm, bias, params, scale, shift, upper, c + F * 1);
                Save<term, activation>(dst + srcC, d12, norm, bias, params, scale, shift, upper, c + F * 2);
                Save<term, activation>(dst + srcC, d13, norm, bias, params, scale, shift, upper, c + F * 3);
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
                Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, ct);
                Save<term, activation>(dst + srcC, d10, norm, bias, params, scale, shift, upper, ct);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType activation> SIMD_INLINE void ConvolutionNhwcDepthwise3x3(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm,
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            size_t srcS = p.srcC * p.srcW;
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstC = p.dstC * a.size;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    ConvolutionNhwcDepthwise3x3Edge<term, activation>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    ConvolutionNhwcDepthwise3x3Edge<term, activation>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW2; dx += 2)
                    ConvolutionNhwcDepthwise3x3Main2<term, activation>(src + offset, p, a, weight, norm, bias, params, scale, shift, dst), dst += dstC * 2, offset += srcX * 2;
                for (; dx < dstW; dx += 1)
                    ConvolutionNhwcDepthwise3x3Main1<term, activation>(src + offset, p, a, weight, norm, bias, params, scale, shift, dst), dst += dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    ConvolutionNhwcDepthwise3x3Edge<term, activation>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    ConvolutionNhwcDepthwise3x3Edge<term, activation>(src, p, a, dy, dx, weight, norm, bias, params, scale, shift, dst), dst += dstC;
        }

        //---------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr & d)
        {
            if(p.IsKernel(3) && p.IsDilation(1))
                d = ConvolutionNhwcDepthwise3x3<term, activation>;
            else
                d = ConvolutionNhwcDepthwiseDefault<term, activation>;
        }
        
        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr & d)
        {
            if(p.dstT == SimdTensorData8u)
                Set<Term8iLast8u, activation>(p, d);
            else
                Set<Term8iLast32f, activation>(p, d);
        }

        static void Set(const ConvParam8i& p, ConvolutionPtr & d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, d); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, d); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, d); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, d); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, d); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, d); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, d); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, d); break;
            default: assert(0);
            }
        }

        SynetConvolution8iNhwcDepthwise::SynetConvolution8iNhwcDepthwise(const ConvParam8i& p)
            : Base::SynetConvolution8iNhwcDepthwise(p)
        {
            Set(p, _convolution);
            _convertSrc = Sse2::SynetConvert32fTo8u;
        }

        bool SynetConvolution8iNhwcDepthwise::Preferable(const ConvParam8i& p)
        {
            if (p.trans != SimdTrue || p.srcC != p.dstC || p.srcC != p.group)
                return false;
            if (p.group < Sse2::F)
                return false;
            return true;
        }
    }
#endif
}

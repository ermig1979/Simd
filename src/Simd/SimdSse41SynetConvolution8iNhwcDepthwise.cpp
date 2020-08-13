/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#if defined(SIMD_SSE41_ENABLE)
    namespace Sse41
    {
        using AlgParam = SynetConvolution8iNhwcDepthwise::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDepthwise::ConvolutionPtr;
        using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

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

        template <Term8iType term, SimdConvolutionActivationType activation> void ConvolutionNhwcDepthwiseDefault(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm, 
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m128i zero = _mm_set1_epi32(a.zero);
            __m128i upper = _mm_set1_epi32(a.upper);
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeF4 = AlignLo(size, F * 4);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeF4; i += F*4)
                    {
                        __m128i d00, d01, d02, d03, w0, s0;
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
                                    Madd4<true>(d00, Cvt8uTo32i<0>(s0), Cvt8iTo32i<0>(w0));
                                    Madd4<true>(d01, Cvt8uTo32i<1>(s0), Cvt8iTo32i<1>(w0));
                                    Madd4<true>(d02, Cvt8uTo32i<2>(s0), Cvt8iTo32i<2>(w0));
                                    Madd4<true>(d03, Cvt8uTo32i<3>(s0), Cvt8iTo32i<3>(w0));
                                }
                                else
                                {
                                    Madd4<true>(d00, zero, Cvt8iTo32i<0>(w0));
                                    Madd4<true>(d01, zero, Cvt8iTo32i<1>(w0));
                                    Madd4<true>(d02, zero, Cvt8iTo32i<2>(w0));
                                    Madd4<true>(d03, zero, Cvt8iTo32i<3>(w0));

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
                        __m128i d00, w0, s0;
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
                                Madd4<true>(d00, s0, w0);
                            }
                        }
                        Save<term, activation>(dst, d00, norm, bias, params, scale, shift, upper, ci);
                    }
                    dst += p.dstC * a.size;
                }
            }
        }

        //---------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr & d)
        {
            d = ConvolutionNhwcDepthwiseDefault<term, activation>;
            //if (p.Is1x1())
            //{
            //    switch (a.microD)
            //    {
            //    case 2 * F: d[term] = ConvolutionNhwcDirect1x1_2<term, activation>; break;
            //    default:
            //        assert(0);
            //    }
            //}
            //else
            //{
            //    switch (a.microD)
            //    {
            //    case 2 * F: d[term] = ConvolutionNhwcDirect_2<term, activation>; break;
            //    default:
            //        assert(0);
            //    }
            //}
        }
        
        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr & d)
        {
            if(p.dstT == SimdTensorData8u)
                Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u, activation>(p, d);
            else
                Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f, activation>(p, d);
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
            if (p.group < Sse::F)
                return false;
            return true;
        }
    }
#endif
}

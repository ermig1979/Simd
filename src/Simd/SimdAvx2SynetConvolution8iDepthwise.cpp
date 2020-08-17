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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE)   
    namespace Avx2
    {
        using AlgParam = SynetConvolution8iNhwcDepthwise::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDepthwise::ConvolutionPtr;
        using Term8iType = Base::SynetConvolution8iNhwcDirect::Term8iType;

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

        template <Term8iType term, SimdConvolutionActivationType activation, bool nofma> void ConvolutionNhwcDepthwiseDefault(
            const uint8_t* src, const ConvParam8i& p, const AlgParam& a, const int8_t* weight, const float* norm,
            const float* bias, const float* params, const float* scale, const float* shift, uint8_t* dst)
        {
            __m256i zero = _mm256_set1_epi32(a.zero);
            __m256i upper = _mm256_set1_epi32(a.upper);
            __m128i w01, w23, s01, s23;
            __m256i d00, d01, d02, d03, w0, s0;
            size_t size = p.group;
            size_t sizeF = AlignLo(size, F);
            size_t sizeDF = AlignLo(size, DF);
            size_t sizeQF = AlignLo(size, QF);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t i = 0;
                    for (; i < sizeQF; i += QF)
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
                                    Madd4<true>(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                                    Madd4<true>(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                                    s23 = _mm_loadu_si128((__m128i*)(src + os) + 1);
                                    Madd4<true>(d02, Cvt8uTo32i<0>(s23), Cvt8iTo32i<0>(w23));
                                    Madd4<true>(d03, Cvt8uTo32i<1>(s23), Cvt8iTo32i<1>(w23));
                                }
                                else
                                {
                                    Madd4<true>(d00, zero, Cvt8iTo32i<0>(w01));
                                    Madd4<true>(d01, zero, Cvt8iTo32i<1>(w01));
                                    Madd4<true>(d02, zero, Cvt8iTo32i<0>(w23));
                                    Madd4<true>(d03, zero, Cvt8iTo32i<1>(w23));
                                }
                            }
                        }
                        Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
                        Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
                        Save<term, activation, nofma>(dst, d02, norm, bias, params, scale, shift, upper, i + F * 2);
                        Save<term, activation, nofma>(dst, d03, norm, bias, params, scale, shift, upper, i + F * 3);
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
                                w01 = _mm_loadu_si128((__m128i*)(weight + (ky * p.kernelX + kx) * size + i));
                                if (sy < p.srcH && sx < p.srcW)
                                {
                                    s01 = _mm_loadu_si128((__m128i*)(src + (sy * p.srcW + sx) * size + i));
                                    Madd4<true>(d00, Cvt8uTo32i<0>(s01), Cvt8iTo32i<0>(w01));
                                    Madd4<true>(d01, Cvt8uTo32i<1>(s01), Cvt8iTo32i<1>(w01));
                                }
                                else
                                {
                                    Madd4<true>(d00, zero, Cvt8iTo32i<0>(w01));
                                    Madd4<true>(d01, zero, Cvt8iTo32i<1>(w01));
                                }
                            }
                        }
                        Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, i + F * 0);
                        Save<term, activation, nofma>(dst, d01, norm, bias, params, scale, shift, upper, i + F * 1);
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
                                    s0 = zero;
                                Madd4<true>(d00, s0, w0);
                            }
                        }
                        Save<term, activation, nofma>(dst, d00, norm, bias, params, scale, shift, upper, ci);
                    }
                    dst += p.dstC * a.size;
                }
            }
        }

        //---------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation, bool nofma> void Set(const ConvParam8i& p, ConvolutionPtr& d)
        {
            //if (p.IsKernel(3) && p.IsDilation(1))
            //    d = ConvolutionNhwcDepthwise3x3<term, activation, nofma>;
            //else
                d = ConvolutionNhwcDepthwiseDefault<term, activation, nofma>;
        }

        template<Term8iType term, SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr& d)
        {
            if (Base::FmaAvoid(p.compatibility))
                Set<term, activation, true>(p, d);
            else
                Set<term, activation, false>(p, d);
        }

        template<SimdConvolutionActivationType activation> void Set(const ConvParam8i& p, ConvolutionPtr& d)
        {
            if (p.dstT == SimdTensorData8u)
                Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u, activation>(p, d);
            else
                Set<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f, activation>(p, d);
        }

        static void Set(const ConvParam8i& p, ConvolutionPtr& d)
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
            : Sse41::SynetConvolution8iNhwcDepthwise(p)
        {
            Set(p, _convolution);
            _convertSrc = Avx2::SynetConvert32fTo8u;
        }
    }
#endif
}

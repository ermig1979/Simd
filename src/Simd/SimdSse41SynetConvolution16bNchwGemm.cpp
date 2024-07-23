/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        typedef Base::SynetConvolution16bNchwGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNchwGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        //static void Convert16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        //{
        //    const float* src = (float*)src8;
        //    size_t srcC8 = Simd::AlignLo(p.srcC, 8);
        //    size_t srcC4 = Simd::AlignLo(p.srcC, 4);
        //    size_t gap = a.bufK - a.K;
        //    for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
        //    {
        //        for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
        //        {
        //            uint16_t* row = dst + dr * a.bufK;
        //            for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
        //            {
        //                size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
        //                if (sy < p.srcH)
        //                {
        //                    for (size_t kx = 0; kx < p.kernelX; kx++)
        //                    {
        //                        size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
        //                        if (sx < p.srcW)
        //                        {
        //                            const float* ps = src + (sy * p.srcW + sx) * p.srcC;
        //                            size_t sc = 0;
        //                            for (; sc < srcC8; sc += 8)
        //                            {
        //                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 0));
        //                                __m128i d1 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 4));
        //                                _mm_storeu_si128((__m128i*)(row + sc), _mm_packus_epi32(d0, d1));
        //                            }
        //                            for (; sc < srcC4; sc += 4)
        //                            {
        //                                __m128i d0 = Sse41::Float32ToBFloat16(_mm_loadu_ps(ps + sc + 0));
        //                                _mm_storel_epi64((__m128i*)(row + sc), _mm_packus_epi32(d0, Sse41::K_ZERO));
        //                            }
        //                            for (; sc < p.srcC; ++sc)
        //                                row[sc] = Base::Float32ToBFloat16(ps[sc]);
        //                            row += p.srcC;
        //                        }
        //                        else
        //                        {
        //                            memset(row, 0, p.srcC * 2);
        //                            row += p.srcC;
        //                        }
        //                    }
        //                }
        //                else
        //                {
        //                    memset(row, 0, p.kernelX * p.srcC * 2);
        //                    row += p.kernelX * p.srcC;
        //                }
        //            }
        //            for (size_t g = 0; g < gap; ++g)
        //                *(row++) = 0;
        //        }
        //    }
        //}

        //static void Reorder16bNhwcGemm(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        //{
        //    const uint16_t* src = (uint16_t*)src8;
        //    size_t gap = a.bufK - a.K;
        //    for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
        //    {
        //        for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
        //        {
        //            uint16_t* row = dst + dr * a.bufK;
        //            for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
        //            {
        //                size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
        //                if (sy < p.srcH)
        //                {
        //                    for (size_t kx = 0; kx < p.kernelX; kx++)
        //                    {
        //                        size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
        //                        if (sx < p.srcW)
        //                        {
        //                            const uint16_t* ps = src + (sy * p.srcW + sx) * p.srcC;
        //                            memcpy(row, ps, p.srcC * 2);
        //                            row += p.srcC;
        //                        }
        //                        else
        //                        {
        //                            memset(row, 0, p.srcC * 2);
        //                            row += p.srcC;
        //                        }
        //                    }
        //                }
        //                else
        //                {
        //                    memset(row, 0, p.kernelX * p.srcC * 2);
        //                    row += p.kernelX * p.srcC;
        //                }
        //            }
        //            for (size_t g = 0; g < gap; ++g)
        //                *(row++) = 0;
        //        }
        //    }
        //}

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M> void Convolution16bNchwGemm_2xM(const uint16_t* weight0, const ConvParam& p, const AlgParam& a, 
            size_t K, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            __m128 d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, w0, s00, s01, s10, s11, m = _mm_castsi128_ps(Bf16::MASK);
            size_t dB = a.N, dD = a.N * a.elem;
            const uint16_t* src1 = src0 + K * F;
            const uint16_t* weight1 = weight0 + 1 * K;
            const uint16_t* weight2 = weight0 + 2 * K;
            const uint16_t* weight3 = weight0 + 3 * K;
            const uint16_t* weight4 = weight0 + 4 * K;
            if (dstS > F)
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps(), d01 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps(), d11 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps(), d21 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps(), d31 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps(), d41 = _mm_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(buf + 0 * dB + 0), d01 = _mm_loadu_ps(buf + 0 * dB + F);
                    if (M > 1) d10 = _mm_loadu_ps(buf + 1 * dB + 0), d11 = _mm_loadu_ps(buf + 1 * dB + F);
                    if (M > 2) d20 = _mm_loadu_ps(buf + 2 * dB + 0), d21 = _mm_loadu_ps(buf + 2 * dB + F);
                    if (M > 3) d30 = _mm_loadu_ps(buf + 3 * dB + 0), d31 = _mm_loadu_ps(buf + 3 * dB + F);
                    if (M > 4) d40 = _mm_loadu_ps(buf + 4 * dB + 0), d41 = _mm_loadu_ps(buf + 4 * dB + F);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    s01 = _mm_loadu_ps((float*)src0);
                    s00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(s01), Base::Bf16::SHIFT));
                    s01 = _mm_and_ps(s01, m);
                    s11 = _mm_loadu_ps((float*)src1);
                    s10 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(s11), Base::Bf16::SHIFT));
                    s11 = _mm_and_ps(s11, m);
                    if (M > 0)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight0 + k - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(w0, s00), d00);
                        d01 = _mm_add_ps(_mm_mul_ps(w0, s10), d01);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight0 + k - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(w0, s01), d00);
                        d01 = _mm_add_ps(_mm_mul_ps(w0, s11), d01);
                    }
                    if (M > 1)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight1 + k - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(w0, s00), d10);
                        d11 = _mm_add_ps(_mm_mul_ps(w0, s10), d11);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight1 + k - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(w0, s01), d10);
                        d11 = _mm_add_ps(_mm_mul_ps(w0, s11), d11);
                    }
                    if (M > 2)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight2 + k - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(w0, s00), d20);
                        d21 = _mm_add_ps(_mm_mul_ps(w0, s10), d21);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight2 + k - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(w0, s01), d20);
                        d21 = _mm_add_ps(_mm_mul_ps(w0, s11), d21);
                    }
                    if (M > 3)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight3 + k - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(w0, s00), d30);
                        d31 = _mm_add_ps(_mm_mul_ps(w0, s10), d31);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight3 + k - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(w0, s01), d30);
                        d31 = _mm_add_ps(_mm_mul_ps(w0, s11), d31);
                    }
                    if (M > 4)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight4 + k - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(w0, s00), d40);
                        d41 = _mm_add_ps(_mm_mul_ps(w0, s10), d41);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight4 + k - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(w0, s01), d40);
                        d41 = _mm_add_ps(_mm_mul_ps(w0, s11), d41);
                    }
                    src0 += DF;
                    src1 += DF;
                }
                if (dstS == DF)
                {
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias, params, 0), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias, params, 1), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias, params, 2), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias, params, 3), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias, params, 4), dst += dD, buf += dB;
                }
                else
                {
                    dstS -= F;
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, bias, params, 0, dstS), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, bias, params, 1, dstS), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, bias, params, 2, dstS), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, bias, params, 3, dstS), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, bias, params, 4, dstS), dst += dD, buf += dB;
                }
            }
            else
            {
                if (zero)
                {
                    if (M > 0) d00 = _mm_setzero_ps();
                    if (M > 1) d10 = _mm_setzero_ps();
                    if (M > 2) d20 = _mm_setzero_ps();
                    if (M > 3) d30 = _mm_setzero_ps();
                    if (M > 4) d40 = _mm_setzero_ps();
                }
                else
                {
                    if (M > 0) d00 = _mm_loadu_ps(buf + 0 * dB + 0);
                    if (M > 1) d10 = _mm_loadu_ps(buf + 1 * dB + 0);
                    if (M > 2) d20 = _mm_loadu_ps(buf + 2 * dB + 0);
                    if (M > 3) d30 = _mm_loadu_ps(buf + 3 * dB + 0);
                    if (M > 4) d40 = _mm_loadu_ps(buf + 4 * dB + 0);
                }
                for (size_t k = 0; k < K; k += 2)
                {
                    s01 = _mm_loadu_ps((float*)src0);
                    s00 = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(s01), Base::Bf16::SHIFT));
                    s01 = _mm_and_ps(s01, m);
                    if (M > 0)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight0 + k - 1)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(w0, s00), d00);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight0 + k - 0)), m);
                        d00 = _mm_add_ps(_mm_mul_ps(w0, s01), d00);
                    }
                    if (M > 1)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight1 + k - 1)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(w0, s00), d10);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight1 + k - 0)), m);
                        d10 = _mm_add_ps(_mm_mul_ps(w0, s01), d10);
                    }
                    if (M > 2)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight2 + k - 1)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(w0, s00), d20);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight2 + k - 0)), m);
                        d20 = _mm_add_ps(_mm_mul_ps(w0, s01), d20);
                    }
                    if (M > 3)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight3 + k - 1)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(w0, s00), d30);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight3 + k - 0)), m);
                        d30 = _mm_add_ps(_mm_mul_ps(w0, s01), d30);
                    }
                    if (M > 4)
                    {
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight4 + k - 1)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(w0, s00), d40);
                        w0 = _mm_and_ps(_mm_set1_ps(*(float*)(weight4 + k - 0)), m);
                        d40 = _mm_add_ps(_mm_mul_ps(w0, s01), d40);
                    }
                    src0 += DF;
                }
                if (dstS == F)
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias, params, 0), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias, params, 1), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias, params, 2), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias, params, 3), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias, params, 4), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, bias, params, 0, dstS), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, bias, params, 1, dstS), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, bias, params, 2, dstS), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, bias, params, 3, dstS), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, bias, params, 4, dstS), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*Convolution16bNchwGemm_2xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int zero, const uint16_t* weight0, const float* bias, const float* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> Convolution16bNchwGemm_2xM_Ptr GetConvolution16bNchwGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return Convolution16bNchwGemm_2xM<term, type, 1>;
            case 2: return Convolution16bNchwGemm_2xM<term, type, 2>;
            case 3: return Convolution16bNchwGemm_2xM<term, type, 3>;
            case 4: return Convolution16bNchwGemm_2xM<term, type, 4>;
            case 5: return Convolution16bNchwGemm_2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNchwGemm_2(const uint16_t* weight, const ConvParam& p, const AlgParam& a, 
            size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* src, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.N, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            Convolution16bNchwGemm_2xM_Ptr convolution_2xN = GetConvolution16bNchwGemm_2xM<term, type>(n);
            Convolution16bNchwGemm_2xM_Ptr convolution_2xM = GetConvolution16bNchwGemm_2xM<term, type>(m);

            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const uint16_t* w = weight;
                float* b = buf + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                    convolution_2xN(w, p, a, K, dS, zero, src, bias + i, params + i * dp, b, d);
                for (; i < n1; i += m, w += m * dW, b += m * dB, d += m * dD)
                    convolution_2xM(w, p, a, K, dS, zero, src, bias + i, params + i * dp, b, d);
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam & a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNchwGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if(p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNchwGemm::SynetConvolution16bNchwGemm(const ConvParam & p)
            : Base::SynetConvolution16bNchwGemm(p)
        {
            SetAlgParam(F, F * 2, 5, 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src16b)
            {
                AlgParam& a = _alg;
                //if (_is1x1 && a.K == a.bufK)
                //    _convert = NULL;
                //else
                //    _convert = Reorder16bNhwcGemm;
            }
            //else
            //    _convert = Convert16bNhwcGemm;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationGelu: Set<SimdConvolutionActivationGelu>(p, _alg, _convolutions); break;
            default: assert(0);
            }
        }
    }
#endif
}

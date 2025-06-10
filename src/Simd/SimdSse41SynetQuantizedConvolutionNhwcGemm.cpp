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
#include "Simd/SimdSynetQuantizedConvolution.h"
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
        typedef Base::SynetQuantizedConvolutionNhwcGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmReorder(const uint8_t* src, const uint8_t* zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t gap = a.bufK - a.K;
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint8_t* row = dst + dr * a.bufK;
                    for (size_t ky = 0, k = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    memcpy(row, ps, p.srcC);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, zero[0], p.srcC);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, zero[0], p.kernelX * p.srcC);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, int M> void QuantizedConvolutionNhwcGemm_2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight0, const __m128i* bias, const __m128* norm, const __m128i* zero, int32_t* buf, uint8_t* dst)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, s0, w0, w1;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            const int8_t* weight1 = weight0 + a.bufK * F;
            const uint8_t* src1 = src0 + 1 * dS;
            const uint8_t* src2 = src0 + 2 * dS;
            const uint8_t* src3 = src0 + 3 * dS;
            const uint8_t* src4 = src0 + 4 * dS;
            if (dstC > F)
            {
                if (update)
                {
                    if (M > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 0), d01 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 1);
                    if (M > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 0), d11 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 1);
                    if (M > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 0), d21 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 1);
                    if (M > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 0), d31 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 1);
                    if (M > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 0), d41 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 1);
                }
                else
                {
                    if (M > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                    if (M > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                    if (M > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                    if (M > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                    if (M > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm_loadu_si128((__m128i*)weight0);
                    w1 = _mm_loadu_si128((__m128i*)weight1);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s0, w1);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s0, w1);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s0, w1);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s0, w1);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s0, w1);
                    weight0 += A, weight1 += A;
                }
                if (dstC == DF)
                {
                    if (M > 0) Save2<term>(dst, buf, d00, d01, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 1) Save2<term>(dst, buf, d10, d11, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 2) Save2<term>(dst, buf, d20, d21, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 3) Save2<term>(dst, buf, d30, d31, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 4) Save2<term>(dst, buf, d40, d41, bias, norm, zero), dst += dD, buf += dB;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term>(dst, buf, d00, d01, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 1) Save2<term>(dst, buf, d10, d11, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 2) Save2<term>(dst, buf, d20, d21, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 3) Save2<term>(dst, buf, d30, d31, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 4) Save2<term>(dst, buf, d40, d41, bias, norm, zero, dstC), dst += dD, buf += dB;
                }
            }
            else
            {
                if (update)
                {
                    if (M > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 0);
                    if (M > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 0);
                    if (M > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 0);
                    if (M > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 0);
                    if (M > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 0);
                }
                else
                {
                    if (M > 0) d00 = _mm_setzero_si128();
                    if (M > 1) d10 = _mm_setzero_si128();
                    if (M > 2) d20 = _mm_setzero_si128();
                    if (M > 3) d30 = _mm_setzero_si128();
                    if (M > 4) d40 = _mm_setzero_si128();
                }
                for (size_t offs = 0; offs < srcC; offs += 4)
                {
                    w0 = _mm_loadu_si128((__m128i*)weight0);
                    if (M > 0) s0 = Set4(src0 + offs), Madd4<true>(d00, s0, w0);
                    if (M > 1) s0 = Set4(src1 + offs), Madd4<true>(d10, s0, w0);
                    if (M > 2) s0 = Set4(src2 + offs), Madd4<true>(d20, s0, w0);
                    if (M > 3) s0 = Set4(src3 + offs), Madd4<true>(d30, s0, w0);
                    if (M > 4) s0 = Set4(src4 + offs), Madd4<true>(d40, s0, w0);
                    weight0 += A;
                }
                if (dstC == F)
                {
                    if (M > 0) Save1<term>(dst, buf, d00, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 1) Save1<term>(dst, buf, d10, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 2) Save1<term>(dst, buf, d20, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 3) Save1<term>(dst, buf, d30, bias, norm, zero), dst += dD, buf += dB;
                    if (M > 4) Save1<term>(dst, buf, d40, bias, norm, zero), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term>(dst, buf, d00, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term>(dst, buf, d10, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term>(dst, buf, d20, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term>(dst, buf, d30, bias, norm, zero, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term>(dst, buf, d40, bias, norm, zero, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*QuantizedConvolutionNhwcGemm_2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstC, int update, const int8_t* weight, const __m128i* bias, const __m128* norm, const __m128i* zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> QuantizedConvolutionNhwcGemm_2xM_Ptr GetQuantizedConvolutionNhwcGemm_2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return QuantizedConvolutionNhwcGemm_2xM<term, 1>;
            case 2: return QuantizedConvolutionNhwcGemm_2xM<term, 2>;
            case 3: return QuantizedConvolutionNhwcGemm_2xM<term, 3>;
            case 4: return QuantizedConvolutionNhwcGemm_2xM<term, 4>;
            case 5: return QuantizedConvolutionNhwcGemm_2xM<term, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term> void QuantizedConvolutionNhwcGemm_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH,
            size_t srcC, int update, const int8_t* weight, const int32_t* bias, const float* norm, const int32_t* zero, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            QuantizedConvolutionNhwcGemm_2xM_Ptr convolution_2xN = GetQuantizedConvolutionNhwcGemm_2xM<term>(n);
            QuantizedConvolutionNhwcGemm_2xM_Ptr convolution_2xM = GetQuantizedConvolutionNhwcGemm_2xM<term>(m);

            __m128 _norm[2];
            __m128i _bias[2], _zero[2];
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm_loadu_si128((__m128i*)(bias + dc) + 0);
                _bias[1] = _mm_loadu_si128((__m128i*)(bias + dc) + 1);
                _norm[0] = _mm_loadu_ps(norm + dc + 0);
                _norm[1] = _mm_loadu_ps(norm + dc + F);
                _zero[0] = _mm_loadu_si128((__m128i*)(zero + dc) + 0);
                _zero[1] = _mm_loadu_si128((__m128i*)(zero + dc) + 1);
                const uint8_t* s = src;
                int32_t* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_2xN(s, p, a, srcC, dC, update, weight, _bias, _norm, _zero, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_2xM(s, p, a, srcC, dC, update, weight, _bias, _norm, _zero, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_2<Term8iInterim>;
            if (p.dstT == SimdTensorData8u)
                convolutions[1] = QuantizedConvolutionNhwcGemm_2<Term8iLast8u>;
            else
                convolutions[1] = NULL;// QuantizedConvolutionNhwcGemm_2<Term8iLast32f>;
        }

        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, 5, 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            if (_src8u)
            {
                AlgParam& a = _alg;
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                    _convert = QuantizedConvolutionNhwcGemmReorder;
            }
            else
                assert(0);
            Set(p, _alg, _convolutions);
        }
    }
#endif
}

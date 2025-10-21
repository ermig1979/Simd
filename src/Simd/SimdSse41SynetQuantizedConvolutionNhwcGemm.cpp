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
#include "Simd/SimdSynetQuantizedActivation.h"
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

        static void QuantizedConvolutionNhwcGemmReorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
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
                                    memset(row, zero, p.srcC);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, zero, p.kernelX * p.srcC);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    for (size_t g = 0; g < gap; ++g)
                        *(row++) = 0;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M> void QuantizedConvolutionNhwcGemm_i2xM(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, 
            int update, const int8_t* weight0, const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, int32_t* buf, uint8_t* dst)
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
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    dstC -= F;
                    if (M > 0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
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
                    if (M > 0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero), dst += dD, buf += dB;
                }
                else
                {
                    if (M > 0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                    if (M > 4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, dstC), dst += dD, buf += dB;
                }
            }
        }

        typedef void(*QuantizedConvolutionNhwcGemm_i2xM_Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstC, int update, const int8_t* weight, 
            const __m128i* sBias, const __m128* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, const __m128* params, const __m128& dNorm, const __m128i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> QuantizedConvolutionNhwcGemm_i2xM_Ptr GetQuantizedConvolutionNhwcGemm_i2xM(size_t M)
        {
            switch (M)
            {
            case 0: return NULL;
            case 1: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 1>;
            case 2: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 2>;
            case 3: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 3>;
            case 4: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 4>;
            case 5: return QuantizedConvolutionNhwcGemm_i2xM<term, type, 5>;
            }
            assert(0);
            return NULL;
        }

        template<Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcGemm_i2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int update, const int8_t* weight,
            const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* buf, uint8_t* dst)
        {
            size_t n1 = dstH * p.dstW, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.dB, dD = p.dstC * a.elem, dS = a.bufK;
            QuantizedConvolutionNhwcGemm_i2xM_Ptr convolution_i2xN = GetQuantizedConvolutionNhwcGemm_i2xM<term, type>(n);
            QuantizedConvolutionNhwcGemm_i2xM_Ptr convolution_i2xM = GetQuantizedConvolutionNhwcGemm_i2xM<term, type>(m);

            __m128 _sNorm[2], _iScale, _params[2], _dNorm;
            __m128i _sBias[2], _dZero = _mm_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm_set1_epi32(-iZero);
                _iHi = _mm_set1_epi32(255 - iZero);
                _iScale = _mm_set1_ps(iScale);
                _dNorm = _mm_set1_ps(dNorm);
                _params[0] = _mm_set1_ps(params[0]);
                _params[1] = _mm_set1_ps(params[1]);
            }
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _sBias[0] = _mm_loadu_si128((__m128i*)(sBias + dc) + 0);
                _sBias[1] = _mm_loadu_si128((__m128i*)(sBias + dc) + 1);
                _sNorm[0] = _mm_loadu_ps(sNorm + dc + 0);
                _sNorm[1] = _mm_loadu_ps(sNorm + dc + F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm_loadu_ps(params + dc + 0);
                    _params[1] = _mm_loadu_ps(params + dc + F);
                }
                const uint8_t* s = src;
                int32_t* b = buf + dc;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, s += n * dS, b += n * dB, d += n * dD)
                    convolution_i2xN(s, p, a, srcC, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                for (; i < n1; i += m, s += m * dS, b += m * dB, d += m * dD)
                    convolution_i2xM(s, p, a, srcC, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                weight += dW;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_i2<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                convolutions[1] = NULL;
            }
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

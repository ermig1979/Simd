/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
        typedef Base::SynetQuantizedConvolutionNchwGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNchwGemm::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------
 
        template<Term8iType term, SimdConvolutionActivationType type, int N> void SynetQuantizedConvolutionNchwGemm_Gemm2xN(const int8_t* weight0, const ConvParam& p, 
            const AlgParam& a, size_t K, size_t M, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m128i& iLo, const __m128i& iHi, 
            const __m128& iScale, const float* params, const __m128* _params, const __m128& dNorm, const __m128i& dZero, int32_t* buf, uint8_t* dst)
        {
            __m128i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, w0, s0, s1;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem;
            const uint8_t* src1 = src0 + K * F;
            const int8_t* weight1 = weight0 + 1 * K;
            const int8_t* weight2 = weight0 + 2 * K;
            const int8_t* weight3 = weight0 + 3 * K;
            const int8_t* weight4 = weight0 + 4 * K;
            if (M > F)
            {
                if (update)
                {
                    if (N > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 0), d01 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 1);
                    if (N > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 0), d11 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 1);
                    if (N > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 0), d21 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 1);
                    if (N > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 0), d31 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 1);
                    if (N > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 0), d41 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 1);
                }
                else
                {
                    if (N > 0) d00 = _mm_setzero_si128(), d01 = _mm_setzero_si128();
                    if (N > 1) d10 = _mm_setzero_si128(), d11 = _mm_setzero_si128();
                    if (N > 2) d20 = _mm_setzero_si128(), d21 = _mm_setzero_si128();
                    if (N > 3) d30 = _mm_setzero_si128(), d31 = _mm_setzero_si128();
                    if (N > 4) d40 = _mm_setzero_si128(), d41 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    s0 = _mm_loadu_si128((__m128i*)src0);
                    s1 = _mm_loadu_si128((__m128i*)src1);
                    if (N > 0) w0 = Set4(weight0 + k), Madd4<true>(d00, s0, w0), Madd4<true>(d01, s1, w0);
                    if (N > 1) w0 = Set4(weight1 + k), Madd4<true>(d10, s0, w0), Madd4<true>(d11, s1, w0);
                    if (N > 2) w0 = Set4(weight2 + k), Madd4<true>(d20, s0, w0), Madd4<true>(d21, s1, w0);
                    if (N > 3) w0 = Set4(weight3 + k), Madd4<true>(d30, s0, w0), Madd4<true>(d31, s1, w0);
                    if (N > 4) w0 = Set4(weight4 + k), Madd4<true>(d40, s0, w0), Madd4<true>(d41, s1, w0);
                    src0 += A, src1 += A;
                }
            }
            else
            {
                if (update)
                {
                    if (N > 0) d00 = _mm_loadu_si128((__m128i*)(buf + 0 * dB) + 0);
                    if (N > 1) d10 = _mm_loadu_si128((__m128i*)(buf + 1 * dB) + 0);
                    if (N > 2) d20 = _mm_loadu_si128((__m128i*)(buf + 2 * dB) + 0);
                    if (N > 3) d30 = _mm_loadu_si128((__m128i*)(buf + 3 * dB) + 0);
                    if (N > 4) d40 = _mm_loadu_si128((__m128i*)(buf + 4 * dB) + 0);
                }
                else
                {
                    if (N > 0) d00 = _mm_setzero_si128();
                    if (N > 1) d10 = _mm_setzero_si128();
                    if (N > 2) d20 = _mm_setzero_si128();
                    if (N > 3) d30 = _mm_setzero_si128();
                    if (N > 4) d40 = _mm_setzero_si128();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    s0 = _mm_loadu_si128((__m128i*)src0);
                    if (N > 0) w0 = Set4(weight0 + k), Madd4<true>(d00, s0, w0);
                    if (N > 1) w0 = Set4(weight1 + k), Madd4<true>(d10, s0, w0);
                    if (N > 2) w0 = Set4(weight2 + k), Madd4<true>(d20, s0, w0);
                    if (N > 3) w0 = Set4(weight3 + k), Madd4<true>(d30, s0, w0);
                    if (N > 4) w0 = Set4(weight4 + k), Madd4<true>(d40, s0, w0);
                    src0 += A;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        typedef void(*SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr)(const int8_t* weight0, const ConvParam& p, const AlgParam& a, size_t K, size_t M, int update, 
            const uint8_t* src, const int32_t* sBias, const float* sNorm, const __m128i& iLo, const __m128i& iHi, const __m128& iScale, 
            const float* params, const __m128* _params, const __m128& dNorm, const __m128i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr GetSynetQuantizedConvolutionNchwGemm_Gemm2xN(size_t N)
        {
            switch (N)
            {
            case 0: return NULL;
            case 1: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 1>;
            case 2: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 2>;
            case 3: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 3>;
            case 4: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 4>;
            case 5: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 5>;
            default: assert(0);  return NULL;
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static void SynetQuantizedConvolutionNchwGemm_Gemm(const int8_t* weight, 
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t K, int update, const uint8_t* src, const int32_t* sBias,
            const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 5;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xN = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(n);
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xM = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(m);

            __m128 _iScale, _params[2], _dNorm;
            __m128i _dZero = _mm_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm_set1_epi32(-iZero);
                _iHi = _mm_set1_epi32(255 - iZero);
                _iScale = _mm_set1_ps(iScale);
                _dNorm = _mm_set1_ps(dNorm);
                _params[0] = _mm_set1_ps(params[0]);
                _params[1] = _mm_set1_ps(params[1]);
            }
            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const int8_t* w = weight;
                int32_t* b = buf + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                    gemm_2xN(w, p, a, K, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                for (; i < n1; i += m, w += m * dW, b += m * dB, d += m * dD)
                    gemm_2xM(w, p, a, K, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemm[1] = SynetQuantizedConvolutionNchwGemm_Gemm<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                gemm[1] = NULL;
            }
        }

         //-----------------------------------------------------------------------------------------


        SynetQuantizedConvolutionNchwGemm::SynetQuantizedConvolutionNchwGemm(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNchwGemm(p)
        {
            SetAlgParam(F, F * 2, 5, 4);
            //if (_is1x1)
            //    _conv = QuantizedConvolutionNchwGemm_Reorder1x1;
            //else
            //    _conv = NULL;
            SetGemm(p, _alg, _gemm);
        }
    }
#endif
}

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
#if defined(SIMD_AVX512VNNI_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512vnni
    {
        typedef Base::SynetQuantizedConvolutionNchwGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNchwGemm::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------
 
        template<Term8iType term, SimdConvolutionActivationType type, int N> void SynetQuantizedConvolutionNchwGemm_Gemm2xN(const int8_t* weight0, const ConvParam& p, 
            const AlgParam& a, size_t K, size_t M, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi, 
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            __m512i d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51, d60, d61, d70, d71, d80, d81, d90, d91, dA0, dA1, dB0, dB1, w0, s0, s1;
            size_t dB = a.bufN, dD = a.N * a.elem;
            const uint8_t* src1 = src0 + K * F;
            const int8_t* weight1 = weight0 + 1 * K;
            const int8_t* weight2 = weight0 + 2 * K;
            const int8_t* weight3 = weight0 + 3 * K;
            const int8_t* weight4 = weight0 + 4 * K;
            const int8_t* weight5 = weight0 + 5 * K;
            if (M > F)
            {
                if (update)
                {
                    if (N > 0x0) d00 = _mm512_loadu_si512((__m512i*)(buf + 0x0 * dB) + 0), d01 = _mm512_loadu_si512((__m512i*)(buf + 0x0 * dB) + 1);
                    if (N > 0x1) d10 = _mm512_loadu_si512((__m512i*)(buf + 0x1 * dB) + 0), d11 = _mm512_loadu_si512((__m512i*)(buf + 0x1 * dB) + 1);
                    if (N > 0x2) d20 = _mm512_loadu_si512((__m512i*)(buf + 0x2 * dB) + 0), d21 = _mm512_loadu_si512((__m512i*)(buf + 0x2 * dB) + 1);
                    if (N > 0x3) d30 = _mm512_loadu_si512((__m512i*)(buf + 0x3 * dB) + 0), d31 = _mm512_loadu_si512((__m512i*)(buf + 0x3 * dB) + 1);
                    if (N > 0x4) d40 = _mm512_loadu_si512((__m512i*)(buf + 0x4 * dB) + 0), d41 = _mm512_loadu_si512((__m512i*)(buf + 0x4 * dB) + 1);
                    if (N > 0x5) d50 = _mm512_loadu_si512((__m512i*)(buf + 0x5 * dB) + 0), d51 = _mm512_loadu_si512((__m512i*)(buf + 0x5 * dB) + 1);
                    if (N > 0x6) d60 = _mm512_loadu_si512((__m512i*)(buf + 0x6 * dB) + 0), d61 = _mm512_loadu_si512((__m512i*)(buf + 0x6 * dB) + 1);
                    if (N > 0x7) d70 = _mm512_loadu_si512((__m512i*)(buf + 0x7 * dB) + 0), d71 = _mm512_loadu_si512((__m512i*)(buf + 0x7 * dB) + 1);
                    if (N > 0x8) d80 = _mm512_loadu_si512((__m512i*)(buf + 0x8 * dB) + 0), d81 = _mm512_loadu_si512((__m512i*)(buf + 0x8 * dB) + 1);
                    if (N > 0x9) d90 = _mm512_loadu_si512((__m512i*)(buf + 0x9 * dB) + 0), d91 = _mm512_loadu_si512((__m512i*)(buf + 0x9 * dB) + 1);
                    if (N > 0xA) dA0 = _mm512_loadu_si512((__m512i*)(buf + 0xA * dB) + 0), dA1 = _mm512_loadu_si512((__m512i*)(buf + 0xA * dB) + 1);
                    if (N > 0xB) dB0 = _mm512_loadu_si512((__m512i*)(buf + 0xB * dB) + 0), dB1 = _mm512_loadu_si512((__m512i*)(buf + 0xB * dB) + 1);
                }
                else
                {
                    if (N > 0x0) d00 = _mm512_setzero_si512(), d01 = _mm512_setzero_si512();
                    if (N > 0x1) d10 = _mm512_setzero_si512(), d11 = _mm512_setzero_si512();
                    if (N > 0x2) d20 = _mm512_setzero_si512(), d21 = _mm512_setzero_si512();
                    if (N > 0x3) d30 = _mm512_setzero_si512(), d31 = _mm512_setzero_si512();
                    if (N > 0x4) d40 = _mm512_setzero_si512(), d41 = _mm512_setzero_si512();
                    if (N > 0x5) d50 = _mm512_setzero_si512(), d51 = _mm512_setzero_si512();
                    if (N > 0x6) d60 = _mm512_setzero_si512(), d61 = _mm512_setzero_si512();
                    if (N > 0x7) d70 = _mm512_setzero_si512(), d71 = _mm512_setzero_si512();
                    if (N > 0x8) d80 = _mm512_setzero_si512(), d81 = _mm512_setzero_si512();
                    if (N > 0x9) d90 = _mm512_setzero_si512(), d91 = _mm512_setzero_si512();
                    if (N > 0xA) dA0 = _mm512_setzero_si512(), dA1 = _mm512_setzero_si512();
                    if (N > 0xB) dB0 = _mm512_setzero_si512(), dB1 = _mm512_setzero_si512();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    s0 = _mm512_loadu_si512((__m512i*)src0);
                    s1 = _mm512_loadu_si512((__m512i*)src1);
                    if (N > 0x0) w0 = Set4(weight0 + k), Madd4<false>(d00, s0, w0), Madd4<false>(d01, s1, w0);
                    if (N > 0x1) w0 = Set4(weight1 + k), Madd4<false>(d10, s0, w0), Madd4<false>(d11, s1, w0);
                    if (N > 0x2) w0 = Set4(weight2 + k), Madd4<false>(d20, s0, w0), Madd4<false>(d21, s1, w0);
                    if (N > 0x3) w0 = Set4(weight3 + k), Madd4<false>(d30, s0, w0), Madd4<false>(d31, s1, w0);
                    if (N > 0x4) w0 = Set4(weight4 + k), Madd4<false>(d40, s0, w0), Madd4<false>(d41, s1, w0);
                    if (N > 0x5) w0 = Set4(weight5 + k), Madd4<false>(d50, s0, w0), Madd4<false>(d51, s1, w0);
                    if (N > 0x6) w0 = Set4(weight0 + k + 6 * K), Madd4<false>(d60, s0, w0), Madd4<false>(d61, s1, w0);
                    if (N > 0x7) w0 = Set4(weight1 + k + 6 * K), Madd4<false>(d70, s0, w0), Madd4<false>(d71, s1, w0);
                    if (N > 0x8) w0 = Set4(weight2 + k + 6 * K), Madd4<false>(d80, s0, w0), Madd4<false>(d81, s1, w0);
                    if (N > 0x9) w0 = Set4(weight3 + k + 6 * K), Madd4<false>(d90, s0, w0), Madd4<false>(d91, s1, w0);
                    if (N > 0xA) w0 = Set4(weight4 + k + 6 * K), Madd4<false>(dA0, s0, w0), Madd4<false>(dA1, s1, w0);
                    if (N > 0xB) w0 = Set4(weight5 + k + 6 * K), Madd4<false>(dB0, s0, w0), Madd4<false>(dB1, s1, w0);
                    src0 += A, src1 += A;
                }
                __mmask32 mask = TailMask32(M);
                if (N > 0x0) Save2<term, type>(dst, buf, d00, d01, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x0, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x1) Save2<term, type>(dst, buf, d10, d11, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x1, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x2) Save2<term, type>(dst, buf, d20, d21, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x2, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x3) Save2<term, type>(dst, buf, d30, d31, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x3, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x4) Save2<term, type>(dst, buf, d40, d41, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x4, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x5) Save2<term, type>(dst, buf, d50, d51, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x5, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x6) Save2<term, type>(dst, buf, d60, d61, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x6, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x7) Save2<term, type>(dst, buf, d70, d71, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x7, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x8) Save2<term, type>(dst, buf, d80, d81, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x8, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x9) Save2<term, type>(dst, buf, d90, d91, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x9, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0xA) Save2<term, type>(dst, buf, dA0, dA1, sBias, sNorm, iLo, iHi, iScale, _params, params, 0xA, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0xB) Save2<term, type>(dst, buf, dB0, dB1, sBias, sNorm, iLo, iHi, iScale, _params, params, 0xB, dNorm, dZero, mask), dst += dD, buf += dB;
            }
            else
            {
                if (update)
                {
                    if (N > 0x0) d00 = _mm512_loadu_si512((__m512i*)(buf + 0x0 * dB) + 0);
                    if (N > 0x1) d10 = _mm512_loadu_si512((__m512i*)(buf + 0x1 * dB) + 0);
                    if (N > 0x2) d20 = _mm512_loadu_si512((__m512i*)(buf + 0x2 * dB) + 0);
                    if (N > 0x3) d30 = _mm512_loadu_si512((__m512i*)(buf + 0x3 * dB) + 0);
                    if (N > 0x4) d40 = _mm512_loadu_si512((__m512i*)(buf + 0x4 * dB) + 0);
                    if (N > 0x5) d50 = _mm512_loadu_si512((__m512i*)(buf + 0x5 * dB) + 0);
                    if (N > 0x6) d60 = _mm512_loadu_si512((__m512i*)(buf + 0x6 * dB) + 0);
                    if (N > 0x7) d70 = _mm512_loadu_si512((__m512i*)(buf + 0x7 * dB) + 0);
                    if (N > 0x8) d80 = _mm512_loadu_si512((__m512i*)(buf + 0x8 * dB) + 0);
                    if (N > 0x9) d90 = _mm512_loadu_si512((__m512i*)(buf + 0x9 * dB) + 0);
                    if (N > 0xA) dA0 = _mm512_loadu_si512((__m512i*)(buf + 0xA * dB) + 0);
                    if (N > 0xB) dB0 = _mm512_loadu_si512((__m512i*)(buf + 0xB * dB) + 0);
                }
                else
                {
                    if (N > 0x0) d00 = _mm512_setzero_si512();
                    if (N > 0x1) d10 = _mm512_setzero_si512();
                    if (N > 0x2) d20 = _mm512_setzero_si512();
                    if (N > 0x3) d30 = _mm512_setzero_si512();
                    if (N > 0x4) d40 = _mm512_setzero_si512();
                    if (N > 0x5) d50 = _mm512_setzero_si512();
                    if (N > 0x6) d60 = _mm512_setzero_si512();
                    if (N > 0x7) d70 = _mm512_setzero_si512();
                    if (N > 0x8) d80 = _mm512_setzero_si512();
                    if (N > 0x9) d90 = _mm512_setzero_si512();
                    if (N > 0xA) dA0 = _mm512_setzero_si512();
                    if (N > 0xB) dB0 = _mm512_setzero_si512();
                }
                for (size_t k = 0; k < K; k += 4)
                {
                    s0 = _mm512_loadu_si512((__m512i*)src0);
                    if (N > 0x0) w0 = Set4(weight0 + k), Madd4<false>(d00, s0, w0);
                    if (N > 0x1) w0 = Set4(weight1 + k), Madd4<false>(d10, s0, w0);
                    if (N > 0x2) w0 = Set4(weight2 + k), Madd4<false>(d20, s0, w0);
                    if (N > 0x3) w0 = Set4(weight3 + k), Madd4<false>(d30, s0, w0);
                    if (N > 0x4) w0 = Set4(weight4 + k), Madd4<false>(d40, s0, w0);
                    if (N > 0x5) w0 = Set4(weight5 + k), Madd4<false>(d50, s0, w0);
                    if (N > 0x6) w0 = Set4(weight0 + k + 6 * K), Madd4<false>(d60, s0, w0);
                    if (N > 0x7) w0 = Set4(weight1 + k + 6 * K), Madd4<false>(d70, s0, w0);
                    if (N > 0x8) w0 = Set4(weight2 + k + 6 * K), Madd4<false>(d80, s0, w0);
                    if (N > 0x9) w0 = Set4(weight3 + k + 6 * K), Madd4<false>(d90, s0, w0);
                    if (N > 0xA) w0 = Set4(weight4 + k + 6 * K), Madd4<false>(dA0, s0, w0);
                    if (N > 0xB) w0 = Set4(weight5 + k + 6 * K), Madd4<false>(dB0, s0, w0);
                    src0 += A;
                }
                __mmask16 mask = TailMask16(M);
                if (N > 0x0) Save1<term, type>(dst, buf, d00, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x0, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x1) Save1<term, type>(dst, buf, d10, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x1, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x2) Save1<term, type>(dst, buf, d20, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x2, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x3) Save1<term, type>(dst, buf, d30, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x3, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x4) Save1<term, type>(dst, buf, d40, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x4, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x5) Save1<term, type>(dst, buf, d50, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x5, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x6) Save1<term, type>(dst, buf, d60, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x6, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x7) Save1<term, type>(dst, buf, d70, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x7, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x8) Save1<term, type>(dst, buf, d80, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x8, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0x9) Save1<term, type>(dst, buf, d90, sBias, sNorm, iLo, iHi, iScale, _params, params, 0x9, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0xA) Save1<term, type>(dst, buf, dA0, sBias, sNorm, iLo, iHi, iScale, _params, params, 0xA, dNorm, dZero, mask), dst += dD, buf += dB;
                if (N > 0xB) Save1<term, type>(dst, buf, dB0, sBias, sNorm, iLo, iHi, iScale, _params, params, 0xB, dNorm, dZero, mask), dst += dD, buf += dB;
            }
        }

        //-----------------------------------------------------------------------------------------

        typedef void(*SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr)(const int8_t* weight0, const ConvParam& p, const AlgParam& a, size_t K, size_t M, int update, 
            const uint8_t* src, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, 
            const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr GetSynetQuantizedConvolutionNchwGemm_Gemm2xN(size_t N)
        {
            switch (N)
            {
            case 0x0: return NULL;
            case 0x1: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x1>;
            case 0x2: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x2>;
            case 0x3: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x3>;
            case 0x4: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x4>;
            case 0x5: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x5>;
            case 0x6: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x6>;
            case 0x7: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x7>;
            case 0x8: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x8>;
            case 0x9: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0x9>;
            case 0xA: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0xA>;
            case 0xB: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0xB>;
            case 0xC: return SynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type, 0xC>;
            default: assert(0);  return NULL;
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> static void SynetQuantizedConvolutionNchwGemm_Gemm(const int8_t* weight, 
            const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t K, int update, const uint8_t* src, const int32_t* sBias,
            const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 12;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.bufN, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xN = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(n);
            SynetQuantizedConvolutionNchwGemm_Gemm2xN_Ptr gemm_2xM = GetSynetQuantizedConvolutionNchwGemm_Gemm2xN<term, type>(m);

            __m512 _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const int8_t* w = weight;
                int32_t* b = sum + ds;
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
            : Avx512bw::SynetQuantizedConvolutionNchwGemm(p)
        {
            SetAlgParam(F, F * 2, 12, 4);
            SetGemm(p, _alg, _gemm);
        }
    }
#endif
}

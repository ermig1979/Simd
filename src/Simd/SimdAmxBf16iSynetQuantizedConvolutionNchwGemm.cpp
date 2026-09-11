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
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNchwGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNchwGemm::GemmPtr GemmPtr;

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNchwGemm_32x32(const int8_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.bufN, dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = 64, strideW = (int)K;
            const int8_t* weight1 = weight0 + K * F;
            const uint8_t* src1 = src0 + K * F;

            if (cfg)
                SetTileConf2x2(dstC, dstS);
            if (update)
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
                _tile_stream_loadd(3, buf + 16 * dB + F, strideB);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, weight0, strideW);
            _tile_loadd(6, src0 + k * 16, strideS);
            for (; k < K64; weight1 += stepW)
            {
                _tile_loadd(7, src1 + k * 16, strideS);
                _tile_stream_loadd(5, weight1, strideW);
                _tile_dpbsud(0, 4, 6);
                _tile_dpbsud(1, 4, 7);
                weight0 += stepW;
                _tile_stream_loadd(4, weight0, strideW);
                _tile_dpbsud(2, 5, 6);
                k += 64;
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbsud(3, 5, 7);
            }
            _tile_loadd(7, src1 + k * 16, strideS);
            _tile_stream_loadd(5, weight1, strideW);
            _tile_dpbsud(0, 4, 6);
            _tile_dpbsud(1, 4, 7);
            _tile_dpbsud(2, 5, 6);
            _tile_dpbsud(3, 5, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstS - F);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply2x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
                for (; dc < dstC; ++dc)
                    Apply2<term, type>(dst + dc * dD, buf + dc * dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
                TileMoveToMemory(buf + 16 * dB + F, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNchwGemm_32x16(const int8_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.bufN, dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = 64, strideW = (int)K;
            const int8_t* weight1 = weight0 + K * F;

            if (cfg)
                SetTileConf2x1(dstC, dstS);
            if (update)
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(2);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_stream_loadd(4, weight0, strideW);
            for (; k < K64; k += 64, weight1 += stepW)
            {
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_stream_loadd(5, weight1, strideW);
                _tile_dpbsud(0, 4, 6);
                weight0 += stepW;
                _tile_stream_loadd(4, weight0, strideW);
                _tile_dpbsud(2, 5, 6);
            }
            _tile_loadd(6, src0 + k * 16, strideS);
            _tile_stream_loadd(5, weight1, strideW);
            _tile_dpbsud(0, 4, 6);
            _tile_dpbsud(2, 5, 6);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstS);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply1x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
                for (; dc < dstC; ++dc)
                    Apply1<term, type>(dst + dc * dD, buf + dc * dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNchwGemm_16x32(const int8_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.bufN, dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = 64, strideW = (int)K;
            const uint8_t* src1 = src0 + K * F;

            if (cfg)
                SetTileConf1x2(dstC, dstS);
            if (update)
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int K64 = (int)K - 64, k = 0;
            _tile_loadd(6, src0 + k * 16, strideS);
            for (; k < K64; weight0 += stepW)
            {
                _tile_stream_loadd(4, weight0, strideW);
                _tile_loadd(7, src1 + k * 16, strideS);
                _tile_dpbsud(0, 4, 6);
                k += 64;
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbsud(1, 4, 7);
            }
            _tile_stream_loadd(4, weight0, strideW);
            _tile_loadd(7, src1 + k * 16, strideS);
            _tile_dpbsud(0, 4, 6);
            _tile_dpbsud(1, 4, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstS - F);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply2x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
                for (; dc < dstC; ++dc)
                    Apply2<term, type>(dst + dc * dD, buf + dc * dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNchwGemm_16x16(const int8_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.bufN, dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = 64, strideW = (int)K;

            if (cfg)
                SetTileConf1x1(dstC, dstS);
            if (update)
            {
                _tile_stream_loadd(0, buf + 0, strideB);
            }
            else
            {
                _tile_zero(0);
            }
            for (size_t k = 0; k < K; k += 64, weight0 += stepW)
            {
                _tile_stream_loadd(4, weight0, strideW);
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbsud(0, 4, 6);
            }
            _tile_stored(0, buf + 0, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstS);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply1x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
                for (; dc < dstC; ++dc)
                    Apply1<term, type>(dst + dc * dD, buf + dc * dB, sBias, sNorm, iLo, iHi, iScale, _params, params, dc, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
            }
        }

        typedef void (*QuantizedConvolutionNchwGemmPtr)(const int8_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int update, const uint8_t* src0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const float* params, const __m512* _params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNchwGemm_2(const int8_t* weight, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t K, int update, const uint8_t* src, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale,
            const float* params, float dNorm, int32_t dZero, int32_t* sum, int32_t* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.bufN, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
            QuantizedConvolutionNchwGemmPtr body_2 = QuantizedConvolutionNchwGemm_32x32<term, type, 0>;
            QuantizedConvolutionNchwGemmPtr tail_2 = m > 16 ? QuantizedConvolutionNchwGemm_32x32<term, type, 0> : QuantizedConvolutionNchwGemm_16x32<term, type, 0>;
            QuantizedConvolutionNchwGemmPtr body_1 = QuantizedConvolutionNchwGemm_32x16<term, type, 0>;
            QuantizedConvolutionNchwGemmPtr tail_1 = m > 16 ? QuantizedConvolutionNchwGemm_32x16<term, type, 0> : QuantizedConvolutionNchwGemm_16x16<term, type, 0>;

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

            SetTileConfFull();
            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const int8_t* w = weight;
                int32_t* b = sum + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                if (dS > F)
                {
                    for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                        body_2(w, p, a, K, n, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                    if (m)
                        tail_2(w, p, a, K, m, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                }
                else
                {
                    for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                        body_1(w, p, a, K, n, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                    if (m)
                        tail_1(w, p, a, K, m, dS, update, src, sBias + i, sNorm + i, _iLo, _iHi, _iScale, params + i * dp, _params, _dNorm, _dZero, b, d);
                }
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void SetGemm(const ConvParam& p, const AlgParam& a, GemmPtr* gemm)
        {
            gemm[0] = QuantizedConvolutionNchwGemm_2<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: gemm[1] = QuantizedConvolutionNchwGemm_2<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                gemm[1] = NULL;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNchwGemm::SynetQuantizedConvolutionNchwGemm(const ConvParam& p)
            : Avx512vnni::SynetQuantizedConvolutionNchwGemm(p)
        {
            if (_alg.K <= 32)
                return;
            SetAlgParam(F, F * 2, F * 2, 64);
            SetGemm(p, _alg, _gemm);
        }
    }
#endif
}

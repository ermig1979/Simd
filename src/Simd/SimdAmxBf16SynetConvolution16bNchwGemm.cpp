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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
	{
		typedef Base::SynetConvolution16bNchwGemm::AlgParam AlgParam;
		typedef Base::SynetConvolution16bNchwGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNchwGemm_32x32(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            int dB = int(a.sumBuf ? a.bufN : a.N), dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = a.reorderType ? 512 : 32, strideW = a.reorderType ? 64 : (int)K * 2;
            const uint16_t* weight1 = weight0 + K * F;
            const uint16_t* src1 = src0 + K * F;
            if (cfg)
                SetTileConf2x2(dstC, dstS);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
                _tile_stream_loadd(3, buf + 16 * dB + F, strideB);
            }

            size_t K32 = K - 32, k = 0;
            _tile_stream_loadd(4, weight0, strideW);
            _tile_loadd(6, src0 + k * 16, strideS);
            for (; k < K32; weight1 += stepW)
            {
                _tile_loadd(7, src1 + k * 16, strideS);
                _tile_stream_loadd(5, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                weight0 += stepW;
                _tile_stream_loadd(4, weight0, strideW);
                _tile_dpbf16ps(2, 5, 6);
                k += 32;
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, src1 + k * 16, strideS);
            _tile_stream_loadd(5, weight1, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstS - F);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply2x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, bias, params, dc, tailD);
                for (; dc < dstC; ++dc)
                    Apply2<term, type>(dst + dc * dD, buf + dc * dB, bias, params, dc, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
                TileMoveToMemory(buf + 16 * dB + F, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNchwGemm_32x16(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            int dB = int(a.sumBuf ? a.bufN : a.N), dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = a.reorderType ? 512 : 32, strideW = a.reorderType ? 64 : (int)K * 2;
            const uint16_t* weight1 = weight0 + K * F;
            if (cfg)
                SetTileConf2x1(dstC, dstS);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(2, buf + 16 * dB + 0, strideB);
            }

            size_t K32 = K - 32, k = 0;
            _tile_stream_loadd(4, weight0, strideW);
            for (; k < K32; k += 32, weight1 += stepW)
            {
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_stream_loadd(5, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += stepW;
                _tile_stream_loadd(4, weight0, strideW);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, src0 + k * 16, strideS);
            _tile_stream_loadd(5, weight1, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstS);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply1x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, bias, params, dc, tailD);
                for (; dc < dstC; ++dc)
                    Apply1<term, type>(dst + dc * dD, buf + dc * dB, bias, params, dc, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNchwGemm_16x32(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            int dB = int(a.sumBuf ? a.bufN : a.N), dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = a.reorderType ? 512 : 32, strideW = a.reorderType ? 64 : (int)K * 2;
            const uint16_t* src1 = src0 + K * F;

            if (cfg)
                SetTileConf1x2(dstC, dstS);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
                _tile_stream_loadd(1, buf + F, strideB);
            }

            size_t K32 = K - 32, k = 0;
            _tile_loadd(6, src0 + k * 16, strideS);
            for (; k < K32; weight0 += stepW)
            {
                _tile_stream_loadd(4, weight0, strideW);
                _tile_loadd(7, src1 + k * 16, strideS);
                _tile_dpbf16ps(0, 4, 6);
                k += 32;
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, weight0, strideW);
            _tile_loadd(7, src1 + k * 16, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstS - F);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply2x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, bias, params, dc, tailD);
                for (; dc < dstC; ++dc)
                    Apply2<term, type>(dst + dc * dD, buf + dc * dB, bias, params, dc, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void Convolution16bNchwGemm_16x16(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            int dB = int(a.sumBuf ? a.bufN : a.N), dD = int(a.N * a.elem), strideB = dB * 4, strideS = 64;
            int stepW = a.reorderType ? 512 : 32, strideW = a.reorderType ? 64 : (int)K * 2;

            if (cfg)
                SetTileConf1x1(dstC, dstS);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, buf + 0, strideB);
            }
            for (size_t k = 0; k < K; k += 32, weight0 += stepW)
            {
                _tile_stream_loadd(4, weight0, strideW);
                _tile_loadd(6, src0 + k * 16, strideS);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, buf + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstS);
                size_t dstC8 = AlignLo(dstC, 8), dc = 0;
                for (; dc < dstC8; dc += 8)
                    Apply1x8<term, type>(dst + dc * dD, dD, buf + dc * dB, dB, bias, params, dc, tailD);
                for (; dc < dstC; ++dc)
                    Apply1<term, type>(dst + dc * dD, buf + dc * dB, bias, params, dc, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
            }
        }

        typedef void (*Convolution16bNchwGemmPtr)(const uint16_t* weight0, const ConvParam& p, const AlgParam& a,
            size_t K, size_t dstC, size_t dstS, int zero, const uint16_t* src0, const float* bias, const float* params, float* buf, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> void Convolution16bNchwGemm_2(const uint16_t* weight, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* src, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t dstS = dstH * p.dstW, n1 = dstC, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dB = a.sumBuf ? a.bufN : a.N, dD = a.N * a.elem, dW = K, dp = type == ::SimdConvolutionActivationPrelu ? 1 : 0;
#if 1
            Convolution16bNchwGemmPtr body_2 = Convolution16bNchwGemm_32x32<term, type, 0>;
            Convolution16bNchwGemmPtr tail_2 = m > 16 ? Convolution16bNchwGemm_32x32<term, type, 0> : Convolution16bNchwGemm_16x32<term, type, 0>;
            Convolution16bNchwGemmPtr body_1 = Convolution16bNchwGemm_32x16<term, type, 0>;
            Convolution16bNchwGemmPtr tail_1 = m > 16 ? Convolution16bNchwGemm_32x16<term, type, 0> : Convolution16bNchwGemm_16x16<term, type, 0>;

            SetTileConfFull();
#else
            Convolution16bNchwGemmPtr body_2 = Convolution16bNchwGemm_32x32<term, type, 1>;
            Convolution16bNchwGemmPtr tail_2 = m > 16 ? Convolution16bNchwGemm_32x32<term, type, 1> : Convolution16bNchwGemm_16x32<term, type, 1>;
            Convolution16bNchwGemmPtr body_1 = Convolution16bNchwGemm_32x16<term, type, 1>;
            Convolution16bNchwGemmPtr tail_1 = m > 16 ? Convolution16bNchwGemm_32x16<term, type, 1> : Convolution16bNchwGemm_16x16<term, type, 1>;
#endif

            for (size_t ds = 0; ds < dstS; ds += DF)
            {
                size_t dS = Simd::Min(DF, dstS - ds);
                const uint16_t* w = weight;
                float* b = buf + ds;
                uint8_t* d = dst + ds * a.elem;
                size_t i = 0;
                if (dS > F)
                {
                    for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                        body_2(w, p, a, K, n, dS, zero, src, bias + i, params + i * dp, b, d);
                    if (m)
                        tail_2(w, p, a, K, m, dS, zero, src, bias + i, params + i * dp, b, d);
                }
                else
                {
                    for (; i < nn; i += n, w += n * dW, b += n * dB, d += n * dD)
                        body_1(w, p, a, K, n, dS, zero, src, bias + i, params + i * dp, b, d);
                    if (m)
                        tail_1(w, p, a, K, m, dS, zero, src, bias + i, params + i * dp, b, d);
                }
                src += K * DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = Convolution16bNchwGemm_2<Term16bInterim, SimdConvolutionActivationIdentity>;
            if (p.dstT == SimdTensorData16b)
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast16b, type>;
            else
                convolutions[1] = Convolution16bNchwGemm_2<Term16bLast32f, type>;
        }

        SynetConvolution16bNchwGemm::SynetConvolution16bNchwGemm(const ConvParam& p)
            : Avx512bw::SynetConvolution16bNchwGemm(p)
        {
            SetAlgParam(F, F * 2, F * 2, F * 2, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
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

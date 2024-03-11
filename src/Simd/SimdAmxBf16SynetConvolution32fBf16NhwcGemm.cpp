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
#include "Simd/SimdSynetConvolution32fBf16.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)
    namespace AmxBf16
    {
        typedef Base::SynetConvolution32fBf16NhwcGemm::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16NhwcGemm::ConvolutionPtr Convolution;

        //-------------------------------------------------------------------------------------------------

        static void ConvertBf16NhwcGemm1x1(const float* src, const ConvParam32f& p, const SynetConvolution32fBf16NhwcGemm::AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (srcC32 < p.srcC)
            {
                srcMask[0] = TailMask16(p.srcC - srcC32 - F * 0);
                srcMask[1] = TailMask16(p.srcC - srcC32 - F * 1);
                dstMask[0] = __mmask32(-1);
            }
            src += yBeg * p.srcW * p.srcC;
            if (a.macroK < a.bufK)
            {
                //SIMD_PERF_BEG("reorder");
                size_t bodyK = AlignLoAny(a.bufK, a.macroK), tailK = a.bufK - bodyK;
                for (size_t dy = yBeg, dr = dy * p.dstW; dy < yEnd; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                    {
                        size_t sc = 0, mak = 0;
                        for (; mak < bodyK; mak += a.macroK)
                        {
                            uint16_t* buf = dst + mak * a.bufM + dr * a.macroK;
                            for (size_t scE = mak + a.macroK; sc < scE; sc += 32)
                                Float32ToBFloat16<false, false>(src + sc, buf + sc - mak, srcMask, dstMask);
                        }
                        if(tailK)
                        {
                            uint16_t* buf = dst + mak * a.bufM + dr * tailK;
                            for (; sc < srcC32; sc += 32)
                                Float32ToBFloat16<false, false>(src + sc, buf + sc - mak, srcMask, dstMask);
                            if (srcC32 < p.srcC)
                                Float32ToBFloat16<false, true>(src + sc, buf + sc - mak, srcMask, dstMask);
                        }
                        src += p.srcC;
                    }
                }
            }
            else if (srcC32 < p.srcC)
            {
                //SIMD_PERF_BEG("direct");
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        size_t sc = 0;
                        for (; sc < srcC32; sc += 32)
                            Float32ToBFloat16<false, false>(src + sc, dst + sc, srcMask, dstMask);
                        if (srcC32 < p.srcC)
                            Float32ToBFloat16<false, true>(src + sc, dst + sc, srcMask, dstMask);
                        src += p.srcC;
                        dst += a.bufK;
                    }
                }
            }
            else
            {
                //SIMD_PERF_BEG("solid");
                for (size_t n = (yEnd - yBeg) * p.srcW * p.srcC, i = 0; i < n; i += 32)
                    Float32ToBFloat16<false, false>(src + i, dst + i, srcMask, dstMask);
            }
        }

        static void ConvertBf16NhwcGemm(const float* src, const ConvParam32f& p, const SynetConvolution32fBf16NhwcGemm::AlgParam& a, size_t yBeg, size_t yEnd, uint16_t* dst)
        {
            //SIMD_PERF_FUNC();

            size_t srcC32 = AlignLo(p.srcC, 32);
            __mmask16 srcMask[2];
            __mmask32 dstMask[1];
            if (srcC32 < p.srcC)
            {
                srcMask[0] = TailMask16(p.srcC - srcC32 - F * 0);
                srcMask[1] = TailMask16(p.srcC - srcC32 - F * 1);
                dstMask[0] = TailMask32(p.srcC - srcC32);
            }
            uint16_t* buf = dst + a.bufM * a.bufK;
            size_t gap = a.bufK - a.K;
            __mmask32 gapMask = TailMask32(gap);
            for (size_t dy = yBeg, dr = a.macroK < a.bufK ? dy * p.dstW : 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    uint16_t* row = a.macroK < a.bufK ? buf : dst + dr * a.bufK;
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
                                    const float* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    size_t sc = 0;
                                    for (; sc < srcC32; sc += 32)
                                        Float32ToBFloat16<false, false>(ps + sc, row + sc, srcMask, dstMask);
                                    if (srcC32 < p.srcC)
                                        Float32ToBFloat16<false, true>(ps + sc, row + sc, srcMask, dstMask);
                                    row += p.srcC;
                                }
                                else
                                {
                                    memset(row, 0, p.srcC * 2);
                                    row += p.srcC;
                                }
                            }
                        }
                        else
                        {
                            memset(row, 0, p.kernelX * p.srcC * 2);
                            row += p.kernelX * p.srcC;
                        }
                    }
                    if (gap)
                    {
                        _mm512_mask_storeu_epi16(row, gapMask, _mm512_setzero_si512());
                        row += gap;
                    }
                    if (a.macroK < a.bufK)
                    {
                        for (size_t mak = 0; mak < a.bufK; mak += a.macroK)
                        {
                            size_t macroK = Simd::Min(a.bufK, mak + a.macroK) - mak;
                            memcpy(dst + mak * a.bufM + dr * macroK, buf + mak, macroK * 2);
                        }
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            int dD = (int)p.dstC, strideS = (int)srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* src1 = src0 + srcC * 16, *weight1 = weight0 + 32;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[1] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[3] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((dstC - 16) * 4);
            conf.colsb[2] = 64;
            conf.colsb[3] = uint16_t((dstC - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((dstC - 16) * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
                _tile_stream_loadd(1, dst + F, strideD);
                _tile_stream_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_stream_loadd(3, dst + 16 * dD + F, strideD);
            }
            for (size_t sc = 0; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for(; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC;
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + srcC * 16;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[2] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
                _tile_stream_loadd(2, dst + 16 * dD + 0, strideD);
            }
            for (size_t sc = 0; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC;
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* weight1 = weight0 + 32;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[1] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t(dstC - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t(dstC - 16) * 4;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
                _tile_stream_loadd(1, dst + F, strideD);
            }
            for (size_t sc = 0; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC;
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst + 0, strideD);
            }
            for (size_t sc = 0; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, dst + 0, strideD);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        typedef void (*ConvolutionBf16NhwcGemmPtr)(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst);

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            //SIMD_PERF_FUNC();

            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 2) * DF;
            ConvolutionBf16NhwcGemmPtr body_2 = ConvolutionBf16NhwcGemm_2x2<type>;
            ConvolutionBf16NhwcGemmPtr tail_2 = m > 16 ? ConvolutionBf16NhwcGemm_2x2<type> : ConvolutionBf16NhwcGemm_1x2<type>;
            ConvolutionBf16NhwcGemmPtr body_1 = ConvolutionBf16NhwcGemm_2x1<type>;
            ConvolutionBf16NhwcGemmPtr tail_1 = m > 16 ? ConvolutionBf16NhwcGemm_2x1<type> : ConvolutionBf16NhwcGemm_1x1<type>;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                        body_2(s, p, srcC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_2(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                else
                {
                    for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                        body_1(s, p, srcC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_1(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[TermLast] = ConvolutionBf16NhwcGemm_2<type>;
            convolutions[TermInterim] = ConvolutionBf16NhwcGemm_2<SimdConvolutionActivationIdentity>;
        }

        SynetConvolution32fBf16NhwcGemm::SynetConvolution32fBf16NhwcGemm(const ConvParam32f & p)
            : Avx512bw::SynetConvolution32fBf16NhwcGemm(p)
        {
            size_t microD = 16 * 2;
            size_t microM = 16 * 2;
            size_t microC = 16 * 2;
#if !defined(SIMD_AMX_EMULATE)
            if (p.srcC* p.kernelX * p.kernelY < 1 * microC)
                return;
#endif
            SetAlgParam(microD, microM, microC, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
#if !defined(SIMD_AMX_EMULATE)
            if(p.Is1x1())
                _convert = ConvertBf16NhwcGemm1x1;
            else
                _convert = ConvertBf16NhwcGemm;
#endif
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

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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution16b::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_2x2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t * dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* src1 = src0 + dS * 16, * weight1 = weight0 + AlignHi(srcC, a.miK) * F;
            float* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(1, buf0 + F, strideB);
                _tile_stream_loadd(2, buf1 + 0, strideB);
                _tile_stream_loadd(3, buf1 + F, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC32;)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                sc += 32;
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            _tile_stored(3, buf1 + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply2<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_2x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* src1 = src0 + dS * 16;
            float* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(2, buf1 + 0, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC32;)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply1<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_1x2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* weight1 = weight0 + AlignHi(srcC, a.miK) * F;
            if (cfg)
                SetTileConf1x2(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(1, buf0 + F, strideB);
            }

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC32;)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, src0 + sc, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply2<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_1x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
            }

            size_t sc = 0;
            for (; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, buf0 + 0, strideB);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply1<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        typedef void (*OutputConvolutionPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dW = AlignHi(maC, a.miK) * DF, dS = a.maC, dB = AlignHi(p.dstC, F), dD = p.dstC * a.elem[1];
            if (buf == NULL && p.dstT == SimdTensorData32f)
                buf = (float*)dst;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (nn)
            {
                OutputConvolutionPtr body_2 = OutputConvolution_2x2<term, type, 0>;
                OutputConvolutionPtr tail_2 = m > 16 ? OutputConvolution_2x2<term, type, 0> : OutputConvolution_1x2<term, type, 0>;
                OutputConvolutionPtr body_1 = OutputConvolution_2x1<term, type, 0>;
                OutputConvolutionPtr tail_1 = m > 16 ? OutputConvolution_2x1<term, type, 0> : OutputConvolution_1x1<term, type, 0>;
                SetTileConfFull();
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    float* b = buf + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, maC, n, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                        if (m)
                            tail_2(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, maC, n, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                        if (m)
                            tail_1(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                OutputConvolutionPtr tail_2 = m > 16 ? OutputConvolution_2x2<term, type, 0> : OutputConvolution_1x2<term, type, 0>;
                OutputConvolutionPtr tail_1 = m > 16 ? OutputConvolution_2x1<term, type, 0> : OutputConvolution_1x1<term, type, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    float* b = buf + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                    size_t i = 0;
                    if (dC > F)
                         tail_2(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    else
                         tail_1(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    weight += dW;
                }
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            if (p.dstT == SimdTensorData16b)
                output[0] = OutputConvolution1x1_2<Term16bLast16b, type>;
            else
                output[0] = OutputConvolution1x1_2<Term16bLast32f, type>;
            output[1] = OutputConvolution1x1_2<Term16bInterim, SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam& p, OutputPtr* output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationRelu: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutput<SimdConvolutionActivationRestrictRange>(p, output); break;
            case SimdConvolutionActivationPrelu: SetOutput<SimdConvolutionActivationPrelu>(p, output); break;
            case SimdConvolutionActivationElu: SetOutput<SimdConvolutionActivationElu>(p, output); break;
            case SimdConvolutionActivationHswish: SetOutput<SimdConvolutionActivationHswish>(p, output); break;
            case SimdConvolutionActivationMish: SetOutput<SimdConvolutionActivationMish>(p, output); break;
            case SimdConvolutionActivationHardSigmoid: SetOutput<SimdConvolutionActivationHardSigmoid>(p, output); break;
            case SimdConvolutionActivationSwish: SetOutput<SimdConvolutionActivationSwish>(p, output); break;
            case SimdConvolutionActivationGelu: SetOutput<SimdConvolutionActivationGelu>(p, output); break;
            }
        }
    }
#endif
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution32f.h"
#include "Simd/SimdSynetConvolution32fBf16Common.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace Amx
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution32fBf16::OutputConvolutionPtr;

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void OutputConvolution_2x2(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = a.maC * p.strideX, dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + dS * 16, * weight1 = weight0 + 32;

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
            conf.colsb[1] = uint16_t(dstC - 16) * 4;
            conf.colsb[2] = 64;
            conf.colsb[3] = uint16_t(dstC - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t(dstC - 16) * 4;
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
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_loadd(3, dst + 16 * dD + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.rows[7] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                conf.colsb[5] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
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
                for (; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void OutputConvolution_2x1(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = a.maC * p.strideX, dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + dS * 16;

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
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                conf.colsb[5] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
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

        template<SimdConvolutionActivationType type> void OutputConvolution_1x2(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = a.maC * p.strideX, dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
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
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.rows[7] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
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

        template<SimdConvolutionActivationType type> void OutputConvolution_1x1(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = a.maC * p.strideX, dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;

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
                _tile_loadd(0, dst + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
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

        typedef void (*OutputConvolutionPtr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst);

        template<SimdConvolutionActivationType type> void OutputConvolution1x1_2(const uint16_t* src,
            const ConvParam32f& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight,
            const float* bias, const float* params, float* dst, int zero)
        {
            size_t n = 32, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(maC, 2) * DF;
            OutputConvolutionPtr body_2 = OutputConvolution_2x2<type>;
            OutputConvolutionPtr tail_2 = m > 16 ? OutputConvolution_2x2<type> : OutputConvolution_1x2<type>;
            OutputConvolutionPtr body_1 = OutputConvolution_2x1<type>;
            OutputConvolutionPtr tail_1 = m > 16 ? OutputConvolution_2x1<type> : OutputConvolution_1x1<type>;

            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
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
                float* d = dst + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n, s += n * a.maC, d += n * p.dstC)
                        body_2(s, p, a, maC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_2(s, p, a, maC, m, dC, zero, weight, _bias, _params, d);
                }
                else
                {
                    for (; i < nn; i += n, s += n * a.maC, d += n * p.dstC)
                        body_1(s, p, a, maC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_1(s, p, a, maC, m, dC, zero, weight, _bias, _params, d);
                }
                weight += dW;
            }
        }

        //---------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetOutput(const ConvParam32f& p, OutputPtr* output)
        {
            output[0] = OutputConvolution1x1_2<type>;
            output[1] = OutputConvolution1x1_2<SimdConvolutionActivationIdentity>;
        }

        void SetOutput(const ConvParam32f& p, OutputPtr* output)
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
            }
        }
    }
#endif
}

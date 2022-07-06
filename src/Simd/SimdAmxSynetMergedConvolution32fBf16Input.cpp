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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace Amx
    {
        using AlgParam = Base::SynetMergedConvolution32fBf16::AlgParam;
        using InputPtr = Base::SynetMergedConvolution32fBf16::InputConvolutionPtr;

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2x2(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst0, float *dst1)
        {
            size_t dD = p.dstC, sC = p.srcC, sC32 = AlignLo(sC, 32);
            int strideS = (int)sC * 2, strideW = 128, strideD = 64;
            const uint16_t* src1 = src0 + sC * 16, * weight1 = weight0 + 32;

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

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            size_t sc = 0;
            for (; sc < sC32; sc += 32)
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
            if (sc < sC)
            {
                size_t tailC = AlignHi(sC - sc, 2);
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
            _tile_stored(0, dst0, strideD);
            _tile_stored(2, dst0 + 256, strideD);
            _tile_stored(1, dst1, strideD);
            _tile_stored(3, dst1 + 256, strideD);
            if (type)
            {
                for (size_t s = 0; s < dstS; ++s, dst0 += F)
                    Apply<type, 0>(dst0, dst0, bias, params);
                for (size_t s = 0; s < dstS; ++s, dst1 += F)
                    Apply<type, 1>(dst1, dst1, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2x1(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t dD = p.dstC, sC = p.srcC, sC32 = AlignLo(sC, 32);
            int strideS = (int)sC * 2, strideW = 128, strideD = 64;
            const uint16_t* src1 = src0 + sC * 16;

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

            _tile_zero(0);
            _tile_zero(2);
            size_t sc = 0;
            for (; sc < sC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            if (sc < sC)
            {
                size_t tailC = AlignHi(sC - sc, 2);
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
            _tile_stored(0, dst0, strideD);
            _tile_stored(2, dst0 + 256, strideD);
            if (type)
            {
                for (size_t s = 0; s < dstS; ++s, dst0 += F)
                    Apply<type, 0>(dst0, dst0, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_1x2(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t dD = p.dstC, sC = p.srcC, sC32 = AlignLo(sC, 32);
            int strideS = (int)sC * 2, strideW = 128, strideD = 64;
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

            _tile_zero(0);
            _tile_zero(1);
            size_t sc = 0;
            for (; sc < sC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (sc < sC)
            {
                size_t tailC = AlignHi(sC - sc, 2);
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
            _tile_stored(0, dst0, strideD);
            _tile_stored(1, dst1, strideD);
            if (type)
            {
                if (type)
                {
                    for (size_t s = 0; s < dstS; ++s, dst0 += F)
                        Apply<type, 0>(dst0, dst0, bias, params);
                    for (size_t s = 0; s < dstS; ++s, dst1 += F)
                        Apply<type, 1>(dst1, dst1, bias, params);
                }
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_1x1(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t dD = p.dstC, sC = p.srcC, sC32 = AlignLo(sC, 32);
            int strideS = (int)sC * 2, strideW = 128, strideD = 64;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            _tile_zero(0);
            size_t sc = 0;
            for (; sc < sC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            if (sc < sC)
            {
                size_t tailC = AlignHi(sC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, dst0, strideD);
            if (type)
            {
                for (size_t s = 0; s < dstS; ++s, dst0 += F)
                    Apply<type, 0>(dst0, dst0, bias, params);
            }
        }

        typedef void (*InputConvolution1x1Ptr)(const uint16_t* src0, const ConvParam32f& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint16_t* src, const ConvParam32f& p,
            const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 32;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
            InputConvolution1x1Ptr conv_2x2 = InputConvolution1x1_2x2<type>;
            InputConvolution1x1Ptr conv_2x1 = InputConvolution1x1_2x1<type>;
            InputConvolution1x1Ptr conv_Ix2 = i > 16 ? InputConvolution1x1_2x2<type> : InputConvolution1x1_1x2<type>;
            InputConvolution1x1Ptr conv_Ix1 = i > 16 ? InputConvolution1x1_2x1<type> : InputConvolution1x1_1x1<type>;
            InputConvolution1x1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2<type> : InputConvolution1x1_1x2<type>;
            InputConvolution1x1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1<type> : InputConvolution1x1_1x1<type>;
            for (size_t dc = 0; dc < maC; dc += DF)
            {
                size_t dC = Simd::Min(DF, maC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                if (dC > F)
                {
                    if (yInt > yBeg)
                    {
                        const uint16_t* src0 = src + (yBeg & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            conv_2x2(src0, p, a, n, dC, weight, _bias, _params, dst0, dst1);
                        if (in < i1)
                            conv_Ix2(src0, p, a, i, dC, weight, _bias, _params, dst0, dst1);
                    }
                    if (yEnd > yInt)
                    {
                        const uint16_t* src0 = src + (yInt & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                            conv_2x2(src0, p, a, n, dC, weight, _bias, _params, dst0, dst1);
                        if (en < e1)
                            conv_Ex2(src0, p, a, e, dC, weight, _bias, _params, dst0, dst1);
                    }
                }
                else
                {
                    if (yInt > yBeg)
                    {
                        const uint16_t* src0 = src + (yBeg & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                        for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n)
                            conv_2x1(src0, p, a, n, dC, weight, _bias, _params, dst0, NULL);
                        if (in < i1)
                            conv_Ix1(src0, p, a, i, dC, weight, _bias, _params, dst0, NULL);
                    }
                    if (yEnd > yInt)
                    {
                        const uint16_t* src0 = src + (yInt & srcM) * p.srcW * p.srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                        for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n)
                            conv_2x1(src0, p, a, n, dC, weight, _bias, _params, dst0, NULL);
                        if (en < e1)
                            conv_Ex1(src0, p, a, e, dC, weight, _bias, _params, dst0, NULL);
                    }
                }
                dst += a.bufH[1] * p.dstW * DF;
                weight += DivHi(p.srcC, 2) * QF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            if (Is1x1(p))
                input = InputConvolution1x1_2<type>;
            else
                assert(0);
        }

        void SetInput(const ConvParam32f& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInput<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInput<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInput<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInput<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInput<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInput<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            }
        }
    }
#endif
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynetMergedConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution8i::AlgParam;
        using InputConvolutionPtr = Base::SynetMergedConvolution8i::InputConvolutionPtr;

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Apply1(float* dst, const __m512* norm, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_loadu_si512((__m512i*)dst)), norm[0], bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Apply2(float* dst0, float* dst1, const __m512* norm, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst0, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_loadu_si512((__m512i*)dst0)), norm[0], bias[0]), params, 0));
            _mm512_storeu_ps(dst1, Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_loadu_si512((__m512i*)dst1)), norm[1], bias[1]), params, 1));
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2x2(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            int strideW = 128, strideS = (int)p.srcC;
            const int8_t* weight1 = weight0 + A;
            const uint8_t* src1 = src0 + 16 * p.srcC;

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
            conf.colsb[1] = 64;
            conf.colsb[2] = 64;
            conf.colsb[3] = 64;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = 64;
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbusd(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
                _tile_dpbusd(3, 5, 7);
            }
            if (sc < p.srcC)
            {
                size_t tailC = AlignHi(p.srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.rows[7] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                conf.colsb[5] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbusd(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_stored(0, dst0, A);
            _tile_stored(1, dst1, A);
            _tile_stored(2, dst0 + 16 * F, A);
            _tile_stored(3, dst1 + 16 * F, A);
            size_t dstS8 = AlignLo(dstS, 8), s = 0;
            if (Base::FmaAvoid(p.compatibility))
            {
                for (; s < dstS; ++s, dst0 += F, dst1 += F)
                    Apply2<type, true>(dst0, dst1, norm, bias, params);
            }
            else
            {
                for (; s < dstS; ++s, dst0 += F, dst1 += F)
                    Apply2<type, false>(dst0, dst1, norm, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2x1(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            int strideW = 128, strideS = (int)p.srcC;
            const uint8_t* src1 = src0 + 16 * p.srcC;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = 64;
            conf.colsb[2] = 64;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(2);
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            if (sc < p.srcC)
            {
                size_t tailC = AlignHi(p.srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                conf.colsb[5] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_stored(0, dst0, A);
            _tile_stored(2, dst0 + 16 * F, A);
            size_t dstS8 = AlignLo(dstS, 8), s = 0;
            if (Base::FmaAvoid(p.compatibility))
            {
                for (; s < dstS; ++s, dst0 += F)
                    Apply1<type, true>(dst0, norm, bias, params);
            }
            else
            {
                for (; s < dstS; ++s, dst0 += F)
                    Apply1<type, false>(dst0, norm, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_1x2(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            int strideW = 128, strideS = (int)p.srcC;
            const int8_t* weight1 = weight0 + A;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[1] = 16;
            conf.rows[4] = 16;
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = 64;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = 64;
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(1);
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            if (sc < p.srcC)
            {
                size_t tailC = AlignHi(p.srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.rows[7] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stored(0, dst0, A);
            _tile_stored(1, dst1, A);
            size_t dstS8 = AlignLo(dstS, 8), s = 0;
            if (Base::FmaAvoid(p.compatibility))
            {
                for (; s < dstS; ++s, dst0 += F, dst1 += F)
                    Apply2<type, true>(dst0, dst1, norm, bias, params);
            }
            else
            {
                for (; s < dstS; ++s, dst0 += F, dst1 += F)
                    Apply2<type, false>(dst0, dst1, norm, bias, params);
            }
        }

        template<SimdConvolutionActivationType type> void InputConvolution1x1_1x1(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1)
        {
            size_t srcC64 = AlignLo(p.srcC, 64);
            int strideW = 128, strideS = (int)p.srcC;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[4] = 16;
            conf.rows[6] = 16;
            conf.colsb[0] = 64;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            _tile_loadconfig(&conf);

            _tile_zero(0);
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
            }
            if (sc < p.srcC)
            {
                size_t tailC = AlignHi(p.srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbusd(0, 4, 6);
            }
            _tile_stored(0, dst0, A);
            size_t dstS8 = AlignLo(dstS, 8), s = 0;
            if (Base::FmaAvoid(p.compatibility))
            {
                for (; s < dstS; ++s, dst0 += F)
                    Apply1<type, true>(dst0, norm, bias, params);
            }
            else
            {
                for (; s < dstS; ++s, dst0 += F)
                    Apply1<type, false>(dst0, norm, bias, params);
            }
        }

        typedef void(*InputConvolution1x1_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t dstS, 
            const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, float* dst0, float* dst1);

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2(const uint8_t* src, const ConvParam8i& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const int8_t* weight, const float* norm, const float* bias, const float* params, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcM = a.bufH[0] - 1;
            __m512 _bias[2], _norm[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (a.bufH[0] == 0)
            {
                size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 32;
                size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLoAny(i1, n), i = i1 - in;
                size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLoAny(e1, n), e = e1 - en;
                InputConvolution1x1_Ptr conv_n2 = InputConvolution1x1_2x2<type>;
                InputConvolution1x1_Ptr conv_n1 = InputConvolution1x1_2x1<type>;
                InputConvolution1x1_Ptr conv_i2 = i > 16 ? InputConvolution1x1_2x2<type> : InputConvolution1x1_1x2<type>;
                InputConvolution1x1_Ptr conv_i1 = i > 16 ? InputConvolution1x1_2x1<type> : InputConvolution1x1_1x1<type>;
                InputConvolution1x1_Ptr conv_e2 = e > 16 ? InputConvolution1x1_2x2<type> : InputConvolution1x1_1x2<type>;
                InputConvolution1x1_Ptr conv_e1 = e > 16 ? InputConvolution1x1_2x1<type> : InputConvolution1x1_1x1<type>;
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    if (dC > F)
                    {
                        _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                        _norm[1] = _mm512_loadu_ps(norm + dc + F);
                        _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                        _bias[1] = _mm512_loadu_ps(bias + dc + F);
                        if (type == ::SimdConvolutionActivationPrelu)
                        {
                            _params[0] = _mm512_loadu_ps(params + dc + 0);
                            _params[1] = _mm512_loadu_ps(params + dc + F);
                        }
                        if (yInt > yBeg)
                        {
                            const uint8_t* src0 = src + yBeg * p.srcW * p.srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_n2(src0, p, a, n, weight, _norm, _bias, _params, dst0, dst1);
                            if (in < i1)
                                conv_i2(src0, p, a, i, weight, _norm, _bias, _params, dst0, dst1);
                        }
                        if (yEnd > yInt)
                        {
                            const uint8_t* src0 = src + yInt * p.srcW * p.srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_n2(src0, p, a, n, weight, _norm, _bias, _params, dst0, dst1);
                            if (en < e1)
                                conv_e2(src0, p, a, e, weight, _norm, _bias, _params, dst0, dst1);
                        }
                    }
                    else
                    {
                        _norm[0] = _mm512_loadu_ps(norm + dc);
                        _bias[0] = _mm512_loadu_ps(bias + dc);
                        if (type == ::SimdConvolutionActivationPrelu)
                            _params[0] = _mm512_loadu_ps(params + dc);
                        if (yInt > yBeg)
                        {
                            const uint8_t* src0 = src + yBeg * p.srcW * p.srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            for (size_t j = 0; j < in; j += n, src0 += p.srcC * n, dst0 += F * n)
                                conv_n1(src0, p, a, n, weight, _norm, _bias, _params, dst0, NULL);
                            if (in < i1)
                                conv_i1(src0, p, a, i, weight, _norm, _bias, _params, dst0, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            const uint8_t* src0 = src + yInt * p.srcW * p.srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            for (size_t j = 0; j < en; j += n, src0 += p.srcC * n, dst0 += F * n)
                                conv_n1(src0, p, a, n, weight, _norm, _bias, _params, dst0, NULL);
                            if (en < e1)
                                conv_e1(src0, p, a, e, weight, _norm, _bias, _params, dst0, NULL);
                        }
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
            else
            {
                size_t n = 32, bodyW = p.dstW, bodyWn = AlignLoAny(bodyW, n), e = bodyW - bodyWn;
                InputConvolution1x1_Ptr conv_n2 = InputConvolution1x1_2x2<type>;
                InputConvolution1x1_Ptr conv_n1 = InputConvolution1x1_2x1<type>;
                InputConvolution1x1_Ptr conv_e2 = e > 16 ? InputConvolution1x1_2x2<type> : InputConvolution1x1_1x2<type>;
                InputConvolution1x1_Ptr conv_e1 = e > 16 ? InputConvolution1x1_2x1<type> : InputConvolution1x1_1x1<type>;
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    if (dC > F)
                    {
                        _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                        _norm[1] = _mm512_loadu_ps(norm + dc + F);
                        _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                        _bias[1] = _mm512_loadu_ps(bias + dc + F);
                        if (type == ::SimdConvolutionActivationPrelu)
                        {
                            _params[0] = _mm512_loadu_ps(params + dc + 0);
                            _params[1] = _mm512_loadu_ps(params + dc + F);
                        }
                        for (size_t dy = yBeg; dy < yEnd; dy++)
                        {
                            const uint8_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                            float* dst0 = dst + (dy & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            size_t dx = 0;
                            for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_n2(src0, p, a, n, weight, _norm, _bias, _params, dst0, dst1);
                            if (dx < bodyW)
                                conv_e2(src0, p, a, e, weight, _norm, _bias, _params, dst0, dst1);
                        }
                    }
                    else
                    {
                        _norm[0] = _mm512_loadu_ps(norm + dc);
                        _bias[0] = _mm512_loadu_ps(bias + dc);
                        if (type == ::SimdConvolutionActivationPrelu)
                            _params[0] = _mm512_loadu_ps(params + dc);
                        for (size_t dy = yBeg; dy < yEnd; dy++)
                        {
                            const uint8_t* src0 = src + (dy & srcM) * p.srcW * p.srcC;
                            float* dst0 = dst + (dy & dstM) * p.dstW * F;
                            size_t dx = 0;
                            for (; dx < bodyWn; dx += n, src0 += p.srcC * n, dst0 += F * n)
                                conv_n1(src0, p, a, n, weight, _norm, _bias, _params, dst0, NULL);
                            if (dx < bodyW)
                                conv_e1(src0, p, a, e, weight, _norm, _bias, _params, dst0, NULL);
                        }
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += DivHi(p.srcC, 4) * DA;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
        {
            if (p.Is1x1())
                input = InputConvolution1x1_2<type>;
            else
                assert(0);
        }

        void SetInput(const ConvParam8i& p, InputConvolutionPtr& input)
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
            case SimdConvolutionActivationSwish: SetInput<SimdConvolutionActivationSwish>(p, input); break;
            case SimdConvolutionActivationGelu: SetInput<SimdConvolutionActivationGelu>(p, input); break;
            }
        }
    }
#endif
}

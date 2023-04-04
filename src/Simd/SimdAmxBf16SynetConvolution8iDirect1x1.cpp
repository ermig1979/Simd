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
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        using AlgParam = SynetConvolution8iNhwcDirect::AlgParam;
        using ConvolutionPtr = SynetConvolution8iNhwcDirect::ConvolutionPtr;

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2x2(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, const int8_t* weight0, const __m512* norm, 
            const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, srcC64 = AlignLo(srcC, 64);
            int strideW = 64, strideS = (int)dS, strideB = (int)dB * 4;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            const uint8_t* src1 = src0 + 16 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);

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

            if (first)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_loadd(0, buf + 0, strideB);
                _tile_loadd(1, buf + F, strideB);
                _tile_loadd(2, buf + 16 * dB + 0, strideB);
                _tile_loadd(3, buf + 16 * dB + F, strideB);
            }
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
                _tile_dpbusd(3, 5, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.rows[7] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                conf.colsb[5] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (term < Term8iInterim)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply2<term, type, true>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
                else
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply2<term, type, false>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, const int8_t* weight0, const __m512* norm,
            const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dW = 64, srcC64 = AlignLo(srcC, 64);
            int strideW = 64, strideS = (int)dS, strideB = (int)dB * 4;
            const uint8_t* src1 = src0 + 16 * dS;
            __m128i upper = _mm_set1_epi32(a.upper);

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

            if (first)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_loadd(0, buf + 0, strideB);
                _tile_loadd(2, buf + 16 * dB + 0, strideB);
            }
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                conf.colsb[5] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply1<term, type, true>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
                else
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply1<term, type, false>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_1x2(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, const int8_t* weight0, const __m512* norm,
            const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dW = 64, srcC64 = AlignLo(srcC, 64);
            int strideW = 64, strideS = (int)dS, strideB = (int)dB * 4;
            const int8_t* weight1 = weight0 + DivHi(p.srcC, 4) * A;
            __m128i upper = _mm_set1_epi32(a.upper);

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[1] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((dstC - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((dstC - 16) * 4);
            _tile_loadconfig(&conf);

            if (first)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_loadd(0, buf + 0, strideB);
                _tile_loadd(1, buf + F, strideB);
            }
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.rows[7] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply2<term, type, true>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
                else
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply2<term, type, false>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_1x1(const uint8_t* src0,
            const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, const int8_t* weight0, const __m512* norm,
            const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t dS = p.srcC * p.strideX, dD = p.dstC * a.size, dB = p.dstC, dW = 64, srcC64 = AlignLo(srcC, 64);
            int strideW = 64, strideS = (int)dS, strideB = (int)dB * 4;
            __m128i upper = _mm_set1_epi32(a.upper);

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            if (first)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_loadd(0, buf + 0, strideB);
            }
            size_t sc = 0;
            for (; sc < srcC64; sc += 64)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 4);
                conf.rows[6] = uint8_t(tailC / 4);
                conf.colsb[4] = uint16_t(tailC);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
            }
            _tile_stored(0, buf + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                if (Base::FmaAvoid(p.compatibility))
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply1<term, type, true>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
                else
                {
                    for (; s < dstS; ++s, buf += dB, dst += dD)
                        Apply1<term, type, false>(dst, buf, norm, bias, params, scale, shift, upper, tailD);
                }
            }
        }

        typedef void(*ConvolutionNhwcDirect1x1_Ptr)(const uint8_t* src0, const ConvParam8i& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC,
            const int8_t* weight0, const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, int32_t* buf, uint8_t* dst, int first);

        template<Term8iType term, SimdConvolutionActivationType type> void ConvolutionNhwcDirect1x1_2(const uint8_t* src, 
            const ConvParam8i& p, const AlgParam& a, size_t dstC, size_t yBeg, size_t yEnd, size_t srcC, const int8_t* weight, 
            const float* norm, const float* bias, const float* params, const float* scale, const float* shift, int32_t* buf, uint8_t* dst, int first)
        {
            size_t n = 32, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 4) * A;
            ConvolutionNhwcDirect1x1_Ptr body_2 = ConvolutionNhwcDirect1x1_2x2<term, type>;
            ConvolutionNhwcDirect1x1_Ptr tail_2 = m > 16 ? ConvolutionNhwcDirect1x1_2x2<term, type> : ConvolutionNhwcDirect1x1_1x2<term, type>;
            ConvolutionNhwcDirect1x1_Ptr body_1 = ConvolutionNhwcDirect1x1_2x1<term, type>;
            ConvolutionNhwcDirect1x1_Ptr tail_1 = m > 16 ? ConvolutionNhwcDirect1x1_2x1<term, type> : ConvolutionNhwcDirect1x1_1x1<term, type>;

            __m512 _norm[2], _bias[2], _params[2], _scale[2], _shift[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                _norm[1] = _mm512_loadu_ps(norm + dc + F);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                _scale[0] = _mm512_loadu_ps(scale + dc + 0);
                _scale[1] = _mm512_loadu_ps(scale + dc + F);
                _shift[0] = _mm512_loadu_ps(shift + dc + 0);
                _shift[1] = _mm512_loadu_ps(shift + dc + F);
                const uint8_t* s = src + yBeg * p.srcW * p.srcC;
                uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.size;
                int32_t* b = buf + dc + yBeg * p.dstW * p.dstC;
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                        body_2(s, p, a, srcC, n, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                    for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                        tail_2(s, p, a, srcC, m, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                }
                else
                {
                    for (; i < nn; i += n, s += p.srcC * n, b += p.dstC * n, d += p.dstC * a.size * n)
                        body_1(s, p, a, srcC, n, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                    for (; i < n1; i += m, s += p.srcC * m, b += p.dstC * m, d += p.dstC * a.size * m)
                        tail_1(s, p, a, srcC, m, dC, weight, _norm, _bias, _params, _scale, _shift, b, d, first);
                }
                weight += DivHi(p.srcC, 4) * DA;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            assert(a.microD == 2 * F && p.Is1x1() == true);
            d[term] = ConvolutionNhwcDirect1x1_2<term, activation>;
        }

        template<SimdConvolutionActivationType activation> void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            SetDirect1x1<Term8iLast8u, activation>(p, a, d);
            SetDirect1x1<Term8iLast32f, activation>(p, a, d);
            SetDirect1x1<Term8iInterim, SimdConvolutionActivationIdentity>(p, a, d);
        }

        void SetDirect1x1(const ConvParam8i& p, const AlgParam& a, ConvolutionPtr* d)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationRelu: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationLeakyRelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationRestrictRange: SetDirect1x1<SimdConvolutionActivationRestrictRange>(p, a, d); break;
            case SimdConvolutionActivationPrelu: SetDirect1x1<SimdConvolutionActivationPrelu>(p, a, d); break;
            case SimdConvolutionActivationElu: SetDirect1x1<SimdConvolutionActivationElu>(p, a, d); break;
            case SimdConvolutionActivationHswish: SetDirect1x1<SimdConvolutionActivationHswish>(p, a, d); break;
            case SimdConvolutionActivationMish: SetDirect1x1<SimdConvolutionActivationMish>(p, a, d); break;
            case SimdConvolutionActivationHardSigmoid: SetDirect1x1<SimdConvolutionActivationHardSigmoid>(p, a, d); break;
            case SimdConvolutionActivationSwish: SetDirect1x1<SimdConvolutionActivationSwish>(p, a, d); break;
            case SimdConvolutionActivationGelu: SetDirect1x1<SimdConvolutionActivationGelu>(p, a, d); break;
            default: assert(0);
            }
        }
    }
#endif
}

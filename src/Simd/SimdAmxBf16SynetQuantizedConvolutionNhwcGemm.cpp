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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcGemm::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcGemm::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcGemmReorderR(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            assert(Aligned(p.srcC, 64));
            size_t K = a.bufK, C = p.srcC, kcX = p.kernelX * C;
            __m512i _zero = _mm512_set1_epi8(zero);
            for (size_t dy = yBeg, dr = 0; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx, ++dr)
                {
                    size_t drB = dr & (~15), drO = dr & 15;
                    uint8_t* row = dst + drB * K + drO * 64;
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                        if (sy < p.srcH)
                        {
                            for (size_t kx = 0; kx < p.kernelX; kx++)
                            {
                                size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                if (sx < p.srcW)
                                {
                                    const uint8_t* ps = src + (sy * p.srcW + sx) * p.srcC;
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
                                        Avx512bw::Copy(ps + sc, row);
                                }
                                else
                                {
                                    for (size_t sc = 0; sc < C; sc += 64, row += 1024)
                                        SetZero(row, _zero);
                                }
                            }
                        }
                        else
                        {
                            for (size_t sc = 0; sc < kcX; sc += 64, row += 1024)
                                SetZero(row, _zero);
                        }
                    }
                }
            }
        }

        static void QuantizedConvolutionNhwcGemmReorder1x1D(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64), n = (yEnd - yBeg) * p.dstW;
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            src += yBeg * p.srcW * p.srcC;
            for (size_t i = 0; i < n; ++i)
            {
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                    Avx512bw::Copy(src + sc, dst + sc);
                if(srcMask)
                    Avx512bw::Copy(src + sc, dst + sc, srcMask);
                src += p.srcC;
                dst += a.bufK;
            }
        }

        static void QuantizedConvolutionNhwcGemmReorder1x1R(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64), n = (yEnd - yBeg) * p.dstW;
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
            __m512i _zero = _mm512_set1_epi8(zero);
            src += yBeg * p.srcW * p.srcC;
            for (size_t i = 0; i < n; i += 16)
            {
                size_t m = Min(i + 16, n) - i;
                size_t sc = 0;
                for (; sc < srcC64; sc += 64)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                if (srcC64 < p.srcC)
                {
                    size_t j = 0;
                    for (; j < m; ++j)
                        Avx512bw::Copy(src + sc + j * p.srcC, dst + j * 64 + sc * 16, srcMask);
                    for (; j < 16; ++j)
                        SetZero(dst + j * 64 + sc * 16, _mm512_setzero_si512());
                }
                src += p.srcC * 16;
                dst += a.bufK * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNhwcGemm_i32x32(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const uint8_t* src1 = src0 + 16 * dS;
            const int8_t* weight1 = weight0 + a.bufK * F;

            if (cfg)
                SetTileConf2x2(dstS, dstC);
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

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC64; src1 += stepS)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbusd(0, 4, 6);
                _tile_dpbusd(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbusd(2, 5, 6);
                sc += 64;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);
            _tile_dpbusd(2, 5, 6);
            _tile_dpbusd(3, 5, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            _tile_stored(3, buf + 16 * dB + F, strideB);
            if (term == Term8iLast8u)
            {
                __mmask32 tailD = TailMask32(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply8u2x8<type>(dst + ds * dD, dD, buf + ds * dB, dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    Apply8u2<type>(dst + ds * dD, buf + ds * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
            else if (term == Term8iLast32f)
            {
                //__mmask16 tailD = TailMask16(dstC - F);
                //size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                //for (; ds < dstS8; ds += 8)
                //    Apply2x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, bias, params, tailD);
                //for (; ds < dstS; ++ds)
                //    Apply2<term, type>(dst + ds * dD, buf + ds * dB, bias, params, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
                TileMoveToMemory(buf + 16 * dB + F, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNhwcGemm_i32x16(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const uint8_t* src1 = src0 + 16 * dS;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
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

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC64; sc += 64, src1 += stepS)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1, strideS);
                _tile_dpbusd(0, 4, 6);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                _tile_dpbusd(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1, strideS);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(2, 5, 6);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(2, buf + 16 * dB + 0, strideB);
            if (term == Term8iLast8u || term == Term8iLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply1x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term, type>(dst + ds * dD, buf + ds * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNhwcGemm_i16x32(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;
            const int8_t* weight1 = weight0 + a.bufK * F;

            if (cfg)
                SetTileConf1x2(dstS, dstC);
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

            int srcC64 = (int)srcC - 64, sc = 0;
            _tile_loadd(6, weight0 + sc * 16, strideW);
            for (; sc < srcC64; src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
                sc += 64;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbusd(0, 4, 6);
            _tile_dpbusd(1, 4, 7);

            _tile_stored(0, buf + 0, strideB);
            _tile_stored(1, buf + F, strideB);
            if (term == Term8iLast8u || term == Term8iLast32f)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply2x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    Apply2<term, type>(dst + ds * dD, buf + ds * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
            }
        }

        template<Term8iType term, SimdConvolutionActivationType type, int cfg> void QuantizedConvolutionNhwcGemm_i16x16(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, 
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst)
        {
            int dB = (int)a.dB, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, strideW = 64;
            int stepS = a.reorderType ? 1024 : 64, strideS = a.reorderType ? 64 : dS;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (update)
            {
                _tile_stream_loadd(0, buf + 0, strideB);
            }
            else
            {
                _tile_zero(0);
            }

            for (size_t sc = 0; sc < srcC; sc += 64, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
            }

            _tile_stored(0, buf + 0, strideB);
            if (term == Term8iLast8u || term == Term8iLast32f)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), ds = 0;
                for (; ds < dstS8; ds += 8)
                    Apply1x8<term, type>(dst + ds * dD, dD, buf + ds * dB, dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term, type>(dst + ds * dD, buf + ds * dB, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
            }
        }

        typedef void (*QuantizedConvolutionNhwcGemm_iPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, int32_t* buf, uint8_t* dst);

        template<Term8iType term, SimdConvolutionActivationType type> void QuantizedConvolutionNhwcGemm_i2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int update, const int8_t* weight,
            const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale, const float* params, float dNorm, int32_t dZero, int32_t* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.macroK < a.bufK ? a.dB : 0, dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _sNorm[2], _iScale, _params[2], _dNorm;
            __m512i _sBias[2], _dZero = _mm512_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                _params[1] = _mm512_set1_ps(params[1]);
            }
            if (nn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 1 && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), nn = n1 - m;
                QuantizedConvolutionNhwcGemm_iPtr body_2 = QuantizedConvolutionNhwcGemm_i32x32<term, type, 0>;
                QuantizedConvolutionNhwcGemm_iPtr tail_2 = m > 16 ? QuantizedConvolutionNhwcGemm_i32x32<term, type, 0> : QuantizedConvolutionNhwcGemm_i16x32<term, type, 0>;
                QuantizedConvolutionNhwcGemm_iPtr body_1 = QuantizedConvolutionNhwcGemm_i32x16<term, type, 0>;
                QuantizedConvolutionNhwcGemm_iPtr tail_1 = m > 16 ? QuantizedConvolutionNhwcGemm_i32x16<term, type, 0> : QuantizedConvolutionNhwcGemm_i16x16<term, type, 0>;
                SetTileConfFull();
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + dc + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + dc + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint8_t* s = src;
                    int32_t* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, srcC, n, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b + i * dB, d + i * dD);
                        if (m)
                            tail_2(s + nn * dS, p, a, srcC, m, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b + i * dB, d + nn * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, srcC, n, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b + i * dB, d + i * dD);
                        if (m)
                            tail_1(s + nn * dS, p, a, srcC, m, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b + i * dB, d + nn * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                QuantizedConvolutionNhwcGemm_iPtr tail_2 = m > 16 ? QuantizedConvolutionNhwcGemm_i32x32<term, type, 0> : QuantizedConvolutionNhwcGemm_i16x32<term, type, 0>;
                QuantizedConvolutionNhwcGemm_iPtr tail_1 = m > 16 ? QuantizedConvolutionNhwcGemm_i32x16<term, type, 0> : QuantizedConvolutionNhwcGemm_i16x16<term, type, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _sBias[0] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 0);
                    _sBias[1] = _mm512_loadu_si512((__m512i*)(sBias + dc) + 1);
                    _sNorm[0] = _mm512_loadu_ps(sNorm + dc + 0);
                    _sNorm[1] = _mm512_loadu_ps(sNorm + dc + F);
                    if (type == SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint8_t* s = src;
                    int32_t* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                        tail_2(s, p, a, srcC, m, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                    else
                        tail_1(s, p, a, srcC, m, dC, update, weight, _sBias, _sNorm, _iLo, _iHi, _iScale, _params, _dNorm, _dZero, b, d);
                    weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_i2<Term8iInterim, SimdConvolutionActivationIdentity>;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationIdentity>; break;
            case SimdConvolutionActivationRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRelu>; break;
            case SimdConvolutionActivationLeakyRelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationLeakyRelu>; break;
            case SimdConvolutionActivationRestrictRange: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationRestrictRange>; break;
            case SimdConvolutionActivationPrelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationPrelu>; break;
            case SimdConvolutionActivationElu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationElu>; break;
            case SimdConvolutionActivationHswish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHswish>; break;
            case SimdConvolutionActivationMish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationMish>; break;
            case SimdConvolutionActivationHardSigmoid: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationHardSigmoid>; break;
            case SimdConvolutionActivationSwish: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationSwish>; break;
            case SimdConvolutionActivationGelu: convolutions[1] = QuantizedConvolutionNhwcGemm_i2<Term8iLast8u, SimdConvolutionActivationGelu>; break;
            default:
                convolutions[1] = NULL;
            }
        }

        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : Avx512vnni::SynetQuantizedConvolutionNhwcGemm(p)
        {
            if (_alg.K <= 32)
                return;
            SetAlgParam(F, F * 2, F * 2, 64, Base::AlgCacheL1(), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            AlgParam& a = _alg;
            if (_src8u)
            {
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                {
                    if (_is1x1)
                    {
                        if (a.batch == 1)
                        {
                            _convert = QuantizedConvolutionNhwcGemmReorder1x1R;
                            a.reorderType = 1;
                        }
                        else
                        {
                            _convert = QuantizedConvolutionNhwcGemmReorder1x1D;
                            a.reorderType = 0;
                        }
                    }
                    else
                    {
                        if (Aligned(p.srcC, 64) && a.batch == 1 && Aligned(p.dstW, a.F))
                        {
                            _convert = QuantizedConvolutionNhwcGemmReorderR;
                            a.reorderType = 1;
                        }
                    }
                }
            }
            else
                assert(0);
            Set(p, _alg, _convolutions);
        }
    }
#endif
}

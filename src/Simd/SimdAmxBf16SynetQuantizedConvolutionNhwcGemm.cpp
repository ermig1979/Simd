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

        template<Term8iType term, int cfg> void QuantizedConvolutionNhwcGemm_32x32(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst)
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
                    Apply8u2x8(dst + ds * dD, dD, buf + ds * dB, dB, bias, norm, zero, tailD);
                for (; ds < dstS; ++ds)
                    Apply8u2(dst + ds * dD, buf + ds * dB, bias, norm, zero, tailD);
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

        template<Term8iType term, int cfg> void QuantizedConvolutionNhwcGemm_32x16(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst)
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
                    Apply1x8<term>(dst + ds * dD, dD, buf + ds * dB, dB, bias, norm, zero, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term>(dst + ds * dD, buf + ds * dB, bias, norm, zero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + 16 * dB + 0, dB);
            }
        }

        template<Term8iType term, int cfg> void QuantizedConvolutionNhwcGemm_16x32(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst)
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
                    Apply2x8<term>(dst + ds * dD, dD, buf + ds * dB, dB, bias, norm, zero, tailD);
                for (; ds < dstS; ++ds)
                    Apply2<term>(dst + ds * dD, buf + ds * dB, bias, norm, zero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
                TileMoveToMemory(buf + F, dB);
            }
        }

        template<Term8iType term, int cfg> void QuantizedConvolutionNhwcGemm_16x16(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst)
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
                    Apply1x8<term>(dst + ds * dD, dD, buf + ds * dB, dB, bias, norm, zero, tailD);
                for (; ds < dstS; ++ds)
                    Apply1<term>(dst + ds * dD, buf + ds * dB, bias, norm, zero, tailD);
            }
            else
            {
                TileMoveToMemory(buf + 0, dB);
            }
        }

        typedef void (*QuantizedConvolutionNhwcGemmPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> void QuantizedConvolutionNhwcGemm_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH,
                size_t srcC, int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.bufK * DF;
            size_t dB = a.macroK < a.bufK ? a.dB : 0, dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            if (nn)
            {
                bool avoidSrcOverflow = !(a.reorderType == 1 && p.Is1x1());
                if (avoidSrcOverflow)
                    m = AlignHi(m, 16), nn = n1 - m;
                QuantizedConvolutionNhwcGemmPtr body_2 = QuantizedConvolutionNhwcGemm_32x32<term, 0>;
                QuantizedConvolutionNhwcGemmPtr tail_2 = m > 16 ? QuantizedConvolutionNhwcGemm_32x32<term, 0> : QuantizedConvolutionNhwcGemm_16x32<term, 0>;
                QuantizedConvolutionNhwcGemmPtr body_1 = QuantizedConvolutionNhwcGemm_32x16<term, 0>;
                QuantizedConvolutionNhwcGemmPtr tail_1 = m > 16 ? QuantizedConvolutionNhwcGemm_32x16<term, 0> : QuantizedConvolutionNhwcGemm_16x16<term, 0>;
                SetTileConfFull();
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    const uint8_t* s = src;
                    int32_t* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, srcC, n, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                        if (m)
                            tail_2(s + nn * dS, p, a, srcC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + nn * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, srcC, n, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                        if (m)
                            tail_1(s + nn * dS, p, a, srcC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + nn * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                QuantizedConvolutionNhwcGemmPtr tail_2 = m > 16 ? QuantizedConvolutionNhwcGemm_32x32<term, 0> : QuantizedConvolutionNhwcGemm_16x32<term, 0>;
                QuantizedConvolutionNhwcGemmPtr tail_1 = m > 16 ? QuantizedConvolutionNhwcGemm_32x16<term, 0> : QuantizedConvolutionNhwcGemm_16x16<term, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, dstC - dc);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    const uint8_t* s = src;
                    int32_t* b = buf + dc;
                    uint8_t* d = dst + dc * a.elem;
                    size_t i = 0;
                    if (dC > F)
                        tail_2(s, p, a, srcC, m, dC, update, weight, _bias, _norm, _zero, b, d);
                    else
                        tail_1(s, p, a, srcC, m, dC, update, weight, _bias, _norm, _zero, b, d);
                    weight += dW;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Set(const ConvParam& p, const AlgParam& a, Convolution* convolutions)
        {
            convolutions[0] = QuantizedConvolutionNhwcGemm_2<Term8iInterim>;
            if (p.dstT == SimdTensorData8u)
                convolutions[1] = QuantizedConvolutionNhwcGemm_2<Term8iLast8u>;
            else
                convolutions[1] = NULL;// QuantizedConvolutionNhwcGemm_2<Term8iLast32f>;
        }

        SynetQuantizedConvolutionNhwcGemm::SynetQuantizedConvolutionNhwcGemm(const ConvParam& p)
            : Avx512vnni::SynetQuantizedConvolutionNhwcGemm(p)
        {
            SetAlgParam(F, F * 2, F * 2, 64, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            AlgParam& a = _alg;
            if (_src8u)
            {
                if (_is1x1 && a.K == a.bufK)
                    _convert = NULL;
                else
                {
                    if (_is1x1 && a.batch == 1)
                    {
                        _convert = QuantizedConvolutionNhwcGemmReorder1x1R;
                        a.reorderType = 1;
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

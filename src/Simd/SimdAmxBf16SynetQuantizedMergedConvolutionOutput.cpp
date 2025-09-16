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
#include "Simd/SimdSynetQuantizedMergedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //------------------------------------------------------------------------------------------------

        template<Term8iType term, int cfg> void QuantizedMergedConvolutionOutput_2x2(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = a.obStep ? (int)a.owStep : 32, dD = int(p.dstC);
            int strideS = dS, strideW = 64, strideB = dB * 4, stepS = 64;
            const uint8_t* src1 = src0 + dS * 16;
            const int8_t * weight1 = weight0 + AlignHi(srcC, a.miK) * F;
            int32_t* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x2(dstS, dstC);
            if (update)
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(1, buf0 + F, strideB);
                _tile_stream_loadd(2, buf1 + 0, strideB);
                _tile_stream_loadd(3, buf1 + F, strideB);
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            _tile_stored(3, buf1 + F, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term>(dst + s * dD, dD, buf0 + s * dB, dB, bias, norm, zero, tailD);
                for (; s < dstS; ++s)
                    Apply2<term>(dst + s * dD, buf0 + s * dB, bias, norm, zero, tailD);
            }
        }

        template<Term8iType term, int cfg> void QuantizedMergedConvolutionOutput_2x1(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = a.obStep ? (int)a.owStep : 32, dD = int(p.dstC);
            int strideS = dS, strideW = 64, strideB = dB * 4, stepS = 64;
            const uint8_t* src1 = src0 + dS * 16;
            int32_t* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
            if (update)
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(2, buf1 + 0, strideB);
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term>(dst + s * dD, dD, buf0 + s * dB, dB, bias, norm, zero, tailD);
                for (; s < dstS; ++s)
                    Apply1<term>(dst + s * dD, buf0 + s * dB, bias, norm, zero, tailD);
            }
        }

        template<Term8iType term, int cfg> void QuantizedMergedConvolutionOutput_1x2(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = a.obStep ? (int)a.owStep : 32, dD = int(p.dstC);
            int strideS = dS, strideW = 64, strideB = dB * 4, stepS = 64;
            const int8_t* weight1 = weight0 + AlignHi(srcC, a.miK) * F;

            if (cfg)
                SetTileConf1x2(dstS, dstC);
            if (update)
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
                _tile_stream_loadd(1, buf0 + F, strideB);
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term>(dst + s * dD, dD, buf0 + s * dB, dB, bias, norm, zero, tailD);
                for (; s < dstS; ++s)
                    Apply2<term>(dst + s * dD, buf0 + s * dB, bias, norm, zero, tailD);
            }
        }

        template<Term8iType term, int cfg> void QuantizedMergedConvolutionOutput_1x1(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = a.obStep ? (int)a.owStep : 32, dD = int(p.dstC);
            int strideS = dS, strideW = 64, strideB = dB * 4, stepS = 64;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (update)
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
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

            _tile_stored(0, buf0 + 0, strideB);
            if (term == Term8iLast8u)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term>(dst + s * dD, dD, buf0 + s * dB, dB, bias, norm, zero, tailD);
                for (; s < dstS; ++s)
                    Apply1<term>(dst + s * dD, buf0 + s * dB, bias, norm, zero, tailD);
            }
        }

        typedef void (*QuantizedMergedConvolutionOutputPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int update, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst);

        template<Term8iType term> void QuantizedMergedConvolutionOutputConvolution_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            int update, const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* buf, uint8_t* dst)
        {
            size_t n = 32, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dW = AlignHi(maC, a.miK) * DF, dS = a.maC, dB = a.obStep, dD = p.dstC;
            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            if (nn)
            {
                QuantizedMergedConvolutionOutputPtr body_2 = QuantizedMergedConvolutionOutput_2x2<term, 0>;
                QuantizedMergedConvolutionOutputPtr tail_2 = m > 16 ? QuantizedMergedConvolutionOutput_2x2<term, 0> : QuantizedMergedConvolutionOutput_1x2<term, 0>;
                QuantizedMergedConvolutionOutputPtr body_1 = QuantizedMergedConvolutionOutput_2x1<term, 0>;
                QuantizedMergedConvolutionOutputPtr tail_1 = m > 16 ? QuantizedMergedConvolutionOutput_2x1<term, 0> : QuantizedMergedConvolutionOutput_1x1<term, 0>;
                SetTileConfFull();
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    const uint8_t* s = src;
                    int32_t* b = buf + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + dc + yBeg * p.dstW * p.dstC;
                    size_t i = 0;
                    if (dC > F)
                    {
                        for (; i < nn; i += n)
                            body_2(s + i * dS, p, a, maC, n, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                        if (m)
                            tail_2(s + i * dS, p, a, maC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                    }
                    else
                    {
                        for (; i < nn; i += n)
                            body_1(s + i * dS, p, a, maC, n, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                        if (m)
                            tail_1(s + i * dS, p, a, maC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                    }
                    weight += dW;
                }
            }
            else
            {
                QuantizedMergedConvolutionOutputPtr tail_2 = m > 16 ? QuantizedMergedConvolutionOutput_2x2<term, 0> : QuantizedMergedConvolutionOutput_1x2<term, 0>;
                QuantizedMergedConvolutionOutputPtr tail_1 = m > 16 ? QuantizedMergedConvolutionOutput_2x1<term, 0> : QuantizedMergedConvolutionOutput_1x1<term, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    const uint8_t* s = src;
                    int32_t* b = buf + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + dc + yBeg * p.dstW * p.dstC;
                    size_t i = 0;
                    if (dC > F)
                        tail_2(s + i * dS, p, a, maC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                    else
                        tail_1(s + i * dS, p, a, maC, m, dC, update, weight, _bias, _norm, _zero, b + i * dB, d + i * dD);
                    weight += dW;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i QuantizedAdd(const __m512i& a, const __m512i& aBias, const __m512& aNorm, const __m512i& b, const __m512i& bBias, const __m512& bNorm, const __m512& dNorm, const __m512i& dZero)
        {
            return QuantizeLinear(_mm512_add_ps(DequantizeLinear(a, aBias, aNorm), DequantizeLinear(b, bBias, bNorm)), dNorm, dZero);
        }

        SIMD_INLINE void AddInputToOutput16(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero, __mmask16 tail = -1)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, a)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(tail, b)), bBias, bNorm, dNorm, dZero);
            _mm_mask_storeu_epi8(dst, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0), K_ZERO)));
        }

        SIMD_INLINE void AddInputToOutput64(const uint8_t* a, const __m512i& aBias, const __m512& aNorm, const uint8_t* b, const __m512i& bBias, const __m512& bNorm, uint8_t* dst, const __m512& dNorm, const __m512i& dZero)
        {
            __m512i d0 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 0)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 0)), bBias, bNorm, dNorm, dZero);
            __m512i d1 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 1)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 1)), bBias, bNorm, dNorm, dZero);
            __m512i d2 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 2)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 2)), bBias, bNorm, dNorm, dZero);
            __m512i d3 = QuantizedAdd(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)a + 3)), aBias, aNorm, _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)b + 3)), bBias, bNorm, dNorm, dZero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
        }

        void QuantizedMergedConvolutionAddInputToOutput(const uint8_t* a, int aBias, float aNorm, const uint8_t* b, int bBias, float bNorm,
            const ConvParam& p, size_t yBeg, size_t yEnd, float dNorm, int dZero, uint8_t* dst)
        {
            __m512i _aBias = _mm512_set1_epi32(aBias), _bBias = _mm512_set1_epi32(bBias), _dZero = _mm512_set1_epi32(dZero);
            __m512 _aNorm = _mm512_set1_ps(aNorm), _bNorm = _mm512_set1_ps(bNorm), _dNorm = _mm512_set1_ps(dNorm);
            size_t beg = yBeg * p.dstW * p.dstC, end = yEnd * p.dstW * p.dstC;
            size_t i = beg, end16 = beg + AlignLo(end - beg, 16), end64 = beg + AlignLo(end - beg, 64);
            __mmask16 tail = TailMask16(end - end16);
            for (; i < end64; i += 64)
                AddInputToOutput64(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, dst + i, _dNorm, _dZero);
            for (; i < end16; i += 16)
                AddInputToOutput16(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, dst + i, _dNorm, _dZero);
            if (i < end)
                AddInputToOutput16(a + i, _aBias, _aNorm, b + i, _bBias, _bNorm, dst + i, _dNorm, _dZero, tail);

        }

        //------------------------------------------------------------------------------------------------

        void SetOutputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::OutputConvolutionPtr* funcs)
        {
            if (p.srcC < 16)
                Avx512vnni::SetOutputConvolution(p, a, funcs);
            else
            {
                funcs[0] = QuantizedMergedConvolutionOutputConvolution_2<Term8iInterim>;
                funcs[1] = QuantizedMergedConvolutionOutputConvolution_2<Term8iLast8u>;
            }
        }

        void SetAddInputToOutput(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::AddInputToOutputPtr& func)
        {
            func = QuantizedMergedConvolutionAddInputToOutput;
        }
    }
#endif
}

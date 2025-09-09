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
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedMergedConvolution::AlgParam AlgParam;

        //-------------------------------------------------------------------------------------------------

        void QuantizedMergedConvolutionInputPreprocess(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t yBeg, size_t yEnd, uint8_t* dst)
        {
            size_t srcC64 = AlignLo(p.srcC, 64), n = (yEnd - yBeg) * p.dstW;
            __mmask64 srcMask = TailMask64(p.srcC - srcC64);
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
                dst += a.iwStep * 16;
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void Apply1(const int32_t* src, uint8_t* dst, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            __m512i i0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 0), bias)), norm)), zero);
            _mm_storeu_si128((__m128i*)dst, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i0, K_ZERO), K_ZERO)));
        }

        SIMD_INLINE void Apply2(const int32_t* src, uint8_t* dst, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            __m512i i0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 0), bias)), norm)), zero);
            __m512i i1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 1), bias)), norm)), zero);
            _mm256_storeu_si256((__m256i*)dst, _mm512_castsi512_si256(PackI16ToU8(PackI32ToI16(i0, i1), K_ZERO)));
        }

       SIMD_INLINE void Apply4(const int32_t* src, uint8_t* dst, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            __m512i i0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 0), bias)), norm)), zero);
            __m512i i1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 1), bias)), norm)), zero);
            __m512i i2 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 2), bias)), norm)), zero);
            __m512i i3 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512((__m512i*)src + 3), bias)), norm)), zero);
            _mm512_storeu_si512((__m512i*)dst, PackI16ToU8(PackI32ToI16(i0, i1), PackI32ToI16(i2, i3)));
            _mm_prefetch((const char*)dst, _MM_HINT_NTA);
        }

        SIMD_INLINE void Apply8(const int32_t* src, uint8_t* dst, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            Apply4(src + 0 * F, dst + 0 * F, bias, norm, zero);
            Apply4(src + 4 * F, dst + 4 * F, bias, norm, zero);
        }

        SIMD_INLINE void Apply16(const int32_t* src, uint8_t* dst, const __m512i& bias, const __m512& norm, const __m512i& zero)
        {
            Apply8(src + 0 * F, dst + 0 * F, bias, norm, zero);
            Apply8(src + 8 * F, dst + 8 * F, bias, norm, zero);
        }

        //------------------------------------------------------------------------------------------------

        template<int cfg> void QuantizedMergedConvolutionInput_2x2(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst0, uint8_t* dst1)
        {
            int srcC = (int)a.iwStep, strideW = 64, stepS = a.isB ? 1024 : 64, strideS = a.isB ? 64 : srcC;
            const uint8_t* src1 = src0 + 16 * srcC;
            const int8_t* weight1 = weight0 + a.iwStep * F;

            if (cfg)
                SetTileConf2x2(dstS, 32);
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

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

            _tile_stored(0, buf + 0 * 256, 64);
            _tile_stored(1, buf + 1 * 256, 64);
            Apply16(buf + 0 * 256, dst0 + 0 * 256, bias[0], norm[0], zero);
            Apply16(buf + 1 * 256, dst1 + 0 * 256, bias[1], norm[1], zero);
            _tile_stored(2, buf + 2 * 256, 64);
            _tile_stored(3, buf + 3 * 256, 64);
            if (dstS == 32)
            {
                Apply16(buf + 2 * 256, dst0 + 1 * 256, bias[0], norm[0], zero);
                Apply16(buf + 3 * 256, dst1 + 1 * 256, bias[1], norm[1], zero);
            }
            else
            {
                for (size_t s = 16; s < dstS; ++s)
                {
                    Apply1(buf + 1 * 256 + s * 16, dst0 + s * 16, bias[0], norm[0], zero);
                    Apply1(buf + 2 * 256 + s * 16, dst1 + s * 16, bias[1], norm[1], zero);
                }
            }
        }

        template<int cfg> void QuantizedMergedConvolutionInput_2x1(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst0, uint8_t* dst1)
        {
            int srcC = (int)a.iwStep, strideW = 64, stepS = a.isB ? 1024 : 64, strideS = a.isB ? 64 : srcC;
            const uint8_t* src1 = src0 + 16 * srcC;

            if (cfg)
                SetTileConf2x1(dstS, 16);
            _tile_zero(0);
            _tile_zero(2);

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

            _tile_stored(0, buf + 0 * 256, 64);
            Apply16(buf + 0 * 256, dst0 + 0 * 256, bias[0], norm[0], zero);
            _tile_stored(2, buf + 2 * 256, 64);
            if (dstS == 32)
                Apply16(buf + 2 * 256, dst0 + 1 * 256, bias[0], norm[0], zero);
            else
            {
                for (size_t s = 16; s < dstS; ++s)
                    Apply1(buf + 1 * 256 + s * 16, dst0 + s * 16, bias[0], norm[0], zero);
            }
        }

        template<int cfg> void QuantizedMergedConvolutionInput_1x2(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst0, uint8_t* dst1)
        {
            int srcC = (int)a.iwStep, strideW = 64, stepS = a.isB ? 1024 : 64, strideS = a.isB ? 64 : srcC;
            const int8_t* weight1 = weight0 + a.iwStep * F;

            if (cfg)
                SetTileConf1x2(dstS, 32);
            _tile_zero(0);
            _tile_zero(1);

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

            _tile_stored(0, buf + 0 * 256, 64);
            _tile_stored(1, buf + 1 * 256, 64);
            if (dstS == 16)
            {
                Apply16(buf + 0 * 256, dst0 + 0 * 256, bias[0], norm[0], zero);
                Apply16(buf + 1 * 256, dst1 + 0 * 256, bias[1], norm[1], zero);
            }
            else
            {
                for (size_t s = 0; s < dstS; ++s)
                {
                    Apply1(buf + 0 * 256 + s * 16, dst0 + s * 16, bias[0], norm[0], zero);
                    Apply1(buf + 1 * 256 + s * 16, dst1 + s * 16, bias[1], norm[1], zero);
                }
            }
        }

        template<int cfg> void QuantizedMergedConvolutionInput_1x1(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst0, uint8_t* dst1)
        {
            int srcC = (int)a.iwStep, strideW = 64, stepS = a.isB ? 1024 : 64, strideS = a.isB ? 64 : srcC;

            if (cfg)
                SetTileConf1x1(dstS, 16);
            _tile_zero(0);

            for (size_t sc = 0; sc < srcC; sc += 64, src0 += stepS)
            {
                _tile_stream_loadd(4, src0, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbusd(0, 4, 6);
            }

            _tile_stored(0, buf + 0 * 256, 64);
            if (dstS == 16)
                Apply16(buf + 0 * 256, dst0 + 0 * 256, bias[0], norm[0], zero);
            else
            {
                for (size_t s = 0; s < dstS; ++s)
                    Apply1(buf + 0 * 256 + s * 16, dst0 + s * 16, bias[0], norm[0], zero);
            }
        }

        typedef void (*QuantizedMergedConvolutionInputPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const int8_t* weight0, const __m512i* bias, const __m512* norm, const __m512i& zero, int32_t* buf, uint8_t* dst0, uint8_t* dst1);

        void QuantizedMergedConvolutionInput_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd,
            const int8_t* weight, const int32_t* bias, const float* norm, int32_t zero, int32_t* sum, uint8_t* dst)
        {
            size_t dstM = a.dsH - 1, dstS = a.dsH * p.dstW * F, srcC = a.isB ? a.iwStep : p.srcC, y0 = a.isB ? yBeg : 0;
            __m512 _norm[2];
            __m512i _bias[2], _zero = _mm512_set1_epi32(zero);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.dsH)), n = 32;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLo(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLo(e1, n), e = e1 - en;

            if (yInt == yBeg)
            {
                if (en || a.isB)
                {
                    if(a.isB == 0)
                        e = AlignHi(e, 16), en = e1 - e;
                    QuantizedMergedConvolutionInputPtr conv_2x2 = QuantizedMergedConvolutionInput_2x2<0>;
                    QuantizedMergedConvolutionInputPtr conv_2x1 = QuantizedMergedConvolutionInput_2x1<0>;
                    QuantizedMergedConvolutionInputPtr conv_Ex2 = e > 16 ? QuantizedMergedConvolutionInput_2x2<0> : QuantizedMergedConvolutionInput_1x2<0>;
                    QuantizedMergedConvolutionInputPtr conv_Ex1 = e > 16 ? QuantizedMergedConvolutionInput_2x1<0> : QuantizedMergedConvolutionInput_1x1<0>;
                    SetTileConfFull();
                    for (size_t dc = 0; dc < maC; dc += DF)
                    {
                        size_t dC = Simd::Min(DF, maC - dc);
                        _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                        _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                        _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                        _norm[1] = _mm512_loadu_ps(norm + dc + F);
                        if (dC > F)
                        {
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            size_t j = 0;
                            for (; j < en; j += n)
                                conv_2x2(src0 + j * srcC, p, a, n, weight, _bias, _norm, _zero, sum, dst0 + j * F, dst1 + j * F);
                            if (en < e1)
                                conv_Ex2(src0 + en * srcC, p, a, e, weight, _bias, _norm, _zero, sum, dst0 + en * F, dst1 + en * F);
                        }
                        else
                        {
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            size_t j = 0;
                            for (; j < en; j += n)
                                conv_2x1(src0 + j * srcC, p, a, n, weight, _bias, _norm, _zero, sum, dst0 + j * F, NULL);
                            if (en < e1)
                                conv_Ex1(src0 + en * srcC, p, a, e, weight, _bias, _norm, _zero, sum, dst0 + en * F, NULL);
                        }
                        dst += a.dsH * p.dstW * DF;
                        weight += srcC * DF;
                    }
                }
                else if (e1)
                {
                    QuantizedMergedConvolutionInputPtr conv_Ex2 = e > 16 ? QuantizedMergedConvolutionInput_2x2<0> : QuantizedMergedConvolutionInput_1x2<0>;
                    QuantizedMergedConvolutionInputPtr conv_Ex1 = e > 16 ? QuantizedMergedConvolutionInput_2x1<0> : QuantizedMergedConvolutionInput_1x1<0>;
                    if (e > 16)
                        SetTileConf2x2(e, 32);
                    else
                        SetTileConf1x2(e, 32);
                    for (size_t dc = 0; dc < maC; dc += DF)
                    {
                        size_t dC = Simd::Min(DF, maC - dc);
                        _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                        _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                        _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                        _norm[1] = _mm512_loadu_ps(norm + dc + F);
                        if (dC > F)
                        {
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            conv_Ex2(src0, p, a, e, weight, _bias, _norm, _zero, sum, dst0, dst1);
                        }
                        else
                        {
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            conv_Ex1(src0, p, a, e, weight, _bias, _norm, _zero, sum, dst0, NULL);
                        }
                        dst += a.dsH * p.dstW * DF;
                        weight += srcC * DF;
                    }
                }
            }
            else
            {
                QuantizedMergedConvolutionInputPtr conv_2x2 = QuantizedMergedConvolutionInput_2x2<0>;
                QuantizedMergedConvolutionInputPtr conv_2x1 = QuantizedMergedConvolutionInput_2x1<0>;
                QuantizedMergedConvolutionInputPtr conv_Ix2 = i > 16 ? QuantizedMergedConvolutionInput_2x2<1> : QuantizedMergedConvolutionInput_1x2<1>;
                QuantizedMergedConvolutionInputPtr conv_Ix1 = i > 16 ? QuantizedMergedConvolutionInput_2x1<1> : QuantizedMergedConvolutionInput_1x1<1>;
                QuantizedMergedConvolutionInputPtr conv_Ex2 = e > 16 ? QuantizedMergedConvolutionInput_2x2<1> : QuantizedMergedConvolutionInput_1x2<1>;
                QuantizedMergedConvolutionInputPtr conv_Ex1 = e > 16 ? QuantizedMergedConvolutionInput_2x1<1> : QuantizedMergedConvolutionInput_1x1<1>;
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_si512((__m512i*)(bias + dc) + 0);
                    _bias[1] = _mm512_loadu_si512((__m512i*)(bias + dc) + 1);
                    _norm[0] = _mm512_loadu_ps(norm + dc + 0);
                    _norm[1] = _mm512_loadu_ps(norm + dc + F);
                    if (dC > F)
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint8_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_2x2(src0, p, a, n, weight, _bias, _norm, _zero, sum, dst0, dst1);
                            if (in < i1)
                                conv_Ix2(src0, p, a, i, weight, _bias, _norm, _zero, sum, dst0, dst1);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_2x2(src0, p, a, n, weight, _bias, _norm, _zero, sum, dst0, dst1);
                            if (en < e1)
                                conv_Ex2(src0, p, a, e, weight, _bias, _norm, _zero, sum, dst0, dst1);
                        }
                    }
                    else
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint8_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n)
                                conv_2x1(src0, p, a, n, weight, _bias, _norm, _zero, sum, dst0, NULL);
                            if (in < i1)
                                conv_Ix1(src0, p, a, i, weight, _bias, _norm, _zero, sum, dst0, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint8_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            uint8_t* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n)
                                conv_2x1(src0, p, a, n, weight, _bias, _norm, _zero, sum, dst0, NULL);
                            if (en < e1)
                                conv_Ex1(src0, p, a, e, weight, _bias, _norm, _zero, sum, dst0, NULL);
                        }
                    }
                    dst += a.dsH * p.dstW * DF;
                    weight += srcC * DF;
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        void SetInputPreprocess(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputPreprocessPtr& func)
        {
            func = QuantizedMergedConvolutionInputPreprocess;
        }

        void SetInputConvolution(const ConvParam& p, const Base::SynetQuantizedMergedConvolution::AlgParam& a, Base::SynetQuantizedMergedConvolution::InputConvolutionPtr& func)
        {
            func = QuantizedMergedConvolutionInput_2;
        }
    }
#endif
}

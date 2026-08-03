/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSynetQuantizedActivation.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV1::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV1::LastConvPtr LastConvPtr;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcSpecV1_Reorder(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            assert(a.microC == A);
            size_t sC = p.srcC, sCA = Simd::AlignLo(sC, A), asC = a.srcC, sW = p.srcW, asW = a.srcW;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad), apH = a.padH;
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            __m512i _zero = _mm512_set1_epi8(zero);
            __mmask64 tailC = TailMask64(sC - sCA);
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * asW; s < n; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
                dst += a.padV * asW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * sW * sC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * asW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (apH)
                {
                    for (size_t s = 0; s < apH; ++s)
                        for (size_t c = 0; c < asC; c += A)
                            _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
                    dst += apH * sD;
                }
                for (size_t sx = 0; sx < sW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < sCA; sc += A)
                        Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Copy(src + sc, dst + sc * cD, tailC);
                    src += sC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0; s < apH; ++s)
                    for (size_t c = 0; c < asC; c += A)
                        _mm512_storeu_si512((__m512i*)(dst + c * cD + s * sD), _zero);
            }
        }

        static void QuantizedConvolutionNhwcSpecV1_Reorder64(const uint8_t* src, uint8_t zero, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint8_t* dst)
        {
            assert(a.microC == A && p.srcC <= A);
            size_t sC = p.srcC, sW = p.srcW, asW = a.srcW;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad), apH = a.padH;
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            __m512i _zero = _mm512_set1_epi8(zero);
            __mmask64 tailC = TailMask64(sC);
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * asW; s < n; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * sW * sC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * asW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                for (size_t s = 0; s < apH; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
                for (size_t sx = 0; sx < sW; ++sx, src += sC, dst += A)
                    Copy(src, dst, tailC);
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0; s < apH; ++s, dst += A)
                    _mm512_storeu_si512((__m512i*)dst, _zero);
            }
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void QuantizedConvolutionNhwcSpecV1_Body32x32(const uint8_t* src0, const ConvParam& p, 
            const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 16 * dS;
            int32_t* buf1 = buf0 + 16 * dB;

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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_stream_loadd(4, src0 + o, strideS);
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i, weight1 += dW)
            {
                _tile_stream_loadd(5, src1 + o, strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbusd(0, 4, 6);
                _tile_dpbusd(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                _tile_dpbusd(2, 5, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbusd(3, 5, 7);
            }
            _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbusd(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);
            TileMoveToMemory(buf0 + 0, dB);

            _tile_dpbusd(1, 4, 7);
            _tile_stored(1, buf0 + F, strideB);
            TileMoveToMemory(buf0 + F, dB);

            _tile_dpbusd(2, 5, 6);
            _tile_stored(2, buf1 + 0, strideB);
            TileMoveToMemory(buf1 + 0, dB);

            _tile_dpbusd(3, 5, 7);
            _tile_stored(3, buf1 + F, strideB);
            TileMoveToMemory(buf1 + F, dB);
        }

        SIMD_INLINE void QuantizedConvolutionNhwcSpecV1_Body32x16(const uint8_t* src0, const ConvParam& p, 
            const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;
            const uint8_t* src1 = src0 + 16 * dS;
            int32_t* buf1 = buf0 + 16 * dB;

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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_loadd(4, src0 + o, strideS);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(6, weight0, strideW);
                _tile_loadd(5, src1 + o, strideS);
                _tile_dpbusd(0, 4, 6);
                o = offs[i + 1];
                _tile_loadd(4, src0 + o, strideS);
                _tile_dpbusd(2, 5, 6);
                weight0 += dW;
            }
            _tile_stream_loadd(6, weight0, strideW);
            _tile_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbusd(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);
            TileMoveToMemory(buf0 + 0, dB);

            _tile_dpbusd(2, 5, 6);
            _tile_stored(2, buf1 + 0, strideB);
            TileMoveToMemory(buf1 + 0, dB);
        }

        SIMD_INLINE void QuantizedConvolutionNhwcSpecV1_Body16x32(const uint8_t* src0, const ConvParam& p, 
            const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;
            const int8_t* weight1 = weight0 + a.K * F;

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

            int n1 = (int)nK - 1;
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i, weight1 += dW)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbusd(0, 4, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbusd(1, 4, 7);
            }
            _tile_stream_loadd(4, src0 + offs[n1], strideS);
            _tile_loadd(7, weight1, strideW);

            _tile_dpbusd(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);
            TileMoveToMemory(buf0 + 0, dB);

            _tile_dpbusd(1, 4, 7);
            _tile_stored(1, buf0 + F, strideB);
            TileMoveToMemory(buf0 + F, dB);
        }

        SIMD_INLINE void QuantizedConvolutionNhwcSpecV1_Body16x16(const uint8_t* src0, const ConvParam& p, 
            const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;

            if (update)
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
            }
            else
            {
                _tile_zero(0);
            }

            int n = (int)nK;
            for (int i = 0; i < n; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(6, weight0, strideW);
                _tile_dpbusd(0, 4, 6);
                weight0 += dW;
            }

            _tile_stored(0, buf0 + 0, strideB);
            TileMoveToMemory(buf0 + 0, dB);
        }

        static void QuantizedConvolutionNhwcSpecV1_Body(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* offs,
            size_t dstC, size_t dstS, size_t nK, int update, const int8_t* weight, int32_t* buf)
        {
            size_t n1 = AlignHi(dstS, 16), n = 32;
            size_t nn = AlignLo(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dB = a.macroD, dS = a.microC;

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        QuantizedConvolutionNhwcSpecV1_Body32x32(src + i * dS, p, a, offs, nK, update, weight, buf + i * dB);
                    if (m)
                        QuantizedConvolutionNhwcSpecV1_Body16x32(src + i * dS, p, a, offs, nK, update, weight, buf + i * dB);
                }
                else
                {
                    for (; i < nn; i += n)
                        QuantizedConvolutionNhwcSpecV1_Body32x16(src + i * dS, p, a, offs, nK, update, weight, buf + i * dB);
                    if (m)
                        QuantizedConvolutionNhwcSpecV1_Body16x16(src + i * dS, p, a, offs, nK, update, weight, buf + i * dB);
                }
                weight += dW;
                buf += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M, int flush> static SIMD_INLINE void ApplyMx1(
            uint8_t*& ptr, int dP, int32_t* buf, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, const int* mask, __mmask32 tail = -1)
        {
            uint32_t msk = mask[0];
            tail = tail & msk;
            if (M == 1)
            {
                __m512i d0 = Avx512bw::ToSave32i<type, 0>(_mm512_loadu_si512(buf + 0), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                if (term == Term8iLast8u)
                {
                    _mm_mask_storeu_epi8(ptr, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0, K_ZERO), K_ZERO)));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
                else if (term == Term8iLast32f)
                {
                    assert(0);
                }
            }
            else if (M == 2)
            {
                __m512i d0 = Avx512bw::ToSave32i<type, 0>(_mm512_loadu_si512(buf + 0), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                __m512i d1 = Avx512bw::ToSave32i<type, 1>(_mm512_loadu_si512(buf + F), sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero);
                if (term == Term8iLast8u)
                {
                    _mm256_mask_storeu_epi8(ptr, tail, _mm512_castsi512_si256(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
                    if (flush) _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                }
                else if (term == Term8iLast32f)
                {
                    assert(0);
                }
            }
            ptr += dP & msk;
        }

        template<Term8iType term, SimdConvolutionActivationType type, int M, int N, int flush> static SIMD_INLINE void ApplyMxN(
            uint8_t*& ptr, int dP, int32_t* buf, const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi,
            const __m512& iScale, const __m512* params, const __m512& dNorm, const __m512i& dZero, const int* mask, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 0 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 0, tail);
            if (N > 1) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 1 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 1, tail);
            if (N > 2) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 2 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 2, tail);
            if (N > 3) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 3 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 3, tail);
            if (N > 4) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 4 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 4, tail);
            if (N > 5) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 5 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 5, tail);
            if (N > 6) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 6 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 6, tail);
            if (N > 7) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 7 * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + 7, tail);
        }

        //-----------------------------------------------------------------------------------------

        template<Term8iType term, SimdConvolutionActivationType type, int M, int apply, int flush> void QuantizedConvolutionNhwcSpecV1_1x32x32(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params, 
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf1, int32_t* buf2, const int* mask, uint8_t*& dst, __mmask32 tail)
        {
            int dD = int(p.dstC * a.elem), dB = (int)DF, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 16 * dS;

            if (update)
            {
                int strideB = (int)a.macroD * 4;
                if (M > 0) _tile_stream_loadd(0, sum0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, sum0 + F, strideB);
                sum0 += 16 * a.macroD;
                if (M > 0) _tile_stream_loadd(2, sum0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(3, sum0 + F, strideB);
            }
            else
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
                if (M > 0) _tile_zero(2);
                if (M > 1) _tile_zero(3);
            }

            int n1 = (int)nK - 1, i = 0, o = offs[0], na = apply ? (8 / apply - 1) : 0, ds = 0;
            _tile_stream_loadd(4, src0 + o, strideS);
            if (M > 0) _tile_loadd(6, weight0, strideW);
            for (; i < na; ++i, weight1 += dW)
            {
                if (M > 1) _tile_loadd(7, weight1, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
                _tile_stream_loadd(5, src1 + o, strideS);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                if (M > 0) _tile_dpbusd(2, 5, 6);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
                if (M > 1) _tile_dpbusd(3, 5, 7);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
            }
            for (; i < n1; ++i, weight1 += dW)
            {
                if (M > 1) _tile_loadd(7, weight1, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                _tile_stream_loadd(5, src1 + o, strideS);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                if (M > 0) _tile_dpbusd(2, 5, 6);
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
                if (M > 1) _tile_dpbusd(3, 5, 7);
            }
            if (M > 1) _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            if (M > 0) _tile_dpbusd(0, 4, 6);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);

            if (M > 1) _tile_dpbusd(1, 4, 7);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
            buf2 += 16 * DF;
            if (M > 0) _tile_dpbusd(2, 5, 6);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
            if (M > 0) _tile_stored(2, buf2 + 0, strideB);

            if (M > 1) _tile_dpbusd(3, 5, 7);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += apply;
            if (M > 1) _tile_stored(3, buf2 + F, strideB);
        }

        template<Term8iType term, SimdConvolutionActivationType type, int M, int apply, int flush> void QuantizedConvolutionNhwcSpecV1_1x16x32(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0,
            const __m512i* sBias, const __m512* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const __m512* params,
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf1, int32_t* buf2, const int* mask, uint8_t*& dst, __mmask32 tail)
        {
            int dD = int(p.dstC * a.elem), dB = (int)DF, dS = (int)a.microC, strideS = dS, dW = 1024, strideW = 64, strideB = dB * 4;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 16 * dS;

            if (update)
            {
                int strideB = (int)a.macroD * 4;
                if (M > 0) _tile_stream_loadd(0, sum0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, sum0 + F, strideB);
            }
            else
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
            }

            int n1 = (int)nK - 1, i = 0, o = offs[0], na = apply ? (8 / apply - 1) : 0, ds = 0;
            _tile_stream_loadd(4, src0 + o, strideS);
            if (M > 0) _tile_loadd(6, weight0, strideW);
            for (; i < na; ++i, weight1 += dW)
            {
                if (M > 1) _tile_loadd(7, weight1, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                ApplyMxN<term, type, M, 2 * apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += 2 * apply;
                if (M > 1) _tile_dpbusd(1, 4, 7);
                ApplyMxN<term, type, M, 2 * apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += 2 * apply;
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
            }
            for (; i < n1; ++i, weight1 += dW)
            {
                if (M > 1) _tile_loadd(7, weight1, strideW);
                if (M > 0) _tile_dpbusd(0, 4, 6);
                if (M > 1) _tile_dpbusd(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
            }
            if (M > 1) _tile_loadd(7, weight1, strideW);

            if (M > 0) _tile_dpbusd(0, 4, 6);
            ApplyMxN<term, type, M, 2 * apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += 2 * apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);

            if (M > 1) _tile_dpbusd(1, 4, 7);
            ApplyMxN<term, type, M, 2 * apply, flush>(dst, dD, buf1 + ds * DF, sBias, sNorm, iLo, iHi, iScale, params, dNorm, dZero, mask + ds, tail), ds += 2 * apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
        }

        template<Term8iType term, SimdConvolutionActivationType type, int M, int apply, int flush> void QuantizedConvolutionNhwcSpecV1_Nx32x32M(
            const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, const int* offs, size_t nK, int update, const int8_t* weight0,
            const int32_t* sBias, const float* sNorm, const __m512i& iLo, const __m512i& iHi, const __m512& iScale, const float* params, __m512* _params,
            const __m512& dNorm, const __m512i& dZero, int32_t* sum0, int32_t* buf1, const int* mask, uint8_t* dst, __mmask32 tail)
        {
            int dB = (int)a.macroD, dD = int(p.dstC * a.elem), dS = (int)a.microC;
            int32_t* buf2 = buf1 + 1024;

            __m512i _sBias[2];
            if (M > 0) _sBias[0] = _mm512_loadu_si512((__m512i*)sBias + 0);
            if (M > 1) _sBias[1] = _mm512_loadu_si512((__m512i*)sBias + 1);
            __m512 _sNorm[2];
            if (M > 0) _sNorm[0] = _mm512_loadu_ps(sNorm + 0 * F);
            if (M > 1) _sNorm[1] = _mm512_loadu_ps(sNorm + 1 * F);
            if (type == SimdConvolutionActivationPrelu)
            {
                if (M > 0) _params[0] = _mm512_loadu_ps(params + 0 * F);
                if (M > 1) _params[1] = _mm512_loadu_ps(params + 1 * F);
            }

            size_t pds = 0;
            QuantizedConvolutionNhwcSpecV1_1x32x32<term, type, M, 0, flush>(src0, p, a, offs, nK, update, weight0, 
                _sBias, _sNorm, iLo, iHi, iScale, _params, dNorm, dZero, sum0, buf1, buf2, mask, dst, tail);
            for (size_t cds = 32; cds < dstS; pds += 32)
            {
                Swap(buf1, buf2);
                if (cds + 16 >= dstS)
                {
                    QuantizedConvolutionNhwcSpecV1_1x16x32<term, type, M, apply, flush>(src0 + cds * dS, p, a, offs, nK, update, weight0, 
                        _sBias, _sNorm, iLo, iHi, iScale, _params, dNorm, dZero, sum0 + cds * dB, buf1, buf2, mask + pds, dst, tail);
                    cds += 16;
                }
                else
                {
                    QuantizedConvolutionNhwcSpecV1_1x32x32<term, type, M, apply, flush>(src0 + cds * dS, p, a, offs, nK, update, weight0, 
                        _sBias, _sNorm, iLo, iHi, iScale, _params, dNorm, dZero, sum0 + cds * dB, buf1, buf2, mask + pds, dst, tail);
                    cds += 32;
                }
            }
            size_t dstS8 = dstS & (~7), ds = 0;
            for (; pds < dstS8; pds += 8, ds += 8)
            {
                ApplyMxN<term, type, M, 8, flush>(dst, dD, buf2 + ds * DF, _sBias, _sNorm, iLo, iHi, iScale, _params, dNorm, dZero, mask + pds, tail);
            }
            for (; pds < dstS; ++pds, ++ds)
            {
                ApplyMxN<term, type, M, 1, flush>(dst, dD, buf2 + ds * DF, _sBias, _sNorm, iLo, iHi, iScale, _params, dNorm, dZero, mask + pds, tail);
            }
        }

        //-------------------------------------------------------------------------------------------------

        typedef void (*QuantizedConvolutionNhwcSpecV1_LastPtr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, 
            const int* offs, size_t nK, int update, const int8_t* weight0, const int32_t* sBias, const float* sNorm, const __m512i& iLo, 
            const __m512i& iHi, const __m512& iScale, const float* params, __m512* _params, const __m512& dNorm, const __m512i& dZero,
            int32_t*sum, int32_t* buf, const int* mask, uint8_t* dst, __mmask32 tail);

        template<Term8iType term, SimdConvolutionActivationType type, int apply, int flush> void QuantizedConvolutionNhwcSpecV1_Last(
            const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* srcOffs, size_t dstC, size_t dstS, size_t nK, int update, 
            const int8_t* weight, int32_t* sum, int32_t* buf, const int32_t* sBias, const float* sNorm, int32_t iZero, float iScale,
            const float* params, float dNorm, int32_t dZero, const int* dstMask, const int* dstOffs, uint8_t* dst)
        {
            size_t n = 256, n1 = dstS, nn = AlignLoAny(n1, n), dW = a.K * a.microD;
            size_t dB = a.macroD, dD = p.dstC * a.elem, dS = a.microC;

            size_t dstC32 = AlignLo(dstC, 32), dstCt = dstC - dstC32;
            __mmask32 tailD = term == Term8iLast8u ? TailMask32(dstCt) : (__mmask32)TailMask16(dstCt - AlignLo(dstCt - 1, 16));
            QuantizedConvolutionNhwcSpecV1_LastPtr mainConv = QuantizedConvolutionNhwcSpecV1_Nx32x32M<term, type, 2, apply, flush>;
            QuantizedConvolutionNhwcSpecV1_LastPtr tailConv = dstCt > 16 ? QuantizedConvolutionNhwcSpecV1_Nx32x32M<term, type, 2, apply, flush> :
                QuantizedConvolutionNhwcSpecV1_Nx32x32M<term, type, 1, apply, flush>;

            __m512 _iScale, _params[2], _dNorm;
            __m512i _dZero = _mm512_set1_epi32(dZero), _iLo, _iHi;
            if (type != SimdConvolutionActivationIdentity)
            {
                _iLo = _mm512_set1_epi32(-iZero);
                _iHi = _mm512_set1_epi32(255 - iZero);
                _iScale = _mm512_set1_ps(iScale);
                _dNorm = _mm512_set1_ps(dNorm);
                _params[0] = _mm512_set1_ps(params[0]);
                if (type == SimdConvolutionActivationRestrictRange ||
                    type == SimdConvolutionActivationHswish ||
                    type == SimdConvolutionActivationHardSigmoid)
                    _params[1] = _mm512_set1_ps(params[1]);
            }

            SetTileConfFull();
            for (size_t i = 0; i < n1;)
            {
                size_t dn = (n1 - i >= n + 32 ? n : n1 - i);
                const uint8_t* s = src + i * dS;
                const int8_t* w = weight;
                int32_t* b = sum + i * dB;
                uint8_t* d = dst + (dstOffs[i / 32] - dstOffs[0]) * dD;
                size_t dc = 0;
                for (; dc < dstC32; dc += DF, w += dW)
                    mainConv(s, p, a, dn, srcOffs, nK, update, w, sBias + dc, sNorm + dc, _iLo, _iHi, _iScale, params + dc, 
                        _params, _dNorm, _dZero, b + dc, buf, dstMask + i, d + dc * a.elem, __mmask32(-1));
                if (dc < dstC)
                    tailConv(s, p, a, dn, srcOffs, nK, update, w, sBias + dc, sNorm + dc, _iLo, _iHi, _iScale, params + dc, 
                        _params, _dNorm, _dZero, b + dc, buf, dstMask + i, d + dc * a.elem, tailD);
                i += dn;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <Term8iType term, SimdConvolutionActivationType type, int flush> SIMD_INLINE void SetLastConvV3(const ConvParam& p, size_t nK, LastConvPtr& lastConv)
        {
            if (nK >= 8)
                lastConv = QuantizedConvolutionNhwcSpecV1_Last<term, type, 1, flush>;
            else if (nK >= 4)
                lastConv = QuantizedConvolutionNhwcSpecV1_Last<term, type, 2, flush>;
            else if (nK >= 2)
                lastConv = QuantizedConvolutionNhwcSpecV1_Last<term, type, 4, flush>;
            else
                lastConv = NULL;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetLastConvV3(const ConvParam& p, size_t nK, LastConvPtr& lastConv)
        {
            if (p.dstT == SimdTensorData8u)
                SetLastConvV3<Term8iLast8u, type, 0>(p, nK, lastConv);
            else
            {
                assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcSpecV1::SynetQuantizedConvolutionNhwcSpecV1(const ConvParam& p)
            : Base::SynetQuantizedConvolutionNhwcSpecV1(p)
        {
            SetAlgParam();
            AlgParam& a = _alg;
            if (_src8u)
            {
                if (p.srcC <= 64)
                    _preprocess = QuantizedConvolutionNhwcSpecV1_Reorder64;
                else
                    _preprocess = QuantizedConvolutionNhwcSpecV1_Reorder;
            }
            else
                assert(0);
            _bodyConv = QuantizedConvolutionNhwcSpecV1_Body;
            size_t nK = _nK[_nK.size - 1];
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetLastConvV3<SimdConvolutionActivationIdentity>(p, nK, _lastConv); break;
            case SimdConvolutionActivationRelu: SetLastConvV3<SimdConvolutionActivationRelu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationLeakyRelu: SetLastConvV3<SimdConvolutionActivationLeakyRelu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationRestrictRange: SetLastConvV3<SimdConvolutionActivationRestrictRange>(p, nK, _lastConv); break;
            case SimdConvolutionActivationPrelu: SetLastConvV3<SimdConvolutionActivationPrelu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationElu: SetLastConvV3<SimdConvolutionActivationElu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationHswish: SetLastConvV3<SimdConvolutionActivationHswish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationMish: SetLastConvV3<SimdConvolutionActivationMish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationHardSigmoid: SetLastConvV3<SimdConvolutionActivationHardSigmoid>(p, nK, _lastConv); break;
            case SimdConvolutionActivationSwish: SetLastConvV3<SimdConvolutionActivationSwish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationGelu: SetLastConvV3<SimdConvolutionActivationGelu>(p, nK, _lastConv); break;
            default: assert(0);
            }
        }
    }
#endif
}

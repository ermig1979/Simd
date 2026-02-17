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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        typedef Base::SynetConvolution16bNhwcSpecV2::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV2::LastConvPtr LastConvPtr;

        //-------------------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV2(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const float* src = (float*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD);
                    if (tailC)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        static void Reorder16bNhwcSpecV2(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            assert(a.microC == DF);
            const uint16_t* src = (uint16_t*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW + a.padE, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = a.padV * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += a.padV * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1 + a.padV - p.padY) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (a.padH)
                {
                    for (size_t s = 0; s < a.padH; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += a.padH * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        Avx512bw::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Avx512bw::Copy(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
            }
            if (end)
            {
                for (size_t s = 0, n = a.padE; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
            else if (dyEnd != p.dstH)
            {
                for (size_t s = 0, n = a.padH; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void Convolution16bNhwcSpecV2Body32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS * 2, dW = (int)a.microD * 32, strideW = (int)a.microD * 4, strideB = dB * 4;
            const uint16_t* weight1 = weight0 + 32;
            const uint16_t* src1 = src0 + 16 * dS;
            float* buf1 = buf0 + 16 * dB;

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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_stream_loadd(4, src0 + o, strideS);
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(5, src1 + o, strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(3, 5, 7);
                weight1 += dW;
            }
            _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);

            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf0 + F, strideB);

            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf1 + 0, strideB);

            _tile_dpbf16ps(3, 5, 7);
            _tile_stored(3, buf1 + F, strideB);
        }

        static void Convolution16bNhwcSpecV2Body32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS * 2, dW = (int)a.microD * 32, strideW = (int)a.microD * 4, strideB = dB * 4;
            const uint16_t* src1 = src0 + 16 * dS;
            float* buf1 = buf0 + 16 * dB;

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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_loadd(4, src0 + o, strideS);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(6, weight0, strideW);
                _tile_loadd(5, src1 + o, strideS);
                _tile_dpbf16ps(0, 4, 6);
                o = offs[i + 1];
                _tile_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
            }
            _tile_stream_loadd(6, weight0, strideW);
            _tile_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);

            _tile_dpbf16ps(2, 5, 6);
            _tile_stored(2, buf1 + 0, strideB);
        }

        static void Convolution16bNhwcSpecV2Body16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS * 2, dW = (int)a.microD * 32, strideW = (int)a.microD * 4, strideB = dB * 4;
            const uint16_t* weight1 = weight0 + 32;

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

            int n1 = (int)nK - 1;
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(1, 4, 7);
                weight1 += dW;
            }
            _tile_stream_loadd(4, src0 + offs[n1], strideS);
            _tile_loadd(7, weight1, strideW);

            _tile_dpbf16ps(0, 4, 6);
            _tile_stored(0, buf0 + 0, strideB);

            _tile_dpbf16ps(1, 4, 7);
            _tile_stored(1, buf0 + F, strideB);
        }

        static void Convolution16bNhwcSpecV2Body16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* buf0)
        {
            int dB = (int)a.macroD, dS = (int)a.microC, strideS = dS * 2, dW = (int)a.microD * 32, strideW = (int)a.microD * 4, strideB = dB * 4;

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
            }

            int n = (int)nK;
            for (int i = 0; i < n; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
            }

            _tile_stored(0, buf0 + 0, strideB);
        }

        typedef void (*Convolution16bNhwcSpecV2BodyPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, int zero, const uint16_t* weight0, float* buf0);

        static void Convolution16bNhwcSpecV2_Body(const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstS, size_t nK, int zero, const uint16_t* weight, float* buf)
        {
            size_t n1 = dstS, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcSpecV2BodyPtr body_2 = Convolution16bNhwcSpecV2Body32x32;
            Convolution16bNhwcSpecV2BodyPtr tail_2 = m > 16 ? Convolution16bNhwcSpecV2Body32x32 : Convolution16bNhwcSpecV2Body16x32;
            Convolution16bNhwcSpecV2BodyPtr body_1 = Convolution16bNhwcSpecV2Body32x16;
            Convolution16bNhwcSpecV2BodyPtr tail_1 = m > 16 ? Convolution16bNhwcSpecV2Body32x16 : Convolution16bNhwcSpecV2Body16x16;

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        body_2(src + i * dS, p, a, offs, nK, zero, weight, buf + i * dD);
                    if (m)
                        tail_2(src + i * dS, p, a, offs, nK, zero, weight, buf + i * dD);
                }
                else
                {
                    for (; i < nn; i += n)
                        body_1(src + i * dS, p, a, offs, nK, zero, weight, buf + i * dD);
                    if (m)
                        tail_1(src + i * dS, p, a, offs, nK, zero, weight, buf + i * dD);
                }
                weight += dW;
                buf += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------


        template<Term16bType term, SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply2x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0), bias[0]), params, 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + F), bias[1]), params, 1);
            if (term == Term16bLast16b)
            {
                _mm512_mask_storeu_epi16((uint16_t*)ptr, tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                if (flush == 1)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)ptr);
            }
            else
            {
                _mm512_storeu_ps((float*)ptr, f0);
                if (flush == 1)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)ptr + 0);
                _mm512_mask_storeu_ps((float*)(ptr + A), (__mmask16)tail, f1);
                if (flush == 1)
                    _mm_prefetch((const char*)(ptr + A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)ptr + A);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply1x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf), bias[0]), params, 0);
            if (term == Term16bLast16b)
            {
                _mm256_mask_storeu_epi16((uint16_t*)ptr, (__mmask16)tail, (__m256i)_mm512_cvtneps_pbh(f0));
                if (flush == 1)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)ptr);
            }
            else
            {
                _mm512_mask_storeu_ps((float*)ptr, (__mmask16)tail, f0);
                if (flush == 1)
                    _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)ptr);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int flush> static SIMD_INLINE void ApplyMx1(
            uint8_t *& ptr, int dP, float* buf, const __m512* bias, const __m512* params, const int* mask, __mmask32 tail = __mmask32(-1))
        {
            uint32_t msk = mask[0];
            switch (M)
            {
            case 1: Apply1x1<term, type, flush>(ptr, buf, bias, params, tail & msk); break;
            case 2: Apply2x1<term, type, flush>(ptr, buf, bias, params, tail & msk); break;
            }
            ptr += dP & msk;
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int N, int flush> static SIMD_INLINE void ApplyMxN(
            uint8_t*& ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, const int* mask, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 0 * dB, bias, params, mask + 0, tail);
            if (N > 1) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 1 * dB, bias, params, mask + 1, tail);
            if (N > 2) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 2 * dB, bias, params, mask + 2, tail);
            if (N > 3) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 3 * dB, bias, params, mask + 3, tail);
            if (N > 4) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 4 * dB, bias, params, mask + 4, tail);
            if (N > 5) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 5 * dB, bias, params, mask + 5, tail);
            if (N > 6) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 6 * dB, bias, params, mask + 6, tail);
            if (N > 7) ApplyMx1<term, type, M, flush>(ptr, dP, buf + 7 * dB, bias, params, mask + 7, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M, int apply, int flush> void Convolution16bNhwcSpecV2_1x32x32(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, 
            const __m512* bias, const __m512* params, float* buf2, const int* mask, uint8_t * &dst, __mmask32 tail)
        {
            int dD = int(p.dstC * a.elem), dB = (int)a.macroD, dS = (int)a.microC, strideS = dS * 2, dW = (int)a.microD * 32, strideW = (int)a.microD * 4, strideB = dB * 4;
            const uint16_t* weight1 = weight0 + 32;
            const uint16_t* src1 = src0 + 16 * dS;
            float* buf0 = buf2 - 32 * dB;
            float* buf3 = buf2 + 16 * dB;

            if (zero)
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
                if (M > 0) _tile_zero(2);
                if (M > 1) _tile_zero(3);
            }
            else
            {
                if (M > 0) _tile_stream_loadd(0, buf2 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf2 + F, strideB);
                if (M > 0) _tile_stream_loadd(2, buf3 + 0, strideB);
                if (M > 1) _tile_stream_loadd(3, buf3 + F, strideB);
            }


            int n1 = (int)nK - 1, i = 0, o = offs[0], na = apply ? (8 / apply - 1) : 0, ds = 0;
            _tile_stream_loadd(4, src0 + o, strideS);
            if (M > 0) _tile_loadd(6, weight0, strideW);
            for (; i < na; ++i)
            {
                if (M > 1) _tile_loadd(7, weight1, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
                _tile_stream_loadd(5, src1 + o, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(3, 5, 7);
                ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
                weight1 += dW;
            }
            for (; i < n1; ++i)
            {
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                if (M > 1) _tile_loadd(7, weight1, strideW);
                _tile_stream_loadd(5, src1 + o, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
                if (M > 0) _tile_loadd(6, weight0, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
                weight1 += dW;
            }
            if (M > 1) _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            if (M > 0) _tile_dpbf16ps(0, 4, 6);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);

            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);

            if (M > 0) _tile_dpbf16ps(2, 5, 6);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
            if (M > 0) _tile_stored(2, buf3 + 0, strideB);

            if (M > 1) _tile_dpbf16ps(3, 5, 7);
            ApplyMxN<term, type, M, apply, flush>(dst, dD, buf0 + ds * dB, dB, bias, params, mask + ds, tail), ds += apply;
            if (M > 1) _tile_stored(3, buf3 + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int apply, int flush> void Convolution16bNhwcSpecV2_Nx32x32M(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, const int* offs, size_t nK, int zero, const uint16_t* weight0,
            const float* bias, const float* params, __m512* _params, float* buf, const int* mask, uint8_t* dst, __mmask32 tail)
        {
            int dB = (int)a.macroD, dD = int(p.dstC * a.elem), dS = (int)a.microC;

            __m512 _bias[2];
            if (M > 0) _bias[0] = _mm512_loadu_ps(bias + 0 * F);
            if (M > 1) _bias[1] = _mm512_loadu_ps(bias + 1 * F);
            if (type == SimdConvolutionActivationPrelu)
            {
                if (M > 0) _params[0] = _mm512_loadu_ps(params + 0 * F);
                if (M > 1) _params[1] = _mm512_loadu_ps(params + 1 * F);
            }

            size_t cds = 0, pds = 0;
            Convolution16bNhwcSpecV2_1x32x32<term, type, M, 0, flush>(src0, p, a, offs, nK, zero, weight0, _bias, _params, buf, mask, dst, tail), cds += 32;
            for (; cds < dstS; pds += 32, cds += 32)
            {
            //    if (cds + 16 >= dstS)
            //    {
            //        cds = Simd::Min(dstS - 16, cds);
            //        Convolution16bNhwcGemm_1x16x32<term, type, M, apply, flush>(src0 + cds * dS, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst + pds * dD, tailD);
            //    }
            //    else
                {
                    Convolution16bNhwcSpecV2_1x32x32<term, type, M, apply, flush>(src0 + cds * dS, p, a, offs, nK, zero, weight0, _bias, _params, buf + cds * dB, mask + pds, dst, tail);
                }
            }
            //std::cout << " dstS " << dstS << " cds " << cds << " pds " << pds << std::endl << std::flush;
            size_t dstS8 = dstS & (~7);
            //for (; pds < dstS8; pds += 8)
            //{
            //    ApplyMxN<term, type, M, 8, flush>(dst, dD, buf + pds * dB, dB, _bias, _params, mask + pds, tail);
            //}
            for (; pds < dstS; ++pds)
            {
                ApplyMxN<term, type, M, 1, flush>(dst, dD, buf + pds * dB, dB, _bias, _params, mask + pds, tail);
            }
        }

        //-------------------------------------------------------------------------------------------------

        typedef void (*Convolution16bNhwcSpecV2TailPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, const int* offs, size_t nK, int zero, 
            const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, const int* mask, uint8_t* dst, __mmask32 tail);


        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> void Convolution16bNhwcSpecV2_Last(
            const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstS, size_t nK, int zero,
            const uint16_t* weight, float* buf, const float* bias, const float* params, const int* mask, uint8_t* dst)
        {
            size_t n = 256, n1 = dstS, nn = AlignLoAny(n1, n), dW = a.K * a.microD;
            size_t dB = a.macroD, dD = p.dstC * a.elem, dS = a.microC;

            size_t dstC32 = AlignLo(dstC, 32), dstCt = dstC - dstC32;
            __mmask32 tailD = term == Term16bLast16b ? TailMask32(dstCt) : (__mmask32)TailMask16(dstCt - AlignLo(dstCt - 1, 16));
            Convolution16bNhwcSpecV2TailPtr mainConv = Convolution16bNhwcSpecV2_Nx32x32M<term, type, 2, apply, flush>;
            Convolution16bNhwcSpecV2TailPtr tailConv = dstCt > 16 ? Convolution16bNhwcSpecV2_Nx32x32M<term, type, 2, apply, flush> : 
                Convolution16bNhwcSpecV2_Nx32x32M<term, type, 1, apply, flush>;

            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            //std::cout << " Last: &dst " << (size_t)dst << std::endl << std::flush;

            SetTileConfFull();
            for (size_t i = 0; i < n1;)
            {
                size_t dn = (n1 - i >= n + 32 ? n : n1 - i);
                const uint16_t* s = src + i * dS;
                const uint16_t* w = weight;
                float* b = buf + i * dB;
                uint8_t* d = dst + i * dD;
                size_t dc = 0;
                for (; dc < dstC32; dc += DF, w += dW)
                    mainConv(s, p, a, dn, offs, nK, zero, w, bias + dc, params + dc, _params, b + dc, mask + i, d + dc * a.elem, __mmask32(-1));
                if (dc < dstC)
                    tailConv(s, p, a, dn, offs, nK, zero, w, bias + dc, params + dc, _params, b + dc, mask + i, d + dc * a.elem, tailD);
                i += dn;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush> SIMD_INLINE void SetLastConv(const ConvParam& p, size_t nK, LastConvPtr& lastConv)
        {
            //if (a.bufK >= 256)
                lastConv = Convolution16bNhwcSpecV2_Last<term, type, 1, flush>;
            //else if (a.bufK >= 128)
            //    convolution = Convolution16bNhwcGemm_Macro32x32<term, type, 2, flush>;
            //else
            //    convolution = NULL;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetLastConv(const ConvParam& p, size_t nK, LastConvPtr& lastConv)
        {
            if (p.dstT == SimdTensorData16b)
                SetLastConv<Term16bLast16b, type, 0>(p, nK, lastConv);
            else
                SetLastConv<Term16bLast32f, type, 0>(p, nK, lastConv);
        }

        SynetConvolution16bNhwcSpecV2::SynetConvolution16bNhwcSpecV2(const ConvParam & p)
            : Base::SynetConvolution16bNhwcSpecV2(p)
        {
            int L1 = int(Base::AlgCacheL1() * (p.IsKernel(5) ? 1.05 : 1.00)), L2 = int(Base::AlgCacheL2() * 0.5);
            SetAlgParam(F, F * 2, F * 2, F * 2, L1, L2, Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV2;
            else
                _preprocess = Convert16bNhwcSpecV2;
            _bodyConv = Convolution16bNhwcSpecV2_Body;
            size_t nK = _nK[_nK.size - 1];
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetLastConv<SimdConvolutionActivationRestrictRange>(p, nK, _lastConv); break;
            case SimdConvolutionActivationRelu: SetLastConv<SimdConvolutionActivationRestrictRange>(p, nK, _lastConv); break;
            case SimdConvolutionActivationLeakyRelu: SetLastConv<SimdConvolutionActivationPrelu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationRestrictRange: SetLastConv<SimdConvolutionActivationRestrictRange>(p, nK, _lastConv); break;
            case SimdConvolutionActivationPrelu: SetLastConv<SimdConvolutionActivationPrelu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationElu: SetLastConv<SimdConvolutionActivationElu>(p, nK, _lastConv); break;
            case SimdConvolutionActivationHswish: SetLastConv<SimdConvolutionActivationHswish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationMish: SetLastConv<SimdConvolutionActivationMish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationHardSigmoid: SetLastConv<SimdConvolutionActivationHardSigmoid>(p, nK, _lastConv); break;
            case SimdConvolutionActivationSwish: SetLastConv<SimdConvolutionActivationSwish>(p, nK, _lastConv); break;
            case SimdConvolutionActivationGelu: SetLastConv<SimdConvolutionActivationGelu>(p, nK, _lastConv); break;
            default: assert(0);
            }
        }
    }
#endif
}

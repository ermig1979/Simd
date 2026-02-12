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
#include "Simd/SimdSynetConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
	{
		typedef Base::SynetConvolution16bNhwcGemmV1::AlgParam AlgParam;
		typedef Base::SynetConvolution16bNhwcGemmV1::ConvolutionPtr Convolution;

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int start> static SIMD_INLINE void Apply2x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[start + 0]), params, start + 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 1) * F), bias[start + 1]), params, start + 1);
            if (term == Term16bLast16b)
            {
                _mm512_mask_storeu_epi16((uint16_t*)(ptr + start * DF), tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + start * DF), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + start * DF));
            }
            else
            {
                _mm512_storeu_ps((float*)(ptr + (start + 0) * A), f0);
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 0) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 0) * A));
                _mm512_mask_storeu_ps((float*)(ptr + (start + 1) * A), (__mmask16)tail, f1);
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 1) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 1) * A));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int start> static SIMD_INLINE void Apply1x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[start + 0]), params + start, 0);
            if (term == Term16bLast16b)
            {
                _mm256_mask_storeu_epi16((uint16_t*)(ptr + start * DF), (__mmask16)tail, (__m256i)_mm512_cvtneps_pbh(f0));
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr + start * DF), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + start * DF));
            }
            else
            {
                _mm512_mask_storeu_ps((float*)(ptr + (start + 0) * A), (__mmask16)tail, f0);
                if (flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 0) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 0) * A));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int flush> static SIMD_INLINE void ApplyMx1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            switch (M)
            {
            case 1: Apply1x1<term, type, flush, 0>(ptr, buf, bias, params, tail); break;
            case 2: Apply2x1<term, type, flush, 0>(ptr, buf, bias, params, tail); break;
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int N, int flush> static SIMD_INLINE void ApplyMxN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) ApplyMx1<term, type, M, flush>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) ApplyMx1<term, type, M, flush>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) ApplyMx1<term, type, M, flush>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) ApplyMx1<term, type, M, flush>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) ApplyMx1<term, type, M, flush>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) ApplyMx1<term, type, M, flush>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) ApplyMx1<term, type, M, flush>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) ApplyMx1<term, type, M, flush>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> SIMD_INLINE void Convolution16bNhwcGemm_32x32x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            int dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst0)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.miniD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + 2 * F;

            int srcC32 = (int)a.bufK - 32, applyC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            if (N > 0) _tile_zero(0);
            if (N > 0) _tile_zero(1);
            if (N > 1) _tile_zero(2);
            if (N > 1) _tile_zero(3);

            _tile_stream_loadd(6, weight0 + sc * dW, strideW);
            if (N > 0) _tile_loadd(4, src0, strideS);
            for (; sc < applyC32;)
            {
                _tile_stream_loadd(7, weight1 + sc * dW, strideW);
                if (N > 0) _tile_dpbf16ps(0, 4, 6);
                if (N > 0) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 1) _tile_loadd(5, src1, strideS);
                if (N > 1) src1 += stepS;
                if (N > 0) _tile_dpbf16ps(1, 4, 7);
                if (N > 0) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 0) src0 += stepS;
                if (N > 0) _tile_loadd(4, src0, strideS);
                if (N > 1) _tile_dpbf16ps(2, 5, 6);
                if (N > 1) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                sc += 32;
                _tile_stream_loadd(6, weight0 + sc * dW, strideW);
                if (N > 1) _tile_dpbf16ps(3, 5, 7);
                if (N > 1) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            }
            for (; sc < srcC32;)
            {
                _tile_stream_loadd(7, weight1 + sc * dW, strideW);
                if (N > 0) _tile_dpbf16ps(0, 4, 6);
                if (N > 1) _tile_loadd(5, src1, strideS);
                if (N > 1) src1 += stepS;
                if (N > 0) _tile_dpbf16ps(1, 4, 7);
                if (N > 0) src0 += stepS;
                if (N > 0) _tile_loadd(4, src0, strideS);
                if (N > 1) _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_stream_loadd(6, weight0 + sc * dW, strideW);
                if (N > 1) _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stream_loadd(7, weight1 + sc * dW, strideW);
            if (N > 0) _tile_dpbf16ps(0, 4, 6);
            if (N > 0) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            if (N > 1) _tile_loadd(5, src1, strideS);
            if (N > 0) _tile_stored(0, buf1 + 0, strideB);
            if (N > 0) _tile_dpbf16ps(1, 4, 7);
            if (N > 0) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            if (N > 0) _tile_stored(1, buf1 + F, strideB);
            if (N > 1) _tile_dpbf16ps(2, 5, 6);
            if (N > 1) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            if (N > 1) _tile_stored(2, buf1 + 16 * dB + 0, strideB);
            if (N > 1) _tile_dpbf16ps(3, 5, 7);
            if (N > 1) ApplyMxN<term, type, 2, apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            if (N > 1) _tile_stored(3, buf1 + 16 * dB + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> SIMD_INLINE void Convolution16bNhwcGemm_32x16x1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            int dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst0)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.miniD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;
            const uint16_t* src1 = src0 + 16 * dS;

            int srcC32 = (int)a.bufK - 32, applyC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            if (N > 0) _tile_zero(0);
            if (N > 1) _tile_zero(2);

            _tile_stream_loadd(6, weight0 + sc * dW, strideW);
            if (N > 0) _tile_loadd(4, src0, strideS);
            for (; sc < applyC32;)
            {
                if (N > 0) _tile_dpbf16ps(0, 4, 6);
                if (N > 0) ApplyMxN<term, type, 2, 2 * apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += 2 *apply;
                if (N > 1) _tile_loadd(5, src1, strideS);
                if (N > 1) src1 += stepS;
                if (N > 0) src0 += stepS;
                if (N > 0) _tile_loadd(4, src0, strideS);
                if (N > 1) _tile_dpbf16ps(2, 5, 6);
                if (N > 1) ApplyMxN<term, type, 2, 2 * apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += 2 * apply;
                sc += 32;
                _tile_stream_loadd(6, weight0 + sc * dW, strideW);
            }
            for (; sc < srcC32;)
            {
                if (N > 0) _tile_dpbf16ps(0, 4, 6);
                if (N > 1) _tile_loadd(5, src1, strideS);
                if (N > 1) src1 += stepS;
                if (N > 0) src0 += stepS;
                if (N > 0) _tile_loadd(4, src0, strideS);
                if (N > 1) _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                _tile_stream_loadd(6, weight0 + sc * dW, strideW);
            }
            if (N > 0) _tile_dpbf16ps(0, 4, 6);
            if (N > 0) ApplyMxN<term, type, 2, 2 * apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += 2 * apply;
            if (N > 1) _tile_loadd(5, src1, strideS);
            if (N > 0) _tile_stored(0, buf1 + 0, strideB);
            if (N > 1) _tile_dpbf16ps(2, 5, 6);
            if (N > 1) ApplyMxN<term, type, 2, 2 * apply, flush>(dst0 + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += 2 * apply;
            if (N > 1) _tile_stored(2, buf1 + 16 * dB + 0, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> void Convolution16bNhwcGemm_Nx32x32xM(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            int dstS, size_t dstC, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dW = (int)(a.miniD * a.bufK), dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 32 * dB;
            int offs0 = N == 1 ? dstS - 1 * F : 0 * F, offs1 = N == 2 ? dstS - 1 * F : 1 * F;

            __m512 _bias[2];
            Convolution16bNhwcGemm_32x32x1<term, type, N, 0, flush>(src0, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst), weight0 += dW;
            size_t dc = 32;
            for (; dc < dstC; dc += 32, weight0 += dW, bias += 32, params += 32, dst += 32 * a.elem)
            {
                Swap(buf0, buf1);
                _bias[0] = _mm512_loadu_ps(bias + 0 * F);
                _bias[1] = _mm512_loadu_ps(bias + 1 * F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + 0 * F);
                    _params[1] = _mm512_loadu_ps(params + 1 * F);
                }
                if (dc + 16 >= dstC)
                    Convolution16bNhwcGemm_32x16x1<term, type, N, apply, flush>(src0, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst);
                else
                    Convolution16bNhwcGemm_32x32x1<term, type, N, apply, flush>(src0, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst);
            }
            if (dc >= dstC + 16)
            {
                _bias[0] = _mm512_loadu_ps(bias);
                if (type == SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params);
                if (N > 0) ApplyMxN<term, type, 1, 8, flush>(dst + (offs0 + 0) * dD, dD, buf1 + (offs0 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 0) ApplyMxN<term, type, 1, 8, flush>(dst + (offs0 + 8) * dD, dD, buf1 + (offs0 + 8) * dB, dB, _bias, _params, tailD);
                if (N > 1) ApplyMxN<term, type, 1, 8, flush>(dst + (offs1 + 0) * dD, dD, buf1 + (offs1 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 1) ApplyMxN<term, type, 1, 8, flush>(dst + (offs1 + 8) * dD, dD, buf1 + (offs1 + 8) * dB, dB, _bias, _params, tailD);
            }
            else
            {
                _bias[0] = _mm512_loadu_ps(bias + 0 * F);
                _bias[1] = _mm512_loadu_ps(bias + 1 * F);
                if (type == SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + 0 * F);
                    _params[1] = _mm512_loadu_ps(params + 1 * F);
                }
                if (N > 0) ApplyMxN<term, type, 2, 8, flush>(dst + (offs0 + 0) * dD, dD, buf1 + (offs0 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 0) ApplyMxN<term, type, 2, 8, flush>(dst + (offs0 + 8) * dD, dD, buf1 + (offs0 + 8) * dB, dB, _bias, _params, tailD);
                if (N > 1) ApplyMxN<term, type, 2, 8, flush>(dst + (offs1 + 0) * dD, dD, buf1 + (offs1 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 1) ApplyMxN<term, type, 2, 8, flush>(dst + (offs1 + 8) * dD, dD, buf1 + (offs1 + 8) * dB, dB, _bias, _params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> void Convolution16bNhwcGemm_Nx32x64(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            int dstS, size_t dstC, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dW = (int)(a.miniD * a.bufK), dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 32 * dB;
            int offs0 = N == 1 ? dstS - 1 * F : 0 * F, offs1 = N == 2 ? dstS - 1 * F : 1 * F;
            const uint16_t* weight1 = weight0 + dW;

            __m512 _bias[4];
            _bias[0] = _mm512_loadu_ps(bias + 0 * F);
            _bias[1] = _mm512_loadu_ps(bias + 1 * F);
            _bias[2] = _mm512_loadu_ps(bias + 2 * F);
            _bias[3] = _mm512_loadu_ps(bias + 3 * F);
            if (type == SimdConvolutionActivationPrelu)
            {
                _params[0] = _mm512_loadu_ps(params + 0 * F);
                _params[1] = _mm512_loadu_ps(params + 1 * F);
                _params[2] = _mm512_loadu_ps(params + 2 * F);
                _params[3] = _mm512_loadu_ps(params + 3 * F);
            }
            size_t cds = 0, pds = 0;
            Convolution16bNhwcGemm_32x32x1<term, type, N, 0, flush>(src0, p, a, dstS, weight0, _bias + 2, _params + 2, buf0, buf1, dst + 32 * a.elem);
            Convolution16bNhwcGemm_32x32x1<term, type, N, apply, flush>(src0, p, a, dstS, weight1, _bias + 0, _params + 0, buf1, buf0, dst + 0 * a.elem), cds += 32;
            for (; cds < dstS; pds = cds, cds += 32)
            {
                if (cds + 16 >= dstS)
                {
                    //cds = Simd::Min(dstS - 16, cds);
                    //Convolution16bNhwcGemm_1x16x32<term, type, M, apply, flush>(src0 + cds * dS, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst + pds * dD, tailD);
                }
                else
                {
                    cds = Simd::Min((size_t)dstS - 32, cds);
                    Convolution16bNhwcGemm_32x32x1<term, type, N, apply, flush>(src0 + cds * dS, p, a, dstS, weight0, _bias + 2, _params + 2, buf0, buf1, dst + pds * dD + 32 * a.elem);
                    Convolution16bNhwcGemm_32x32x1<term, type, N, apply, flush>(src0 + cds * dS, p, a, dstS, weight1, _bias + 0, _params + 0, buf1, buf0, dst + cds * dD + 0 * a.elem);
                }
            }
            uint8_t* dst1 = dst + pds * dD + 32 * a.elem;
            dstS -= pds;
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                {
                    ApplyMxN<term, type, 2, 8, flush>(dst1 + ds * dD, dD, buf0 + ds * dB, dB, _bias + 2, _params + 2, tailD);
                }
                for (; ds < dstS; ++ds)
                {
                    ApplyMxN<term, type, 2, 1, flush>(dst1 + ds * dD, dD, buf0 + ds * dB, dB, _bias + 2, _params + 2, tailD);
                }
            }
        }

        //------------------------------------------------------------------------------------------------

        typedef void (*Convolution16bNhwcGemm_Nx32x32xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            int dstS, size_t dstC, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD);

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> void Convolution16bNhwcGemm_MacroNx32(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t dD = p.dstC * a.elem, dS = a.bufK, dStep = a.miniD * 8, dW = a.bufK * dStep;
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            assert(n1 >= 16);

            Convolution16bNhwcGemm_Nx32x32xM_Ptr nConv = Convolution16bNhwcGemm_Nx32x32xM<term, type, 2, apply, flush>;
            Convolution16bNhwcGemm_Nx32x32xM_Ptr mConv = m > 16 ? Convolution16bNhwcGemm_Nx32x32xM<term, type, 2, apply, flush> :
                Convolution16bNhwcGemm_Nx32x32xM<term, type, 1, apply, flush>;

            m = AlignHi(m, 16);
            nn = n1 - m;

            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC;)
            {
                size_t dC = (dstC - dc >= dStep + 32 ? dStep : dstC - dc);
                __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC - AlignLo(dC - 1, 32)) : (__mmask32)TailMask16(dC - AlignLo(dC - 1, 16));
                const uint16_t* s = src;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nn; i += n)
                    nConv(s + i * dS, p, a, (int)n, dC, weight, bias, params, _params, buf, d + i * dD, tailD);
                if (m)
                    mConv(s + nn * dS, p, a, (int)m, dC, weight, bias, params, _params, buf, d + nn * dD, tailD);
                weight += a.bufK * dC;
                bias += dC;
                params += dC;
                dc += dC;
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> void Convolution16bNhwcGemm_Macro64x32(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 256, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), dW = a.bufK * a.miniD * 2;
            size_t dD = p.dstC * a.elem, dS = a.bufK;

            size_t dstC64 = AlignLo(dstC, 64), dstCt = dstC - dstC64;
            __mmask32 tailD = __mmask32(-1);// term == Term16bLast16b ? TailMask32(dstCt) : (__mmask32)TailMask16(dstCt - AlignLo(dstCt - 1, 16));
            Convolution16bNhwcGemm_Nx32x32xM_Ptr mainConv = Convolution16bNhwcGemm_Nx32x64<term, type, 2, apply, flush>;
            //Convolution16bNhwcGemm_Nx32x32_Ptr tailConv = dstCt > 16 ? Convolution16bNhwcGemm_Nx32x32M<term, type, 2, apply, flush> : Convolution16bNhwcGemm_Nx32x32M<term, type, 1, apply, flush>;

            __m512 _params[4];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[2] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
            {
                _params[1] = _mm512_set1_ps(params[1]);
                _params[3] = _mm512_set1_ps(params[1]);
            }

            SetTileConfFull();
            for (size_t i = 0; i < n1;)
            {
                size_t dn = (i == nn ? n1 - i : n);
                const uint16_t* s = src + i * dS;
                const uint16_t* w = weight;
                uint8_t* d = dst + i * dD;
                size_t dc = 0;
                for (; dc < dstC64; dc += QF, w += dW)
                    mainConv(s, p, a, dn, 64, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, __mmask32(-1));
                //if (dc < dstC)
                //    tailConv(s, p, a, dn, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, tailD);
                i += dn;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush> SIMD_INLINE void SetMacro32x32i(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_MacroNx32<term, type, 1, flush>;
            else if (a.bufK >= 128)
                convolution = Convolution16bNhwcGemm_MacroNx32<term, type, 2, flush>;
            else
                convolution = NULL;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetMacro32x32i(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (p.dstT == SimdTensorData16b)
                SetMacro32x32i<Term16bLast16b, type, 0>(p, a, convolution);
            else
                SetMacro32x32i<Term16bLast32f, type, 0>(p, a, convolution);
        }

        void SynetConvolution16bNhwcGemmV1::SetMacro32x32i()
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: AmxBf16::SetMacro32x32i<SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRelu: AmxBf16::SetMacro32x32i<SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: AmxBf16::SetMacro32x32i<SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRestrictRange: AmxBf16::SetMacro32x32i<SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
            case SimdConvolutionActivationPrelu: AmxBf16::SetMacro32x32i<SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationElu: AmxBf16::SetMacro32x32i<SimdConvolutionActivationElu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHswish: AmxBf16::SetMacro32x32i<SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationMish: AmxBf16::SetMacro32x32i<SimdConvolutionActivationMish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: AmxBf16::SetMacro32x32i<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
            case SimdConvolutionActivationSwish: AmxBf16::SetMacro32x32i<SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationGelu: AmxBf16::SetMacro32x32i<SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
            default: assert(0);
            }
        }
    }
#endif
}

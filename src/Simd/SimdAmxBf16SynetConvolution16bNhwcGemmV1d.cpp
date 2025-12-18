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
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[start + 0]), params, start + 0);
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
                _mm512_mask_storeu_ps((float*)(ptr + (start + 0) * A), tail, f0);
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
            case 3: Apply2x1<term, type, flush, 0>(ptr, buf, bias, params); Apply1x1<term, type, flush, 2>(ptr, buf, bias, params, tail); break;
            case 4: Apply2x1<term, type, flush, 0>(ptr, buf, bias, params); Apply2x1<term, type, flush, 2>(ptr, buf, bias, params, tail); break;
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int N, int flush> static SIMD_INLINE void ApplyMxN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) ApplyMx1<term, type, M, flush>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) ApplyMx1<term, type, M, flush>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) ApplyMx1<term, type, M, flush>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) ApplyMx1<term, type, M, flush>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int M, int apply, int flush, int order> SIMD_INLINE void Convolution16bNhwcGemm_1x16xM(const uint16_t* src0, 
            const ConvParam& p, const AlgParam& a, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            int srcC64 = (int)(a.bufK - 32)&(~63), applyC64 = apply ? (8 * 32 / apply - 64) : 0, sc = 0, ds = 0;

            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);
            if (M > 2) _tile_zero(2);
            if (M > 3) _tile_zero(3);

            if (M > 0) _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
            for (; sc < applyC64; sc += 64)
            {
                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                weight0 += 32 * dW;
                if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(4, src0, strideS);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                weight0 += 32 * dW;
                if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
            }
            for (; sc < srcC64; sc += 64)
            {
                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                weight0 += 32 * dW;
                if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 6, 5);
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(4, src0, strideS);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                weight0 += 32 * dW;
                if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
            }
            if (a.bufK - srcC64 == 64)
            {
                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                weight0 += 32 * dW;
                if (M > 0) _tile_loadd(5, weight0 + 0 * DF, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 0) _tile_stored(0, buf1 + 0 * F, strideB);
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                if (M > 1) _tile_stored(1, buf1 + 1 * F, strideB);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 2) _tile_stored(2, buf1 + 2 * F, strideB);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
                if (M > 3) _tile_stored(3, buf1 + 3 * F, strideB);
            }
            else
            {
                if (M > 1) _tile_loadd(7, weight0 + 1 * DF, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 0) _tile_stored(0, buf1 + 0 * F, strideB);
                if (M > 2) _tile_loadd(5, weight0 + 2 * DF, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 1) _tile_stored(1, buf1 + 1 * F, strideB);
                if (M > 3) _tile_loadd(7, weight0 + 3 * DF, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 2) _tile_stored(2, buf1 + 2 * F, strideB);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);
                ApplyMxN<term, type, M, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                if (M > 3) _tile_stored(3, buf1 + 3 * F, strideB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int M, int apply, int flush, int order> static void Convolution16bNhwcGemm_Nx16xM(const uint16_t* src0, 
            const ConvParam& p, const AlgParam& a, size_t dstS, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            __m512 _bias[4];
            if (M > 0) _bias[0] = _mm512_loadu_ps(bias + 0 * F);
            if (M > 1) _bias[1] = _mm512_loadu_ps(bias + 1 * F);
            if (M > 2) _bias[2] = _mm512_loadu_ps(bias + 2 * F);
            if (M > 3) _bias[3] = _mm512_loadu_ps(bias + 3 * F);
            if (type == SimdConvolutionActivationPrelu)
            {
                if (M > 0) _params[0] = _mm512_loadu_ps(params + 0 * F);
                if (M > 1) _params[1] = _mm512_loadu_ps(params + 1 * F);
                if (M > 2) _params[2] = _mm512_loadu_ps(params + 2 * F);
                if (M > 3) _params[3] = _mm512_loadu_ps(params + 3 * F);
            }
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dW = (int)a.microD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 16 * dB;
            size_t cds = 0, pds = 0;
            Convolution16bNhwcGemm_1x16xM<term, type, M, 0, 0, order>(src0, p, a, weight0, _bias, _params, buf0, buf1, dst, tailD), cds += 16;
            for (; cds < dstS; pds = cds, cds += 16)
            {
                cds = Simd::Min(dstS - 16, cds);
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_1x16xM<term, type, M, apply, flush, order>(src0 + cds * dS, p, a, weight0, _bias, _params, buf0, buf1, dst + pds * dD, tailD);
            }
            uint8_t* dst1 = dst + pds * dD;
            dstS -= pds;
            {
                size_t ds = 0, dstS4 = dstS & (~3);
                for (; ds < dstS4; ds += 4)
                    ApplyMxN<term, type, M, 4, flush>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, _bias, _params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, M, 1, flush>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, _bias, _params, tailD);
            }
        }

        typedef  void (*Convolution16bNhwcGemm_Nx16xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, 
            const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD);

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> Convolution16bNhwcGemm_Nx16xM_Ptr GetConvolution16bNhwcGemm_Nx16xM(size_t M)
        {
            switch (M)
            {
            case 1: return Convolution16bNhwcGemm_Nx16xM<term, type, 1, apply, flush, order>;
            case 2: return Convolution16bNhwcGemm_Nx16xM<term, type, 2, apply, flush, order>;
            case 3: return Convolution16bNhwcGemm_Nx16xM<term, type, 3, apply, flush, order>;
            case 4: return Convolution16bNhwcGemm_Nx16xM<term, type, 4, apply, flush, order>;
            default: return NULL;
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Macro16xM(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 256, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), dW = a.bufK * a.microD;
            if (n1 > nn && n1 - nn < 16)
                nn -= n;

            size_t dstC64 = AlignLo(dstC, 64), dstCt = dstC - dstC64;
            __mmask32 tailD = term == Term16bLast16b ? TailMask32(dstCt - AlignLo(dstCt, 32)) : (__mmask32)TailMask16(dstCt - AlignLo(dstCt, 16));
            Convolution16bNhwcGemm_Nx16xM_Ptr mainConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush, order>(4);
            Convolution16bNhwcGemm_Nx16xM_Ptr tailConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush, order>(DivHi(dstCt, 16));

            size_t dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _params[4];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            SetTileConfFull();
            for (size_t i = 0; i < n1;)
            {
                size_t dn = (i == nn ? n1 - i : n);
                const uint16_t* s = src + i * dS;
                const uint16_t* w = weight;
                uint8_t* d = dst + i * dD;
                size_t dc = 0;
                for (; dc < dstC64; dc += QF, w += dW)
                    mainConv(s, p, a, dn, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, __mmask32(-1));
                if (dc < dstC)
                    tailConv(s, p, a, dn, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, tailD);
                i += dn;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush, int order> SIMD_INLINE void SetMacro16x64(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_Macro16xM<term, type, 1, flush, order>;
            else if (a.bufK >= 128)
                convolution = Convolution16bNhwcGemm_Macro16xM<term, type, 2, flush, order>;
            else
                convolution = Convolution16bNhwcGemm_Macro16xM<term, type, 4, flush, order>;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetMacro16x64(const ConvParam& p, const AlgParam & a, Convolution& convolution)
        {
            if (p.dstT == SimdTensorData16b)
                SetMacro16x64<Term16bLast16b, type, 0, 0>(p, a, convolution);
            else
                SetMacro16x64<Term16bLast32f, type, 0, 0>(p, a, convolution);
        }

        void SynetConvolution16bNhwcGemmV1::SetMacro16x64d()
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetMacro16x64<SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRelu: SetMacro16x64<SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: SetMacro16x64<SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRestrictRange: SetMacro16x64<SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
            case SimdConvolutionActivationPrelu: SetMacro16x64<SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationElu: SetMacro16x64<SimdConvolutionActivationElu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHswish: SetMacro16x64<SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationMish: SetMacro16x64<SimdConvolutionActivationMish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: SetMacro16x64<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
            case SimdConvolutionActivationSwish: SetMacro16x64<SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationGelu: SetMacro16x64<SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
            default: assert(0);
            }
        }
    }
#endif
}

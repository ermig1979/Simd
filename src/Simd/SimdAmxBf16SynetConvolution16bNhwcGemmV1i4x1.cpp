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

        template<Term16bType term, SimdConvolutionActivationType type, int flush> static SIMD_INLINE void Apply1x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf), bias[0]), params, 0);
            if (term == Term16bLast16b)
            {
                _mm256_mask_storeu_epi16((uint16_t*)(ptr), tail, (__m256i)_mm512_cvtneps_pbh(f0));
                if(flush == 1)
                    _mm_prefetch((const char*)(ptr), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr));
            }
            else
            {
                _mm512_mask_storeu_ps((float*)(ptr), tail, f0);
                if (flush == 1)
                    _mm_prefetch((const char*)(ptr), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int flush> static SIMD_INLINE void Apply1xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            if (N > 0) Apply1x1<term, type, flush>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply1x1<term, type, flush>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply1x1<term, type, flush>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply1x1<term, type, flush>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply1x1<term, type, flush>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply1x1<term, type, flush>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply1x1<term, type, flush>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply1x1<term, type, flush>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> SIMD_INLINE void Convolution16bNhwcGemm_Nx16x1(const uint16_t* src0, 
            const ConvParam& p, const AlgParam& a, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.miniD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            int srcC64 = (int)(a.bufK - 32)&(~63), applyC64 = apply ? (16 * 32 / apply - 64) : 0, sc = 0, ds = 0;

            if (N > 0) _tile_zero(0);
            if (N > 1) _tile_zero(1);
            if (N > 2) _tile_zero(2);
            if (N > 3) _tile_zero(3);

            if (N > 0) _tile_stream_loadd(5, weight0, strideW);
            if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);

            for (; sc < applyC64; sc += 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                src0 += stepS;
                if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;

                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(5, weight0, strideW);
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                src0 += stepS;
                if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
            }
            for (; sc < srcC64; sc += 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                src0 += stepS;
                if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);

                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(5, weight0, strideW);
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                src0 += stepS;
                if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
            }
            if (a.bufK - srcC64 == 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                src0 += stepS;
                if (N > 0) _tile_loadd(4, src0 + 0 * DF, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;

                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 0) _tile_stored(0, buf1 + 0 * 16 * dB, strideB);
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 1) _tile_stored(1, buf1 + 1 * 16 * dB, strideB);
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 2) _tile_stored(2, buf1 + 2 * 16 * dB, strideB);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply;
                if (N > 3) _tile_stored(3, buf1 + 3 * 16 * dB, strideB);
            }
            else
            {
                if (N > 1) _tile_loadd(6, src0 + 1 * DF, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply * 2, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply * 2;
                if (N > 0) _tile_stored(0, buf1 + 0 * 16 * dB, strideB);
                if (N > 2) _tile_loadd(4, src0 + 2 * DF, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply * 2, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply * 2;
                if (N > 1) _tile_stored(1, buf1 + 1 * 16 * dB, strideB);
                if (N > 3) _tile_loadd(6, src0 + 3 * DF, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply * 2, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply * 2;
                if (N > 2) _tile_stored(2, buf1 + 2 * 16 * dB, strideB);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply * 2, flush>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params), ds += apply * 2;
                if (N > 3) _tile_stored(3, buf1 + 3 * 16 * dB, strideB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> static void Convolution16bNhwcGemm_Nx16xM(const uint16_t* src0, 
            const ConvParam& p, const AlgParam& a, size_t dstC, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask16 tailD)
        {
            __m512 _bias[1];
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dW = (int)a.miniD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + N * 16 * dB;
            //size_t cds = 0, pds = 0;
            //Convolution16bNhwcGemm_Nx16x1<term, type, N, 0, 0>(src0, p, a, weight0, _bias, _params, buf0, buf1, dst, tailD), cds += 16;
            //for (; cds < dstS; pds = cds, cds += 16, bias += 16, params += 16)
            //{
            //    _bias[0] = _mm512_loadu_ps(bias);
            //    if (type == SimdConvolutionActivationPrelu)
            //        _params[0] = _mm512_loadu_ps(params);
            //    cds = Simd::Min(dstS - 16, cds);
            //    Swap(buf0, buf1);
            //    Convolution16bNhwcGemm_Nx16x1<term, type, N, apply, flush>(src0 + cds * dS, p, a, weight0, _bias, _params, buf0, buf1, dst + pds * dD, tailD);
            //}
            //uint8_t* dst1 = dst + pds * dD;
            //dstC -= pds;
            //{
            //    _bias[0] = _mm512_loadu_ps(bias);
            //    if (type == SimdConvolutionActivationPrelu)
            //        _params[0] = _mm512_loadu_ps(params);
            //    size_t ds = 0, dstS8 = dstS & (~7);
            //    for (; ds < dstS8; ds += 8)
            //        Apply1xN<term, type, 8, flush>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, _bias, _params, tailD);
            //    for (; ds < dstS; ++ds)
            //        Apply1xN<term, type, 1, flush>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, _bias, _params, tailD);
            //}
        }

        //typedef  void (*Convolution16bNhwcGemm_Nx16xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t dstS, 
        //    const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask32 tailD);

        //template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> Convolution16bNhwcGemm_Nx16xM_Ptr GetConvolution16bNhwcGemm_Nx16xM(size_t M)
        //{
        //    switch (M)
        //    {
        //    case 1: return Convolution16bNhwcGemm_Nx16xM<term, type, 1, apply, flush>;
        //    case 2: return Convolution16bNhwcGemm_Nx16xM<term, type, 2, apply, flush>;
        //    case 3: return Convolution16bNhwcGemm_Nx16xM<term, type, 3, apply, flush>;
        //    case 4: return Convolution16bNhwcGemm_Nx16xM<term, type, 4, apply, flush>;
        //    default: return NULL;
        //    }
        //}

        //template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> void Convolution16bNhwcGemm_Macro16xM(const uint16_t* src, const ConvParam& p, const AlgParam& a,
        //    size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        //{
        //    size_t n = 256, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), dW = a.bufK * a.miniD;
        //    if (n1 > nn && n1 - nn < 16)
        //        nn -= n;

        //    size_t dstC64 = AlignLo(dstC, 64), dstCt = dstC - dstC64;
        //    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dstCt - AlignLo(dstCt - 1, 32)) : (__mmask32)TailMask16(dstCt - AlignLo(dstCt - 1, 16));
        //    Convolution16bNhwcGemm_Nx16xM_Ptr mainConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush>(4);
        //    Convolution16bNhwcGemm_Nx16xM_Ptr tailConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush>(DivHi(dstCt, 16));

        //    size_t dD = p.dstC * a.elem, dS = a.bufK;

        //    __m512 _params[4];
        //    _params[0] = _mm512_set1_ps(params[0]);
        //    if (type == SimdConvolutionActivationRestrictRange ||
        //        type == SimdConvolutionActivationHswish ||
        //        type == SimdConvolutionActivationHardSigmoid)
        //        _params[1] = _mm512_set1_ps(params[1]);

        //    SetTileConfFull();
        //    for (size_t i = 0; i < n1;)
        //    {
        //        size_t dn = (i == nn ? n1 - i : n);
        //        const uint16_t* s = src + i * dS;
        //        const uint16_t* w = weight;
        //        uint8_t* d = dst + i * dD;
        //        size_t dc = 0;
        //        for (; dc < dstC64; dc += QF, w += dW)
        //            mainConv(s, p, a, dn, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, __mmask32(-1));
        //        if (dc < dstC)
        //            tailConv(s, p, a, dn, w, bias + dc, params + dc, _params, buf, d + dc * a.elem, tailD);
        //        i += dn;
        //    }
        //}

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush> SIMD_INLINE void SetMacro64x16i(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            //if (a.bufK >= 96)
            //    convolution = Convolution16bNhwcGemm_Macro16xM<term, type, 1, flush>;
            //else 
            //    convolution = Convolution16bNhwcGemm_Macro16xM<term, type, 2, flush>;
        }

        template <SimdConvolutionActivationType type> SIMD_INLINE void SetMacro64x16i(const ConvParam& p, const AlgParam & a, Convolution& convolution)
        {
            if (p.dstT == SimdTensorData16b)
                SetMacro64x16i<Term16bLast16b, type, 0>(p, a, convolution);
            else
                SetMacro64x16i<Term16bLast32f, type, 0>(p, a, convolution);
        }

        void SynetConvolution16bNhwcGemmV1::SetMacro64x16i()
        {
            const ConvParam& p = _param;
            const AlgParam& a = _alg;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: AmxBf16::SetMacro64x16i<SimdConvolutionActivationIdentity>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRelu: AmxBf16::SetMacro64x16i<SimdConvolutionActivationRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: AmxBf16::SetMacro64x16i<SimdConvolutionActivationLeakyRelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationRestrictRange: AmxBf16::SetMacro64x16i<SimdConvolutionActivationRestrictRange>(p, _alg, _convolution); break;
            case SimdConvolutionActivationPrelu: AmxBf16::SetMacro64x16i<SimdConvolutionActivationPrelu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationElu: AmxBf16::SetMacro64x16i<SimdConvolutionActivationElu>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHswish: AmxBf16::SetMacro64x16i<SimdConvolutionActivationHswish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationMish: AmxBf16::SetMacro64x16i<SimdConvolutionActivationMish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: AmxBf16::SetMacro64x16i<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolution); break;
            case SimdConvolutionActivationSwish: AmxBf16::SetMacro64x16i<SimdConvolutionActivationSwish>(p, _alg, _convolution); break;
            case SimdConvolutionActivationGelu: AmxBf16::SetMacro64x16i<SimdConvolutionActivationGelu>(p, _alg, _convolution); break;
            default: assert(0);
            }
        }
    }
#endif
}

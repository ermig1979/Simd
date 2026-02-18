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

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> SIMD_INLINE void Convolution16bNhwcGemm_Nx16x1(const uint16_t* src, 
            const ConvParam& p, const AlgParam& a, int dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst)
        {
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), strideB = dB * 4, dW = (int)a.miniD, strideW = dW * 4;
            int dS = (int)a.bufK, stepS = a.reorder ? 512 : 32, strideS = a.reorder ? 64 : dS * 2;
            int offs0 = N == 1 ? dstS - 1 * F : 0 * F, offs1 = N == 2 ? dstS - 1 * F : 1 * F;
            int offs2 = N == 3 ? dstS - 1 * F : 2 * F, offs3 = N == 4 ? dstS - 1 * F : 3 * F;
            int srcC64 = (int)(a.bufK - 32)&(~63), applyC64 = apply ? (16 * 32 / apply - 64) : 0, sc = 0;

            if (N > 0) _tile_zero(0);
            if (N > 1) _tile_zero(1);
            if (N > 2) _tile_zero(2);
            if (N > 3) _tile_zero(3);

            if (N > 0) _tile_stream_loadd(5, weight0, strideW);
            if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
            for (; sc < applyC64; sc += 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + offs0 * dD, dD, buf0 + offs0 * dB, dB, bias, params);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + offs1 * dD, dD, buf0 + offs1 * dB, dB, bias, params);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + offs2 * dD, dD, buf0 + offs2 * dB, dB, bias, params);
                src += stepS;
                if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + offs3 * dD, dD, buf0 + offs3 * dB, dB, bias, params);
                dst += apply * dD;
                buf0 += apply * dB;

                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(5, weight0, strideW);
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + offs0 * dD, dD, buf0 + offs0 * dB, dB, bias, params);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + offs1 * dD, dD, buf0 + offs1 * dB, dB, bias, params);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + offs2 * dD, dD, buf0 + offs2 * dB, dB, bias, params);
                src += stepS;
                if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + offs3 * dD, dD, buf0 + offs3 * dB, dB, bias, params);
                dst += apply * dD;
                buf0 += apply * dB;
            }
            for (; sc < srcC64; sc += 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                src += stepS;
                if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);

                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(5, weight0, strideW);
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                src += stepS;
                if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
            }
            if (a.bufK - srcC64 == 64)
            {
                weight0 += 32 * dW;
                if (N > 0) _tile_stream_loadd(7, weight0, strideW);
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + offs0 * dD, dD, buf0 + offs0 * dB, dB, bias, params);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + offs1 * dD, dD, buf0 + offs1 * dB, dB, bias, params);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + offs2 * dD, dD, buf0 + offs2 * dB, dB, bias, params);
                src += stepS;
                if (N > 0) _tile_loadd(4, src + offs0 * dS, strideS);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + offs3 * dD, dD, buf0 + offs3 * dB, dB, bias, params);
                dst += apply * dD;
                buf0 += apply * dB;

                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 7);
                if (N > 0) Apply1xN<term, type, apply, flush>(dst + offs0 * dD, dD, buf0 + offs0 * dB, dB, bias, params);
                if (N > 0) _tile_stored(0, buf1 + offs0 * dB, strideB);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 7);
                if (N > 1) Apply1xN<term, type, apply, flush>(dst + offs1 * dD, dD, buf0 + offs1 * dB, dB, bias, params);
                if (N > 1) _tile_stored(1, buf1 + offs1 * dB, strideB);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 7);
                if (N > 2) Apply1xN<term, type, apply, flush>(dst + offs2 * dD, dD, buf0 + offs2 * dB, dB, bias, params);
                if (N > 2) _tile_stored(2, buf1 + offs2 * dB, strideB);
                if (N > 3) _tile_dpbf16ps(3, 6, 7);
                if (N > 3) Apply1xN<term, type, apply, flush>(dst + offs3 * dD, dD, buf0 + offs3 * dB, dB, bias, params);
                if (N > 3) _tile_stored(3, buf1 + offs3 * dB, strideB);
            }
            else
            {
                if (N > 1) _tile_loadd(6, src + offs1 * dS, strideS);
                if (N > 0) _tile_dpbf16ps(0, 4, 5);
                if (N > 0) Apply1xN<term, type, apply * 2, flush>(dst + offs0 * dD, dD, buf0 + offs0 * dB, dB, bias, params);
                if (N > 0) _tile_stored(0, buf1 + offs0 * dB, strideB);
                if (N > 2) _tile_loadd(4, src + offs2 * dS, strideS);
                if (N > 1) _tile_dpbf16ps(1, 6, 5);
                if (N > 1) Apply1xN<term, type, apply * 2, flush>(dst + offs1 * dD, dD, buf0 + offs1 * dB, dB, bias, params);
                if (N > 1) _tile_stored(1, buf1 + offs1 * dB, strideB);
                if (N > 3) _tile_loadd(6, src + offs3 * dS, strideS);
                if (N > 2) _tile_dpbf16ps(2, 4, 5);
                if (N > 2) Apply1xN<term, type, apply * 2, flush>(dst + offs2 * dD, dD, buf0 + offs2 * dB, dB, bias, params);
                if (N > 2) _tile_stored(2, buf1 + offs2 * dB, strideB);
                if (N > 3) _tile_dpbf16ps(3, 6, 5);
                if (N > 3) Apply1xN<term, type, apply * 2, flush>(dst + offs3 * dD, dD, buf0 + offs3 * dB, dB, bias, params);
                if (N > 3) _tile_stored(3, buf1 + offs3 * dB, strideB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int apply, int flush> static void Convolution16bNhwcGemm_Nx16xM(const uint16_t* src0, 
            const ConvParam& p, const AlgParam& a, int dstS, size_t dstC, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask16 tailD)
        {
            __m512 _bias[1];
            int dB = (int)a.miniD, dD = int(p.dstC * a.elem), dW = (int)a.miniD, dS = (int)a.bufK;
            int offs0 = N == 1 ? dstS - 1 * F : 0 * F, offs1 = N == 2 ? dstS - 1 * F : 1 * F;
            int offs2 = N == 3 ? dstS - 1 * F : 2 * F, offs3 = N == 4 ? dstS - 1 * F : 3 * F;

            float* buf0 = buf, * buf1 = buf + 16;
            Convolution16bNhwcGemm_Nx16x1<term, type, N, 0, 0>(src0, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst), weight0 += 32;
            for (size_t dc = 16; dc < dstC; dc += 16, weight0 += 32, bias += 16, params += 16, dst += 16 * a.elem)
            {
                _bias[0] = _mm512_loadu_ps(bias);
                if (type == SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params);
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_Nx16x1<term, type, N, apply, flush>(src0, p, a, dstS, weight0, _bias, _params, buf0, buf1, dst);
            }
            {
                _bias[0] = _mm512_loadu_ps(bias);
                if (type == SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params);
                if (N > 0) Apply1xN<term, type, 8, flush>(dst + (offs0 + 0) * dD, dD, buf1 + (offs0 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 0) Apply1xN<term, type, 8, flush>(dst + (offs0 + 8) * dD, dD, buf1 + (offs0 + 8) * dB, dB, _bias, _params, tailD);
                if (N > 1) Apply1xN<term, type, 8, flush>(dst + (offs1 + 0) * dD, dD, buf1 + (offs1 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 1) Apply1xN<term, type, 8, flush>(dst + (offs1 + 8) * dD, dD, buf1 + (offs1 + 8) * dB, dB, _bias, _params, tailD);
                if (N > 2) Apply1xN<term, type, 8, flush>(dst + (offs2 + 0) * dD, dD, buf1 + (offs2 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 2) Apply1xN<term, type, 8, flush>(dst + (offs2 + 8) * dD, dD, buf1 + (offs2 + 8) * dB, dB, _bias, _params, tailD);
                if (N > 3) Apply1xN<term, type, 8, flush>(dst + (offs3 + 0) * dD, dD, buf1 + (offs3 + 0) * dB, dB, _bias, _params, tailD);
                if (N > 3) Apply1xN<term, type, 8, flush>(dst + (offs3 + 8) * dD, dD, buf1 + (offs3 + 8) * dB, dB, _bias, _params, tailD);
            }
        }

        typedef  void (*Convolution16bNhwcGemm_Nx16xM_Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, int dstS, size_t dstC, 
            const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, uint8_t* dst, __mmask16 tailD);

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> Convolution16bNhwcGemm_Nx16xM_Ptr GetConvolution16bNhwcGemm_Nx16xM(size_t N)
        { 
            if (N == 0)
                return NULL;
            if (N <= 16)
                return Convolution16bNhwcGemm_Nx16xM<term, type, 1, apply, flush>;
            if (N <= 32)
                return Convolution16bNhwcGemm_Nx16xM<term, type, 2, apply, flush>;
            if (N <= 48)
                return Convolution16bNhwcGemm_Nx16xM<term, type, 3, apply, flush>;
            if (N <= 64)
                return Convolution16bNhwcGemm_Nx16xM<term, type, 4, apply, flush>;
            return NULL;
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush> void Convolution16bNhwcGemm_MacroNx16(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t dD = p.dstC * a.elem, dS = a.bufK, dW = a.bufK * a.miniD;
            size_t n = 64, n1 = dstH * p.dstW, nl = AlignLoAny(n1, n), l = n1 - nl, m = 0, nm = nl;
            assert(n1 >= 16);
            if (l && l < 16)
                nm = nl - 64, m = 48, nl -= 16, l += 16;

            Convolution16bNhwcGemm_Nx16xM_Ptr nConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush>(n);
            Convolution16bNhwcGemm_Nx16xM_Ptr mConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush>(m);
            Convolution16bNhwcGemm_Nx16xM_Ptr lConv = GetConvolution16bNhwcGemm_Nx16xM<term, type, apply, flush>(l);

            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += a.miniD)
            {
                size_t dC = Simd::Min(a.miniD, dstC - dc);
                __mmask16 tailD = TailMask16(dC - AlignLo(dC - 1, 16));
                const uint16_t* s = src;
                uint8_t* d = dst + dc * a.elem;
                size_t i = 0;
                for (; i < nm; i += n)
                    nConv(s + i * dS, p, a, (int)n, dC, weight, bias, params, _params, buf, d + i * dD, tailD);
                if (m)
                    mConv(s + i * dS, p, a, (int)m, dC, weight, bias, params, _params, buf, d + i * dD, tailD), i += m;
                if (l)
                    lConv(s + i * dS, p, a, (int)l, dC, weight, bias, params, _params, buf, d + i * dD, tailD), i += l;
                weight += dW;
                bias += a.miniD;
                params += a.miniD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush> SIMD_INLINE void SetMacro64x16i(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 512)
                convolution = Convolution16bNhwcGemm_MacroNx16<term, type, 1, flush>;
            else if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_MacroNx16<term, type, 2, flush>;
            //else if (a.bufK >= 128)
            //    convolution = Convolution16bNhwcGemm_MacroNx16<term, type, 4, flush>;
            //else if (a.bufK >= 64)
            //    convolution = Convolution16bNhwcGemm_MacroNx16<term, type, 8, flush>;
            else 
                convolution = NULL;
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

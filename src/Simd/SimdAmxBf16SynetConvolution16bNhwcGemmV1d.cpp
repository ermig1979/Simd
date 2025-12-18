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

        template<int tile, int order> static SIMD_INLINE void LoadS(const void* ptr, int stride)
        {
            switch (tile)
            {
            case 4: if (order) _tile_loadd(4, ptr, stride); else _tile_stream_loadd(4, ptr, stride); break;
            case 5: if (order) _tile_loadd(5, ptr, stride); else _tile_stream_loadd(5, ptr, stride); break;
            case 6: if (order) _tile_loadd(6, ptr, stride); else _tile_stream_loadd(6, ptr, stride); break;
            case 7: if (order) _tile_loadd(7, ptr, stride); else _tile_stream_loadd(7, ptr, stride); break;
            }
        }

        template<int tile, int order> static SIMD_INLINE void LoadW(const void* ptr, int stride)
        {
            switch (tile)
            {
            case 4: if (order) _tile_stream_loadd(4, ptr, stride); else _tile_loadd(4, ptr, stride); break;
            case 5: if (order) _tile_stream_loadd(5, ptr, stride); else _tile_loadd(5, ptr, stride); break;
            case 6: if (order) _tile_stream_loadd(6, ptr, stride); else _tile_loadd(6, ptr, stride); break;
            case 7: if (order) _tile_stream_loadd(7, ptr, stride); else _tile_loadd(7, ptr, stride); break;
            }
        }

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

        template<Term16bType term, SimdConvolutionActivationType type, int N, int flush, int start> static SIMD_INLINE void Apply2xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) Apply2x1<term, type, flush, start>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply2x1<term, type, flush, start>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply2x1<term, type, flush, start>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply2x1<term, type, flush, start>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply2x1<term, type, flush, start>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply2x1<term, type, flush, start>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply2x1<term, type, flush, start>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply2x1<term, type, flush, start>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int start> static SIMD_INLINE void Apply1x1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + (start + 0) * F), bias[0]), params, 0);
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
                _mm512_storeu_ps((float*)(ptr + (start + 0) * A), f0);
                if (flush == 1)
                    _mm_prefetch((const char*)(ptr + (start + 0) * A), _MM_HINT_NTA);
                else if (flush == 2)
                    _m_prefetchw((char*)(ptr + (start + 0) * A));
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int N, int flush, int start> static SIMD_INLINE void Apply1xN(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            if (N > 0) Apply1x1<term, type, flush, start>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            if (N > 1) Apply1x1<term, type, flush, start>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            if (N > 2) Apply1x1<term, type, flush, start>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            if (N > 3) Apply1x1<term, type, flush, start>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            if (N > 4) Apply1x1<term, type, flush, start>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            if (N > 5) Apply1x1<term, type, flush, start>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            if (N > 6) Apply1x1<term, type, flush, start>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            if (N > 7) Apply1x1<term, type, flush, start>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }


        //------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> SIMD_INLINE void Convolution16bNhwcGemm_1x16x64(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dS = (int)a.bufK, strideB = dB * 4, dW = (int)a.microD, strideW = dW * 4;
            int stepS = 32, strideS = dS * 2;

            int srcC64 = (int)(a.bufK - 32)&(~63), applyC64 = apply ? (16 * 32 / apply - 64) : 0, sc = 0, ds = 0;

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            LoadS<4, order>(src0, strideS);
            LoadW<5, order>(weight0 + 0 * DF, strideW);
            for (; sc < applyC64; sc += 64)
            {
                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<6, order>(src0, strideS);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 5);
                weight0 += 32 * dW;
                LoadW<5, order>(weight0 + 0 * DF, strideW);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(3, 4, 7);

                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 6, 5);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 6, 7);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 6, 5);
                weight0 += 32 * dW;
                LoadW<5, order>(weight0 + 0 * DF, strideW);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(3, 6, 7);
            }
            for (; sc < srcC64; sc += 64)
            {
                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<6, order>(src0, strideS);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 5);
                weight0 += 32 * dW;
                LoadW<5, order>(weight0 + 0 * DF, strideW);
                _tile_dpbf16ps(3, 4, 7);

                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 6, 5);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 6, 7);
                src0 += stepS;
                LoadS<4, order>(src0, strideS);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 6, 5);
                weight0 += 32 * dW;
                LoadW<5, order>(weight0 + 0 * DF, strideW);
                _tile_dpbf16ps(3, 6, 7);
            }
            if (a.bufK - srcC64 == 64)
            {
                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                LoadS<6, order>(src0, strideS);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 5);
                weight0 += 32 * dW;
                LoadW<5, order>(weight0 + 0 * DF, strideW);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(3, 4, 7);

                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 6, 5);
                _tile_stored(0, buf1 + 0 * F, strideB);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 6, 7);
                _tile_stored(1, buf1 + 1 * F, strideB);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 6, 5);
                _tile_stored(2, buf1 + 2 * F, strideB);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_dpbf16ps(3, 6, 7);
                _tile_stored(3, buf1 + 3 * F, strideB);
            }
            else
            {
                LoadW<7, order>(weight0 + 1 * DF, strideW);
                _tile_dpbf16ps(0, 4, 5);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                _tile_stored(0, buf1 + 0 * F, strideB);
                LoadW<5, order>(weight0 + 2 * DF, strideW);
                _tile_dpbf16ps(1, 4, 7);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_stored(1, buf1 + 1 * F, strideB);
                LoadW<7, order>(weight0 + 3 * DF, strideW);
                _tile_dpbf16ps(2, 4, 5);
                Apply2xN<term, type, apply, flush, 0>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params);
                _tile_stored(2, buf1 + 2 * F, strideB);
                _tile_dpbf16ps(3, 4, 7);
                Apply2xN<term, type, apply, flush, 2>(dst + ds * dD, dD, buf0 + ds * dB, dB, bias, params, tailD), ds += apply;
                _tile_stored(3, buf1 + 3 * F, strideB);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> static void Convolution16bNhwcGemm_Nx16x64(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, size_t dstC, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)a.microD, dD = int(p.dstC * a.elem), dW = (int)a.microD, dS = (int)a.bufK;
            float* buf0 = buf, * buf1 = buf + 16 * dB;
            size_t cds = 0, pds = 0;
            Convolution16bNhwcGemm_1x16x64<term, type, 0, 0, order>(src0, p, a, dstC, weight0, bias, params, buf0, buf1, dst, tailD), cds += 16;
            for (; cds < dstS; pds = cds, cds += 16)
            {
                cds = Simd::Min(dstS - 16, cds);
                Swap(buf0, buf1);
                Convolution16bNhwcGemm_1x16x64<term, type, apply, flush, order>(src0 + cds * dS, p, a, dstC, weight0, bias, params, buf0, buf1, dst + pds * dD, tailD);
            }
            uint8_t* dst1 = dst + pds * dD;
            dstS -= pds;
            {
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                {
                    Apply2xN<term, type, 8, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params);
                    Apply2xN<term, type, 8, flush, 2>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                }
                for (; ds < dstS; ++ds)
                {
                    Apply2xN<term, type, 1, flush, 0>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params);
                    Apply2xN<term, type, 1, flush, 2>(dst1 + ds * dD, dD, buf1 + ds * dB, dB, bias, params, tailD);
                }
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int apply, int flush, int order> void Convolution16bNhwcGemm_Macro16x64(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t dstC, size_t dstH, const uint16_t* weight, const float* bias, const float* params, float* buf, uint8_t* dst)
        {
            size_t n = 256, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), dW = a.bufK * a.microD;
            if (n1 > nn && n1 - nn < 16)
                nn -= n;

            size_t dD = p.dstC * a.elem, dS = a.bufK;

            __m512 _params[4], _bias[4];
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
                for (size_t dc = 0; dc < dstC; dc += 4 * F)
                {
                    size_t dC = Simd::Min(QF, dstC - dc);
                    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC - 2 * F) : (__mmask32)TailMask16(dC - 3 * F);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0 * F);
                    _bias[1] = _mm512_loadu_ps(bias + dc + 1 * F);
                    _bias[2] = _mm512_loadu_ps(bias + dc + 2 * F);
                    _bias[3] = _mm512_loadu_ps(bias + dc + 3 * F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0 * F);
                        _params[1] = _mm512_loadu_ps(params + dc + 1 * F);
                        _params[2] = _mm512_loadu_ps(params + dc + 2 * F);
                        _params[3] = _mm512_loadu_ps(params + dc + 3 * F);
                    }
                    Convolution16bNhwcGemm_Nx16x64<term, type, apply, flush, order>(s, p, a, dn, dC, w, _bias, _params, buf, d + dc * a.elem, tailD);
                    w += dW;
                }
                i += dn;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term, SimdConvolutionActivationType type, int flush, int order> SIMD_INLINE void SetMacro16x64(const ConvParam& p, const AlgParam& a, Convolution& convolution)
        {
            if (a.bufK >= 512)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 1, flush, order>;
            else if (a.bufK >= 256)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 2, flush, order>;
            else if (a.bufK >= 128)
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 4, flush, order>;
            else
                convolution = Convolution16bNhwcGemm_Macro16x64<term, type, 8, flush, order>;
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

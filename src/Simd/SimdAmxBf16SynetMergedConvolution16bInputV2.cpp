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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using InputPtr = Base::SynetMergedConvolution16b::InputConvolutionPtr;

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M> static SIMD_INLINE void ApplyMx1(const float* src, float* dst0, float* dst1, float* dst2, float* dst3, const __m512* bias, const __m512* params)
        {
            if (M > 0)
            {
                _mm512_storeu_ps(dst0, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 0 * 256), bias[0]), params, 0));
                if (flush) _mm_prefetch((const char*)dst0, _MM_HINT_NTA);
            }
            if (M > 1)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 1 * 256), bias[1]), params, 1));
                if (flush) _mm_prefetch((const char*)dst1, _MM_HINT_NTA);
            }
            if (M > 2)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 2 * 256), bias[2]), params, 2));
                if (flush) _mm_prefetch((const char*)dst2, _MM_HINT_NTA);
            }
            if (M > 3)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 3 * 256), bias[3]), params, 3));
                if (flush) _mm_prefetch((const char*)dst3, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush, int M, int N> static SIMD_INLINE void ApplyMxN(const float* src, float* dst0, float* dst1, float* dst2, float* dst3, const __m512* bias, const __m512* params)
        {
            if (N > 0) ApplyMx1<type, flush, M>(src + 0 * F, dst0 + 0 * F, dst1 + 0 * F, dst2 + 0 * F, dst3 + 0 * F, bias, params);
            if (N > 1) ApplyMx1<type, flush, M>(src + 1 * F, dst0 + 1 * F, dst1 + 1 * F, dst2 + 1 * F, dst3 + 1 * F, bias, params);
            if (N > 2) ApplyMx1<type, flush, M>(src + 2 * F, dst0 + 2 * F, dst1 + 2 * F, dst2 + 2 * F, dst3 + 2 * F, bias, params);
            if (N > 3) ApplyMx1<type, flush, M>(src + 3 * F, dst0 + 3 * F, dst1 + 3 * F, dst2 + 3 * F, dst3 + 3 * F, bias, params);
            if (N > 4) ApplyMx1<type, flush, M>(src + 4 * F, dst0 + 4 * F, dst1 + 4 * F, dst2 + 4 * F, dst3 + 4 * F, bias, params);
            if (N > 5) ApplyMx1<type, flush, M>(src + 5 * F, dst0 + 5 * F, dst1 + 5 * F, dst2 + 5 * F, dst3 + 5 * F, bias, params);
            if (N > 6) ApplyMx1<type, flush, M>(src + 6 * F, dst0 + 6 * F, dst1 + 6 * F, dst2 + 6 * F, dst3 + 6 * F, bias, params);
            if (N > 7) ApplyMx1<type, flush, M>(src + 7 * F, dst0 + 7 * F, dst1 + 7 * F, dst2 + 7 * F, dst3 + 7 * F, bias, params);
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M, int cfg> void InputConvolution1x1_1xMV2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, float* dst0, float* dst1, float* dst2, float* dst3)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideB = 64, stepS = 32, stepW = 32 * 16;
            const uint16_t* weight1 = weight0 + sC * 16, * weight2 = weight0 + sC * 32, * weight3 = weight0 + sC * 48;

            if (cfg)
                SetTileConf1x4(dstS);
            __m512 _bias[4];
            if (M > 0) _bias[0] = _mm512_loadu_ps(bias + 0 * F);
            if (M > 1) _bias[1] = _mm512_loadu_ps(bias + 1 * F);
            if (M > 2) _bias[2] = _mm512_loadu_ps(bias + 2 * F);
            if (M > 3) _bias[3] = _mm512_loadu_ps(bias + 3 * F);
            if (type == ::SimdConvolutionActivationPrelu)
            {
                if (M > 0) _params[0] = _mm512_loadu_ps(params + 0 * F);
                if (M > 1) _params[1] = _mm512_loadu_ps(params + 1 * F);
                if (M > 2) _params[2] = _mm512_loadu_ps(params + 2 * F);
                if (M > 3) _params[3] = _mm512_loadu_ps(params + 3 * F);
            }
            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);
            if (M > 2) _tile_zero(2);
            if (M > 3) _tile_zero(3);

            int sC64 = (int)(sC - 32) & (~63), sc = 0, ow = 0;

            if (M > 0) _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(5, weight0, strideW);
            for (; sc < sC64; sc += 64)
            {
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                src0 += stepS;
                if (M > 0) _tile_stream_loadd(4, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
            }
            if (sC - sC64 == 64)
            {
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                if (M > 0) _tile_stored(0, buf + 0 * 256, strideB);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                if (M > 1) _tile_stored(1, buf + 1 * 256, strideB);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                if (M > 2) _tile_stored(2, buf + 2 * 256, strideB);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
                if (M > 3) _tile_stored(3, buf + 3 * 256, strideB);
            }
            else
            {
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                if (M > 0) _tile_stored(0, buf + 0 * 256, strideB);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                if (M > 1) _tile_stored(1, buf + 1 * 256, strideB);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                if (M > 2) _tile_stored(2, buf + 2 * 256, strideB);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);
                if (M > 3) _tile_stored(3, buf + 3 * 256, strideB);
            }

            if (dstS == 16)
            {
                ApplyMxN<type, flush, M, 8>(buf + 0 * 16, dst0 + 0 * 16, dst1 + 0 * 16, dst2 + 0 * 16, dst3 + 0 * 16, _bias, _params);
                ApplyMxN<type, flush, M, 8>(buf + 8 * 16, dst0 + 8 * 16, dst1 + 8 * 16, dst2 + 8 * 16, dst3 + 8 * 16, _bias, _params);
            }
            else
            {
                for (size_t s = 16; s < dstS; ++s)
                    ApplyMxN<type, flush, M, 1>(buf + s * 16, dst0 + s * 16, dst1 + s * 16, dst2 + s * 16, dst3 + s * 16, _bias, _params);
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M, int apply> void InputConvolution1x1_1xMV2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, float* dst0, float* dst1, float* dst2, float* dst3)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideB = 64, stepS = 32, stepW = 32 * 16;
            const uint16_t* weight1 = weight0 + sC * 16, * weight2 = weight0 + sC * 32, * weight3 = weight0 + sC * 48;

            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);
            if (M > 2) _tile_zero(2);
            if (M > 3) _tile_zero(3);

            int sC64 = (int)(sC - 32) & (~63), aC64 = apply ? (4 * 32 / apply - 64) : 0, sc = 0, ds = 0, ow = 0;

            if (M > 0) _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(5, weight0, strideW);
            for (; sc < aC64; sc += 64)
            {
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;

                src0 += stepS;
                if (M > 0) _tile_stream_loadd(4, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
            }
            for (; sc < sC64; sc += 64)
            {
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);

                src0 += stepS;
                if (M > 0) _tile_stream_loadd(4, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
            }
            if (sC - sC64 == 64)
            {
                src0 += stepS;
                if (M > 0) _tile_stream_loadd(6, src0, strideS);
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                ow += stepW;
                if (M > 0) _tile_loadd(5, weight0 + ow, strideW);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;

                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 6, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 0) _tile_stored(0, buf1 + 0 * 256, strideB);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 6, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 1) _tile_stored(1, buf1 + 1 * 256, strideB);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 6, 5);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 2) _tile_stored(2, buf1 + 2 * 256, strideB);
                if (M > 3) _tile_dpbf16ps(3, 6, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += apply * F;
                if (M > 3) _tile_stored(3, buf1 + 3 * 256, strideB);
            }
            else
            {
                if (M > 1) _tile_loadd(7, weight1 + ow, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 5);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += 2 * apply * F;
                if (M > 0) _tile_stored(0, buf1 + 0 * 256, strideB);
                if (M > 2) _tile_loadd(5, weight2 + ow, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += 2 * apply * F;
                if (M > 1) _tile_stored(1, buf1 + 1 * 256, strideB);
                if (M > 3) _tile_loadd(7, weight3 + ow, strideW);
                if (M > 2) _tile_dpbf16ps(2, 4, 5);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += 2 * apply * F;
                if (M > 2) _tile_stored(2, buf1 + 2 * 256, strideB);
                if (M > 3) _tile_dpbf16ps(3, 4, 7);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, dst2 + ds, dst3 + ds, bias, params), ds += 2 * apply * F;
                if (M > 3) _tile_stored(3, buf1 + 3 * 256, strideB);
            }
        }

        template<SimdConvolutionActivationType type, int flush, int M, int apply> void InputConvolution1x1_NxMV2(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const float* bias, const float* params, __m512* _params, float* buf, float* dst0, float* dst1, float* dst2, float* dst3)
        {
            int dS = (int)AlignHi(p.srcC, a.miK);
            float* buf0 = buf, * buf1 = buf + 1024;

            __m512 _bias[4];
            if (M > 0) _bias[0] = _mm512_loadu_ps(bias + 0 * F);
            if (M > 1) _bias[1] = _mm512_loadu_ps(bias + 1 * F);
            if (M > 2) _bias[2] = _mm512_loadu_ps(bias + 2 * F);
            if (M > 3) _bias[3] = _mm512_loadu_ps(bias + 3 * F);
            if (type == ::SimdConvolutionActivationPrelu)
            {
                if (M > 0) _params[0] = _mm512_loadu_ps(params + 0 * F);
                if (M > 1) _params[1] = _mm512_loadu_ps(params + 1 * F);
                if (M > 2) _params[2] = _mm512_loadu_ps(params + 2 * F);
                if (M > 3) _params[3] = _mm512_loadu_ps(params + 3 * F);
            }
            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);
            if (M > 2) _tile_zero(2);
            if (M > 3) _tile_zero(3);

            size_t cds = 0, pds = 0;
            InputConvolution1x1_1xMV2<type, flush, M, 0>(src0, p, a, weight0, _bias, _params, buf0, buf1, dst0, dst1, dst2, dst3), cds += 16;
            for (; cds < dstS; pds = cds, cds += 16)
            {
                cds = Simd::Min(dstS - 16, cds);
                Swap(buf0, buf1);
                InputConvolution1x1_1xMV2<type, flush, M, apply>(src0 + cds * dS, p, a, weight0, _bias, _params, buf0, buf1, dst0 + pds * F, dst1 + pds * F, dst2 + pds * F, dst3 + pds * F);
            }
            dst0 += pds * F;
            dst1 += pds * F;
            dst2 += pds * F;
            dst3 += pds * F;
            dstS -= pds;
            {
                if (dstS == 16)
                {
                    ApplyMxN<type, flush, M, 8>(buf + 0 * 16, dst0 + 0 * 16, dst1 + 0 * 16, dst2 + 0 * 16, dst3 + 0 * 16, _bias, _params);
                    ApplyMxN<type, flush, M, 8>(buf + 8 * 16, dst0 + 8 * 16, dst1 + 8 * 16, dst2 + 8 * 16, dst3 + 8 * 16, _bias, _params);
                }
                else
                {
                    for (size_t s = 16; s < dstS; ++s)
                        ApplyMxN<type, flush, M, 1>(buf + s * 16, dst0 + s * 16, dst1 + s * 16, dst2 + s * 16, dst3 + s * 16, _bias, _params);
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int apply> void InputConvolution1x1_4V2(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* buf, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcC = AlignHi(p.srcC, a.miK), y0 = a.bufH[0] ? yBeg : 0, dW = srcC * QF, dD = a.bufH[1] * p.dstW * QF;
            __m512 _params[4];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 16;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLo(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLo(e1, n), e = e1 - en;

            if (yInt == yBeg)
            {
                if (en)
                {
                    SetTileConfFull();
                    for (size_t dc = 0; dc < maC; dc += QF)
                    {
                        size_t dC = Simd::Min(QF, maC - dc);
                        const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                        float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                        if (dC > 3 * F)
                            InputConvolution1x1_NxMV2<type, flush, 4, apply>(src0, p, a, e1, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, dst0 + 3 * dstS);
                        else if (dC > 2 * F)
                            InputConvolution1x1_NxMV2<type, flush, 3, apply>(src0, p, a, e1, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, NULL);
                        else if (dC > 1 * F)
                            InputConvolution1x1_NxMV2<type, flush, 2, apply>(src0, p, a, e1, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, NULL, NULL);
                        else
                            InputConvolution1x1_NxMV2<type, flush, 1, apply>(src0, p, a, e1, weight, bias + dc, params + dc, _params, buf, dst0, NULL, NULL, NULL);
                        dst0 += dD;
                        weight += dW;
                    }
                }
                else if (e1)
                {
                    SetTileConf1x4(e);
                    const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                    float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                    for (size_t dc = 0; dc < maC; dc += QF)
                    {
                        size_t dC = Simd::Min(QF, maC - dc);
                        if (dC > 3 * F)
                            InputConvolution1x1_1xMV2<type, flush, 4, 0>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, dst0 + 3 * dstS);
                        else if (dC > 2 * F)
                            InputConvolution1x1_1xMV2<type, flush, 3, 0>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, NULL);
                        else if (dC > 1 * F)
                            InputConvolution1x1_1xMV2<type, flush, 2, 0>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, NULL, NULL);
                        else
                            InputConvolution1x1_1xMV2<type, flush, 1, 0>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, NULL, NULL, NULL);
                        dst0 += dD;
                        weight += dW;
                    }
                }
            }
            else
            {
                for (size_t dc = 0; dc < maC; dc += QF)
                {
                    size_t dC = Simd::Min(QF, maC - dc);
                    if (dC > 3 * F)
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, dst1);
                            else if (i)
                                InputConvolution1x1_1xMV2<type, flush, 4, 1>(src0, p, a, i, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, dst0 + 3 * dstS);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
                            else if (e)
                                InputConvolution1x1_1xMV2<type, flush, 4, 1>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, dst0 + 3 * dstS);
                        }
                    }
                    else if (dC > 2 * F)
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, dst1);
                            else if (i)
                                InputConvolution1x1_1xMV2<type, flush, 3, 1>(src0, p, a, i, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
                            else if (e)
                                InputConvolution1x1_1xMV2<type, flush, 3, 1>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, dst0 + 2 * dstS, NULL);
                        }
                    }
                    else if (dC > 1 * F)
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, dst1);
                            else if (i)
                                InputConvolution1x1_1xMV2<type, flush, 2, 1>(src0, p, a, i, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, NULL, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
                            else if (e)
                                InputConvolution1x1_1xMV2<type, flush, 2, 1>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, dst0 + 1 * dstS, NULL, NULL);
                        }
                    }
                    else
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, i1, weight, _bias, _params, buf, dst0, dst1);
                            else if (i)
                                InputConvolution1x1_1xMV2<type, flush, 1, 1>(src0, p, a, i, weight, bias + dc, params + dc, _params, buf, dst0, NULL, NULL, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            if (in)
                                ;// InputConvolution1x1_Nx32V1<type, 1, 2, apply>(src0, p, a, e1, weight, _bias, _params, buf, dst0, dst1);
                            else if (e)
                                InputConvolution1x1_1xMV2<type, flush, 1, 1>(src0, p, a, e, weight, bias + dc, params + dc, _params, buf, dst0, NULL, NULL, NULL);
                        }
                    }
                    dst += a.bufH[1] * p.dstW * QF;
                    weight += srcC * QF;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInputV2(const ConvParam& p, InputPtr& input)
        {
            if (Is1x1(p))
            {
                if (p.srcC >= 96)
                    input = InputConvolution1x1_4V2<type, 1, 1>;
                else
                    input = InputConvolution1x1_4V2<type, 1, 2>;
            }
            else
                assert(0);
        }

        void SetInputV2(const ConvParam& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInputV2<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInputV2<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInputV2<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInputV2<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInputV2<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInputV2<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInputV2<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInputV2<SimdConvolutionActivationSwish>(p, input); break;
            case SimdConvolutionActivationGelu: SetInputV2<SimdConvolutionActivationGelu>(p, input); break;
            }
        }
    }
#endif
}

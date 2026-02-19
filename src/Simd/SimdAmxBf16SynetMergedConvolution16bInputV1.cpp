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

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply1(const float* src, float* dst, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst + 0x0 * F, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 0x0 * F), bias[index]), params, index));
            _mm_prefetch((const char*)(dst + 0x0 * F), _MM_HINT_NTA);
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply2(const float* src, float* dst, const __m512* bias, const __m512* params)
        {
            Apply1<type, index>(src + 0 * F, dst + 0 * F, bias, params);
            Apply1<type, index>(src + 1 * F, dst + 1 * F, bias, params);
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply4(const float* src, float* dst, const __m512* bias, const __m512* params)
        {
            Apply2<type, index>(src + 0 * F, dst + 0 * F, bias, params);
            Apply2<type, index>(src + 2 * F, dst + 2 * F, bias, params);
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply8(const float* src, float* dst, const __m512* bias, const __m512* params)
        {
            Apply4<type, index>(src + 0 * F, dst + 0 * F, bias, params);
            Apply4<type, index>(src + 4 * F, dst + 4 * F, bias, params);
        }

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply16(const float* src, float* dst, const __m512* bias, const __m512* params)
        {
            Apply8<type, index>(src + 0 * F, dst + 0 * F, bias, params);
            Apply8<type, index>(src + 8 * F, dst + 8 * F, bias, params);
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int cfg> void InputConvolution1x1_2x2V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideD = 64;
            const uint16_t* src1 = src0 + sC * 16, * weight1 = weight0 + sC * 16;

            if (cfg)
                SetTileConf2x2(dstS, 32);
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);

            int sC32 = (int)sC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0, strideW);
            for (; sc < sC32;)
            {
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                sc += 32;
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, dst0 + 0 * 256, strideD);
            _tile_stored(1, dst1 + 0 * 256, strideD);
            Apply16<type, 0>(dst0 + 0 * 256, dst0 + 0 * 256, bias, params);
            Apply16<type, 1>(dst1 + 0 * 256, dst1 + 0 * 256, bias, params);
            if (dstS == 32)
            {
                _tile_stored(2, dst0 + 1 * 256, strideD);
                _tile_stored(3, dst1 + 1 * 256, strideD);
                Apply16<type, 0>(dst0 + 1 * 256, dst0 + 1 * 256, bias, params);
                Apply16<type, 1>(dst1 + 1 * 256, dst1 + 1 * 256, bias, params);
            }
            else
            {
                _tile_stored(2, buf + 1 * 256, strideD);
                _tile_stored(3, buf + 3 * 256, strideD);
                for (size_t s = 16; s < dstS; ++s)
                    Apply1<type, 0>(buf + 0 * 256 + s * F, dst0 + s * F, bias, params);
                for (size_t s = 16; s < dstS; ++s)
                    Apply1<type, 1>(buf + 2 * 256 + s * F, dst1 + s * F, bias, params);
            }
        }

        template<SimdConvolutionActivationType type, int cfg> void InputConvolution1x1_2x1V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideD = 64;
            const uint16_t* src1 = src0 + sC * 16;

            if (cfg)
                SetTileConf2x1(dstS, 16);
            _tile_zero(0);
            _tile_zero(2);

            int sC32 = (int)sC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < sC32;)
            {
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_stream_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_loadd(6, weight0 + sc * 16, strideW);
            _tile_stream_loadd(5, src1 + sc, strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, dst0 + 0 * 256, strideD);
            Apply16<type, 0>(dst0 + 0 * 256, dst0 + 0 * 256, bias, params);
            if (dstS == 32)
            {
                _tile_stored(2, dst0 + 1 * 256, strideD);
                Apply16<type, 0>(dst0 + 1 * 256, dst0 + 1 * 256, bias, params);
            }
            else
            {
                _tile_stored(2, buf + 1 * 256, strideD);
                for (size_t s = 16; s < dstS; ++s)
                    Apply1<type, 0>(buf + 0 * 256 + s * F, dst0 + s * F, bias, params);
            }
        }

        template<SimdConvolutionActivationType type, int cfg> void InputConvolution1x1_1x2V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideD = 64;
            const uint16_t* weight1 = weight0 + sC * 16;

            if (cfg)
                SetTileConf1x2(dstS, 32);
            _tile_zero(0);
            _tile_zero(1);

            int sC32 = (int)sC - 32, sc = 0;
            _tile_loadd(6, weight0, strideW);
            for (; sc < sC32;)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(7, weight1 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stream_loadd(4, src0 + sc, strideS);
            _tile_loadd(7, weight1 + sc * 16, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            if (dstS == 16)
            {
                _tile_stored(0, dst0 + 0 * 256, strideD);
                _tile_stored(1, dst1 + 0 * 256, strideD);
                Apply16<type, 0>(dst0 + 0 * 256, dst0 + 0 * 256, bias, params);
                Apply16<type, 1>(dst1 + 0 * 256, dst1 + 0 * 256, bias, params);
            }
            else
            {
                _tile_stored(0, buf + 0 * 256, strideD);
                _tile_stored(1, buf + 2 * 256, strideD);
                for (size_t s = 0; s < dstS; ++s)
                    Apply1<type, 0>(buf + 0 * 256 + s * F, dst0 + s * F, bias, params);
                for (size_t s = 0; s < dstS; ++s)
                    Apply1<type, 1>(buf + 2 * 256 + s * F, dst1 + s * F, bias, params);
            }
        }

        template<SimdConvolutionActivationType type, int cfg> void InputConvolution1x1_1x1V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideD = 64;

            if (cfg)
                SetTileConf1x1(dstS, 16);
            _tile_zero(0);

            for (size_t sc = 0; sc < sC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }

            if (dstS == 16)
            {
                _tile_stored(0, dst0 + 0 * 256, strideD);
                Apply16<type, 0>(dst0 + 0 * 256, dst0 + 0 * 256, bias, params);
            }
            else
            {
                _tile_stored(0, buf + 0 * 256, strideD);
                for (size_t s = 0; s < dstS; ++s)
                    Apply1<type, 0>(buf + 0 * 256 + s * F, dst0 + s * F, bias, params);
            }
        }

        typedef void (*InputConvolution1x1V1Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf, float* dst0, float* dst1);

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M> static SIMD_INLINE void ApplyMx1(const float* src, float* dst0, float* dst1, const __m512* bias, const __m512* params)
        {
            if (M > 0)
            {
                _mm512_storeu_ps(dst0, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 0 * 512), bias[0]), params, 0));
                if (flush) _mm_prefetch((const char*)dst0, _MM_HINT_NTA);
            }
            if (M > 1)
            {
                _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + 1 * 512), bias[1]), params, 1));
                if (flush) _mm_prefetch((const char*)dst1, _MM_HINT_NTA);
            }
        }

        template<SimdConvolutionActivationType type, int flush, int M, int N> static SIMD_INLINE void ApplyMxN(const float* src, float* dst0, float* dst1, const __m512* bias, const __m512* params)
        {
            if (N > 0) ApplyMx1<type, flush, M>(src + 0 * F, dst0 + 0 * F, dst1 + 0 * F, bias, params);
            if (N > 1) ApplyMx1<type, flush, M>(src + 1 * F, dst0 + 1 * F, dst1 + 1 * F, bias, params);
            if (N > 2) ApplyMx1<type, flush, M>(src + 2 * F, dst0 + 2 * F, dst1 + 2 * F, bias, params);
            if (N > 3) ApplyMx1<type, flush, M>(src + 3 * F, dst0 + 3 * F, dst1 + 3 * F, bias, params);
            if (N > 4) ApplyMx1<type, flush, M>(src + 4 * F, dst0 + 4 * F, dst1 + 4 * F, bias, params);
            if (N > 5) ApplyMx1<type, flush, M>(src + 5 * F, dst0 + 5 * F, dst1 + 5 * F, bias, params);
            if (N > 6) ApplyMx1<type, flush, M>(src + 6 * F, dst0 + 6 * F, dst1 + 6 * F, bias, params);
            if (N > 7) ApplyMx1<type, flush, M>(src + 7 * F, dst0 + 7 * F, dst1 + 7 * F, bias, params);
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int flush, int M, int apply> void InputConvolution1x1_1x32x32V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideB = 64, stepS = 32, dW =16;
            const uint16_t* src1 = src0 + sC * 16, * weight1 = weight0 + sC * 16;
            int sC32 = (int)sC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);
            if (M > 0) _tile_zero(2);
            if (M > 1) _tile_zero(3);

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
                ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
            }
            for (; sc < sC32; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            _tile_stream_loadd(5, src1, strideS);
            if (M > 0) _tile_dpbf16ps(0, 4, 6);
            ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
            if (M > 0) _tile_stored(0, buf1 + 0 * 256, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
            if (M > 1) _tile_stored(1, buf1 + 2 * 256, strideB);
            if (M > 0) _tile_dpbf16ps(2, 5, 6);
            ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
            if (M > 0) _tile_stored(2, buf1 + 1 * 256, strideB);
            if (M > 1) _tile_dpbf16ps(3, 5, 7);
            ApplyMxN<type, flush, M, apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += apply * F;
            if (M > 1) _tile_stored(3, buf1 + 3 * 256, strideB);
        }

        template<SimdConvolutionActivationType type, int flush, int M, int apply> void InputConvolution1x1_1x16x32V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t dstS, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, float* buf1, float* dst0, float* dst1)
        {
            size_t sC = AlignHi(p.srcC, a.miK);
            int strideS = (int)sC * 2, strideW = 64, strideB = 64, stepS = 32, dW = 16;
            const uint16_t* weight1 = weight0 + sC * 16;
            int sC32 = (int)sC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            if (M > 0) _tile_zero(0);
            if (M > 1) _tile_zero(1);

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += 2 * apply * F;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += 2 * apply * F;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            for (; sc < sC32;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
            }
            if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
            if (M > 0) _tile_dpbf16ps(0, 4, 6);
            ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += 2 * apply * F;
            if (M > 0) _tile_stored(0, buf1 + 0 * 256, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            ApplyMxN<type, flush, M, 2 * apply>(buf0 + ds, dst0 + ds, dst1 + ds, bias, params), ds += 2 * apply * F;
            if (M > 1) _tile_stored(1, buf1 + 2 * 256, strideB);
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void InputConvolution1x1_2V1(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const uint16_t* weight, const float* bias, const float* params, float* buf, float* dst)
        {
            size_t dstM = a.bufH[1] - 1, dstS = a.bufH[1] * p.dstW * F, srcC = AlignHi(p.srcC, a.miK), y0 = a.bufH[0] ? yBeg : 0;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            size_t yInt = Simd::Max(yBeg, AlignLo(yEnd, a.bufH[1])), n = 32;
            size_t i1 = (yInt - yBeg) * p.dstW, in = AlignLo(i1, n), i = i1 - in;
            size_t e1 = (yEnd - yInt) * p.dstW, en = AlignLo(e1, n), e = e1 - en;

            if (yInt == yBeg)
            {
                if (en)
                {
                    e = AlignHi(e, 16), en = e1 - e;
                    InputConvolution1x1V1Ptr conv_2x2 = InputConvolution1x1_2x2V1<type, 0>;
                    InputConvolution1x1V1Ptr conv_2x1 = InputConvolution1x1_2x1V1<type, 0>;
                    InputConvolution1x1V1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2V1<type, 0> : InputConvolution1x1_1x2V1<type, 0>;
                    InputConvolution1x1V1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1V1<type, 0> : InputConvolution1x1_1x1V1<type, 0>;
                    SetTileConfFull();
                    for (size_t dc = 0; dc < maC; dc += DF)
                    {
                        size_t dC = Simd::Min(DF, maC - dc);
                        _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                        _bias[1] = _mm512_loadu_ps(bias + dc + F);
                        if (type == ::SimdConvolutionActivationPrelu)
                        {
                            _params[0] = _mm512_loadu_ps(params + dc + 0);
                            _params[1] = _mm512_loadu_ps(params + dc + F);
                        }
                        if (dC > F)
                        {
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            size_t j = 0;
                            for (; j < en; j += n)
                                conv_2x2(src0 + j * srcC, p, a, n, weight, _bias, _params, buf, dst0 + j * F, dst1 + j * F);
                            if (en < e1)
                                conv_Ex2(src0 + en * srcC, p, a, e, weight, _bias, _params, buf, dst0 + en * F, dst1 + en * F);
                        }
                        else
                        {
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            size_t j = 0;
                            for (; j < en; j += n)
                                conv_2x1(src0 + j * srcC, p, a, n, weight, _bias, _params, buf, dst0 + j * F, NULL);
                            if (en < e1)
                                conv_Ex1(src0 + en * srcC, p, a, e, weight, _bias, _params, buf, dst0 + en * F, NULL);
                        }
                        dst += a.bufH[1] * p.dstW * DF;
                        weight += srcC * DF;
                    }
                }
                else if (e1)
                {
                    InputConvolution1x1V1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2V1<type, 0> : InputConvolution1x1_1x2V1<type, 0>;
                    InputConvolution1x1V1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1V1<type, 0> : InputConvolution1x1_1x1V1<type, 0>;
                    if (e > 16)
                        SetTileConf2x2(e, 32);
                    else
                        SetTileConf1x2(e, 32);
                    for (size_t dc = 0; dc < maC; dc += DF)
                    {
                        size_t dC = Simd::Min(DF, maC - dc);
                        _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                        _bias[1] = _mm512_loadu_ps(bias + dc + F);
                        if (type == ::SimdConvolutionActivationPrelu)
                        {
                            _params[0] = _mm512_loadu_ps(params + dc + 0);
                            _params[1] = _mm512_loadu_ps(params + dc + F);
                        }
                        if (dC > F)
                        {
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            conv_Ex2(src0, p, a, e, weight, _bias, _params, buf, dst0, dst1);
                        }
                        else
                        {
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            conv_Ex1(src0, p, a, e, weight, _bias, _params, buf, dst0, NULL);
                        }
                        dst += a.bufH[1] * p.dstW * DF;
                        weight += srcC * DF;
                    }
                }
            }
            else
            {
                InputConvolution1x1V1Ptr conv_2x2 = InputConvolution1x1_2x2V1<type, 0>;
                InputConvolution1x1V1Ptr conv_2x1 = InputConvolution1x1_2x1V1<type, 0>;
                InputConvolution1x1V1Ptr conv_Ix2 = i > 16 ? InputConvolution1x1_2x2V1<type, 1> : InputConvolution1x1_1x2V1<type, 1>;
                InputConvolution1x1V1Ptr conv_Ix1 = i > 16 ? InputConvolution1x1_2x1V1<type, 1> : InputConvolution1x1_1x1V1<type, 1>;
                InputConvolution1x1V1Ptr conv_Ex2 = e > 16 ? InputConvolution1x1_2x2V1<type, 1> : InputConvolution1x1_1x2V1<type, 1>;
                InputConvolution1x1V1Ptr conv_Ex1 = e > 16 ? InputConvolution1x1_2x1V1<type, 1> : InputConvolution1x1_1x1V1<type, 1>;
                for (size_t dc = 0; dc < maC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, maC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    if (dC > F)
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_2x2(src0, p, a, n, weight, _bias, _params, buf, dst0, dst1);
                            if (in < i1)
                                conv_Ix2(src0, p, a, i, weight, _bias, _params, buf, dst0, dst1);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F, * dst1 = dst0 + dstS;
                            for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n, dst1 += F * n)
                                conv_2x2(src0, p, a, n, weight, _bias, _params, buf, dst0, dst1);
                            if (en < e1)
                                conv_Ex2(src0, p, a, e, weight, _bias, _params, buf, dst0, dst1);
                        }
                    }
                    else
                    {
                        if (yInt > yBeg)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yBeg - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yBeg & dstM) * p.dstW * F;
                            for (size_t j = 0; j < in; j += n, src0 += srcC * n, dst0 += F * n)
                                conv_2x1(src0, p, a, n, weight, _bias, _params, buf, dst0, NULL);
                            if (in < i1)
                                conv_Ix1(src0, p, a, i, weight, _bias, _params, buf, dst0, NULL);
                        }
                        if (yEnd > yInt)
                        {
                            SetTileConfFull();
                            const uint16_t* src0 = src + (yInt - y0) * p.srcW * srcC;
                            float* dst0 = dst + (yInt & dstM) * p.dstW * F;
                            for (size_t j = 0; j < en; j += n, src0 += srcC * n, dst0 += F * n)
                                conv_2x1(src0, p, a, n, weight, _bias, _params, buf, dst0, NULL);
                            if (en < e1)
                                conv_Ex1(src0, p, a, e, weight, _bias, _params, buf, dst0, NULL);
                        }
                    }
                    dst += a.bufH[1] * p.dstW * DF;
                    weight += srcC * DF;
                }
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> static void SetInputV1(const ConvParam& p, InputPtr& input)
        {
            if (Is1x1(p))
                input = InputConvolution1x1_2V1<type>;
            else
                assert(0);
        }

        void SetInputV1(const ConvParam& p, InputPtr& input)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetInputV1<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationRelu: SetInputV1<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationLeakyRelu: SetInputV1<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationRestrictRange: SetInputV1<SimdConvolutionActivationRestrictRange>(p, input); break;
            case SimdConvolutionActivationPrelu: SetInputV1<SimdConvolutionActivationPrelu>(p, input); break;
            case SimdConvolutionActivationElu: SetInputV1<SimdConvolutionActivationElu>(p, input); break;
            case SimdConvolutionActivationHswish: SetInputV1<SimdConvolutionActivationHswish>(p, input); break;
            case SimdConvolutionActivationMish: SetInputV1<SimdConvolutionActivationMish>(p, input); break;
            case SimdConvolutionActivationHardSigmoid: SetInputV1<SimdConvolutionActivationHardSigmoid>(p, input); break;
            case SimdConvolutionActivationSwish: SetInputV1<SimdConvolutionActivationSwish>(p, input); break;
            case SimdConvolutionActivationGelu: SetInputV1<SimdConvolutionActivationGelu>(p, input); break;
            }
        }
    }
#endif
}

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
#include "Simd/SimdSynetApply16b.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)  
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using OutputPtr = Base::SynetMergedConvolution16b::OutputConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void OutputConvolution_1xMx32(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, const uint16_t* weight0, 
            const __m512* bias, const __m512* params, float *buf0, float* buf1, float* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem[1]), dS = (int)a.maC, dW = 16, strideW = 64;
            int stepS = 32, strideS = dS * 2, dB = term == Term16bInterim ? (int)AlignHi(p.dstC, F) : DF, strideB = dB * 4;
            const uint16_t *src1 = src0 + 16 * dS;
            const uint16_t *weight1 = weight0 + AlignHi(srcC, a.miK) * F;

            if (zero)
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
                if (M > 0) _tile_zero(2);
                if (M > 1) _tile_zero(3);
            }
            else
            {
                int dB = (int)AlignHi(p.dstC, F), strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
                buf0 += 16 * dB;
                if (M > 0) _tile_stream_loadd(2, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(3, buf0 + F, strideB);
            }

            int sC32 = (int)srcC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32; src1 += stepS)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                _tile_stream_loadd(5, src1, strideS);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                src0 += stepS;
                _tile_stream_loadd(4, src0, strideS);
                if (M > 0) _tile_dpbf16ps(2, 5, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(3, 5, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
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
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
            buf2 += 16 * dB;
            if (M > 0) _tile_dpbf16ps(2, 5, 6);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(2, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(3, 5, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(3, buf2 + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> SIMD_INLINE void OutputConvolution_1xMx16(
            const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, int zero, const uint16_t* weight0,
            const __m512* bias, const __m512* params, float* buf0, float* buf1, float* buf2, uint8_t* dst, __mmask32 tailD)
        {
            int dD = int(p.dstC * a.elem[1]), dS = (int)a.maC, dW = 16, strideW = 64;
            int stepS = 32, strideS = dS * 2, dB = term == Term16bInterim ? (int)AlignHi(p.dstC, F) : DF, strideB = dB * 4;
            const uint16_t* src1 = src0 + 16 * dS;
            const uint16_t* weight1 = weight0 + AlignHi(srcC, a.miK) * F;

            if (zero)
            {
                if (M > 0) _tile_zero(0);
                if (M > 1) _tile_zero(1);
            }
            else
            {
                int dB = (int)AlignHi(p.dstC, F), strideB = dB * 4;
                if (M > 0) _tile_stream_loadd(0, buf0 + 0, strideB);
                if (M > 1) _tile_stream_loadd(1, buf0 + F, strideB);
            }

            int sC32 = (int)srcC - 32, aC32 = apply ? (8 * 32 / apply - 32) : 0, sc = 0, ds = 0;

            _tile_stream_loadd(4, src0, strideS);
            if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
            for (; sc < aC32;)
            {
                if (M > 1) _tile_loadd(7, weight1 + sc * dW, strideW);
                if (M > 0) _tile_dpbf16ps(0, 4, 6);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                sc += 32;
                if (M > 0) _tile_loadd(6, weight0 + sc * dW, strideW);
                if (M > 1) _tile_dpbf16ps(1, 4, 7);
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
                ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
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
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 0) _tile_stored(0, buf2 + 0, strideB);
            if (M > 1) _tile_dpbf16ps(1, 4, 7);
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (apply) ApplyMxN<term, type, flush, M, apply>(dst + ds * dD, dD, buf1 + ds * DF, bias, params, tailD), ds += apply;
            if (M > 1) _tile_stored(1, buf2 + F, strideB);
        }

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int M, int apply> void OutputConvolution_NxMx32(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, int zero, const uint16_t* weight0, const __m512 * bias, const __m512* params, float* buf0, float* buf1, uint8_t* dst, __mmask32 tailD)
        {
            int dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]), dS = (int)a.maC;
            float * buf2 = buf1 + 1024;

            if (term == Term16bInterim)
            {
                for (size_t cds = 0; cds < dstS; cds += 32)
                {
                    if (cds + 16 >= dstS)
                    {
                        cds = Simd::Min(dstS - 16, cds);
                        OutputConvolution_1xMx16<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, buf0 + cds * dB, NULL, buf0 + cds * dB, NULL, tailD);
                    }
                    else
                    {
                        cds = Simd::Min(dstS - 32, cds);
                        OutputConvolution_1xMx32<term, type, flush, M, 0>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, buf0 + cds * dB, NULL, buf0 + cds * dB, NULL, tailD);
                    }
                }
            }  
            else
            {
                size_t cds = 0, pds = 0;
                OutputConvolution_1xMx32<term, type, flush, M, 0>(src0, p, a, srcC, zero, weight0, bias, params, buf0, buf1, buf2, dst, tailD), cds += 32;
                for (; cds < dstS; pds = cds, cds += 32)
                {
                    Swap(buf1, buf2);
                    if (cds + 16 >= dstS)
                    {
                        cds = Simd::Min(dstS - 16, cds);
                        OutputConvolution_1xMx16<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, buf0 + cds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                    else
                    {
                        cds = Simd::Min(dstS - 32, cds);
                        OutputConvolution_1xMx32<term, type, flush, M, apply>(src0 + cds * dS, p, a, srcC, zero, weight0, bias, params, buf0 + cds * dB, buf1, buf2, dst + pds * dD, tailD);
                    }
                }
                uint8_t* dst1 = dst + pds * dD;
                dstS -= pds;
                size_t ds = 0, dstS8 = dstS & (~7);
                for (; ds < dstS8; ds += 8)
                    ApplyMxN<term, type, flush, M, 8>(dst1 + ds * dD, dD, buf2 + ds * DF, bias, params, tailD);
                for (; ds < dstS; ++ds)
                    ApplyMxN<term, type, flush, M, 1>(dst1 + ds * dD, dD, buf2 + ds * DF, bias, params, tailD);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_2x2V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t * dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* src1 = src0 + dS * 16, * weight1 = weight0 + AlignHi(srcC, a.miK) * F;
            float* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x2(dstS, dstC);
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

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC32;)
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            _tile_stored(3, buf1 + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply2<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_2x1V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* src1 = src0 + dS * 16;
            float* buf1 = buf0 + 16 * dB;

            if (cfg)
                SetTileConf2x1(dstS, dstC);
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

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_stream_loadd(4, src0, strideS);
            for (; sc < srcC32;)
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(2, buf1 + 0, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply1<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_1x2V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;
            const uint16_t* weight1 = weight0 + AlignHi(srcC, a.miK) * F;
            if (cfg)
                SetTileConf1x2(dstS, dstC);
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

            int srcC32 = (int)srcC - 32, sc = 0;
            _tile_loadd(6, weight0, strideW);
            for (; sc < srcC32;)
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

            _tile_stored(0, buf0 + 0, strideB);
            _tile_stored(1, buf0 + F, strideB);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply2x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply2<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        template<Term16bType term, SimdConvolutionActivationType type, int cfg> void OutputConvolution_1x1V1(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst)
        {
            int dS = (int)a.maC, dB = (int)AlignHi(p.dstC, F), dD = int(p.dstC * a.elem[1]);
            int strideS = dS * 2, strideW = 64, strideB = dB * 4;

            if (cfg)
                SetTileConf1x1(dstS, dstC);
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, buf0 + 0, strideB);
            }

            size_t sc = 0;
            for (; sc < srcC; sc += 32)
            {
                _tile_stream_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 16, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, buf0 + 0, strideB);

            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8)
                    Apply1x8<term, type>(dst + s * dD, dD, buf0 + s * dB, dB, bias, params, tailD);
                for (; s < dstS; ++s)
                    Apply1<term, type>(dst + s * dD, buf0 + s * dB, bias, params, tailD);
            }
        }

        typedef void (*OutputConvolutionV1Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* buf0, uint8_t* dst);

        template<Term16bType term, SimdConvolutionActivationType type, int flush, int apply> void OutputConvolution1x1_2V1(const uint16_t* src, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, int zero, const uint16_t* weight, const float* bias, const float* params, float* buf0, float* buf1, uint8_t* dst)
        {
            size_t n = 32, n1 = (yEnd - yBeg) * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn;
            size_t dW = AlignHi(maC, a.miK) * DF, dS = a.maC, dB = AlignHi(p.dstC, F), dD = p.dstC * a.elem[1];
            if (buf0 == NULL && p.dstT == SimdTensorData32f)
                buf0 = (float*)dst;
            __m512 _bias[2], _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            _params[1] = _mm512_set1_ps(params[1]);
            if (nn)
            {
                SetTileConfFull();
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    __mmask32 tailD = term == Term16bLast16b ? TailMask32(dC) : (__mmask32)TailMask16(dC - AlignLo(dC - 1, 16));
                    const uint16_t* s = src;
                    float* b = buf0 + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                    size_t i = 0;
                    if (dC > F)
                        OutputConvolution_NxMx32<term, type, flush, 2, apply>(s + i * dS, p, a, maC, n1, zero, weight, _bias, _params, b + i * dB, buf1, d + i * dD, tailD);
                    else
                        OutputConvolution_NxMx32<term, type, flush, 1, apply>(s + i * dS, p, a, maC, n1, zero, weight, _bias, _params, b + i * dB, buf1, d + i * dD, tailD);
                    weight += dW;
                }
            }
            else
            {
                OutputConvolutionV1Ptr tail_2 = m > 16 ? OutputConvolution_2x2V1<term, type, 0> : OutputConvolution_1x2V1<term, type, 0>;
                OutputConvolutionV1Ptr tail_1 = m > 16 ? OutputConvolution_2x1V1<term, type, 0> : OutputConvolution_1x1V1<term, type, 0>;
                if (m > 16)
                    SetTileConf2x2(m, 32);
                else
                    SetTileConf1x2(m, 32);
                for (size_t dc = 0; dc < p.dstC; dc += DF)
                {
                    size_t dC = Simd::Min(DF, p.dstC - dc);
                    _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                    _bias[1] = _mm512_loadu_ps(bias + dc + F);
                    if (type == ::SimdConvolutionActivationPrelu)
                    {
                        _params[0] = _mm512_loadu_ps(params + dc + 0);
                        _params[1] = _mm512_loadu_ps(params + dc + F);
                    }
                    const uint16_t* s = src;
                    float* b = buf0 + dc + yBeg * p.dstW * dB;
                    uint8_t* d = dst + (dc + yBeg * p.dstW * p.dstC) * a.elem[1];
                    size_t i = 0;
                    if (dC > F)
                         tail_2(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    else
                         tail_1(s + i * dS, p, a, maC, m, dC, zero, weight, _bias, _params, b + i * dB, d + i * dD);
                    weight += dW;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> static void SetOutputV1(const ConvParam& p, const AlgParam& a, OutputPtr& output)
        {
            size_t maC = p.srcC - AlignLoAny(p.srcC, a.maC);
            if (maC > 224)
                output = OutputConvolution1x1_2V1<term, type, 1, 1>;
            else if (maC > 96)
                output = OutputConvolution1x1_2V1<term, type, 1, 2>;
            else if (maC > 32)
                output = OutputConvolution1x1_2V1<term, type, 1, 4>;
            else
                output = OutputConvolution1x1_2V1<term, type, 1, 8>;
        }

        template<SimdConvolutionActivationType type> static void SetOutputV1(const ConvParam& p, const AlgParam& a, OutputPtr* output)
        {
            if (p.dstT == SimdTensorData16b)
                SetOutputV1<Term16bLast16b, type>(p, a, output[0]);
            else
                SetOutputV1<Term16bLast32f, type>(p, a, output[0]);
            output[1] = OutputConvolution1x1_2V1<Term16bInterim, SimdConvolutionActivationIdentity, 0, 0>;
        }

        void SetOutputV1(const ConvParam& p, const AlgParam& a, OutputPtr* output)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetOutputV1<SimdConvolutionActivationRestrictRange>(p, a, output); break;
            case SimdConvolutionActivationRelu: SetOutputV1<SimdConvolutionActivationRestrictRange>(p, a, output); break;
            case SimdConvolutionActivationLeakyRelu: SetOutputV1<SimdConvolutionActivationPrelu>(p, a, output); break;
            case SimdConvolutionActivationRestrictRange: SetOutputV1<SimdConvolutionActivationRestrictRange>(p, a, output); break;
            case SimdConvolutionActivationPrelu: SetOutputV1<SimdConvolutionActivationPrelu>(p, a, output); break;
            case SimdConvolutionActivationElu: SetOutputV1<SimdConvolutionActivationElu>(p, a, output); break;
            case SimdConvolutionActivationHswish: SetOutputV1<SimdConvolutionActivationHswish>(p, a, output); break;
            case SimdConvolutionActivationMish: SetOutputV1<SimdConvolutionActivationMish>(p, a, output); break;
            case SimdConvolutionActivationHardSigmoid: SetOutputV1<SimdConvolutionActivationHardSigmoid>(p, a, output); break;
            case SimdConvolutionActivationSwish: SetOutputV1<SimdConvolutionActivationSwish>(p, a, output); break;
            case SimdConvolutionActivationGelu: SetOutputV1<SimdConvolutionActivationGelu>(p, a, output); break;
            }
        }
    }
#endif
}

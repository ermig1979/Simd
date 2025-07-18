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
#include "Simd/SimdSynetQuantizedConvolution.h"
#include "Simd/SimdSynetQuantizeLinear.h"
#include "Simd/SimdSynetConvolution8iCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdLog.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::AlgParam AlgParam;
        typedef Base::SynetQuantizedConvolutionNhwcSpecV0::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        static void QuantizedConvolutionNhwcSpecV0_32x32(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX, dW = 1024, strideW = 64, strideD = dD * 4;
            const int8_t* weight1 = weight0 + a.K * F;
            const uint8_t* src1 = src0 + 16 * dX;
            int32_t* dst1 = dst0 + 16 * dD;

            if (update)
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
                _tile_stream_loadd(3, dst1 + F, strideD);
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
            for (int i = 0; i < n1; ++i)
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
                weight1 += dW;
            }
            _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);

            _tile_dpbusd(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbusd(1, 4, 7);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + F, dD);

            _tile_dpbusd(2, 5, 6);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst1 + 0, dD);

            _tile_dpbusd(3, 5, 7);
            _tile_stored(3, dst1 + F, strideD);
            TileMoveToMemory(dst1 + F, dD);
        }

        static void QuantizedConvolutionNhwcSpecV0_32x16(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX, dW = 1024, strideW = 64, strideD = dD * 4;
            const uint8_t* src1 = src0 + 16 * dX;
            int32_t* dst1 = dst0 + 16 * dD;

            if (update)
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
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
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbusd(2, 5, 6);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst1 + 0, dD);
        }

        static void QuantizedConvolutionNhwcSpecV0_16x32(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX, dW = 1024, strideW = 64, strideD = dD * 4;
            const int8_t* weight1 = weight0 + a.K * F;

            if (update)
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
            }
            else
            {
                _tile_zero(0);
                _tile_zero(1);
            }

            int n1 = (int)nK - 1;
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbusd(0, 4, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbusd(1, 4, 7);
                weight1 += dW;
            }
            _tile_stream_loadd(4, src0 + offs[n1], strideS);
            _tile_loadd(7, weight1, strideW);

            _tile_dpbusd(0, 4, 6);
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);

            _tile_dpbusd(1, 4, 7);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + F, dD);
        }

        static void QuantizedConvolutionNhwcSpecV0_16x16(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int update, const int8_t* weight0, int32_t* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, strideS = dX, dW = 1024, strideW = 64, strideD = dD * 4;

            if (update)
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
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

            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
        }

        typedef void (*QuantizedConvolutionNhwcSpecV0Ptr)(const uint8_t* src0, const ConvParam& p, const AlgParam& a, const int* offset, size_t nK, int update, const int8_t* weight0, int32_t* dst0);

        static void QuantizedConvolutionNhwcSpecV0_2(const uint8_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t nK, int update, const int8_t* weight, int32_t* dst)
        {
            size_t n1 = dstH * a.srcW - a.gapH, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            QuantizedConvolutionNhwcSpecV0Ptr body_2 = QuantizedConvolutionNhwcSpecV0_32x32;
            QuantizedConvolutionNhwcSpecV0Ptr tail_2 = m > 16 ? QuantizedConvolutionNhwcSpecV0_32x32 : QuantizedConvolutionNhwcSpecV0_16x32;
            QuantizedConvolutionNhwcSpecV0Ptr body_1 = QuantizedConvolutionNhwcSpecV0_32x16;
            QuantizedConvolutionNhwcSpecV0Ptr tail_1 = m > 16 ? QuantizedConvolutionNhwcSpecV0_32x16 : QuantizedConvolutionNhwcSpecV0_16x16;

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        body_2(src + i * dS, p, a, offs, nK, update, weight, dst + i * dD);
                    if (m)
                        tail_2(src + i * dS, p, a, offs, nK, update, weight, dst + i * dD);
                }
                else
                {
                    for (; i < nn; i += n)
                        body_1(src + i * dS, p, a, offs, nK, update, weight, dst + i * dD);
                    if (m)
                        tail_1(src + i * dS, p, a, offs, nK, update, weight, dst + i * dD);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        SynetQuantizedConvolutionNhwcSpecV0::SynetQuantizedConvolutionNhwcSpecV0(const ConvParam& p)
            : Avx512vnni::SynetQuantizedConvolutionNhwcSpecV0(p)
        {
            SetAlgParam(F, F * 2, F * 2, F * 4, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
            AlgParam& a = _alg;
            _convolution = QuantizedConvolutionNhwcSpecV0_2;
        }
    }
#endif
}

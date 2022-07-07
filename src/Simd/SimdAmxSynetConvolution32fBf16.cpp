/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdAvx512bf16.h"
#include "Simd/SimdAmx.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMX_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE)
    namespace Amx
    {
        typedef Base::SynetConvolution32fBf16Nhwc::AlgParam AlgParam;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvertPtr Convert;
        typedef Base::SynetConvolution32fBf16Nhwc::ConvolutionPtr Convolution;

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, size_t dstW, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY, 
                dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX, dW = AlignHi(srcC, 2) * 32;
            size_t srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + dS * 16, * weight1 = weight0 + 32;

            TileConf body, tail;
            body.rows[0] = 16;
            body.rows[1] = 16;
            body.rows[2] = uint8_t(dstW - 16);
            body.rows[3] = uint8_t(dstW - 16);
            body.rows[4] = 16;
            body.rows[5] = uint8_t(dstW - 16);
            body.rows[6] = 16;
            body.rows[7] = 16;
            body.colsb[0] = 64;
            body.colsb[1] = uint16_t((dstC - 16) * 4);
            body.colsb[2] = 64;
            body.colsb[3] = uint16_t((dstC - 16) * 4);
            body.colsb[4] = 64;
            body.colsb[5] = 64;
            body.colsb[6] = 64;
            body.colsb[7] = uint16_t((dstC - 16) * 4);
            if (srcC32 < srcC)
            {
                size_t tailC = AlignHi(srcC - srcC32, 2);
                tail = body;
                tail.rows[6] = uint8_t(tailC / 2);
                tail.rows[7] = uint8_t(tailC / 2);
                tail.colsb[4] = uint16_t(tailC * 2);
                tail.colsb[5] = uint16_t(tailC * 2);
            }
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_loadconfig(&body);
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_loadd(3, dst + 16 * dD + F, strideD);
            }
            for (size_t ky = 0; ky < kY; ++ky)
            {
                for (size_t kx = 0; kx < kX; ++kx)
                {
                    size_t sc = 0, offs = ky * dY + kx * dX;
                    _tile_loadconfig(&body);
                    for (;sc < srcC32; sc += 32)
                    {
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    if (sc < srcC)
                    {
                        _tile_loadconfig(&tail);
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                    }
                    weight0 += dW;
                    weight1 += dW;
                }
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstW8 = AlignLo(dstW, 8), w = 0;
                for (; w < dstW8; w += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for (; w < dstW; w += 1, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_1x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, size_t dstW, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY,
                dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX, dW = AlignHi(srcC, 2) * 32;
            size_t srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* weight1 = weight0 + 32;

            TileConf body, tail;
            body.rows[0] = uint8_t(dstW);
            body.rows[1] = uint8_t(dstW);
            body.rows[4] = uint8_t(dstW);
            body.rows[6] = 16;
            body.rows[7] = 16;
            body.colsb[0] = 64;
            body.colsb[1] = uint16_t((dstC - 16) * 4);
            body.colsb[4] = 64;
            body.colsb[6] = 64;
            body.colsb[7] = uint16_t((dstC - 16) * 4);
            if (srcC32 < srcC)
            {
                size_t tailC = AlignHi(srcC - srcC32, 2);
                tail = body;
                tail.rows[6] = uint8_t(tailC / 2);
                tail.rows[7] = uint8_t(tailC / 2);
                tail.colsb[4] = uint16_t(tailC * 2);
            }
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_loadconfig(&body);
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
            }
            for (size_t ky = 0; ky < kY; ++ky)
            {
                for (size_t kx = 0; kx < kX; ++kx)
                {
                    size_t sc = 0, offs = ky * dY + kx * dX;
                    _tile_loadconfig(&body);
                    for (; sc < srcC32; sc += 32)
                    {
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                    }
                    if (sc < srcC)
                    {
                        _tile_loadconfig(&tail);
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1 + sc * 32, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                    }
                    weight0 += dW;
                    weight1 += dW;
                }
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstW8 = AlignLo(dstW, 8), w = 0;
                for (; w < dstW8; w += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for (; w < dstW; ++w, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, size_t dstW, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY,
                dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX, dW = AlignHi(srcC, 2) * 32;
            size_t srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + dS * 16;

            TileConf body, tail;
            body.rows[0] = 16;
            body.rows[2] = uint8_t(dstW - 16);
            body.rows[4] = 16;
            body.rows[5] = uint8_t(dstW - 16);
            body.rows[6] = 16;
            body.colsb[0] = uint16_t(dstC * 4);
            body.colsb[2] = uint16_t(dstC * 4);
            body.colsb[4] = 64;
            body.colsb[5] = 64;
            body.colsb[6] = uint16_t(dstC * 4);
            if (srcC32 < srcC)
            {
                size_t tailC = AlignHi(srcC - srcC32, 2);
                tail = body;
                tail.rows[6] = uint8_t(tailC / 2);
                tail.colsb[4] = uint16_t(tailC * 2);
                tail.colsb[5] = uint16_t(tailC * 2);
            }
            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_loadconfig(&body);
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
            }
            for (size_t ky = 0; ky < kY; ++ky)
            {
                for (size_t kx = 0; kx < kX; ++kx)
                {
                    size_t sc = 0, offs = ky * dY + kx * dX;
                    _tile_loadconfig(&body);
                    for (; sc < srcC32; sc += 32)
                    {
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                    }
                    if (sc < srcC)
                    {
                        _tile_loadconfig(&tail);
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(5, src1 + offs + sc, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                    }
                    weight0 += dW;
                }
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstW8 = AlignLo(dstW, 8), w = 0;
                for (; w < dstW8; w += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; w < dstW; ++w, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_1x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstC, size_t dstW, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dS = srcC * p.strideX, dY = (p.srcW + p.padX + p.padW) * srcC * p.dilationY,
                dX = srcC * p.dilationX, dD = p.dstC, kY = p.kernelY, kX = p.kernelX, dW = AlignHi(srcC, 2) * 32;
            size_t srcC32 = AlignLo(srcC, 32);
            int strideS = (int)dS * 2, strideW = 128, strideD = (int)dD * 4;

            TileConf body, tail;
            body.rows[0] = uint8_t(dstW);
            body.rows[4] = uint8_t(dstW);
            body.rows[6] = 16;
            body.colsb[0] = uint16_t(dstC * 4);
            body.colsb[4] = 64;
            body.colsb[6] = uint16_t(dstC * 4);
            if (srcC32 < srcC)
            {
                size_t tailC = AlignHi(srcC - srcC32, 2);
                tail = body;
                tail.rows[6] = uint8_t(tailC / 2);
                tail.colsb[4] = uint16_t(tailC * 2);
            }
            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_loadconfig(&body);
                _tile_loadd(0, dst + 0, strideD);
            }
            for (size_t ky = 0; ky < kY; ++ky)
            {
                for (size_t kx = 0; kx < kX; ++kx)
                {
                    size_t sc = 0, offs = ky * dY + kx * dX;
                    _tile_loadconfig(&body);
                    for (; sc < srcC32; sc += 32)
                    {
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                    }
                    if (sc < srcC)
                    {
                        _tile_loadconfig(&tail);
                        _tile_loadd(4, src0 + offs + sc, strideS);
                        _tile_loadd(6, weight0 + sc * 32, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                    }
                    weight0 += dW;
                }
            }
            _tile_stored(0, dst + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstW8 = AlignLo(dstW, 8), w = 0;
                for (; w < dstW8; w += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; w < dstW; ++w, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        typedef void (*ConvolutionBf16NhwcConvPtr)(const uint16_t* src0, const ConvParam32f& p, size_t srcC, size_t dstC, 
            size_t dstW, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst);

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcConv_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 32, dstWn = AlignLoAny(p.dstW, n), m = p.dstW - dstWn;
            size_t dW = p.kernelY * p.kernelX * AlignHi(srcC, 2) * DF, dD = p.dstW * p.dstC;
            size_t dS = p.strideY * (p.srcW + p.padX + p.padW) * srcC;

            ConvolutionBf16NhwcConvPtr body_2 = ConvolutionBf16NhwcConv_2x2<type>;
            ConvolutionBf16NhwcConvPtr tail_2 = m > 16 ? ConvolutionBf16NhwcConv_2x2<type> : ConvolutionBf16NhwcConv_1x2<type>;
            ConvolutionBf16NhwcConvPtr body_1 = ConvolutionBf16NhwcConv_2x1<type>;
            ConvolutionBf16NhwcConvPtr tail_1 = m > 16 ? ConvolutionBf16NhwcConv_2x1<type> : ConvolutionBf16NhwcConv_1x1<type>;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                if (dC > F)
                {
                    for (size_t dy = 0; dy < dstH; dy++)
                    {
                        float* d = dst + dy * dD;
                        const uint16_t* s = src + dy * dS;
                        size_t dx = 0;
                        for (; dx < dstWn; dx += n, d += n * p.dstC, s += n * p.strideX * srcC)
                            body_2(s, p, srcC, dC, n, zero, weight, _bias, _params, d);
                        if(m)
                            tail_2(s, p, srcC, dC, m, zero, weight, _bias, _params, d);
                    }
                }
                else
                {
                    for (size_t dy = 0; dy < dstH; dy++)
                    {
                        float* d = dst + dy * dD;
                        const uint16_t* s = src + dy * dS;
                        size_t dx = 0;
                        for (; dx < dstWn; dx += n, d += n * p.dstC, s += n * p.strideX * srcC)
                            body_1(s, p, srcC, dC, n, zero, weight, _bias, _params, d);
                        if (m)
                            tail_1(s, p, srcC, dC, m, zero, weight, _bias, _params, d);
                    }
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t srcC32 = AlignLo(srcC, 32);
            int dD = (int)p.dstC, strideS = (int)srcC * 2, strideW = 128, strideD = dD * 4;
            const uint16_t* src1 = src0 + srcC * 16, *weight1 = weight0 + 32;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[1] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[3] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((dstC - 16) * 4);
            conf.colsb[2] = 64;
            conf.colsb[3] = uint16_t((dstC - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((dstC - 16) * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
                _tile_zero(2);
                _tile_zero(3);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
                _tile_loadd(3, dst + 16 * dD + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            if(sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.rows[7] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                conf.colsb[5] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
                _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            _tile_stored(3, dst + 16 * dD + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for(; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* src1 = src0 + srcC * 16;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[2] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(2, dst + 16 * dD + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                conf.colsb[5] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(5, src1 + sc, strideS);
                _tile_dpbf16ps(2, 5, 6);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(2, dst + 16 * dD + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x2(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;
            const uint16_t* weight1 = weight0 + 32;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[1] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t(dstC - 16) * 4;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t(dstC - 16) * 4;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
                _tile_loadd(1, dst + F, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.rows[7] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
                _tile_loadd(7, weight1 + sc * 32, strideW);
                _tile_dpbf16ps(1, 4, 7);
            }
            _tile_stored(0, dst + 0, strideD);
            _tile_stored(1, dst + F, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC - F);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply2x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply2<type>(dst, bias, params, tailD);
            }
        }

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_1x1(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst)
        {
            size_t dD = p.dstC, srcC32 = AlignLo(srcC, 32);
            int strideS = (int)srcC * 2, strideW = 128, strideD = (int)dD * 4;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(dstC * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(dstC * 4);
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_loadd(0, dst + 0, strideD);
            }
            size_t sc = 0;
            for (; sc < srcC32; sc += 32)
            {
                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            if (sc < srcC)
            {
                size_t tailC = AlignHi(srcC - sc, 2);
                conf.rows[6] = uint8_t(tailC / 2);
                conf.colsb[4] = uint16_t(tailC * 2);
                _tile_loadconfig(&conf);

                _tile_loadd(4, src0 + sc, strideS);
                _tile_loadd(6, weight0 + sc * 32, strideW);
                _tile_dpbf16ps(0, 4, 6);
            }
            _tile_stored(0, dst + 0, strideD);
            if (type)
            {
                __mmask16 tailD = TailMask16(dstC);
                size_t dstS8 = AlignLo(dstS, 8), s = 0;
                for (; s < dstS8; s += 8, dst += 8 * dD)
                    Apply1x8<type>(dst, dD, bias, params, tailD);
                for (; s < dstS; ++s, dst += dD)
                    Apply1<type>(dst, bias, params, tailD);
            }
        }

        typedef void (*ConvolutionBf16NhwcGemmPtr)(const uint16_t* src0, const ConvParam32f& p,
            size_t srcC, size_t dstS, size_t dstC, int zero, const uint16_t* weight0, const __m512* bias, const __m512* params, float* dst);

        template<SimdConvolutionActivationType type> void ConvolutionBf16NhwcGemm_2(const uint16_t* src, const ConvParam32f& p,
            size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, const float* bias, const float* params, float* dst)
        {
            size_t n = 32, n1 = dstH * p.dstW, nn = AlignLoAny(n1, n), m = n1 - nn, dW = AlignHi(srcC, 2) * DF;
            ConvolutionBf16NhwcGemmPtr body_2 = ConvolutionBf16NhwcGemm_2x2<type>;
            ConvolutionBf16NhwcGemmPtr tail_2 = m > 16 ? ConvolutionBf16NhwcGemm_2x2<type> : ConvolutionBf16NhwcGemm_1x2<type>;
            ConvolutionBf16NhwcGemmPtr body_1 = ConvolutionBf16NhwcGemm_2x1<type>;
            ConvolutionBf16NhwcGemmPtr tail_1 = m > 16 ? ConvolutionBf16NhwcGemm_2x1<type> : ConvolutionBf16NhwcGemm_1x1<type>;

            __m512 _params[2], _bias[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                _bias[0] = _mm512_loadu_ps(bias + dc + 0);
                _bias[1] = _mm512_loadu_ps(bias + dc + F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + dc + 0);
                    _params[1] = _mm512_loadu_ps(params + dc + F);
                }
                float* d = dst;
                const uint16_t* s = src;
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                        body_2(s, p, srcC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_2(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                else
                {
                    for (; i < nn; i += n, s += n * srcC, d += n * p.dstC)
                        body_1(s, p, srcC, n, dC, zero, weight, _bias, _params, d);
                    if (m)
                        tail_1(s, p, srcC, m, dC, zero, weight, _bias, _params, d);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <SimdConvolutionActivationType type> SIMD_INLINE void Set(const ConvParam32f& p, const AlgParam& a, Convolution* convolutions)
        {
            if (p.Is1x1() || a.mode)
            {
                convolutions[TermLast] = ConvolutionBf16NhwcGemm_2<type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcGemm_2<SimdConvolutionActivationIdentity>;
            }
            else
            {
                convolutions[TermLast] = ConvolutionBf16NhwcConv_2<type>;
                convolutions[TermInterim] = ConvolutionBf16NhwcConv_2<SimdConvolutionActivationIdentity>;
            }
        }

        SynetConvolution32fBf16Nhwc::SynetConvolution32fBf16Nhwc(const ConvParam32f & p)
#if defined(SIMD_AMX_EMULATE)
            : Avx512bw::SynetConvolution32fBf16Nhwc(p)
#else
            : Avx512bf16::SynetConvolution32fBf16Nhwc(p)
#endif
        {
            size_t microD = 16 * 2;
            size_t microHW = 16 * 2;
            size_t microC = 16 * 2;
            SetAlgParam(microD, microHW, microC, Base::AlgCacheL1(), Base::AlgCacheL2(), Base::AlgCacheL3());
#if defined(SIMD_AMX_EMULATE)
            if (_alg.mode)
                _convert = Avx512bw::ConvolutionBf16NhwcConvertGemm;
            else
                _convert = Avx512bw::ConvolutionBf16NhwcConvertConv;
#else
            if (_alg.mode)
                _convert = Avx512bf16::ConvolutionBf16NhwcConvertGemm;
            else
                _convert = Avx512bf16::ConvolutionBf16NhwcConvertConv;
#endif
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRelu: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationLeakyRelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationRestrictRange: Set<SimdConvolutionActivationRestrictRange>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationPrelu: Set<SimdConvolutionActivationPrelu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationElu: Set<SimdConvolutionActivationElu>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHswish: Set<SimdConvolutionActivationHswish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationMish: Set<SimdConvolutionActivationMish>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationHardSigmoid: Set<SimdConvolutionActivationHardSigmoid>(p, _alg, _convolutions); break;
            case SimdConvolutionActivationSwish: Set<SimdConvolutionActivationSwish>(p, _alg, _convolutions); break;
            default: assert(0);
            }
        }

        //-----------------------------------------------------------------------------------------

        void* SynetConvolution32fInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility)
        {
            ConvParam32f param(batch, conv, compatibility);
            if (!param.Valid())
                return NULL;
            else if (Base::Bf16Soft(compatibility) || Base::Bf16Hard(compatibility))
            {
                if (Base::SynetConvolution32fBf16Nhwc::Preferable(param))
                    return new Amx::SynetConvolution32fBf16Nhwc(param);
                else
                    return new Base::SynetConvolution32fBf16Gemm(param);
            }
#if defined(SIMD_AMX_EMULATE)
            return Avx512bw::SynetConvolution32fInit(batch, conv, compatibility);
#else
            return Avx512bf16::SynetConvolution32fInit(batch, conv, compatibility);
#endif
        }
    }
#endif
}

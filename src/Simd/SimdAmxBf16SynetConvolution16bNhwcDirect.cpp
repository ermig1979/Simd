/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))
    namespace AmxBf16
    {
        typedef Base::SynetConvolution16bNhwcDirect::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcDirect::PostprocessPtr PostprocessPtr;

        //-----------------------------------------------------------------------------------------

        static void Convert16bNhwcDirect(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint16_t* dst)
        {
            assert(a.microC == DF);
            const float* src = (float*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = p.padY * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += p.padY * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (p.padX)
                {
                    for (size_t s = 0; s < p.padX; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padX * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD);
                    if (tailC)
                        AmxBf16::Float32ToBFloat16(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
                if (p.padW)
                {
                    for (size_t s = 0; s < p.padW; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padW * sD;
                }
            }
            if (dyEnd == p.dstH)
            {
                for (size_t s = 0, n = p.padH * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += p.padH * a.srcW * sD;
            }
        }

        static void Reorder16bNhwcDirect(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, uint16_t* dst)
        {
            assert(a.microC == DF);
            const uint16_t* src = (uint16_t*)src8;
            size_t srcCDF = Simd::AlignLo(p.srcC, DF);
            __mmask32 tailC = TailMask32(p.srcC - srcCDF);
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t cD = a.batch * a.srcH * a.srcW, sD = a.microC;
            if (dyBeg == 0)
            {
                for (size_t s = 0, n = p.padY * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += p.padY * a.srcW * sD;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * sD;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (p.padX)
                {
                    for (size_t s = 0; s < p.padX; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padX * sD;
                }
                for (size_t sx = 0; sx < p.srcW; ++sx)
                {
                    size_t sc = 0;
                    for (; sc < srcCDF; sc += DF)
                        Avx512bw::Copy(src + sc, dst + sc * cD);
                    if (tailC)
                        Avx512bw::Copy(src + sc, dst + sc * cD, tailC);
                    src += p.srcC;
                    dst += sD;
                }
                if (p.padW)
                {
                    for (size_t s = 0; s < p.padW; ++s)
                        for (size_t c = 0; c < a.srcC; c += a.microC)
                            Avx512bw::SetZero(dst + c * cD + s * sD);
                    dst += p.padW * sD;
                }
            }
            if (dyEnd == p.dstH)
            {
                for (size_t s = 0, n = p.padH * a.srcW; s < n; ++s)
                    for (size_t c = 0; c < a.srcC; c += a.microC)
                        Avx512bw::SetZero(dst + c * cD + s * sD);
                dst += p.padH * a.srcW * sD;
            }
        }

        //-----------------------------------------------------------------------------------------

        static void Convolution16bNhwcDirect_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, dY = (int)a.srcW * dX, dC = dY * int(a.srcH * a.batch);
            int kX = (int)p.kernelX, kY = (int)p.kernelY, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.srcC * a.K * F;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

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
            conf.colsb[1] = 64;
            conf.colsb[2] = 64;
            conf.colsb[3] = 64;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = 64;
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
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
                _tile_stream_loadd(3, dst1 + F, strideD);
            }
            for (size_t c = 0, offsS = 0; c < srcC; c += dX, offsS += dC)
            {
                for (size_t y = 0, offsY = offsS; y < kY; y += 1, offsY += dY)
                {
                    for (size_t offsX = offsY, endX = offsY + kX * dX; offsX < endX; offsX += dX)
                    {
                        _tile_stream_loadd(4, src0 + offsX, strideS);
                        _tile_loadd(6, weight0, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_stream_loadd(5, src1 + offsX, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_dpbf16ps(3, 5, 7);
                        weight0 += dW;
                        weight1 += dW;
                    }
                }
            }
            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(1, dst0 + F, strideD);
            _tile_stored(2, dst1 + 0, strideD);
            _tile_stored(3, dst1 + F, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst0 + F, dD);
            TileMoveToMemory(dst1 + 0, dD);
            TileMoveToMemory(dst1 + F, dD);
        }

        static void Convolution16bNhwcDirect_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, dY = (int)a.srcW * dX, dC = dY * int(a.srcH * a.batch);
            int kX = (int)p.kernelX, kY = (int)p.kernelY, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* src1 = src0 + 16 * dX;
            float* dst1 = dst0 + 16 * dD;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(dstS - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(dstS - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = 64;
            conf.colsb[2] = 64;
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(2);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(2, dst1 + 0, strideD);
            }
            for (size_t c = 0, offsS = 0; c < srcC; c += dX, offsS += dC)
            {
                for (size_t y = 0, offsY = offsS; y < kY; y += 1, offsY += dY)
                {
                    for (size_t x = 0, offsX = offsY; x < kX; x += 1, offsX += dX)
                    {
                        _tile_stream_loadd(4, src0 + offsX, strideS);
                        _tile_loadd(6, weight0, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_stream_loadd(5, src1 + offsX, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        weight0 += dW;
                    }
                }
            }
            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst1 + 0, dD);
        }

        static void Convolution16bNhwcDirect_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, dY = (int)a.srcW * dX, dC = dY * int(a.srcH * a.batch);
            int kX = (int)p.kernelX, kY = (int)p.kernelY, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.srcC * a.K * F;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[1] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = 64;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = 64;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
                _tile_zero(1);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
                _tile_stream_loadd(1, dst0 + F, strideD);
            }
            for (size_t c = 0, offsS = 0; c < srcC; c += dX, offsS += dC)
            {
                for (size_t y = 0, offsY = offsS; y < kY; y += 1, offsY += dY)
                {
                    for (size_t x = 0, offsX = offsY; x < kX; x += 1, offsX += dX)
                    {
                        _tile_stream_loadd(4, src0 + offsX, strideS);
                        _tile_loadd(6, weight0, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_loadd(7, weight1, strideW);
                        _tile_dpbf16ps(1, 4, 7);
                        weight0 += dW;
                        weight1 += dW;
                    }
                }
            }
            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst0 + F, dD);
        }

        static void Convolution16bNhwcDirect_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dX = (int)a.microC, dY = (int)a.srcW * dX, dC = dY * int(a.srcH * a.batch);
            int kX = (int)p.kernelX, kY = (int)p.kernelY, strideS = dX * 2, dW = 512, strideW = 64, strideD = dD * 4;

            TileConf conf;
            conf.rows[0] = uint8_t(dstS);
            conf.rows[4] = uint8_t(dstS);
            conf.rows[6] = 16;
            conf.colsb[0] = 64;
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            _tile_loadconfig(&conf);

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
            }
            for (size_t c = 0, offsS = 0; c < srcC; c += dX, offsS += dC)
            {
                for (size_t y = 0, offsY = offsS; y < kY; y += 1, offsY += dY)
                {
                    for (size_t x = 0, offsX = offsY; x < kX; x += 1, offsX += dX)
                    {
                        _tile_stream_loadd(4, src0 + offsX, strideS);
                        _tile_loadd(6, weight0, strideW);
                        _tile_dpbf16ps(0, 4, 6);
                        weight0 += dW;
                    }
                }
            }
            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
        }

        typedef void (*Convolution16bNhwcDirectPtr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, size_t srcC, size_t dstS, int zero, const uint16_t* weight0, float* dst0);

        static void Convolution16bNhwcDirect_2(const uint16_t* src, const ConvParam& p, const AlgParam& a, size_t dstC, size_t dstH, size_t srcC, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW + 1 - p.kernelX, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.srcC * a.K * DF;
            size_t dD = a.macroD, dS = a.microC;
            Convolution16bNhwcDirectPtr body_2 = Convolution16bNhwcDirect_32x32;
            Convolution16bNhwcDirectPtr tail_2 = m > 16 ? Convolution16bNhwcDirect_32x32 : Convolution16bNhwcDirect_16x32;
            Convolution16bNhwcDirectPtr body_1 = Convolution16bNhwcDirect_32x16;
            Convolution16bNhwcDirectPtr tail_1 = m > 16 ? Convolution16bNhwcDirect_32x16 : Convolution16bNhwcDirect_16x16;

            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        body_2(src + i * dS, p, a, srcC, n, zero, weight, dst + i * dD);
                    if (m)
                        tail_2(src + i * dS, p, a, srcC, m, zero, weight, dst + i * dD);
                }
                else
                {
                    for (; i < nn; i += n)
                        body_1(src + i * dS, p, a, srcC, n, zero, weight, dst + i * dD);
                    if (m)
                        tail_1(src + i * dS, p, a, srcC, m, zero, weight, dst + i * dD);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type>  void Postprocess16bNhwcDirect(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F);
            __mmask16 tailD = TailMask16(dstC - dstCF);
            size_t rowGap = (p.kernelX - 1) * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                        Avx512bw::Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tailD)
                        Avx512bw::Postprocess<term, type>(src, bias, params, dc, dst, tailD);
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        template<SimdConvolutionActivationType type> void SetPostprocess(const ConvParam& p, const AlgParam& a, PostprocessPtr & postprocess)
        {
            if (p.dstT == SimdTensorData16b)
                postprocess = Postprocess16bNhwcDirect<Term16bLast16b, type>;
            else
                postprocess = Postprocess16bNhwcDirect<Term16bLast32f, type>;
        }

        //-----------------------------------------------------------------------------------------

        SynetConvolution16bNhwcDirect::SynetConvolution16bNhwcDirect(const ConvParam & p)
            : Avx512bw::SynetConvolution16bNhwcDirect(p)
        {
            SetAlgParam(F, F * 2, 32, F * 2, int(Base::AlgCacheL1() * 1.05), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcDirect;
            else
                _preprocess = Convert16bNhwcDirect;
            _convolution = Convolution16bNhwcDirect_2;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRelu: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationLeakyRelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationRestrictRange: SetPostprocess<SimdConvolutionActivationRestrictRange>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationPrelu: SetPostprocess<SimdConvolutionActivationPrelu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationElu: SetPostprocess<SimdConvolutionActivationElu>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHswish: SetPostprocess<SimdConvolutionActivationHswish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationMish: SetPostprocess<SimdConvolutionActivationMish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationHardSigmoid: SetPostprocess<SimdConvolutionActivationHardSigmoid>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationSwish: SetPostprocess<SimdConvolutionActivationSwish>(p, _alg, _postprocess); break;
            case SimdConvolutionActivationGelu: SetPostprocess<SimdConvolutionActivationGelu>(p, _alg, _postprocess); break;
            default: assert(0);
            }
        }
    }
#endif
}

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
#include "Simd/SimdAmxBf16.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))) && defined(SIMD_SYNET_ENABLE) 
    namespace AmxBf16
    {
        typedef Base::SynetConvolution16bNhwcSpecV1::AlgParam AlgParam;
        typedef Base::SynetConvolution16bNhwcSpecV1::PostprocessPtr PostprocessPtr;

        //-------------------------------------------------------------------------------------------------

        static void Convert16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const float* src = (float*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t padH = a.padH * p.srcC, padHDF = AlignLo(padH, DF);
            size_t sizeH = p.srcW * p.srcC, sizeHDF = AlignLo(sizeH, DF);
            __mmask32 padM = TailMask32(padH - padHDF), sizeM = TailMask32(sizeH - sizeHDF);
            if (dyBeg == 0)
            {
                size_t padV = a.padV * a.srcW * p.srcC;
                memset(dst, 0, padV * sizeof(uint16_t));
                dst += padV;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * p.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (padH)
                {
                    size_t i = 0;
                    for (; i < padHDF; i += DF)
                        Avx512bw::SetZero(dst + i);
                    if (padM)
                        Avx512bw::SetZero(dst + i, padM);
                    dst += padH;
                }
                size_t x = 0;
                for (; x < sizeH; x += DF)
                    AmxBf16::Float32ToBFloat16(src + x, dst + x);
                if (sizeM)
                    AmxBf16::Float32ToBFloat16(src + x, dst + x, sizeM, sizeM);
                src += sizeH;
                dst += sizeH;
            }
            if (end)
            {
                size_t padE = a.padE * p.srcC;
                memset(dst, 0, padE * sizeof(uint16_t));
                dst += padE;
            }
        }

        static void Reorder16bNhwcSpecV1(const uint8_t* src8, const ConvParam& p, const AlgParam& a, size_t dyBeg, size_t dyEnd, int end, uint16_t* dst)
        {
            const uint16_t* src = (uint16_t*)src8;
            size_t syPad = p.kernelY - 1 - p.padY, syBeg, syEnd = (dyEnd == p.dstH ? p.srcH : dyEnd + syPad);
            size_t padH = a.padH * p.srcC, padHDF = AlignLo(padH, DF);
            size_t sizeH = p.srcW * p.srcC, sizeHDF = AlignLo(sizeH, DF);
            __mmask32 padM = TailMask32(padH - padHDF), sizeM = TailMask32(sizeH - sizeHDF);
            if (dyBeg == 0)
            {
                size_t padV = a.padV * a.srcW * p.srcC;
                memset(dst, 0, padV * sizeof(uint16_t));
                dst += padV;
                syBeg = 0;
            }
            else
            {
                syBeg = dyBeg + syPad;
                src += syBeg * p.srcW * p.srcC;
                dst += (dyBeg + p.kernelY - 1) * a.srcW * p.srcC;
            }
            for (size_t sy = syBeg; sy < syEnd; ++sy)
            {
                if (padH)
                {
                    size_t i = 0;
                    for (; i < padHDF; i += DF)
                        Avx512bw::SetZero(dst + i);
                    if (padM)
                        Avx512bw::SetZero(dst + i, padM);
                    dst += padH;
                }
                memcpy(dst, src, sizeH * sizeof(uint16_t));
                src += sizeH;
                dst += sizeH;
            }
            if (end)
            {
                size_t padE = a.padE * p.srcC;
                memset(dst, 0, padE * sizeof(uint16_t));
                dst += padE;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void Convolution16bNhwcSpecV1_32x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dS = (int)p.srcC, strideS = dS * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;
            const uint16_t* src1 = src0 + 16 * dS;
            float* dst1 = dst0 + 16 * dD;
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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_stream_loadd(4, src0 + o, strideS);
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(5, src1 + o, strideS);
                _tile_loadd(7, weight1, strideW);                        
                _tile_dpbf16ps(0, 4, 6);
                _tile_dpbf16ps(1, 4, 7);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(3, 5, 7);
                weight1 += dW;
            }
            _tile_loadd(7, weight1, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);

            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(1, dst0 + F, strideD);
            _tile_stored(2, dst1 + 0, strideD);
            _tile_stored(3, dst1 + F, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst0 + F, dD);
            TileMoveToMemory(dst1 + 0, dD);
            TileMoveToMemory(dst1 + F, dD);
        }

        static void Convolution16bNhwcSpecV1_32x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dS = (int)p.srcC, strideS = dS * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* src1 = src0 + 16 * dS;
            float* dst1 = dst0 + 16 * dD;

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

            int n1 = (int)nK - 1, o = offs[0];
            _tile_stream_loadd(4, src0 + o, strideS);
            for (int i = 0; i < n1; ++i)
            {
                _tile_loadd(6, weight0, strideW);
                _tile_stream_loadd(5, src1 + o, strideS);
                _tile_dpbf16ps(0, 4, 6);
                o = offs[i + 1];
                _tile_stream_loadd(4, src0 + o, strideS);
                _tile_dpbf16ps(2, 5, 6);
                weight0 += dW;
            }
            _tile_loadd(6, weight0, strideW);
            _tile_stream_loadd(5, src1 + offs[n1], strideS);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(2, 5, 6);

            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(2, dst1 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst1 + 0, dD);
        }

        static void Convolution16bNhwcSpecV1_16x32(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dS = (int)p.srcC, strideS = dS * 2, dW = 512, strideW = 64, strideD = dD * 4;
            const uint16_t* weight1 = weight0 + a.K * F;

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

            int n1 = (int)nK - 1;
            _tile_loadd(6, weight0, strideW);
            for (int i = 0; i < n1; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(7, weight1, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(1, 4, 7);
                weight1 += dW;
            }
            _tile_stream_loadd(4, src0 + offs[n1], strideS);
            _tile_loadd(7, weight1, strideW);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);

            _tile_stored(0, dst0 + 0, strideD);
            _tile_stored(1, dst0 + F, strideD);
            TileMoveToMemory(dst0 + 0, dD);
            TileMoveToMemory(dst0 + F, dD);
        }

        static void Convolution16bNhwcSpecV1_16x16(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0)
        {
            int dD = (int)a.macroD, dS = (int)p.srcC, strideS = dS * 2, dW = 512, strideW = 64, strideD = dD * 4;

            if (zero)
            {
                _tile_zero(0);
            }
            else
            {
                _tile_stream_loadd(0, dst0 + 0, strideD);
            }

            int n = (int)nK;
            for (int i = 0; i < n; ++i)
            {
                _tile_stream_loadd(4, src0 + offs[i], strideS);
                _tile_loadd(6, weight0, strideW);
                _tile_dpbf16ps(0, 4, 6);
                weight0 += dW;
            }

            _tile_stored(0, dst0 + 0, strideD);
            TileMoveToMemory(dst0 + 0, dD);
        }

        typedef void (*Convolution16bNhwcSpecV1Ptr)(const uint16_t* src0, const ConvParam& p, const AlgParam& a, const int* offs, size_t nK, int zero, const uint16_t* weight0, float* dst0);

        static void Convolution16bNhwcSpecV1_2(const uint16_t* src, const ConvParam& p, const AlgParam& a, const int* offs, size_t dstC, size_t dstH, size_t K, int zero, const uint16_t* weight, float* dst)
        {
            size_t n1 = dstH * a.srcW - a.padH, n = 32;
            size_t nn = AlignLoAny(n1, n), m = n1 - nn, dW = a.K * DF;
            size_t dD = a.macroD, dS = p.srcC, nK = DivHi(K, 32);
            Convolution16bNhwcSpecV1Ptr body_2 = Convolution16bNhwcSpecV1_32x32;
            Convolution16bNhwcSpecV1Ptr tail_2 = m > 16 ? Convolution16bNhwcSpecV1_32x32 : Convolution16bNhwcSpecV1_16x32;
            Convolution16bNhwcSpecV1Ptr body_1 = Convolution16bNhwcSpecV1_32x16;
            Convolution16bNhwcSpecV1Ptr tail_1 = m > 16 ? Convolution16bNhwcSpecV1_32x16 : Convolution16bNhwcSpecV1_16x16;

            SetTileConfFull();
            for (size_t dc = 0; dc < dstC; dc += DF)
            {
                size_t dC = Simd::Min(DF, dstC - dc);
                size_t i = 0;
                if (dC > F)
                {
                    for (; i < nn; i += n)
                        body_2(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                    if (m)
                        tail_2(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                }
                else
                {
                    for (; i < nn; i += n)
                        body_1(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                    if (m)
                        tail_1(src + i * dS, p, a, offs, nK, zero, weight, dst + i * dD);
                }
                weight += dW;
                dst += DF;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type>  void Postprocess16bNhwcSpecV1(const float* src, const ConvParam& p,
            const AlgParam& a, size_t dstC, size_t dyBeg, size_t dyEnd, const float* bias, const float* params, uint8_t* dst)
        {
            size_t dstCF = AlignLo(dstC, F);
            __mmask16 tailD = TailMask16(dstC - dstCF);
            size_t rowGap = a.padH * a.macroD;
            src += dyBeg * a.srcW * a.macroD;
            dst += dyBeg * p.dstW * p.dstC * a.elem;
            for (size_t dy = dyBeg; dy < dyEnd; ++dy)
            {
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstCF; dc += F)
                        AmxBf16::Postprocess<term, type>(src, bias, params, dc, dst);
                    if (tailD)
                        AmxBf16::Postprocess<term, type>(src, bias, params, dc, dst, tailD);
                    src += a.macroD;
                    dst += p.dstC * a.elem;
                }
                src += rowGap;
            }
        }

        template<SimdConvolutionActivationType type> void SetPostprocess(const ConvParam& p, const AlgParam& a, PostprocessPtr & postprocess)
        {
            if (p.dstT == SimdTensorData16b)
                postprocess = Postprocess16bNhwcSpecV1<Term16bLast16b, type>;
            else
                postprocess = Postprocess16bNhwcSpecV1<Term16bLast32f, type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcSpecV1::SynetConvolution16bNhwcSpecV1(const ConvParam & p)
            : Avx512bw::SynetConvolution16bNhwcSpecV1(p)
        {
            SetAlgParam(F, F * 2, 32, F * 2, int(Base::AlgCacheL1() * (p.IsKernel(5) ? 1.05 : 1.00)), int(Base::AlgCacheL2() * 0.5), Base::AlgCacheL3());
            if (_src16b)
                _preprocess = Reorder16bNhwcSpecV1;
            else
                _preprocess = Convert16bNhwcSpecV1;
            _convolution = Convolution16bNhwcSpecV1_2;
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

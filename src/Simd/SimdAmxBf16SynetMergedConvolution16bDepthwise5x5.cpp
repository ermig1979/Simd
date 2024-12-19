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
#include "Simd/SimdSynetMergedConvolution16b.h"
#include "Simd/SimdSynetConvolution16bCommon.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#if defined(SIMD_AMXBF16_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace AmxBf16
    {
        using AlgParam = Base::SynetMergedConvolution16b::AlgParam;
        using DepthwisePtr = Base::SynetMergedConvolution16b::DepthwiseConvolutionPtr;

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE bool Preferable_k5p2d1s1w6(const ConvParam& p)
        {
            return p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 8;
                //(p.srcW > 10 && AlignHiAny(p.srcW, 8) < AlignHiAny(p.srcW, 6) * 1.2);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k5p2d1s1w6(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 8);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 25 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 6;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, w3, w4, d0, d1, d2, d3, d4, d5;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstCe; dc += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + dc);
                __mmask16 tailS = TailMask16(dstC - dc);
                __mmask32 tailC = (dc == dstCF && a.bufH[2]) ? TailMask32(dstCe - dstCF) : tailS;
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    for (size_t dx = 0;; dx += Min<size_t>(6, endW - dx))
                    {
                        d0 = _mm512_setzero_ps();
                        d1 = _mm512_setzero_ps();
                        d2 = _mm512_setzero_ps();
                        d3 = _mm512_setzero_ps();
                        d4 = _mm512_setzero_ps();
                        d5 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < 5; ++ky)
                        {
                            size_t sy = dy + ky - 2;
                            const T* ps = src + (sy & sM) * sY + (dx - 2) * sX;
                            const float* pw = weight + ky * 5 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                if (dx)
                                {
                                    s0 = LoadSrc(ps + 0 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = LoadSrc(ps + 1 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);
                                }
                                s0 = LoadSrc(ps + 2 * sX, tailS);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                d0 = _mm512_fmadd_ps(s0, w2, d0);
                                d1 = _mm512_fmadd_ps(s0, w1, d1);
                                d2 = _mm512_fmadd_ps(s0, w0, d2);

                                s1 = LoadSrc(ps + 3 * sX, tailS);
                                w3 = _mm512_maskz_loadu_ps(tailS, pw + 3 * F);
                                d0 = _mm512_fmadd_ps(s1, w3, d0);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = LoadSrc(ps + 4 * sX, tailS);
                                w4 = _mm512_maskz_loadu_ps(tailS, pw + 4 * F);
                                d0 = _mm512_fmadd_ps(s0, w4, d0);
                                d1 = _mm512_fmadd_ps(s0, w3, d1);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);
                                d4 = _mm512_fmadd_ps(s0, w0, d4);

                                s1 = LoadSrc(ps + 5 * sX, tailS);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);

                                s1 = LoadSrc(ps + 7 * sX, tailS);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);
                                if (dx < endW)
                                {
                                    s0 = LoadSrc(ps + 8 * sX, tailS);
                                    d4 = _mm512_fmadd_ps(s0, w4, d4);
                                    d5 = _mm512_fmadd_ps(s0, w3, d5); 
                                
                                    s1 = LoadSrc(ps + 9 * sX, tailS);
                                    d5 = _mm512_fmadd_ps(s1, w4, d5);
                                }
                            }
                        }
                        uint8_t* pd = dst + (dy - dy0) * dY + dx * dX;
                        Save1<term, type>(pd + 0 * dX, dD, d0, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d1, _bias, _params, tailC);
                        Save1<term, type>(pd + 2 * dX, dD, d2, _bias, _params, tailC);
                        Save1<term, type>(pd + 3 * dX, dD, d3, _bias, _params, tailC);
                        Save1<term, type>(pd + 4 * dX, dD, d4, _bias, _params, tailC);
                        Save1<term, type>(pd + 5 * dX, dD, d5, _bias, _params, tailC);
                        if (dx == endW)
                            break;
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE bool Preferable_k5p2d1s1w8(const ConvParam& p)
        {
            return p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && 
                (p.srcW >= 10 && AlignHiAny(p.srcW, 8) < AlignHiAny(p.srcW, 6) * 1.2);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k5p2d1s1w8(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 10);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 25 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 8;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, w3, w4, d0, d1, d2, d3, d4, d5, d6, d7;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dc = 0; dc < dstCe; dc += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + dc);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + dc);
                __mmask16 tailS = TailMask16(dstC - dc);
                __mmask32 tailC = (dc == dstCF && a.bufH[2]) ? TailMask32(dstCe - dstCF) : tailS;
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    for (size_t dx = 0;; dx += Min<size_t>(8, endW - dx))
                    {
                        d0 = _mm512_setzero_ps();
                        d1 = _mm512_setzero_ps();
                        d2 = _mm512_setzero_ps();
                        d3 = _mm512_setzero_ps();
                        d4 = _mm512_setzero_ps();
                        d5 = _mm512_setzero_ps();
                        d6 = _mm512_setzero_ps();
                        d7 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < 5; ++ky)
                        {
                            size_t sy = dy + ky - 2;
                            const T* ps = src + (sy & sM) * sY + (dx - 2) * sX;
                            const float* pw = weight + ky * 5 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                if (dx)
                                {
                                    s0 = LoadSrc(ps + 0 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = LoadSrc(ps + 1 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);
                                }
                                s0 = LoadSrc(ps + 2 * sX, tailS);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                d0 = _mm512_fmadd_ps(s0, w2, d0);
                                d1 = _mm512_fmadd_ps(s0, w1, d1);
                                d2 = _mm512_fmadd_ps(s0, w0, d2);

                                s1 = LoadSrc(ps + 3 * sX, tailS);
                                w3 = _mm512_maskz_loadu_ps(tailS, pw + 3 * F);
                                d0 = _mm512_fmadd_ps(s1, w3, d0);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = LoadSrc(ps + 4 * sX, tailS);
                                w4 = _mm512_maskz_loadu_ps(tailS, pw + 4 * F);
                                d0 = _mm512_fmadd_ps(s0, w4, d0);
                                d1 = _mm512_fmadd_ps(s0, w3, d1);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);
                                d4 = _mm512_fmadd_ps(s0, w0, d4);

                                s1 = LoadSrc(ps + 5 * sX, tailS);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);
                                d6 = _mm512_fmadd_ps(s0, w0, d6);

                                s1 = LoadSrc(ps + 7 * sX, tailS);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);
                                d6 = _mm512_fmadd_ps(s1, w1, d6);
                                d7 = _mm512_fmadd_ps(s1, w0, d7);

                                s0 = LoadSrc(ps + 8 * sX, tailS);
                                d4 = _mm512_fmadd_ps(s0, w4, d4);
                                d5 = _mm512_fmadd_ps(s0, w3, d5);
                                d6 = _mm512_fmadd_ps(s0, w2, d6);
                                d7 = _mm512_fmadd_ps(s0, w1, d7);

                                s1 = LoadSrc(ps + 9 * sX, tailS);
                                d5 = _mm512_fmadd_ps(s1, w4, d5);
                                d6 = _mm512_fmadd_ps(s1, w3, d6);
                                d7 = _mm512_fmadd_ps(s1, w2, d7);
                                if (dx < endW)
                                {
                                    s0 = LoadSrc(ps + 10 * sX, tailS);
                                    d6 = _mm512_fmadd_ps(s0, w4, d6);
                                    d7 = _mm512_fmadd_ps(s0, w3, d7);

                                    s1 = LoadSrc(ps + 11 * sX, tailS);
                                    d7 = _mm512_fmadd_ps(s1, w4, d7);
                                }
                            }
                        }
                        uint8_t* pd = dst + (dy - dy0) * dY + dx * dX;
                        Save1<term, type>(pd + 0 * dX, dD, d0, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d1, _bias, _params, tailC);
                        Save1<term, type>(pd + 2 * dX, dD, d2, _bias, _params, tailC);
                        Save1<term, type>(pd + 3 * dX, dD, d3, _bias, _params, tailC);
                        Save1<term, type>(pd + 4 * dX, dD, d4, _bias, _params, tailC);
                        Save1<term, type>(pd + 5 * dX, dD, d5, _bias, _params, tailC);
                        Save1<term, type>(pd + 6 * dX, dD, d6, _bias, _params, tailC);
                        Save1<term, type>(pd + 7 * dX, dD, d7, _bias, _params, tailC);
                        if (dx == endW)
                            break;
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static bool SetDepthwise5x5(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (Preferable_k5p2d1s1w8(p))
            {
                depthwise = DepthwiseConvolution_k5p2d1s1w8<T, term, type>;
                return true;
            } 
            else if (Preferable_k5p2d1s1w6(p))
            {
                depthwise = DepthwiseConvolution_k5p2d1s1w6<T, term, type>;
                return true;
            }
            else
                return false;
        }

        template<typename T, SimdConvolutionActivationType type> static bool SetDepthwise5x5(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return (p.dstT == SimdTensorData32f ? SetDepthwise5x5<T, Term16bLast32f, type>(p, depthwise) : SetDepthwise5x5<T, Term16bLast16b, type>(p, depthwise));
        }

        template<SimdConvolutionActivationType type> static bool SetDepthwise5x5(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return (p.srcT == SimdTensorData16b ? SetDepthwise5x5<uint16_t, type>(p, depthwise) : SetDepthwise5x5<float, type>(p, depthwise));
        }

        bool SetDepthwise5x5(const ConvParam& p, DepthwisePtr& depthwise)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: return SetDepthwise5x5<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationRelu: return SetDepthwise5x5<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationLeakyRelu: return SetDepthwise5x5<SimdConvolutionActivationPrelu>(p, depthwise);
            case SimdConvolutionActivationRestrictRange: return SetDepthwise5x5<SimdConvolutionActivationRestrictRange>(p, depthwise);
            case SimdConvolutionActivationPrelu: return SetDepthwise5x5<SimdConvolutionActivationPrelu>(p, depthwise);
            case SimdConvolutionActivationElu: return SetDepthwise5x5<SimdConvolutionActivationElu>(p, depthwise);
            case SimdConvolutionActivationHswish: return SetDepthwise5x5<SimdConvolutionActivationHswish>(p, depthwise);
            case SimdConvolutionActivationMish: return SetDepthwise5x5<SimdConvolutionActivationMish>(p, depthwise);
            case SimdConvolutionActivationHardSigmoid: return SetDepthwise5x5<SimdConvolutionActivationHardSigmoid>(p, depthwise);
            case SimdConvolutionActivationSwish: return SetDepthwise5x5<SimdConvolutionActivationSwish>(p, depthwise);
            case SimdConvolutionActivationGelu: return SetDepthwise5x5<SimdConvolutionActivationGelu>(p, depthwise);
            default:
                return false;
            }
        }
    }
#endif
}

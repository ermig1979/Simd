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

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void DepthwiseConvolutionDefault(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW, kernelX = p.kernelX, kernelY = p.kernelY;
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
            size_t noseY = NoseH(p), bodyY = BodyH(p), noseX = NoseW(p), bodyX = BodyW(p);
            size_t bodyX2 = AlignLo(bodyX - noseX, 2) + noseX;
            size_t bodyX4 = AlignLo(bodyX - noseX, 4) + noseX;
            size_t bodyX8 = AlignLo(bodyX - noseX, 8) + noseX;
            size_t dstCF = AlignLo(dstC, F), dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstCe; c += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);
                if (c == dstCF)
                {
                    __mmask16 tail = TailMask16(dstC - c);
                    __mmask32 gapMask = a.bufH[2] ? TailMask32(dstCe - dstC) : 0;
                    for (size_t dy = yBeg; dy < yEnd; ++dy)
                    {
                        uint8_t* pd = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; ++dx, pd += dX)
                        {
                            __m512 sum = _mm512_setzero_ps();
                            for (size_t ky = 0; ky < kernelY; ++ky)
                            {
                                size_t sy = dy * strideY + ky - padY;
                                if (sy < srcH)
                                {
                                    for (size_t kx = 0; kx < kernelX; ++kx)
                                    {
                                        size_t sx = dx * strideX + kx - padX;
                                        if (sx < srcW)
                                        {
                                            const float* pw = weight + (ky * kernelX + kx) * F;
                                            const T* ps = src + (sy & sM) * sY + sx * sX;
                                            sum = _mm512_fmadd_ps(LoadSrc(ps, tail), _mm512_maskz_loadu_ps(tail, pw), sum);
                                        }
                                    }
                                }
                            }
                            Save1<term, type>(pd, NULL, sum, _bias, _params, tail);
                            if (gapMask)
                                SetZero((uint16_t*)pd + dstC - dstCF, gapMask);
                        }
                    }
                    return;
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    size_t dx = 0;
                    for (; dx < noseX; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = _mm512_fmadd_ps(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                    for (; dx < bodyX8; dx += 8, pd += 8 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        __m512 sum2 = _mm512_setzero_ps();
                        __m512 sum3 = _mm512_setzero_ps();
                        __m512 sum4 = _mm512_setzero_ps();
                        __m512 sum5 = _mm512_setzero_ps();
                        __m512 sum6 = _mm512_setzero_ps();
                        __m512 sum7 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = _mm512_fmadd_ps(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = _mm512_fmadd_ps(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = _mm512_fmadd_ps(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = _mm512_fmadd_ps(LoadSrc(ps + 3 * ssX), w0, sum3);
                                    sum4 = _mm512_fmadd_ps(LoadSrc(ps + 4 * ssX), w0, sum4);
                                    sum5 = _mm512_fmadd_ps(LoadSrc(ps + 5 * ssX), w0, sum5);
                                    sum6 = _mm512_fmadd_ps(LoadSrc(ps + 6 * ssX), w0, sum6);
                                    sum7 = _mm512_fmadd_ps(LoadSrc(ps + 7 * ssX), w0, sum7);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                        Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                        Save1<term, type>(pd + 4 * dX, NULL, sum4, _bias, _params);
                        Save1<term, type>(pd + 5 * dX, NULL, sum5, _bias, _params);
                        Save1<term, type>(pd + 6 * dX, NULL, sum6, _bias, _params);
                        Save1<term, type>(pd + 7 * dX, NULL, sum7, _bias, _params);
                    }
                    for (; dx < bodyX4; dx += 4, pd += 4 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        __m512 sum2 = _mm512_setzero_ps();
                        __m512 sum3 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = _mm512_fmadd_ps(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = _mm512_fmadd_ps(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = _mm512_fmadd_ps(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = _mm512_fmadd_ps(LoadSrc(ps + 3 * ssX), w0, sum3);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                        Save1<term, type>(pd + 2 * dX, NULL, sum2, _bias, _params);
                        Save1<term, type>(pd + 3 * dX, NULL, sum3, _bias, _params);
                    }
                    for (; dx < bodyX2; dx += 2, pd += 2 * dX)
                    {
                        __m512 sum0 = _mm512_setzero_ps();
                        __m512 sum1 = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum0 = _mm512_fmadd_ps(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = _mm512_fmadd_ps(LoadSrc(ps + 1 * ssX), w0, sum1);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd + 0 * dX, NULL, sum0, _bias, _params);
                        Save1<term, type>(pd + 1 * dX, NULL, sum1, _bias, _params);
                    }
                    for (; dx < bodyX; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        const float* pw = weight;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* ps = src + (sy & sM) * sY + (dx * strideX - padX) * sX;
                                for (size_t kx = 0; kx < kernelX; ++kx, ps += sX, pw += F)
                                {
                                    __m512 w0 = _mm512_loadu_ps(pw);
                                    sum = _mm512_fmadd_ps(LoadSrc(ps), w0, sum);
                                }
                            }
                            else
                                pw += kernelX * F;
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                    for (; dx < p.dstW; dx += 1, pd += dX)
                    {
                        __m512 sum = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * F;
                                        const T* ps = src + (sy & sM) * sY + sx * sX;
                                        sum = _mm512_fmadd_ps(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
                                    }
                                }
                            }
                        }
                        Save1<term, type>(pd, NULL, sum, _bias, _params);
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void DepthwiseConvolutionLargePad(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW, kernelX = p.kernelX, kernelY = p.kernelY;
            size_t strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = strideX * sX;
            size_t dstCF = AlignLo(dstC, F), dstC2F = AlignLo(dstC, 2 * F), dstC4F = AlignLo(dstC, 4 * F), dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;
            size_t dstW = p.dstW, dstW2 = AlignLo(dstW, 2), dstW4 = AlignLo(dstW, 4);

            __m512 d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, w0;
            __m512 _params[4], _bias[4];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            size_t c = 0;
            for (; c < dstC4F; c += 4 * F)
            {
                _bias[0] = _mm512_loadu_ps(bias + c + 0 * F);
                _bias[1] = _mm512_loadu_ps(bias + c + 1 * F);
                _bias[2] = _mm512_loadu_ps(bias + c + 2 * F);
                _bias[3] = _mm512_loadu_ps(bias + c + 3 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + c + 0 * F);
                    _params[1] = _mm512_loadu_ps(params + c + 1 * F);
                    _params[2] = _mm512_loadu_ps(params + c + 2 * F);
                    _params[3] = _mm512_loadu_ps(params + c + 3 * F);
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, pd += 4 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        d20 = _mm512_setzero_ps();
                        d30 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        d11 = _mm512_setzero_ps();
                        d21 = _mm512_setzero_ps();
                        d31 = _mm512_setzero_ps();
                        d02 = _mm512_setzero_ps();
                        d12 = _mm512_setzero_ps();
                        d22 = _mm512_setzero_ps();
                        d32 = _mm512_setzero_ps();
                        d03 = _mm512_setzero_ps();
                        d13 = _mm512_setzero_ps();
                        d23 = _mm512_setzero_ps();
                        d33 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * sD, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * sD, mask3), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * sD, mask1), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 1 * sD, mask2), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 1 * sD, mask3), w0, d31, mask3);
                                    w0 = _mm512_loadu_ps(pw + 2 * wD);
                                    d02 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 2 * sD, mask0), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 2 * sD, mask1), w0, d12, mask1);
                                    d22 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 2 * sD, mask2), w0, d22, mask2);
                                    d32 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 2 * sD, mask3), w0, d32, mask3);
                                    w0 = _mm512_loadu_ps(pw + 3 * wD);
                                    d03 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 3 * sD, mask0), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 3 * sD, mask1), w0, d13, mask1);
                                    d23 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 3 * sD, mask2), w0, d23, mask2);
                                    d33 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 3 * sD, mask3), w0, d33, mask3);
                                }
                            }
                        }
                        Save4<term, type>(pd + 0 * dX, dD, d00, d01, d02, d03, _bias, _params);
                        Save4<term, type>(pd + 1 * dX, dD, d10, d11, d12, d13, _bias, _params);
                        Save4<term, type>(pd + 2 * dX, dD, d20, d21, d22, d23, _bias, _params);
                        Save4<term, type>(pd + 3 * dX, dD, d30, d31, d32, d33, _bias, _params);
                    }
                    for (; dx < dstW2; dx += 2, pd += 2 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        d11 = _mm512_setzero_ps();
                        d02 = _mm512_setzero_ps();
                        d12 = _mm512_setzero_ps();
                        d03 = _mm512_setzero_ps();
                        d13 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * sD, mask1), w0, d11, mask1);
                                    w0 = _mm512_loadu_ps(pw + 2 * wD);
                                    d02 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 2 * sD, mask0), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 2 * sD, mask1), w0, d12, mask1);
                                    w0 = _mm512_loadu_ps(pw + 3 * wD);
                                    d03 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 3 * sD, mask0), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 3 * sD, mask1), w0, d13, mask1);
                                }
                            }
                        }
                        Save4<term, type>(pd + 0 * dX, dD, d00, d01, d02, d03, _bias, _params);
                        Save4<term, type>(pd + 1 * dX, dD, d10, d11, d12, d13, _bias, _params);
                    }
                    for (; dx < dstW; dx += 1, pd += 1 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        d02 = _mm512_setzero_ps();
                        d03 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                    w0 = _mm512_loadu_ps(pw + 2 * wD);
                                    d02 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 2 * sD, mask0), w0, d02, mask0);
                                    w0 = _mm512_loadu_ps(pw + 3 * wD);
                                    d03 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 3 * sD, mask0), w0, d03, mask0);
                                }
                            }
                        }
                        Save4<term, type>(pd + 0 * dX, dD, d00, d01, d02, d03, _bias, _params);
                    }
                }
                src += 4 * sD;
                dst += 4 * dD;
                weight += 4 * wD;
            }
            for (; c < dstC2F; c += 2 * F)
            {
                _bias[0] = _mm512_loadu_ps(bias + c + 0 * F);
                _bias[1] = _mm512_loadu_ps(bias + c + 1 * F);
                if (type == ::SimdConvolutionActivationPrelu)
                {
                    _params[0] = _mm512_loadu_ps(params + c + 0 * F);
                    _params[1] = _mm512_loadu_ps(params + c + 1 * F);
                }
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, pd += 4 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        d20 = _mm512_setzero_ps();
                        d30 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        d11 = _mm512_setzero_ps();
                        d21 = _mm512_setzero_ps();
                        d31 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * sD, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * sD, mask3), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * sD, mask1), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 1 * sD, mask2), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 1 * sD, mask3), w0, d31, mask3);
                                }
                            }
                        }
                        Save2<term, type>(pd + 0 * dX, dD, d00, d01, _bias, _params);
                        Save2<term, type>(pd + 1 * dX, dD, d10, d11, _bias, _params);
                        Save2<term, type>(pd + 2 * dX, dD, d20, d21, _bias, _params);
                        Save2<term, type>(pd + 3 * dX, dD, d30, d31, _bias, _params);
                    }
                    for (; dx < dstW2; dx += 2, pd += 2 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        d11 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * sD, mask1), w0, d11, mask1);
                                }
                            }
                        }
                        Save2<term, type>(pd + 0 * dX, dD, d00, d01, _bias, _params);
                        Save2<term, type>(pd + 1 * dX, dD, d10, d11, _bias, _params);
                    }
                    for (; dx < dstW; dx += 1, pd += 1 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d01 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * sX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_loadu_ps(pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    w0 = _mm512_loadu_ps(pw + 1 * wD);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * sD, mask0), w0, d01, mask0);
                                }
                            }
                        }
                        Save2<term, type>(pd + 0 * dX, dD, d00, d01, _bias, _params);
                    }
                }
                src += 2 * sD;
                dst += 2 * dD;
                weight += 2 * wD;
            }
            for (; c < dstCe; c += F)
            {
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);
                __mmask16 tailS = TailMask16(dstC - c);
                __mmask32 tailC = (c == dstCF && a.bufH[2]) ? TailMask32(dstCe - dstCF) : tailS;
                for (size_t dy = yBeg; dy < yEnd; ++dy)
                {
                    uint8_t* pd = dst + (dy - dy0) * dY;
                    size_t dx = 0;
                    for (; dx < dstW4; dx += 4, pd += 4 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        d20 = _mm512_setzero_ps();
                        d30 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? tailS : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? tailS : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? tailS : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? tailS : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * sD, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * sD, mask3), w0, d30, mask3);
                                }
                            }
                        }
                        Save1<term, type>(pd + 0 * dX, dD, d00, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d10, _bias, _params, tailC);
                        Save1<term, type>(pd + 2 * dX, dD, d20, _bias, _params, tailC);
                        Save1<term, type>(pd + 3 * dX, dD, d30, _bias, _params, tailC);
                    }
                    for (; dx < dstW2; dx += 2, pd += 2 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        d10 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? tailS : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? tailS : 0x0000;
                                    const T* ps0 = psy + sx * sX, * ps1 = ps0 + 1 * ssX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * sD, mask1), w0, d10, mask1);
                                }
                            }
                        }
                        Save1<term, type>(pd + 0 * dX, dD, d00, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d10, _bias, _params, tailC);
                    }
                    for (; dx < dstW; dx += 1, pd += 1 * dX)
                    {
                        size_t sx0 = dx * strideX - padX;
                        d00 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                const T* psy = src + (sy & sM) * sY;
                                const float* pwy = weight + ky * kernelX * F;
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? tailS : 0x0000;
                                    const T* ps0 = psy + sx * sX;
                                    const float* pw = pwy + kx * F;

                                    w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * wD);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * sD, mask0), w0, d00, mask0);
                                }
                            }
                        }
                        Save1<term, type>(pd + 0 * dX, dD, d00, _bias, _params, tailC);
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if(p.padX + p.padW > 2 && p.srcC >= 128)
                depthwise = DepthwiseConvolutionLargePad<T, term, type>;
            else
                depthwise = DepthwiseConvolutionDefault<T, term, type>;
        }

        template<typename T, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return p.dstT == SimdTensorData32f ? SetDepthwise<T, Term16bLast32f, type>(p, depthwise) : SetDepthwise<T, Term16bLast16b, type>(p, depthwise);
        }

        template<SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return p.srcT == SimdTensorData16b ? SetDepthwise<uint16_t, type>(p, depthwise) : SetDepthwise<float, type>(p, depthwise);
        }

        void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (SetDepthwise7x7(p, depthwise))
                return;
            if (SetDepthwise5x5(p, depthwise))
                return;
            if (SetDepthwise3x3(p, depthwise))
                return;
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationRelu: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationLeakyRelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationRestrictRange: SetDepthwise<SimdConvolutionActivationRestrictRange>(p, depthwise); break;
            case SimdConvolutionActivationPrelu: SetDepthwise<SimdConvolutionActivationPrelu>(p, depthwise); break;
            case SimdConvolutionActivationElu: SetDepthwise<SimdConvolutionActivationElu>(p, depthwise); break;
            case SimdConvolutionActivationHswish: SetDepthwise<SimdConvolutionActivationHswish>(p, depthwise); break;
            case SimdConvolutionActivationMish: SetDepthwise<SimdConvolutionActivationMish>(p, depthwise); break;
            case SimdConvolutionActivationHardSigmoid: SetDepthwise<SimdConvolutionActivationHardSigmoid>(p, depthwise); break;
            case SimdConvolutionActivationSwish: SetDepthwise<SimdConvolutionActivationSwish>(p, depthwise); break;
            case SimdConvolutionActivationGelu: SetDepthwise<SimdConvolutionActivationGelu>(p, depthwise); break;
            }
        }
    }
#endif
}

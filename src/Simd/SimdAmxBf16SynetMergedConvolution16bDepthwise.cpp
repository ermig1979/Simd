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

        template <class T> SIMD_INLINE __m512 LoadSrc(const T* src, __mmask16 mask = -1);

        template <> SIMD_INLINE __m512 LoadSrc<float>(const float* src, __mmask16 mask)
        {
            return _mm512_maskz_loadu_ps(mask, src);
        }

        template <> SIMD_INLINE __m512 LoadSrc<uint16_t>(const uint16_t* src, __mmask16 mask)
        {
            return BFloat16ToFloat32(_mm256_maskz_loadu_epi16(mask, src));
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct DepthwiseTerm16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, size_t stride, __m512 value, const __m512* bias, const __m512* params, __mmask32 tail);
        };

        template <> struct DepthwiseTerm16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, size_t stride, __m512 value, const __m512* bias, const __m512* params, __mmask32 tail)
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm512_mask_storeu_epi16(ptr + index * stride, tail, _mm512_castsi256_si512((__m256i)_mm512_cvtneps_pbh(f32)));
            }
        };

        template <> struct DepthwiseTerm16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, size_t stride, __m512 value, const __m512* bias, const __m512* params, __mmask32 tail)
            {
                _mm512_mask_storeu_ps((float*)(ptr + index * stride), __mmask16(tail), Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, size_t stride, __m512 val0, const __m512* bias, const __m512* params, __mmask32 tail)
        {
            DepthwiseTerm16b<term>::template Save<type, 0>(ptr, stride, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, size_t stride, __m512 val0, __m512 val1, const __m512* bias, const __m512* params)
        {
            DepthwiseTerm16b<term>::template Save<type, 0>(ptr, stride, val0, bias, params, 0xFFFF);
            DepthwiseTerm16b<term>::template Save<type, 1>(ptr, stride, val1, bias, params, 0xFFFF);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save4(uint8_t* ptr, size_t stride, __m512 val0, __m512 val1, __m512 val2, __m512 val3, const __m512* bias, const __m512* params)
        {
            DepthwiseTerm16b<term>::template Save<type, 0>(ptr, stride, val0, bias, params, 0xFFFF);
            DepthwiseTerm16b<term>::template Save<type, 1>(ptr, stride, val1, bias, params, 0xFFFF);
            DepthwiseTerm16b<term>::template Save<type, 2>(ptr, stride, val2, bias, params, 0xFFFF);
            DepthwiseTerm16b<term>::template Save<type, 3>(ptr, stride, val3, bias, params, 0xFFFF);
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolutionDefault(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
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
                                            sum = Fmadd<nofma>(LoadSrc(ps, tail), _mm512_maskz_loadu_ps(tail, pw), sum);
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
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
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
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(LoadSrc(ps + 3 * ssX), w0, sum3);
                                    sum4 = Fmadd<nofma>(LoadSrc(ps + 4 * ssX), w0, sum4);
                                    sum5 = Fmadd<nofma>(LoadSrc(ps + 5 * ssX), w0, sum5);
                                    sum6 = Fmadd<nofma>(LoadSrc(ps + 6 * ssX), w0, sum6);
                                    sum7 = Fmadd<nofma>(LoadSrc(ps + 7 * ssX), w0, sum7);
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
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
                                    sum2 = Fmadd<nofma>(LoadSrc(ps + 2 * ssX), w0, sum2);
                                    sum3 = Fmadd<nofma>(LoadSrc(ps + 3 * ssX), w0, sum3);
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
                                    sum0 = Fmadd<nofma>(LoadSrc(ps + 0 * ssX), w0, sum0);
                                    sum1 = Fmadd<nofma>(LoadSrc(ps + 1 * ssX), w0, sum1);
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
                                    sum = Fmadd<nofma>(LoadSrc(ps), w0, sum);
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
                                        sum = Fmadd<nofma>(LoadSrc(ps), _mm512_loadu_ps(pw), sum);
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

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x2(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                Save1<term, type>(dst, NULL, _mm512_add_ps(sum0, sum1), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge2x3(const T* src0,
            const T* src1, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                Save1<term, type>(dst, NULL, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Edge3x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                Save1<term, type>(dst, NULL, _mm512_add_ps(sum0, sum1), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x1(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, __mmask16 tail)
        {
            if (nofma)
            {
                __m512 sum = _mm512_setzero_ps();
                sum = Fmadd<true>(LoadSrc(src0 + 0 * sX), weight[0], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 1 * sX), weight[1], sum);
                sum = Fmadd<true>(LoadSrc(src0 + 2 * sX), weight[2], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 0 * sX), weight[3], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 1 * sX), weight[4], sum);
                sum = Fmadd<true>(LoadSrc(src1 + 2 * sX), weight[5], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 0 * sX), weight[6], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 1 * sX), weight[7], sum);
                sum = Fmadd<true>(LoadSrc(src2 + 2 * sX), weight[8], sum);
                Save1<term, type>(dst, NULL, sum, bias, params, tail);
            }
            else
            {
                __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps();
                sum0 = Fmadd<false>(LoadSrc(src0 + 0 * sX), weight[0], sum0);
                sum1 = Fmadd<false>(LoadSrc(src0 + 1 * sX), weight[1], sum1);
                sum2 = Fmadd<false>(LoadSrc(src0 + 2 * sX), weight[2], sum2);
                sum0 = Fmadd<false>(LoadSrc(src1 + 0 * sX), weight[3], sum0);
                sum1 = Fmadd<false>(LoadSrc(src1 + 1 * sX), weight[4], sum1);
                sum2 = Fmadd<false>(LoadSrc(src1 + 2 * sX), weight[5], sum2);
                sum0 = Fmadd<false>(LoadSrc(src2 + 0 * sX), weight[6], sum0);
                sum1 = Fmadd<false>(LoadSrc(src2 + 1 * sX), weight[7], sum1);
                sum2 = Fmadd<false>(LoadSrc(src2 + 2 * sX), weight[8], sum2);
                Save1<term, type>(dst, NULL, _mm512_add_ps(_mm512_add_ps(sum0, sum1), sum2), bias, params, tail);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x2(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> SIMD_INLINE void DepthwiseConvolution3x3Main1x4(const T* src0,
            const T* src1, const T* src2, size_t sX, const __m512* weight, const __m512* bias, const __m512* params, uint8_t* dst, size_t dX, __mmask16 tail)
        {
            __m512 sum0 = _mm512_setzero_ps(), sum1 = _mm512_setzero_ps(), sum2 = _mm512_setzero_ps(), sum3 = _mm512_setzero_ps(), s0;

            s0 = LoadSrc(src0 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[0], sum0);
            s0 = LoadSrc(src0 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[1], sum0);
            sum1 = Fmadd<nofma>(s0, weight[0], sum1);
            s0 = LoadSrc(src0 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[2], sum0);
            sum1 = Fmadd<nofma>(s0, weight[1], sum1);
            sum2 = Fmadd<nofma>(s0, weight[0], sum2);
            s0 = LoadSrc(src0 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[2], sum1);
            sum2 = Fmadd<nofma>(s0, weight[1], sum2);
            sum3 = Fmadd<nofma>(s0, weight[0], sum3);
            s0 = LoadSrc(src0 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[2], sum2);
            sum3 = Fmadd<nofma>(s0, weight[1], sum3);
            s0 = LoadSrc(src0 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[2], sum3);

            s0 = LoadSrc(src1 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[3], sum0);
            s0 = LoadSrc(src1 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[4], sum0);
            sum1 = Fmadd<nofma>(s0, weight[3], sum1);
            s0 = LoadSrc(src1 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[5], sum0);
            sum1 = Fmadd<nofma>(s0, weight[4], sum1);
            sum2 = Fmadd<nofma>(s0, weight[3], sum2);
            s0 = LoadSrc(src1 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[5], sum1);
            sum2 = Fmadd<nofma>(s0, weight[4], sum2);
            sum3 = Fmadd<nofma>(s0, weight[3], sum3);
            s0 = LoadSrc(src1 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[5], sum2);
            sum3 = Fmadd<nofma>(s0, weight[4], sum3);
            s0 = LoadSrc(src1 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[5], sum3);

            s0 = LoadSrc(src2 + 0 * sX);
            sum0 = Fmadd<nofma>(s0, weight[6], sum0);
            s0 = LoadSrc(src2 + 1 * sX);
            sum0 = Fmadd<nofma>(s0, weight[7], sum0);
            sum1 = Fmadd<nofma>(s0, weight[6], sum1);
            s0 = LoadSrc(src2 + 2 * sX);
            sum0 = Fmadd<nofma>(s0, weight[8], sum0);
            sum1 = Fmadd<nofma>(s0, weight[7], sum1);
            sum2 = Fmadd<nofma>(s0, weight[6], sum2);
            s0 = LoadSrc(src2 + 3 * sX);
            sum1 = Fmadd<nofma>(s0, weight[8], sum1);
            sum2 = Fmadd<nofma>(s0, weight[7], sum2);
            sum3 = Fmadd<nofma>(s0, weight[6], sum3);
            s0 = LoadSrc(src2 + 4 * sX);
            sum2 = Fmadd<nofma>(s0, weight[8], sum2);
            sum3 = Fmadd<nofma>(s0, weight[7], sum3);
            s0 = LoadSrc(src2 + 5 * sX);
            sum3 = Fmadd<nofma>(s0, weight[8], sum3);

            Save1<term, type>(dst + 0 * dX, NULL, sum0, bias, params, tail);
            Save1<term, type>(dst + 1 * dX, NULL, sum1, bias, params, tail);
            Save1<term, type>(dst + 2 * dX, NULL, sum2, bias, params, tail);
            Save1<term, type>(dst + 3 * dX, NULL, sum3, bias, params, tail);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> void DepthwiseConvolution3x3(const uint8_t* src8, const ConvParam& p, const AlgParam& a,
            size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t strideY = p.strideY, padY = p.padY, padX = p.padX, padH = p.padH, padW = p.padW, dstC = maC;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = p.kernelY * p.kernelX * F, ssX = p.strideX * sX, ssX0 = (p.strideX - p.padX) * sX;
            size_t xMainEnd = p.dstW - p.padW, yMainEnd = yEnd == p.dstH && p.padH ? yEnd - 1 : yEnd;
            size_t xMainEnd2 = AlignLo(xMainEnd - padX, 2) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t xMainEnd4 = AlignLo(xMainEnd - padX, 4) * (p.strideX == 1 ? 1 : 0) + padX;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 _params[2], _bias[1];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t c = 0; c < dstCe; c += F)
            {
                __mmask16 tail = TailMask16(dstC - c);
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                _bias[0] = _mm512_loadu_ps(bias + c);
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_loadu_ps(params + c);

                size_t dy = yBeg;
                if (c == dstC)
                {
                    __mmask32 gapMask = a.bufH[2] ? TailMask32(dstCe - dstC) : 0;
                    for (; dy < yEnd; ++dy)
                    {
                        uint8_t* pDst = dst + (dy - dy0) * dY;
                        for (size_t dx = 0; dx < p.dstW; dx++, pDst += dX)
                            SetZero((uint16_t*)pDst, gapMask);
                    }
                    return;
                }
                if (yBeg == 0 && padY)
                {
                    size_t sy = 0, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 4, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 3, _bias, _params, pDst, tail);
                    dy++;
                }
                for (; dy < yMainEnd; ++dy)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    const T* src2 = src + ((sy + 2) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0, src2 += ssX0;
                    for (; dx < xMainEnd4; dx += 4, pDst += dX * 4, src0 += ssX * 4, src1 += ssX * 4, src2 += ssX * 4)
                        DepthwiseConvolution3x3Main1x4<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd2; dx += 2, pDst += dX * 2, src0 += ssX * 2, src1 += ssX * 2, src2 += ssX * 2)
                        DepthwiseConvolution3x3Main1x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, dX, tail);
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX, src2 += ssX)
                        DepthwiseConvolution3x3Main1x1<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge3x2<T, term, type, nofma>(src0, src1, src2, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                if (dy < yEnd)
                {
                    size_t sy = dy * strideY - padY, dx = 0;
                    const T* src0 = src + ((sy + 0) & sM) * sY;
                    const T* src1 = src + ((sy + 1) & sM) * sY;
                    uint8_t* pDst = dst + (dy - dy0) * dY;
                    if (padX)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 1, _bias, _params, pDst, tail),
                        pDst += dX, dx++, src0 += ssX0, src1 += ssX0;
                    for (; dx < xMainEnd; dx++, pDst += dX, src0 += ssX, src1 += ssX)
                        DepthwiseConvolution3x3Edge2x3<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                    if (padW)
                        DepthwiseConvolution3x3Edge2x2<T, term, type, nofma>(src0, src1, sX, _weight + 0, _bias, _params, pDst, tail);
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE bool Preferable_k7p3d1s1w4(const ConvParam& p)
        {
            return p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 7;
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k7p3d1s1w4(const uint8_t* src8, 
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 7);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 49 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 4;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3;

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
                    for (size_t dx = 0;; dx += Min<size_t>(4, endW - dx))
                    {
                        d0 = _mm512_setzero_ps();
                        d1 = _mm512_setzero_ps();
                        d2 = _mm512_setzero_ps();
                        d3 = _mm512_setzero_ps();
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const T* ps = src + (sy & sM) * sY + (dx - 3) * sX;
                            const float* pw = weight + ky * 7 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                if (dx)
                                {
                                    s0 = LoadSrc(ps + 0 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = LoadSrc(ps + 1 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = LoadSrc(ps + 2 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
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

                                s1 = LoadSrc(ps + 5 * sX, tailS);
                                w5 = _mm512_maskz_loadu_ps(tailS, pw + 5 * F);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                w6 = _mm512_maskz_loadu_ps(tailS, pw + 6 * F);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                if (dx < endW)
                                {
                                    s1 = LoadSrc(ps + 7 * sX, tailS);
                                    d1 = _mm512_fmadd_ps(s1, w6, d1);
                                    d2 = _mm512_fmadd_ps(s1, w5, d2);
                                    d3 = _mm512_fmadd_ps(s1, w4, d3);

                                    s0 = LoadSrc(ps + 8 * sX, tailS);
                                    d2 = _mm512_fmadd_ps(s0, w6, d2);
                                    d3 = _mm512_fmadd_ps(s0, w5, d3);

                                    s1 = LoadSrc(ps + 9 * sX, tailS);
                                    d3 = _mm512_fmadd_ps(s1, w6, d3);
                                }
                            }
                        }
                        uint8_t* pd = dst + (dy - dy0) * dY + dx * dX;
                        Save1<term, type>(pd + 0 * dX, dD, d0, _bias, _params, tailC);
                        Save1<term, type>(pd + 1 * dX, dD, d1, _bias, _params, tailC);
                        Save1<term, type>(pd + 2 * dX, dD, d2, _bias, _params, tailC);
                        Save1<term, type>(pd + 3 * dX, dD, d3, _bias, _params, tailC);
                        if(dx == endW)
                            break;
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE bool Preferable_k7p3d1s1w6(const ConvParam& p)
        {
            return p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) &&
                (p.srcW > 8 && AlignHiAny(p.srcW, 6) < AlignHiAny(p.srcW, 4) * 1.2);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k7p3d1s1w6(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 11);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 49 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 6;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5;

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
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const T* ps = src + (sy & sM) * sY + (dx - 3) * sX;
                            const float* pw = weight + ky * 7 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                if (dx)
                                {
                                    s0 = LoadSrc(ps + 0 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = LoadSrc(ps + 1 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = LoadSrc(ps + 2 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
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
                                w5 = _mm512_maskz_loadu_ps(tailS, pw + 5 * F);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                w6 = _mm512_maskz_loadu_ps(tailS, pw + 6 * F);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);

                                s1 = LoadSrc(ps + 7 * sX, tailS);
                                d1 = _mm512_fmadd_ps(s1, w6, d1);
                                d2 = _mm512_fmadd_ps(s1, w5, d2);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);

                                s0 = LoadSrc(ps + 8 * sX, tailS);
                                d2 = _mm512_fmadd_ps(s0, w6, d2);
                                d3 = _mm512_fmadd_ps(s0, w5, d3);
                                d4 = _mm512_fmadd_ps(s0, w4, d4);
                                d5 = _mm512_fmadd_ps(s0, w3, d5);
                                if (dx < endW)
                                {
                                    s1 = LoadSrc(ps + 9 * sX, tailS);
                                    d3 = _mm512_fmadd_ps(s1, w6, d3);
                                    d4 = _mm512_fmadd_ps(s1, w5, d4);
                                    d5 = _mm512_fmadd_ps(s1, w4, d5);

                                    s0 = LoadSrc(ps + 10 * sX, tailS);
                                    d4 = _mm512_fmadd_ps(s0, w6, d4);
                                    d5 = _mm512_fmadd_ps(s0, w5, d5);

                                    s1 = LoadSrc(ps + 11 * sX, tailS);
                                    d5 = _mm512_fmadd_ps(s1, w6, d5);
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

        static SIMD_INLINE bool Preferable_k7p3d1s1w8(const ConvParam& p)
        {
            return p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) &&
                (p.srcW > 12 && AlignHiAny(p.srcW, 8) < AlignHiAny(p.srcW, 6) * 1.2);
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k7p3d1s1w8(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && p.srcW >= 15);
            const T* src = (T*)src8;
            size_t srcH = p.srcH, srcW = p.srcW;
            size_t sM = (a.bufH[1] - 1), sD = a.bufH[1] ? a.bufH[1] * p.srcW * F : F, sX = a.bufH[1] ? F : p.srcC, sY = sX * p.srcW, dstC = maC;
            size_t dX = (a.bufH[2] ? a.maC * 2 : p.dstC * a.elem[1]), dY = p.dstW * dX, dy0 = a.bufH[2] ? yBeg : 0, dD = a.bufH[2] ? F * 2 : F * a.elem[1];
            size_t wD = 49 * F, dstCF = AlignLo(dstC, F), dstW = p.dstW, endW = dstW - 8;
            size_t dstCe = a.bufH[2] ? AlignHi(dstC, DF) : dstC;

            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, d6, d7;

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
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const T* ps = src + (sy & sM) * sY + (dx - 3) * sX;
                            const float* pw = weight + ky * 7 * F;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tailS, pw + 0 * F);
                                w1 = _mm512_maskz_loadu_ps(tailS, pw + 1 * F);
                                w2 = _mm512_maskz_loadu_ps(tailS, pw + 2 * F);
                                if (dx)
                                {
                                    s0 = LoadSrc(ps + 0 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = LoadSrc(ps + 1 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = LoadSrc(ps + 2 * sX, tailS);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
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
                                w5 = _mm512_maskz_loadu_ps(tailS, pw + 5 * F);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = LoadSrc(ps + 6 * sX, tailS);
                                w6 = _mm512_maskz_loadu_ps(tailS, pw + 6 * F);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);
                                d6 = _mm512_fmadd_ps(s0, w0, d6);

                                s1 = LoadSrc(ps + 7 * sX, tailS);
                                d1 = _mm512_fmadd_ps(s1, w6, d1);
                                d2 = _mm512_fmadd_ps(s1, w5, d2);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);
                                d6 = _mm512_fmadd_ps(s1, w1, d6);
                                d7 = _mm512_fmadd_ps(s1, w0, d7);

                                s0 = LoadSrc(ps + 8 * sX, tailS);
                                d2 = _mm512_fmadd_ps(s0, w6, d2);
                                d3 = _mm512_fmadd_ps(s0, w5, d3);
                                d4 = _mm512_fmadd_ps(s0, w4, d4);
                                d5 = _mm512_fmadd_ps(s0, w3, d5);
                                d6 = _mm512_fmadd_ps(s0, w2, d6);
                                d7 = _mm512_fmadd_ps(s0, w1, d7);

                                s1 = LoadSrc(ps + 9 * sX, tailS);
                                d3 = _mm512_fmadd_ps(s1, w6, d3);
                                d4 = _mm512_fmadd_ps(s1, w5, d4);
                                d5 = _mm512_fmadd_ps(s1, w4, d5);
                                d6 = _mm512_fmadd_ps(s1, w3, d6);
                                d7 = _mm512_fmadd_ps(s1, w2, d7);

                                s0 = LoadSrc(ps + 10 * sX, tailS);
                                d4 = _mm512_fmadd_ps(s0, w6, d4);
                                d5 = _mm512_fmadd_ps(s0, w5, d5);
                                d6 = _mm512_fmadd_ps(s0, w4, d6);
                                d7 = _mm512_fmadd_ps(s0, w3, d7);

                                if (dx < endW)
                                {
                                    s1 = LoadSrc(ps + 11 * sX, tailS);
                                    d5 = _mm512_fmadd_ps(s1, w6, d5);
                                    d6 = _mm512_fmadd_ps(s1, w5, d6);
                                    d7 = _mm512_fmadd_ps(s1, w4, d7);

                                    s0 = LoadSrc(ps + 12 * sX, tailS);
                                    d6 = _mm512_fmadd_ps(s0, w6, d6);
                                    d7 = _mm512_fmadd_ps(s0, w5, d7);

                                    s1 = LoadSrc(ps + 13 * sX, tailS);
                                    d7 = _mm512_fmadd_ps(s1, w6, d7);
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

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void DepthwiseConvolution_k5p2d1s1w8(const uint8_t* src8,
            const ConvParam& p, const AlgParam& a, size_t maC, size_t yBeg, size_t yEnd, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8));
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
                    for (size_t dx = 0; dx < dstW; dx += 8)
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
                    }
                }
                src += sD;
                dst += dD;
                weight += wD;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type, bool nofma> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            if (p.IsKernel(5) && p.IsPad(2) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8))
                depthwise = DepthwiseConvolution_k5p2d1s1w8<T, term, type>;
            else if (Preferable_k7p3d1s1w8(p))
                depthwise = DepthwiseConvolution_k7p3d1s1w8<T, term, type>;
            else if (Preferable_k7p3d1s1w6(p))
                depthwise = DepthwiseConvolution_k7p3d1s1w6<T, term, type>;
            else if (Preferable_k7p3d1s1w4(p))
                depthwise = DepthwiseConvolution_k7p3d1s1w4<T, term, type>;
            else if (IsKernel(p, 3) && IsDilation(p, 1) && Aligned(p.dstC, F))
                depthwise = DepthwiseConvolution3x3<T, term, type, nofma>;
            else if(p.padX + p.padW > 2 && p.srcC >= 128)
                depthwise = DepthwiseConvolutionLargePad<T, term, type>;
            else
                depthwise = DepthwiseConvolutionDefault<T, term, type, nofma>;
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetDepthwise(const ConvParam& p, DepthwisePtr& depthwise)
        {
            return Base::FmaAvoid(p.compatibility) ? SetDepthwise<T, term, type, true>(p, depthwise) : SetDepthwise<T, term, type, false>(p, depthwise);
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

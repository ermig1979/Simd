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
#include "Simd/SimdSynetConvolution32f.h"
#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdGemm.h"
#include "Simd/SimdExp.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwiseDefault(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcW = p.srcW, strideX = p.strideX, dilationX = p.dilationX, kernelX = p.kernelY, sX = strideX * p.dstC;
            size_t dstC = p.dstC, dstCF = AlignLo(p.dstC, F), dstC2F = AlignLo(p.dstC, 2 * F), dstC4F = AlignLo(p.dstC, 4 * F);
            size_t dstW2 = AlignLo(p.dstW, 2), dstW4 = AlignLo(p.dstW, 4);
            __m512 d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, w0;
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < dstW4; dx += 4)
                {
                    float* dst0 = dst + 0 * p.dstC, * dst1 = dst + 1 * p.dstC, * dst2 = dst + 2 * p.dstC, * dst3 = dst + 3 * p.dstC;
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t dc = 0;
                    for (; dc < dstC4F; dc += 4 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                            d02 = _mm512_loadu_ps(bias + dc + 2 * F);
                            d03 = _mm512_loadu_ps(bias + dc + 3 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                            d02 = _mm512_setzero_ps();
                            d03 = _mm512_setzero_ps();
                        }
                        d10 = d00; d11 = d01; d12 = d02; d13 = d03;
                        d20 = d00; d21 = d01; d22 = d02; d23 = d03;
                        d30 = d00; d31 = d01; d32 = d02; d33 = d03;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX, * ps2 = ps0 + 2 * sX, * ps3 = ps0 + 3 * sX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 0 * F), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 0 * F), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 1 * F), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 1 * F), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 1 * F), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 1 * F), w0, d31, mask3);
                                    w0 = _mm512_loadu_ps(pw + 2 * F);
                                    d02 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 2 * F), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 2 * F), w0, d12, mask1);
                                    d22 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 2 * F), w0, d22, mask2);
                                    d32 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 2 * F), w0, d32, mask3);
                                    w0 = _mm512_loadu_ps(pw + 3 * F);
                                    d03 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 3 * F), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 3 * F), w0, d13, mask1);
                                    d23 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 3 * F), w0, d23, mask2);
                                    d33 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 3 * F), w0, d33, mask3);
                                }
                            }
                        }
                        _mm512_storeu_ps(dst0 + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst0 + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                        _mm512_storeu_ps(dst0 + dc + 2 * F, Activate<type>(d02, params, dc + 2 * F));
                        _mm512_storeu_ps(dst0 + dc + 3 * F, Activate<type>(d03, params, dc + 3 * F));
                        _mm512_storeu_ps(dst1 + dc + 0 * F, Activate<type>(d10, params, dc + 0 * F));
                        _mm512_storeu_ps(dst1 + dc + 1 * F, Activate<type>(d11, params, dc + 1 * F));
                        _mm512_storeu_ps(dst1 + dc + 2 * F, Activate<type>(d12, params, dc + 2 * F));
                        _mm512_storeu_ps(dst1 + dc + 3 * F, Activate<type>(d13, params, dc + 3 * F));
                        _mm512_storeu_ps(dst2 + dc + 0 * F, Activate<type>(d20, params, dc + 0 * F));
                        _mm512_storeu_ps(dst2 + dc + 1 * F, Activate<type>(d21, params, dc + 1 * F));
                        _mm512_storeu_ps(dst2 + dc + 2 * F, Activate<type>(d22, params, dc + 2 * F));
                        _mm512_storeu_ps(dst2 + dc + 3 * F, Activate<type>(d23, params, dc + 3 * F));
                        _mm512_storeu_ps(dst3 + dc + 0 * F, Activate<type>(d30, params, dc + 0 * F));
                        _mm512_storeu_ps(dst3 + dc + 1 * F, Activate<type>(d31, params, dc + 1 * F));
                        _mm512_storeu_ps(dst3 + dc + 2 * F, Activate<type>(d32, params, dc + 2 * F));
                        _mm512_storeu_ps(dst3 + dc + 3 * F, Activate<type>(d33, params, dc + 3 * F));
                    }
                    for (; dc < dstC2F; dc += 2 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                        }
                        d10 = d00; d11 = d01;
                        d20 = d00; d21 = d01;
                        d30 = d00; d31 = d01;

                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX, * ps2 = ps0 + 2 * sX, * ps3 = ps0 + 3 * sX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 0 * F), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 0 * F), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 1 * F), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 1 * F), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 1 * F), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 1 * F), w0, d31, mask3);
                                }
                            }
                        }
                        _mm512_storeu_ps(dst0 + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst0 + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                        _mm512_storeu_ps(dst1 + dc + 0 * F, Activate<type>(d10, params, dc + 0 * F));
                        _mm512_storeu_ps(dst1 + dc + 1 * F, Activate<type>(d11, params, dc + 1 * F));
                        _mm512_storeu_ps(dst2 + dc + 0 * F, Activate<type>(d20, params, dc + 0 * F));
                        _mm512_storeu_ps(dst2 + dc + 1 * F, Activate<type>(d21, params, dc + 1 * F));
                        _mm512_storeu_ps(dst3 + dc + 0 * F, Activate<type>(d30, params, dc + 0 * F));
                        _mm512_storeu_ps(dst3 + dc + 1 * F, Activate<type>(d31, params, dc + 1 * F));
                    }
                    for (; dc < dstC; dc += F)
                    {
                        __mmask16 tailC = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        d00 = bias ? _mm512_maskz_loadu_ps(tailC, bias + dc) : _mm512_setzero_ps();
                        d10 = d00; d20 = d00; d30 = d00;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? tailC : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? tailC : 0x0000;
                                    __mmask16 mask2 = sx + 2 * strideX < srcW ? tailC : 0x0000;
                                    __mmask16 mask3 = sx + 3 * strideX < srcW ? tailC : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX, * ps2 = ps0 + 2 * sX, * ps3 = ps0 + 3 * sX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask2, ps2 + 0 * F), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask3, ps3 + 0 * F), w0, d30, mask3);
                                }
                            }
                        }
                        _mm512_mask_storeu_ps(dst0 + dc, tailC, Activate<type>(d00, params, dc, tailC));
                        _mm512_mask_storeu_ps(dst1 + dc, tailC, Activate<type>(d10, params, dc, tailC));
                        _mm512_mask_storeu_ps(dst2 + dc, tailC, Activate<type>(d20, params, dc, tailC));
                        _mm512_mask_storeu_ps(dst3 + dc, tailC, Activate<type>(d30, params, dc, tailC));
                    }
                    dst += 4 * p.dstC;
                }
                for (; dx < dstW2; dx += 2)
                {
                    float* dst0 = dst + 0 * p.dstC, *dst1 = dst + 1 * p.dstC;
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t dc = 0;
                    for (; dc < dstC4F; dc += 4 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                            d02 = _mm512_loadu_ps(bias + dc + 2 * F);
                            d03 = _mm512_loadu_ps(bias + dc + 3 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                            d02 = _mm512_setzero_ps();
                            d03 = _mm512_setzero_ps();
                        }
                        d10 = d00; d11 = d01; d12 = d02; d13 = d03;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 1 * F), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 1 * F), w0, d11, mask1);                                    
                                    w0 = _mm512_loadu_ps(pw + 2 * F);
                                    d02 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 2 * F), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 2 * F), w0, d12, mask1);                                    
                                    w0 = _mm512_loadu_ps(pw + 3 * F);
                                    d03 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 3 * F), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 3 * F), w0, d13, mask1);
                                }
                            }
                        }
                        _mm512_storeu_ps(dst0 + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst0 + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                        _mm512_storeu_ps(dst0 + dc + 2 * F, Activate<type>(d02, params, dc + 2 * F));
                        _mm512_storeu_ps(dst0 + dc + 3 * F, Activate<type>(d03, params, dc + 3 * F));
                        _mm512_storeu_ps(dst1 + dc + 0 * F, Activate<type>(d10, params, dc + 0 * F));
                        _mm512_storeu_ps(dst1 + dc + 1 * F, Activate<type>(d11, params, dc + 1 * F));
                        _mm512_storeu_ps(dst1 + dc + 2 * F, Activate<type>(d12, params, dc + 2 * F));
                        _mm512_storeu_ps(dst1 + dc + 3 * F, Activate<type>(d13, params, dc + 3 * F));
                    }
                    for (; dc < dstC2F; dc += 2 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                        }
                        d10 = d00; d11 = d01;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? 0xFFFF : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 1 * F), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 1 * F), w0, d11, mask1);
                                }
                            }
                        }
                        _mm512_storeu_ps(dst0 + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst0 + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                        _mm512_storeu_ps(dst1 + dc + 0 * F, Activate<type>(d10, params, dc + 0 * F));
                        _mm512_storeu_ps(dst1 + dc + 1 * F, Activate<type>(d11, params, dc + 1 * F));
                    }
                    for (; dc < dstC; dc += F)
                    {
                        __mmask16 tailC = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        d00 = bias ? _mm512_maskz_loadu_ps(tailC, bias + dc) : _mm512_setzero_ps();
                        d10 = d00;
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            const float* psy = src + sy * p.srcW * dstC + dc;
                            const float* pwy = weight + ky * p.kernelX * dstC + dc;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dilationX;
                                    const float* pw = pwy + kx * dstC;
                                    __mmask16 mask0 = sx + 0 * strideX < srcW ? tailC : 0x0000;
                                    __mmask16 mask1 = sx + 1 * strideX < srcW ? tailC : 0x0000;
                                    const float* ps0 = psy + sx * dstC, * ps1 = ps0 + 1 * sX;

                                    w0 = _mm512_maskz_loadu_ps(tailC, pw);
                                    d00 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask0, ps0 + 0 * F), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(_mm512_maskz_loadu_ps(mask1, ps1 + 0 * F), w0, d10, mask1);
                                }
                            }
                        }
                        _mm512_mask_storeu_ps(dst0 + dc, tailC, Activate<type>(d00, params, dc, tailC));
                        _mm512_mask_storeu_ps(dst1 + dc, tailC, Activate<type>(d10, params, dc, tailC));
                    }
                    dst += 2 * p.dstC;
                }
                for (; dx < p.dstW; ++dx)
                {
                    size_t dc = 0;
                    for (; dc < dstC4F; dc += 4 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                            d02 = _mm512_loadu_ps(bias + dc + 2 * F);
                            d03 = _mm512_loadu_ps(bias + dc + 3 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                            d02 = _mm512_setzero_ps();
                            d03 = _mm512_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)* dstC + dc;
                                        const float * ps = src + (sy*p.srcW + sx)* dstC + dc;
                                        d00 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                        d01 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                                        d02 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * F), _mm512_loadu_ps(pw + 2 * F), d02);
                                        d03 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 3 * F), _mm512_loadu_ps(pw + 3 * F), d03);
                                    }
                                }
                            }
                        }
                        _mm512_storeu_ps(dst + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                        _mm512_storeu_ps(dst + dc + 2 * F, Activate<type>(d02, params, dc + 2 * F));
                        _mm512_storeu_ps(dst + dc + 3 * F, Activate<type>(d03, params, dc + 3 * F));
                    }
                    for (; dc < dstC2F; dc += 2 * F)
                    {
                        if (bias)
                        {
                            d00 = _mm512_loadu_ps(bias + dc + 0 * F);
                            d01 = _mm512_loadu_ps(bias + dc + 1 * F);
                        }
                        else
                        {
                            d00 = _mm512_setzero_ps();
                            d01 = _mm512_setzero_ps();
                        }
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)* dstC + dc;
                                        const float * ps = src + (sy*p.srcW + sx)* dstC + dc;
                                        d00 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                        d01 = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                                    }
                                }
                            }
                        }
                        _mm512_storeu_ps(dst + dc + 0 * F, Activate<type>(d00, params, dc + 0 * F));
                        _mm512_storeu_ps(dst + dc + 1 * F, Activate<type>(d01, params, dc + 1 * F));
                    }
                    for (; dc < dstC; dc += F)
                    {
                        __mmask16 tailC = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        d00 = bias ? _mm512_maskz_loadu_ps(tailC, bias + dc) : _mm512_setzero_ps();
                        for (size_t ky = 0; ky < p.kernelY; ++ky)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; ++kx)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        const float * pw = weight + (ky*p.kernelX + kx)* dstC + dc;
                                        const float * ps = src + (sy*p.srcW + sx)* dstC + dc;
                                        d00 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tailC, ps), _mm512_maskz_loadu_ps(tailC, pw), d00);
                                    }
                                }
                            }
                        }
                        _mm512_mask_storeu_ps(dst + dc, tailC, Activate<type>(d00, params, dc, tailC));
                    }
                    dst += p.dstC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Edge(const float * src, const ConvParam & p, size_t dy, size_t dx, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcC = p.srcC;
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky * 3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), _mm512_loadu_ps(pw), sum);
                            }
                        }
                    }
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t sy = dy * p.strideY + ky - p.padY;
                    if (sy < p.srcH)
                    {
                        for (size_t kx = 0; kx < 3; ++kx)
                        {
                            size_t sx = dx * p.strideX + kx - p.padX;
                            if (sx < p.srcW)
                            {
                                const float * pw = weight + (ky*3 + kx) * srcC;
                                const float * ps = src + (sy*p.srcW + sx) * srcC;
                                sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps), _mm512_maskz_loadu_ps(tail, pw), sum);
                            }
                        }
                    }
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main1(const float * src, size_t srcS, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 0 * srcC), _mm512_loadu_ps(pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 1 * srcC), _mm512_loadu_ps(pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps + 2 * srcC), _mm512_loadu_ps(pw + 2 * srcC), sum);
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum, params, c));
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps = src + ky * srcS;
                    const float * pw = weight + ky * 3 * srcC;
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 0 * srcC), _mm512_maskz_loadu_ps(tail, pw + 0 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 1 * srcC), _mm512_maskz_loadu_ps(tail, pw + 1 * srcC), sum);
                    sum = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps + 2 * srcC), _mm512_maskz_loadu_ps(tail, pw + 2 * srcC), sum);
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main2(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m512 sum0, sum1, w0;
            for (; c < srcCF; c += F)
            {
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_storeu_ps(dst + c, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + c + srcC, Activate<type>(sum1, params, c));
                src += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                const float * pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const float * ps0 = src + ky * srcS;
                    const float * ps1 = ps0 + srcX;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 0 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 0 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 1 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 1 * srcC), w0, sum1);
                    pw += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + 2 * srcC), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + 2 * srcC), w0, sum1);
                    pw += srcC;
                }
                _mm512_mask_storeu_ps(dst + c, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + c + srcC, tail, Activate<type>(sum1, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main4(const float * src, size_t srcS, size_t srcX, size_t srcC, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_loadu_ps(bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);                    
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);                    
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_loadu_ps(ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_storeu_ps(dst + 0 * srcC, Activate<type>(sum0, params, c));
                _mm512_storeu_ps(dst + 1 * srcC, Activate<type>(sum1, params, c));
                _mm512_storeu_ps(dst + 2 * srcC, Activate<type>(sum2, params, c));
                _mm512_storeu_ps(dst + 3 * srcC, Activate<type>(sum3, params, c));
                src += F;
                dst += F;
            }
            if (c < srcC)
            {
                __mmask16 tail = TailMask16(srcC - c);
                __m512 sum0, sum1, sum2, sum3, w0;
                sum0 = bias ? _mm512_maskz_loadu_ps(tail, bias + c) : _mm512_setzero_ps();
                sum1 = sum0;
                sum2 = sum0;
                sum3 = sum0;
                const float * pw = weight + c;
                const float * ps0 = src + 0 * srcX;
                const float * ps1 = src + 1 * srcX;
                const float * ps2 = src + 2 * srcX;
                const float * ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_maskz_loadu_ps(tail, pw);
                    sum0 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps0 + offset), w0, sum0);
                    sum1 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps1 + offset), w0, sum1);
                    sum2 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps2 + offset), w0, sum2);
                    sum3 = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(tail, ps3 + offset), w0, sum3);
                    pw += srcC, offset += srcC;
                }
                _mm512_mask_storeu_ps(dst + 0 * srcC, tail, Activate<type>(sum0, params, c, tail));
                _mm512_mask_storeu_ps(dst + 1 * srcC, tail, Activate<type>(sum1, params, c, tail));
                _mm512_mask_storeu_ps(dst + 2 * srcC, tail, Activate<type>(sum2, params, c, tail));
                _mm512_mask_storeu_ps(dst + 3 * srcC, tail, Activate<type>(sum3, params, c, tail));
            }
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Edge16(const float * src, const ConvParam & p, size_t dy, size_t dx, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                size_t sy = dy * p.strideY + ky - p.padY;
                if (sy < p.srcH)
                {
                    for (size_t kx = 0; kx < 3; ++kx)
                    {
                        size_t sx = dx * p.strideX + kx - p.padX;
                        if (sx < p.srcW)
                        {
                            const float * ps = src + (sy*p.srcW + sx) * F;
                            sum = _mm512_fmadd_ps(_mm512_loadu_ps(ps), weight[ky * 3 + kx], sum);
                        }
                    }
                }
            }
            _mm512_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main16x1(const float * src, size_t srcS, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum = bias;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum);
            src += srcS;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[3], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[4], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[5], sum);
            src += srcS;
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[6], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[7], sum);
            sum = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[8], sum);
            _mm512_storeu_ps(dst, Activate<type>(sum, params, 0));
        }

        template<::SimdConvolutionActivationType type> 
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main16x2(const float * src, size_t srcS, const __m512 * weight, __m512 bias, const float * params, float * dst)
        {
            __m512 sum0 = bias;
            __m512 sum1 = bias;
            for (size_t ky = 0; ky < 3; ++ky)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0 * F);
                __m512 s1 = _mm512_loadu_ps(src + 1 * F);
                __m512 s2 = _mm512_loadu_ps(src + 2 * F);
                __m512 s3 = _mm512_loadu_ps(src + 3 * F);
                sum0 = _mm512_fmadd_ps(s0, weight[0], sum0);
                sum1 = _mm512_fmadd_ps(s1, weight[0], sum1);
                sum0 = _mm512_fmadd_ps(s1, weight[1], sum0);
                sum1 = _mm512_fmadd_ps(s2, weight[1], sum1);
                sum0 = _mm512_fmadd_ps(s2, weight[2], sum0);
                sum1 = _mm512_fmadd_ps(s3, weight[2], sum1);
                src += srcS;
                weight += 3;
            }
            _mm512_storeu_ps(dst + 0, Activate<type>(sum0, params, 0));
            _mm512_storeu_ps(dst + F, Activate<type>(sum1, params, 0));
        }

        template<::SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution32fNhwcDepthwise3x3Main48(const float * src, size_t srcS, const __m512 * weight, const float * bias, const float * params, float * dst)
        {
            __m512 sum0, sum1, sum2;
            if (bias)
            {
                sum0 = _mm512_loadu_ps(bias + 0 * F);
                sum1 = _mm512_loadu_ps(bias + 1 * F);
                sum2 = _mm512_loadu_ps(bias + 2 * F);
            }
            else
            {
                sum0 = _mm512_setzero_ps();
                sum1 = _mm512_setzero_ps();
                sum2 = _mm512_setzero_ps();
            }
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            src += srcS;
            weight += 9;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            src += srcS;
            weight += 9;
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 0 * F), weight[0], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 1 * F), weight[1], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 2 * F), weight[2], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 3 * F), weight[3], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 4 * F), weight[4], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 5 * F), weight[5], sum2);
            sum0 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 6 * F), weight[6], sum0);
            sum1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 7 * F), weight[7], sum1);
            sum2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 8 * F), weight[8], sum2);
            _mm512_storeu_ps(dst + 0 * F, Activate<type>(sum0, params, 0 * F));
            _mm512_storeu_ps(dst + 1 * F, Activate<type>(sum1, params, 1 * F));
            _mm512_storeu_ps(dst + 2 * F, Activate<type>(sum2, params, 2 * F));
        }

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise3x3(const float * src, const ConvParam & p, const float * weight, const float * bias, const float * params, float * dst)
        {
            size_t srcS = p.srcC*p.srcW;
            size_t srcX = p.srcC*p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            if (p.dstC == F && p.strideX == 1)
            {
                __m512 _weight[9];
                for (size_t i = 0; i < 9; ++i)
                    _weight[i] = _mm512_loadu_ps(weight + i * F);
                __m512 _bias = bias ? _mm512_loadu_ps(bias) : _mm512_setzero_ps();
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    for (; dx < dstW2; dx += 2)
                        Convolution32fNhwcDepthwise3x3Main16x2<type>(src + offset, srcS, _weight, _bias, params, dst), offset += 2*F, dst += 2*F;
                    for (; dx < dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Main16x1<type>(src + offset, srcS, _weight, _bias, params, dst), offset += F, dst += F;
                    for (; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge16<type>(src, p, dy, dx, _weight, _bias, params, dst), dst += F;
            }
            else
            {
                size_t dy = 0;
                for (; dy < p.padY; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                for (; dy < dstH; ++dy)
                {
                    size_t dx = 0;
                    for (; dx < p.padX; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                    size_t offset = ((dy * p.strideY - p.padY)*p.srcW + dx * p.strideX - p.padX)*p.srcC;
                    if (p.srcC == 48)
                    {
                        __m512 _weight[27];
                        for (size_t i = 0; i < 27; ++i)
                            _weight[i] = _mm512_loadu_ps(weight + i * F);
                        for (; dx < dstW; ++dx)
                            Convolution32fNhwcDepthwise3x3Main48<type>(src + offset, srcS, _weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    }
                    else
                        for (; dx < dstW4; dx += 4)
                            Convolution32fNhwcDepthwise3x3Main4<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 4 * p.dstC, offset += 4 * srcX;
                    for (; dx < dstW2; dx += 2)
                        Convolution32fNhwcDepthwise3x3Main2<type>(src + offset, srcS, srcX, p.srcC, weight, bias, params, dst), dst += 2 * p.dstC, offset += 2 * srcX;
                    for (; dx < dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Main1<type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += p.dstC, offset += srcX;
                    for (; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
                }
                for (; dy < p.dstH; ++dy)
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                        Convolution32fNhwcDepthwise3x3Edge<type>(src, p, dy, dx, weight, bias, params, dst), dst += p.dstC;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise_k7p3d1s1w4(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 4));

            size_t dstC = p.dstC, dstCF = AlignLo(p.dstC, F), dstW = p.dstW, srcH = p.srcH, end = dstW - 4;
            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; dx += 4)
                {
                    for (size_t dc = 0; dc < dstC; dc += F)
                    {
                        __mmask16 tail = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _params[0] = _mm512_maskz_loadu_ps(tail, params + dc);
                        d0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                        d1 = d0; d2 = d0; d3 = d0;
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const float* ps = src + (sy * dstW + dx - 3) * dstC + dc;
                            const float* pw = weight + ky * 7 * dstC + dc;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * dstC);
                                w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * dstC);
                                w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * dstC);
                                if (dx)
                                {
                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * dstC);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
                                s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * dstC);
                                w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w3, d0);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * dstC);
                                w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w4, d0);
                                d1 = _mm512_fmadd_ps(s0, w3, d1);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * dstC);
                                w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * dstC);
                                w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                if (dx < end)
                                {
                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * dstC);
                                    d1 = _mm512_fmadd_ps(s1, w6, d1);
                                    d2 = _mm512_fmadd_ps(s1, w5, d2);
                                    d3 = _mm512_fmadd_ps(s1, w4, d3);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * dstC);
                                    d2 = _mm512_fmadd_ps(s0, w6, d2);
                                    d3 = _mm512_fmadd_ps(s0, w5, d3);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * dstC);
                                    d3 = _mm512_fmadd_ps(s1, w6, d3);
                                }
                            }
                        }
                        float* pd = dst + (dy * dstW + dx) * dstC + dc;
                        _mm512_mask_storeu_ps(pd + 0 * dstC, tail, Activate<type>(d0, _params, 0));
                        _mm512_mask_storeu_ps(pd + 1 * dstC, tail, Activate<type>(d1, _params, 0));
                        _mm512_mask_storeu_ps(pd + 2 * dstC, tail, Activate<type>(d2, _params, 0));
                        _mm512_mask_storeu_ps(pd + 3 * dstC, tail, Activate<type>(d3, _params, 0));
                    }
                }
            }
        }

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise_k7p3d1s1w6(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && AlignedAny(p.srcW, 6));

            size_t dstC = p.dstC, dstCF = AlignLo(p.dstC, F), dstW = p.dstW, srcH = p.srcH, end = dstW - 6;
            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; dx += 6)
                {
                    for (size_t dc = 0; dc < dstC; dc += F)
                    {
                        __mmask16 tail = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _params[0] = _mm512_maskz_loadu_ps(tail, params + dc);
                        d0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                        d1 = d0; d2 = d0; d3 = d0, d4 = d0, d5 = d0;
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const float* ps = src + (sy * dstW + dx - 3) * dstC + dc;
                            const float* pw = weight + ky * 7 * dstC + dc;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * dstC);
                                w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * dstC);
                                w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * dstC);
                                if (dx)
                                {
                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * dstC);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
                                s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * dstC);
                                w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w3, d0);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * dstC);
                                w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w4, d0);
                                d1 = _mm512_fmadd_ps(s0, w3, d1);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);
                                d4 = _mm512_fmadd_ps(s0, w0, d4);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * dstC);
                                w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * dstC);
                                w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * dstC);
                                d1 = _mm512_fmadd_ps(s1, w6, d1);
                                d2 = _mm512_fmadd_ps(s1, w5, d2);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * dstC);
                                d2 = _mm512_fmadd_ps(s0, w6, d2);
                                d3 = _mm512_fmadd_ps(s0, w5, d3);
                                d4 = _mm512_fmadd_ps(s0, w4, d4);
                                d5 = _mm512_fmadd_ps(s0, w3, d5);

                                if (dx < end)
                                {
                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * dstC);
                                    d3 = _mm512_fmadd_ps(s1, w6, d3);
                                    d4 = _mm512_fmadd_ps(s1, w5, d4);
                                    d5 = _mm512_fmadd_ps(s1, w4, d5);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 10 * dstC);
                                    d4 = _mm512_fmadd_ps(s0, w6, d4);
                                    d5 = _mm512_fmadd_ps(s0, w5, d5);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 11 * dstC);
                                    d5 = _mm512_fmadd_ps(s1, w6, d5);
                                }
                            }
                        }
                        float* pd = dst + (dy * dstW + dx) * dstC + dc;
                        _mm512_mask_storeu_ps(pd + 0 * dstC, tail, Activate<type>(d0, _params, 0));
                        _mm512_mask_storeu_ps(pd + 1 * dstC, tail, Activate<type>(d1, _params, 0));
                        _mm512_mask_storeu_ps(pd + 2 * dstC, tail, Activate<type>(d2, _params, 0));
                        _mm512_mask_storeu_ps(pd + 3 * dstC, tail, Activate<type>(d3, _params, 0));
                        _mm512_mask_storeu_ps(pd + 4 * dstC, tail, Activate<type>(d4, _params, 0));
                        _mm512_mask_storeu_ps(pd + 5 * dstC, tail, Activate<type>(d5, _params, 0));
                    }
                }
            }
        }

        template<::SimdConvolutionActivationType type> void Convolution32fNhwcDepthwise_k7p3d1s1w8(const float* src, const ConvParam& p, const float* weight, const float* bias, const float* params, float* dst)
        {
            assert(p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8));

            size_t dstC = p.dstC, dstCF = AlignLo(p.dstC, F), dstW = p.dstW, srcH = p.srcH, end = dstW - 8;
            __m512 s0, s1, w0, w1, w2, w3, w4, w5, w6, d0, d1, d2, d3, d4, d5, d6, d7, _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == SimdConvolutionActivationRestrictRange ||
                type == SimdConvolutionActivationHswish ||
                type == SimdConvolutionActivationHardSigmoid)
                _params[1] = _mm512_set1_ps(params[1]);
            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                for (size_t dx = 0; dx < dstW; dx += 8)
                {
                    for (size_t dc = 0; dc < dstC; dc += F)
                    {
                        __mmask16 tail = dc < dstCF ? __mmask16(-1) : TailMask16(dstC - dc);
                        if (type == SimdConvolutionActivationPrelu)
                            _params[0] = _mm512_maskz_loadu_ps(tail, params + dc);
                        d0 = bias ? _mm512_maskz_loadu_ps(tail, bias + dc) : _mm512_setzero_ps();
                        d1 = d0; d2 = d0; d3 = d0, d4 = d0, d5 = d0, d6 = d0, d7 = d0;
                        for (size_t ky = 0; ky < 7; ++ky)
                        {
                            size_t sy = dy + ky - 3;
                            const float* ps = src + (sy * dstW + dx - 3) * dstC + dc;
                            const float* pw = weight + ky * 7 * dstC + dc;
                            if (sy < srcH)
                            {
                                w0 = _mm512_maskz_loadu_ps(tail, pw + 0 * dstC);
                                w1 = _mm512_maskz_loadu_ps(tail, pw + 1 * dstC);
                                w2 = _mm512_maskz_loadu_ps(tail, pw + 2 * dstC);
                                if (dx)
                                {
                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 0 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w0, d0);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 1 * dstC);
                                    d0 = _mm512_fmadd_ps(s1, w1, d0);
                                    d1 = _mm512_fmadd_ps(s1, w0, d1);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 2 * dstC);
                                    d0 = _mm512_fmadd_ps(s0, w2, d0);
                                    d1 = _mm512_fmadd_ps(s0, w1, d1);
                                    d2 = _mm512_fmadd_ps(s0, w0, d2);
                                }
                                s1 = _mm512_maskz_loadu_ps(tail, ps + 3 * dstC);
                                w3 = _mm512_maskz_loadu_ps(tail, pw + 3 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w3, d0);
                                d1 = _mm512_fmadd_ps(s1, w2, d1);
                                d2 = _mm512_fmadd_ps(s1, w1, d2);
                                d3 = _mm512_fmadd_ps(s1, w0, d3);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 4 * dstC);
                                w4 = _mm512_maskz_loadu_ps(tail, pw + 4 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w4, d0);
                                d1 = _mm512_fmadd_ps(s0, w3, d1);
                                d2 = _mm512_fmadd_ps(s0, w2, d2);
                                d3 = _mm512_fmadd_ps(s0, w1, d3);
                                d4 = _mm512_fmadd_ps(s0, w0, d4);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 5 * dstC);
                                w5 = _mm512_maskz_loadu_ps(tail, pw + 5 * dstC);
                                d0 = _mm512_fmadd_ps(s1, w5, d0);
                                d1 = _mm512_fmadd_ps(s1, w4, d1);
                                d2 = _mm512_fmadd_ps(s1, w3, d2);
                                d3 = _mm512_fmadd_ps(s1, w2, d3);
                                d4 = _mm512_fmadd_ps(s1, w1, d4);
                                d5 = _mm512_fmadd_ps(s1, w0, d5);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 6 * dstC);
                                w6 = _mm512_maskz_loadu_ps(tail, pw + 6 * dstC);
                                d0 = _mm512_fmadd_ps(s0, w6, d0);
                                d1 = _mm512_fmadd_ps(s0, w5, d1);
                                d2 = _mm512_fmadd_ps(s0, w4, d2);
                                d3 = _mm512_fmadd_ps(s0, w3, d3);
                                d4 = _mm512_fmadd_ps(s0, w2, d4);
                                d5 = _mm512_fmadd_ps(s0, w1, d5);
                                d6 = _mm512_fmadd_ps(s0, w0, d6);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 7 * dstC);
                                d1 = _mm512_fmadd_ps(s1, w6, d1);
                                d2 = _mm512_fmadd_ps(s1, w5, d2);
                                d3 = _mm512_fmadd_ps(s1, w4, d3);
                                d4 = _mm512_fmadd_ps(s1, w3, d4);
                                d5 = _mm512_fmadd_ps(s1, w2, d5);
                                d6 = _mm512_fmadd_ps(s1, w1, d6);
                                d7 = _mm512_fmadd_ps(s1, w0, d7);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 8 * dstC);
                                d2 = _mm512_fmadd_ps(s0, w6, d2);
                                d3 = _mm512_fmadd_ps(s0, w5, d3);
                                d4 = _mm512_fmadd_ps(s0, w4, d4);
                                d5 = _mm512_fmadd_ps(s0, w3, d5);
                                d6 = _mm512_fmadd_ps(s0, w2, d6);
                                d7 = _mm512_fmadd_ps(s0, w1, d7);

                                s1 = _mm512_maskz_loadu_ps(tail, ps + 9 * dstC);
                                d3 = _mm512_fmadd_ps(s1, w6, d3);
                                d4 = _mm512_fmadd_ps(s1, w5, d4);
                                d5 = _mm512_fmadd_ps(s1, w4, d5);
                                d6 = _mm512_fmadd_ps(s1, w3, d6);
                                d7 = _mm512_fmadd_ps(s1, w2, d7);

                                s0 = _mm512_maskz_loadu_ps(tail, ps + 10 * dstC);
                                d4 = _mm512_fmadd_ps(s0, w6, d4);
                                d5 = _mm512_fmadd_ps(s0, w5, d5);
                                d6 = _mm512_fmadd_ps(s0, w4, d6);
                                d7 = _mm512_fmadd_ps(s0, w3, d7);

                                if (dx < end)
                                {
                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 11 * dstC);
                                    d5 = _mm512_fmadd_ps(s1, w6, d5);
                                    d6 = _mm512_fmadd_ps(s1, w5, d6);
                                    d7 = _mm512_fmadd_ps(s1, w4, d7);

                                    s0 = _mm512_maskz_loadu_ps(tail, ps + 12 * dstC);
                                    d6 = _mm512_fmadd_ps(s0, w6, d6);
                                    d7 = _mm512_fmadd_ps(s0, w5, d7);

                                    s1 = _mm512_maskz_loadu_ps(tail, ps + 13 * dstC);
                                    d7 = _mm512_fmadd_ps(s1, w6, d7);
                                }
                            }
                        }
                        float* pd = dst + (dy * dstW + dx) * dstC + dc;
                        _mm512_mask_storeu_ps(pd + 0 * dstC, tail, Activate<type>(d0, _params, 0));
                        _mm512_mask_storeu_ps(pd + 1 * dstC, tail, Activate<type>(d1, _params, 0));
                        _mm512_mask_storeu_ps(pd + 2 * dstC, tail, Activate<type>(d2, _params, 0));
                        _mm512_mask_storeu_ps(pd + 3 * dstC, tail, Activate<type>(d3, _params, 0));
                        _mm512_mask_storeu_ps(pd + 4 * dstC, tail, Activate<type>(d4, _params, 0));
                        _mm512_mask_storeu_ps(pd + 5 * dstC, tail, Activate<type>(d5, _params, 0));
                        _mm512_mask_storeu_ps(pd + 6 * dstC, tail, Activate<type>(d6, _params, 0));
                        _mm512_mask_storeu_ps(pd + 7 * dstC, tail, Activate<type>(d7, _params, 0));
                    }
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <::SimdConvolutionActivationType type> SynetConvolution32fNhwcDepthwise::ConvolutionPtr Get(const ConvParam& p)
        {
            if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 8))
                return Convolution32fNhwcDepthwise_k7p3d1s1w8<type>;
            else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && AlignedAny(p.srcW, 6))
                return Convolution32fNhwcDepthwise_k7p3d1s1w6<type>;
            else if (p.IsKernel(7) && p.IsPad(3) && p.IsStride(1) && p.IsDilation(1) && Aligned(p.srcW, 4))
                return Convolution32fNhwcDepthwise_k7p3d1s1w4<type>;
            else if (p.IsKernel(3) && p.IsDilation(1))
                return Convolution32fNhwcDepthwise3x3<type>;
            else
                return Convolution32fNhwcDepthwiseDefault<type>;
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution32fNhwcDepthwise::SynetConvolution32fNhwcDepthwise(const ConvParam& p)
            : Avx2::SynetConvolution32fNhwcDepthwise(p)
        {
            if (p.dstC > HF && p.dstC != 24 && p.dstH >= p.padY + p.padH && p.dstW >= p.padX + p.padW)
            {
                switch (p.activation)
                {
                case ::SimdConvolutionActivationIdentity: _convolution = Get<::SimdConvolutionActivationIdentity>(p); break;
                case ::SimdConvolutionActivationRelu: _convolution = Get<::SimdConvolutionActivationRelu>(p); break;
                case ::SimdConvolutionActivationLeakyRelu: _convolution = Get<::SimdConvolutionActivationLeakyRelu>(p); break;
                case ::SimdConvolutionActivationRestrictRange: _convolution = Get<::SimdConvolutionActivationRestrictRange>(p); break;
                case ::SimdConvolutionActivationPrelu: _convolution = Get<::SimdConvolutionActivationPrelu>(p); break;
                case ::SimdConvolutionActivationElu: _convolution = Get<::SimdConvolutionActivationElu>(p); break;
                case ::SimdConvolutionActivationHswish: _convolution = Get<::SimdConvolutionActivationHswish>(p); break;
                case ::SimdConvolutionActivationMish: _convolution = Get<::SimdConvolutionActivationMish>(p); break;
                case ::SimdConvolutionActivationHardSigmoid: _convolution = Get<::SimdConvolutionActivationHardSigmoid>(p); break;
                case ::SimdConvolutionActivationSwish: _convolution = Get<::SimdConvolutionActivationSwish>(p); break;
                case ::SimdConvolutionActivationGelu: _convolution = Get<::SimdConvolutionActivationGelu>(p); break;
                }
            }
        }
    }
#endif
}

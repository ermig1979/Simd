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
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdAlignment.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Avx512bw
    {
        template <Term16bType term> struct DepthwiseTerm16b
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m512 value, const float* params, size_t offset, __mmask16 tail = __mmask16(-1));
        };

        template <> struct DepthwiseTerm16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m512 value, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(value, params, offset);
                _mm256_mask_storeu_epi16(ptr + offset * 2, tail, _mm512_cvtepi32_epi16(Float32ToBFloat16(f32)));
            }
        };

        template <> struct DepthwiseTerm16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m512 value, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(value, params, offset);
                _mm512_mask_storeu_ps((float*)ptr + offset, tail, f32);
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, __m512 val0, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            DepthwiseTerm16b<term>::template Save<type>(ptr, val0, params, offset, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwiseDefault(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.trans && p.IsDepthwise());
            const T* src = (T*)src8;
            size_t kX = p.kernelX, kY = p.kernelY, dX = p.dilationX, dY = p.dilationY, sX = p.strideX, sY = p.strideY, srcH = p.srcH, srcW = p.srcW;
            size_t size = p.group, elem = (term == Term16bLast16b ? 2 : 4), sdS = size * dX, ssX = size * sX, dstC = size * elem;
            size_t sizeF = AlignLo(size, F), size2F = AlignLo(size, 2 * F), size4F = AlignLo(size, 4 * F), size8F = AlignLo(size, 8 * F);
            size_t dstW = p.dstW, dstW2 = AlignLo(dstW, 2), dstW4 = AlignLo(dstW, 4);
            __mmask16 tail = TailMask16(size - sizeF);
            __m512 d00, d01, d02, d03, d10, d11, d12, d13, d20, d21, d22, d23, d30, d31, d32, d33, w0;

            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                size_t sy0 = dy * p.strideY - p.padY;
                size_t dx = 0;
                for (; dx < dstW4; dx += 4)
                {
                    uint8_t* dst0 = dst + 0 * dstC, * dst1 = dst + 1 * dstC, * dst2 = dst + 2 * dstC, * dst3 = dst + 3 * dstC;
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t c = 0;
                    for (; c < size4F; c += 4 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        d02 = _mm512_loadu_ps(bias + c + 2 * F);
                        d03 = _mm512_loadu_ps(bias + c + 3 * F);
                        d10 = d00; d11 = d01; d12 = d02; d13 = d03;
                        d20 = d00; d21 = d01; d22 = d02; d23 = d03;
                        d30 = d00; d31 = d01; d32 = d02; d33 = d03;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * F, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * F, mask3), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * F, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * F, mask1), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 1 * F, mask2), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 1 * F, mask3), w0, d31, mask3);
                                    w0 = _mm512_loadu_ps(pw + 2 * F);
                                    d02 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 2 * F, mask0), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 2 * F, mask1), w0, d12, mask1);
                                    d22 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 2 * F, mask2), w0, d22, mask2);
                                    d32 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 2 * F, mask3), w0, d32, mask3);
                                    w0 = _mm512_loadu_ps(pw + 3 * F);
                                    d03 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 3 * F, mask0), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 3 * F, mask1), w0, d13, mask1);
                                    d23 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 3 * F, mask2), w0, d23, mask2);
                                    d33 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 3 * F, mask3), w0, d33, mask3);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F);
                        Save1<term, type>(dst0, d01, params, c + 1 * F);
                        Save1<term, type>(dst0, d02, params, c + 2 * F);
                        Save1<term, type>(dst0, d03, params, c + 3 * F);
                        Save1<term, type>(dst1, d10, params, c + 0 * F);
                        Save1<term, type>(dst1, d11, params, c + 1 * F);
                        Save1<term, type>(dst1, d12, params, c + 2 * F);
                        Save1<term, type>(dst1, d13, params, c + 3 * F);
                        Save1<term, type>(dst2, d20, params, c + 0 * F);
                        Save1<term, type>(dst2, d21, params, c + 1 * F);
                        Save1<term, type>(dst2, d22, params, c + 2 * F);
                        Save1<term, type>(dst2, d23, params, c + 3 * F);
                        Save1<term, type>(dst3, d30, params, c + 0 * F);
                        Save1<term, type>(dst3, d31, params, c + 1 * F);
                        Save1<term, type>(dst3, d32, params, c + 2 * F);
                        Save1<term, type>(dst3, d33, params, c + 3 * F);
                    }
                    for (; c < size2F; c += 2 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        d10 = d00; d11 = d01;
                        d20 = d00; d21 = d01;
                        d30 = d00; d31 = d01;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * F, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * F, mask3), w0, d30, mask3);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * F, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * F, mask1), w0, d11, mask1);
                                    d21 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 1 * F, mask2), w0, d21, mask2);
                                    d31 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 1 * F, mask3), w0, d31, mask3);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F);
                        Save1<term, type>(dst0, d01, params, c + 1 * F);
                        Save1<term, type>(dst1, d10, params, c + 0 * F);
                        Save1<term, type>(dst1, d11, params, c + 1 * F);
                        Save1<term, type>(dst2, d20, params, c + 0 * F);
                        Save1<term, type>(dst2, d21, params, c + 1 * F);
                        Save1<term, type>(dst3, d30, params, c + 0 * F);
                        Save1<term, type>(dst3, d31, params, c + 1 * F);
                    }
                    for (; c < size; c += F)
                    {
                        __mmask16 tail = c < sizeF ? __mmask16(-1) : TailMask16(size - c);
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d10 = d00;
                        d20 = d00;
                        d30 = d00;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask2 = sx + 2 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask3 = sx + 3 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX, * ps2 = ps0 + 2 * ssX, * ps3 = ps0 + 3 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                    d20 = _mm512_mask3_fmadd_ps(LoadSrc(ps2 + 0 * F, mask2), w0, d20, mask2);
                                    d30 = _mm512_mask3_fmadd_ps(LoadSrc(ps3 + 0 * F, mask3), w0, d30, mask3);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F, tail);
                        Save1<term, type>(dst1, d10, params, c + 0 * F, tail);
                        Save1<term, type>(dst2, d20, params, c + 0 * F, tail);
                        Save1<term, type>(dst3, d30, params, c + 0 * F, tail);
                    }
                    dst += 4 * dstC;
                }
                for (; dx < dstW2; dx += 2)
                {
                    uint8_t* dst0 = dst + 0 * dstC, * dst1 = dst + 1 * dstC;
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t c = 0;
                    for (; c < size4F; c += 4 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        d02 = _mm512_loadu_ps(bias + c + 2 * F);
                        d03 = _mm512_loadu_ps(bias + c + 3 * F);
                        d10 = d00; d11 = d01; d12 = d02; d13 = d03;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * F, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * F, mask1), w0, d11, mask1);
                                    w0 = _mm512_loadu_ps(pw + 2 * F);
                                    d02 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 2 * F, mask0), w0, d02, mask0);
                                    d12 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 2 * F, mask1), w0, d12, mask1);
                                    w0 = _mm512_loadu_ps(pw + 3 * F);
                                    d03 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 3 * F, mask0), w0, d03, mask0);
                                    d13 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 3 * F, mask1), w0, d13, mask1);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F);
                        Save1<term, type>(dst0, d01, params, c + 1 * F);
                        Save1<term, type>(dst0, d02, params, c + 2 * F);
                        Save1<term, type>(dst0, d03, params, c + 3 * F);
                        Save1<term, type>(dst1, d10, params, c + 0 * F);
                        Save1<term, type>(dst1, d11, params, c + 1 * F);
                        Save1<term, type>(dst1, d12, params, c + 2 * F);
                        Save1<term, type>(dst1, d13, params, c + 3 * F);
                    }
                    for (; c < size2F; c += 2 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        d10 = d00; d11 = d01;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                    w0 = _mm512_loadu_ps(pw + 1 * F);
                                    d01 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 1 * F, mask0), w0, d01, mask0);
                                    d11 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 1 * F, mask1), w0, d11, mask1);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F);
                        Save1<term, type>(dst0, d01, params, c + 1 * F);
                        Save1<term, type>(dst1, d10, params, c + 0 * F);
                        Save1<term, type>(dst1, d11, params, c + 1 * F);
                    }
                    for (; c < size; c += F)
                    {
                        __mmask16 tail = c < sizeF ? __mmask16(-1) : TailMask16(size - c);
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d10 = d00;
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pwy = weight + ky * kX * size + c;
                                const T* psy = src + sy * srcW * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    const float* pw = pwy + kx * size;
                                    __mmask16 mask0 = sx + 0 * sX < srcW ? 0xFFFF : 0x0000;
                                    __mmask16 mask1 = sx + 1 * sX < srcW ? 0xFFFF : 0x0000;
                                    const T* ps0 = psy + sx * size, * ps1 = ps0 + 1 * ssX;

                                    w0 = _mm512_loadu_ps(pw + 0 * F);
                                    d00 = _mm512_mask3_fmadd_ps(LoadSrc(ps0 + 0 * F, mask0), w0, d00, mask0);
                                    d10 = _mm512_mask3_fmadd_ps(LoadSrc(ps1 + 0 * F, mask1), w0, d10, mask1);
                                }
                            }
                        }
                        Save1<term, type>(dst0, d00, params, c + 0 * F, tail);
                        Save1<term, type>(dst1, d10, params, c + 0 * F, tail);
                    }
                    dst += 2 * dstC;
                }
                for (; dx < p.dstW; ++dx)
                {
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t c = 0;
                    for (; c < size4F; c += 4 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        d02 = _mm512_loadu_ps(bias + c + 2 * F);
                        d03 = _mm512_loadu_ps(bias + c + 3 * F);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                        d01 = _mm512_fmadd_ps(LoadSrc(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                                        d02 = _mm512_fmadd_ps(LoadSrc(ps + 2 * F), _mm512_loadu_ps(pw + 2 * F), d02);
                                        d03 = _mm512_fmadd_ps(LoadSrc(ps + 3 * F), _mm512_loadu_ps(pw + 3 * F), d03);
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c + 0 * F);
                        Save1<term, type>(dst, d01, params, c + 1 * F);
                        Save1<term, type>(dst, d02, params, c + 2 * F);
                        Save1<term, type>(dst, d03, params, c + 3 * F);
                    }
                    for (; c < size2F; c += 2 * F)
                    {
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        d01 = _mm512_loadu_ps(bias + c + 1 * F);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                        d01 = _mm512_fmadd_ps(LoadSrc(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c + 0 * F);
                        Save1<term, type>(dst, d01, params, c + 1 * F);
                    }
                    for (; c < size; c += F)
                    {
                        __mmask16 tail = c < sizeF ? __mmask16(-1) : TailMask16(size - c);
                        d00 = _mm512_loadu_ps(bias + c + 0 * F);
                        for (size_t ky = 0; ky < kY; ++ky)
                        {
                            size_t sy = sy0 + ky * dY;
                            if (sy < srcH)
                            {
                                const float* pw = weight + ky * kX * size + c;
                                const T* ps = src + (sy * srcW + sx0) * size + c;
                                for (size_t kx = 0; kx < kX; ++kx)
                                {
                                    size_t sx = sx0 + kx * dX;
                                    if (sx < srcW)
                                    {
                                        d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * F, tail), _mm512_maskz_loadu_ps(tail, pw + 0 * F), d00);
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c, tail);
                    }
                    dst += dstC;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Edge(const T* src, const ConvParam& p, size_t dy, size_t dx, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            size_t srcC = p.srcC, srcCF = AlignLo(srcC, F), srcC2F = AlignLo(srcC, 2 * F), srcC4F = AlignLo(srcC, 4 * F);
            size_t c = 0;
            for (; c < srcC4F; c += 4 * F)
            {
                __m512 d00 = _mm512_loadu_ps(bias + c + 0 * F);
                __m512 d01 = _mm512_loadu_ps(bias + c + 1 * F);
                __m512 d02 = _mm512_loadu_ps(bias + c + 2 * F);
                __m512 d03 = _mm512_loadu_ps(bias + c + 3 * F);
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
                                const float* pw = weight + (ky * 3 + kx) * srcC;
                                const T* ps = src + (sy * p.srcW + sx) * srcC;
                                d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                d01 = _mm512_fmadd_ps(LoadSrc(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                                d02 = _mm512_fmadd_ps(LoadSrc(ps + 2 * F), _mm512_loadu_ps(pw + 2 * F), d02);
                                d03 = _mm512_fmadd_ps(LoadSrc(ps + 3 * F), _mm512_loadu_ps(pw + 3 * F), d03);
                            }
                        }
                    }
                }
                Save1<term, type>(dst, d00, params, c + 0 * F);
                Save1<term, type>(dst, d01, params, c + 1 * F);
                Save1<term, type>(dst, d02, params, c + 2 * F);
                Save1<term, type>(dst, d03, params, c + 3 * F);
                src += 4 * F;
                weight += 4 * F;
            }
            for (; c < srcC2F; c += 2 * F)
            {
                __m512 d00 = _mm512_loadu_ps(bias + c + 0 * F);
                __m512 d01 = _mm512_loadu_ps(bias + c + 1 * F);
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
                                const float* pw = weight + (ky * 3 + kx) * srcC;
                                const T* ps = src + (sy * p.srcW + sx) * srcC;
                                d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * F), _mm512_loadu_ps(pw + 0 * F), d00);
                                d01 = _mm512_fmadd_ps(LoadSrc(ps + 1 * F), _mm512_loadu_ps(pw + 1 * F), d01);
                            }
                        }
                    }
                }
                Save1<term, type>(dst, d00, params, c + 0 * F);
                Save1<term, type>(dst, d01, params, c + 1 * F);
                src += 2 * F;
                weight += 2 * F;
            }
            for (; c < srcCF; c += F)
            {
                __m512 d00 = _mm512_loadu_ps(bias + c);
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
                                const float* pw = weight + (ky * 3 + kx) * srcC;
                                const T* ps = src + (sy * p.srcW + sx) * srcC;
                                d00 = _mm512_fmadd_ps(LoadSrc(ps), _mm512_loadu_ps(pw), d00);
                            }
                        }
                    }
                }
                Save1<term, type>(dst, d00, params, c);
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m512 d00 = _mm512_loadu_ps(bias + c);
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
                                const float* pw = weight + (ky * 3 + kx) * srcC;
                                const T* ps = src + (sy * p.srcW + sx) * srcC;
                                d00 = _mm512_fmadd_ps(LoadSrc(ps), _mm512_loadu_ps(pw), d00);
                            }
                        }
                    }
                }
                Save1<term, type>(dst, d00, params, c);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main1(const T* src, size_t srcS, size_t srcC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 d00 = _mm512_loadu_ps(bias + c);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps = src + ky * srcS;
                    const float* pw = weight + ky * 3 * srcC;
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * srcC), _mm512_loadu_ps(pw + 0 * srcC), d00);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 1 * srcC), _mm512_loadu_ps(pw + 1 * srcC), d00);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 2 * srcC), _mm512_loadu_ps(pw + 2 * srcC), d00);
                }
                Save1<term, type>(dst, d00, params, c);
                src += F;
                weight += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                weight -= srcCF - c;
                __m512 d00 = _mm512_loadu_ps(bias + c);
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps = src + ky * srcS;
                    const float* pw = weight + ky * 3 * srcC;
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 0 * srcC), _mm512_loadu_ps(pw + 0 * srcC), d00);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 1 * srcC), _mm512_loadu_ps(pw + 1 * srcC), d00);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps + 2 * srcC), _mm512_loadu_ps(pw + 2 * srcC), d00);
                }
                Save1<term, type>(dst, d00, params, c + 0 * F);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main2(const T* src, size_t srcS, size_t srcX, size_t srcC, size_t dstC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            __m512 d00, d01, w0;
            for (; c < srcCF; c += F)
            {
                d00 = _mm512_loadu_ps(bias + c);
                d01 = d00;
                const float* pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps0 = src + ky * srcS;
                    const T* ps1 = ps0 + srcX;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 0 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 0 * srcC), w0, d01);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 1 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 1 * srcC), w0, d01);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 2 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 2 * srcC), w0, d01);
                    pw += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c);
                Save1<term, type>(dst + 1 * dstC, d01, params, c);
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                d00 = _mm512_loadu_ps(bias + c);
                d01 = d00;
                const float* pw = weight + c;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    const T* ps0 = src + ky * srcS;
                    const T* ps1 = ps0 + srcX;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 0 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 0 * srcC), w0, d01);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 1 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 1 * srcC), w0, d01);
                    pw += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + 2 * srcC), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + 2 * srcC), w0, d01);
                    pw += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c);
                Save1<term, type>(dst + 1 * dstC, d01, params, c);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Convolution16bNhwcDepthwise3x3Main4(const T* src, size_t srcS, size_t srcX, size_t srcC, size_t dstC, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            size_t srcCF = AlignLo(srcC, F);
            size_t c = 0;
            for (; c < srcCF; c += F)
            {
                __m512 d00, d01, d02, d03, w0;
                d00 = _mm512_loadu_ps(bias + c);
                d01 = d00;
                d02 = d00;
                d03 = d00;
                const float* pw = weight + c;
                const T* ps0 = src + 0 * srcX;
                const T* ps1 = src + 1 * srcX;
                const T* ps2 = src + 2 * srcX;
                const T* ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c);
                Save1<term, type>(dst + 1 * dstC, d01, params, c);
                Save1<term, type>(dst + 2 * dstC, d02, params, c);
                Save1<term, type>(dst + 3 * dstC, d03, params, c);
                src += F;
            }
            if (c < srcC)
            {
                c = srcC - F;
                src -= srcCF - c;
                __m512 d00, d01, d02, d03, w0;
                d00 = _mm512_loadu_ps(bias + c);
                d01 = d00;
                d02 = d00;
                d03 = d00;
                const float* pw = weight + c;
                const T* ps0 = src + 0 * srcX;
                const T* ps1 = src + 1 * srcX;
                const T* ps2 = src + 2 * srcX;
                const T* ps3 = src + 3 * srcX;
                for (size_t ky = 0; ky < 3; ++ky)
                {
                    size_t offset = ky * srcS;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                    w0 = _mm512_loadu_ps(pw);
                    d00 = _mm512_fmadd_ps(LoadSrc(ps0 + offset), w0, d00);
                    d01 = _mm512_fmadd_ps(LoadSrc(ps1 + offset), w0, d01);
                    d02 = _mm512_fmadd_ps(LoadSrc(ps2 + offset), w0, d02);
                    d03 = _mm512_fmadd_ps(LoadSrc(ps3 + offset), w0, d03);
                    pw += srcC, offset += srcC;
                }
                Save1<term, type>(dst + 0 * dstC, d00, params, c);
                Save1<term, type>(dst + 1 * dstC, d01, params, c);
                Save1<term, type>(dst + 2 * dstC, d02, params, c);
                Save1<term, type>(dst + 3 * dstC, d03, params, c);
            }
        }

        template<typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwise3x3(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            const T* src = (T*)src8;
            size_t srcS = p.srcC * p.srcW;
            size_t srcX = p.srcC * p.strideX;
            size_t dstH = p.dstH - p.padH;
            size_t dstW = p.dstW - p.padW;
            size_t dstW2 = AlignLo(dstW - p.padX, 2) + p.padX;
            size_t dstW4 = AlignLo(dstW - p.padX, 4) + p.padX;
            size_t dstC = p.dstC * (term == Term16bLast16b ? 2 : 4);
            size_t dy = 0;
            for (; dy < p.padY; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
            for (; dy < dstH; ++dy)
            {
                size_t dx = 0;
                for (; dx < p.padX; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
                size_t offset = ((dy * p.strideY - p.padY) * p.srcW + dx * p.strideX - p.padX) * p.srcC;
                for (; dx < dstW4; dx += 4)
                    Convolution16bNhwcDepthwise3x3Main4<T, term, type>(src + offset, srcS, srcX, p.srcC, dstC, weight, bias, params, dst), dst += 4 * dstC, offset += 4 * srcX;
                for (; dx < dstW2; dx += 2)
                    Convolution16bNhwcDepthwise3x3Main2<T, term, type>(src + offset, srcS, srcX, p.srcC, dstC, weight, bias, params, dst), dst += 2 * dstC, offset += 2 * srcX;
                for (; dx < dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Main1<T, term, type>(src + offset, srcS, p.srcC, weight, bias, params, dst), dst += dstC, offset += srcX;
                for (; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
            }
            for (; dy < p.dstH; ++dy)
                for (size_t dx = 0; dx < p.dstW; ++dx)
                    Convolution16bNhwcDepthwise3x3Edge<T, term, type>(src, p, dy, dx, weight, bias, params, dst), dst += dstC;
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.IsKernel(3) && p.IsDilation(1) && p.srcC >= F)
                convolution = Convolution16bNhwcDepthwise3x3<T, term, type>;
            else
                convolution = Convolution16bNhwcDepthwiseDefault<T, term, type>;
        }

        template<typename T, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.dstT == SimdTensorData32f)
                SetConvolution<T, Term16bLast32f, type>(p, convolution);
            else
                SetConvolution<T, Term16bLast16b, type>(p, convolution);
        }

        template<SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
            if (p.srcT == SimdTensorData16b)
                SetConvolution<uint16_t, type>(p, convolution);
            else
                SetConvolution<float, type>(p, convolution);
        }

        //-------------------------------------------------------------------------------------------------

        SynetConvolution16bNhwcDepthwise::SynetConvolution16bNhwcDepthwise(const ConvParam& p)
            : Avx2::SynetConvolution16bNhwcDepthwise(p)
        {
            switch (p.activation)
            {
            case SimdConvolutionActivationIdentity: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationRelu: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationLeakyRelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationRestrictRange: SetConvolution<SimdConvolutionActivationRestrictRange>(p, _convolution); break;
            case SimdConvolutionActivationPrelu: SetConvolution<SimdConvolutionActivationPrelu>(p, _convolution); break;
            case SimdConvolutionActivationElu: SetConvolution<SimdConvolutionActivationElu>(p, _convolution); break;
            case SimdConvolutionActivationHswish: SetConvolution<SimdConvolutionActivationHswish>(p, _convolution); break;
            case SimdConvolutionActivationMish: SetConvolution<SimdConvolutionActivationMish>(p, _convolution); break;
            case SimdConvolutionActivationHardSigmoid: SetConvolution<SimdConvolutionActivationHardSigmoid>(p, _convolution); break;
            case SimdConvolutionActivationSwish: SetConvolution<SimdConvolutionActivationSwish>(p, _convolution); break;
            case SimdConvolutionActivationGelu: SetConvolution<SimdConvolutionActivationGelu>(p, _convolution); break;
            }
        }
    }
#endif
}

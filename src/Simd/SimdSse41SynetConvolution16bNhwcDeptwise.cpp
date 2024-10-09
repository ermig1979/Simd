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
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE) 
    namespace Sse41
    {
        template <Term16bType term> struct DepthwiseTerm16b
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset);
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset, size_t tail);
        };

        template <> struct DepthwiseTerm16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset)
            {
                __m128 f32 = Activate<type>(value, params, offset);
                _mm_storel_epi64((__m128i*)(ptr + offset * 2), _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset, size_t tail)
            {
                __m128 f32 = Activate<type>(value, params, offset);
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)ptr)[i + offset] = tmp[i];
            }
        };

        template <> struct DepthwiseTerm16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset)
            {
                __m128 f32 = Activate<type>(value, params, offset);
                _mm_storeu_ps((float*)ptr + offset, f32);
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, __m128 value, const float* params, size_t offset, size_t tail)
            {
                __m128 f32 = Activate<type>(value, params, offset);
                float tmp[F];
                _mm_storeu_ps(tmp, f32);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i + offset] = tmp[i];
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, __m128 val0, const float* params, size_t offset)
        {
            DepthwiseTerm16b<term>::template Save<type>(ptr, val0, params, offset);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, __m128 val0, const float* params, size_t offset, size_t tail)
        {
            DepthwiseTerm16b<term>::template Save<type>(ptr, val0, params, offset, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template <typename T, Term16bType term, SimdConvolutionActivationType type> void Convolution16bNhwcDepthwiseDefault(const uint8_t* src8, const ConvParam& p, const float* weight, const float* bias, const float* params, uint8_t* dst)
        {
            assert(p.trans && p.IsDepthwise());
            const T* src = (T*)src8;
            size_t kX = p.kernelX, kY = p.kernelY, dX = p.dilationX, dY = p.dilationY, srcH = p.srcH, srcW = p.srcW;
            size_t size = p.group, elem = (term == Term16bLast16b ? 2 : 4), sdS = size * dX;
            size_t sizeF = AlignLo(size, F), tail = sizeF - size, size2F = AlignLo(size, 2 * F), size4F = AlignLo(size, 4 * F), size8F = AlignLo(size, 8 * F);

            for (size_t dy = 0; dy < p.dstH; ++dy)
            {
                size_t sy0 = dy * p.strideY - p.padY;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    size_t sx0 = dx * p.strideX - p.padX;
                    size_t c = 0;
                    for (; c < size8F; c += 8 * F)
                    {
                        __m128 d00 = _mm_loadu_ps(bias + c + 0 * F);
                        __m128 d01 = _mm_loadu_ps(bias + c + 1 * F);
                        __m128 d02 = _mm_loadu_ps(bias + c + 2 * F);
                        __m128 d03 = _mm_loadu_ps(bias + c + 3 * F);
                        __m128 d04 = _mm_loadu_ps(bias + c + 4 * F);
                        __m128 d05 = _mm_loadu_ps(bias + c + 5 * F);
                        __m128 d06 = _mm_loadu_ps(bias + c + 6 * F);
                        __m128 d07 = _mm_loadu_ps(bias + c + 7 * F);
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
                                        d00 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), d00);
                                        d01 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), d01);
                                        d02 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 2 * F), _mm_loadu_ps(pw + 2 * F)), d02);
                                        d03 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 3 * F), _mm_loadu_ps(pw + 3 * F)), d03);
                                        d04 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 4 * F), _mm_loadu_ps(pw + 4 * F)), d04);
                                        d05 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 5 * F), _mm_loadu_ps(pw + 5 * F)), d05);
                                        d06 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 6 * F), _mm_loadu_ps(pw + 6 * F)), d06);
                                        d07 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 7 * F), _mm_loadu_ps(pw + 7 * F)), d07);
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        Save1<term, type>(dst, d00, params, c + 0 * F);
                        Save1<term, type>(dst, d01, params, c + 1 * F);
                        Save1<term, type>(dst, d02, params, c + 2 * F);
                        Save1<term, type>(dst, d03, params, c + 3 * F);
                        Save1<term, type>(dst, d04, params, c + 4 * F);
                        Save1<term, type>(dst, d05, params, c + 5 * F);
                        Save1<term, type>(dst, d06, params, c + 6 * F);
                        Save1<term, type>(dst, d07, params, c + 7 * F);
                    }
                    for (; c < size4F; c += 4 * F)
                    {
                        __m128 d00 = _mm_loadu_ps(bias + c + 0 * F);
                        __m128 d01 = _mm_loadu_ps(bias + c + 1 * F);
                        __m128 d02 = _mm_loadu_ps(bias + c + 2 * F);
                        __m128 d03 = _mm_loadu_ps(bias + c + 3 * F);
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
                                        d00 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), d00);
                                        d01 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), d01);
                                        d02 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 2 * F), _mm_loadu_ps(pw + 2 * F)), d02);
                                        d03 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 3 * F), _mm_loadu_ps(pw + 3 * F)), d03);
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
                        __m128 d00 = _mm_loadu_ps(bias + c + 0 * F);
                        __m128 d01 = _mm_loadu_ps(bias + c + 1 * F);
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
                                        d00 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), d00);
                                        d01 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 1 * F), _mm_loadu_ps(pw + 1 * F)), d01);
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
                        __m128 d00 = _mm_loadu_ps(bias + c + 0 * F);
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
                                        d00 = _mm_add_ps(_mm_mul_ps(LoadSrc(ps + 0 * F), _mm_loadu_ps(pw + 0 * F)), d00);
                                    }
                                    pw += size, ps += sdS;
                                }
                            }
                        }
                        if(c == sizeF)
                            Save1<term, type>(dst, d00, params, c, tail);
                        else
                            Save1<term, type>(dst, d00, params, c);
                    }
                    dst += size * elem;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<typename T, Term16bType term, SimdConvolutionActivationType type> static void SetConvolution(const ConvParam& p, SynetConvolution16bNhwcDepthwise::ConvolutionPtr& convolution)
        {
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
            : Base::SynetConvolution16bNhwcDepthwise(p)
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

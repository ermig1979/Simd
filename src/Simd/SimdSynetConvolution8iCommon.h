/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __SimdSynetConvolution8iCommon_h__
#define __SimdSynetConvolution8iCommon_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynetConvolution8i.h"
#include "Simd/SimdSynetConvolution32fCommon.h"

namespace Simd
{
    enum Term8iType
    {
        Term8iLast8u,
        Term8iLast32f,
        Term8iInterim,
        Term8iSize
    };

    namespace Base
    {
        template<class S, class D, class F> SIMD_INLINE D Convert(S value, F scale, F shift, int lower, int upper)
        {
            return (D)(F(value) * scale + shift);
        }

        template<> SIMD_INLINE uint8_t Convert<float, uint8_t, float>(float value, float scale, float shift, int lower, int upper)
        {
            return (uint8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<> SIMD_INLINE int8_t Convert<float, int8_t, float>(float value, float scale, float shift, int lower, int upper)
        {
            return (int8_t)Simd::RestrictRange(Round(value * scale + shift), lower, upper);
        }

        template<class S, class D, class F> void Convert(const S* src, size_t batch, size_t channels, size_t height, size_t width,
            SimdTensorFormatType format, const F* scale, const F* shift, int lower, int upper, D* dst)
        {
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        F _scale = scale[c];
                        F _shift = shift[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = Convert<S, D, F>(src[w], _scale, _shift, lower, upper);
                            src += width;
                            dst += width;
                        }
                    }
                }
                else if (format == SimdTensorFormatNhwc)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        for (size_t w = 0; w < width; ++w)
                        {
                            for (size_t c = 0; c < channels; ++c)
                                dst[c] = Convert<S, D, F>(src[c], scale[c], shift[c], lower, upper);
                            src += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //---------------------------------------------------------------------

        inline void ImgToCol(const uint8_t* src, const ConvParam8i& p, const uint8_t* zero, uint8_t* dst)
        {
            assert(!p.trans);
            size_t srcSize = p.srcW * p.srcH;
            if (p.IsDilation(1) && p.IsStride(2) && p.IsPad(0) && p.IsKernel(1))
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const uint8_t* psrc = src + 2 * dy * p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.IsDilation(1) && p.IsStride(1))
            {
                const ptrdiff_t bodySize = p.dstW - p.padX - p.padW;
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t ky = 0; ky < p.kernelY; ++ky)
                    {
                        for (size_t kx = 0; kx < p.kernelX; ++kx)
                        {
                            size_t sy = ky - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy, ++sy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx - p.padX, dx = 0;
                                    const uint8_t* psrc = src + sy * p.srcW;
                                    for (; dx < p.padX; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                    if (bodySize > 0)
                                    {
                                        memcpy(dst, psrc + sx, bodySize * sizeof(uint8_t));
                                        dst += bodySize;
                                        dx += bodySize;
                                        sx += bodySize;
                                    }
                                    for (; dx < p.dstW; ++dx, ++sx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = psrc[sx];
                                        else
                                            *(dst++) = zero[channel];
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                for (size_t channel = 0; channel < p.srcC; ++channel)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t sx = kx * p.dilationX - p.padX;
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                    {
                                        if (sx < p.srcW)
                                            *(dst++) = src[sy * p.srcW + sx];
                                        else
                                            *(dst++) = zero[channel];
                                        sx += p.strideX;
                                    }
                                }
                                else
                                {
                                    for (size_t dx = 0; dx < p.dstW; ++dx)
                                        *(dst++) = zero[channel];
                                }
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
        }

        inline void ImgToRow(const uint8_t* src, const SimdConvolutionParameters & p, const uint8_t* zero, uint8_t* dst)
        {
            assert(p.srcF == SimdTensorFormatNhwc);
            size_t size = p.srcC / p.group;
            for (size_t g = 0; g < p.group; ++g)
            {
                for (size_t dy = 0; dy < p.dstH; ++dy)
                {
                    for (size_t dx = 0; dx < p.dstW; ++dx)
                    {
                        for (size_t ky = 0; ky < p.kernelY; ky++)
                        {
                            size_t sy = dy * p.strideY + ky * p.dilationY - p.padY;
                            if (sy < p.srcH)
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    size_t sx = dx * p.strideX + kx * p.dilationX - p.padX;
                                    if (sx < p.srcW)
                                    {
                                        memcpy(dst, src + (sy * p.srcW + sx) * p.srcC, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                    else
                                    {
                                        memcpy(dst, zero, size * sizeof(uint8_t));
                                        dst += size;
                                    }
                                }
                            }
                            else
                            {
                                for (size_t kx = 0; kx < p.kernelX; kx++)
                                {
                                    memcpy(dst, zero, size * sizeof(uint8_t));
                                    dst += size;
                                }
                            }
                        }
                    }
                }
                src += size;
                zero += size;
            }
        }

        inline void GemmNchw(size_t D, size_t S, size_t C, size_t K, const int8_t* wgt, size_t ldw, const uint8_t* src, size_t lds, int32_t* dst, size_t ldd, bool overflow)
        {
            const size_t C2 = overflow ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < D; ++i)
            {
                for (size_t j = 0; j < S; ++j)
                    dst[j] = 0;
                size_t c = 0;
                for (; c < C2; c += 2)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        int32_t w1 = wgt[(c + 1) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        const uint8_t* s1 = src + ((c + 1) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += Simd::RestrictRange(s0[j] * w0 + s1[j] * w1, SHRT_MIN, SHRT_MAX);
                    }
                }
                for (; c < C; ++c)
                {
                    for (size_t k = 0; k < K; k++)
                    {
                        int32_t w0 = wgt[(c + 0) * K + k];
                        const uint8_t* s0 = src + ((c + 0) * K + k) * lds;
                        for (size_t j = 0; j < S; ++j)
                            dst[j] += s0[j] * w0;
                    }
                }
                wgt += ldw;
                dst += ldd;
            }
        }

        inline void GemmNhwc(size_t S, size_t D, size_t K, size_t C, const uint8_t* src, size_t lds, const int8_t* wgt, size_t ldw, int32_t* dst, size_t ldd, bool overflow)
        {
            const size_t C2 = overflow ? AlignLo(C, 2) : 0;
            for (size_t i = 0; i < S; ++i)
            {
                for (size_t j = 0; j < D; ++j)
                    dst[j] = 0;
                for (size_t k = 0, o = 0; k < K; k++)
                {
                    size_t c = 0;
                    for (; c < C2; c += 2, o += 2)
                    {
                        int32_t s0 = src[o + 0];
                        int32_t s1 = src[o + 1];
                        const int8_t* w0 = wgt + (o + 0) * ldw;
                        const int8_t* w1 = wgt + (o + 1) * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += Simd::RestrictRange(s0 * w0[j] + s1 * w1[j], SHRT_MIN, SHRT_MAX);
                    }
                    for (; c < C; ++c, ++o)
                    {
                        int32_t s0 = src[o];
                        const int8_t* w0 = wgt + o * ldw;
                        for (size_t j = 0; j < D; ++j)
                            dst[j] += s0 * w0[j];
                    }
                }
                src += lds;
                dst += ldd;
            }
        }
    }

#if defined(SIMD_SSE41_ENABLE)   
    namespace Sse41
    {
        template <Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf, __m128i sum, 
                const __m128 * norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf, __m128i sum, 
                const __m128 * norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail);

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum, 
                const __m128* params, const __m128 & scale, const __m128 & shift, __m128i upper);
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum,
                const __m128* params, const __m128 & scale, const __m128 & shift, __m128i upper, size_t tail);
        };

        template <> struct Term8i<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), norm[index]), bias[index]), params, index);
                __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(f32, scale[index]), shift[index]));
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
            {
                uint8_t tmp[F];
                Term8i::Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum,
                const __m128* params, const __m128 & scale, const __m128 & shift, __m128i upper)
            {
                __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse2::Activate<type>(sum, params, 0), scale), shift));
                ((int32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum,
                const __m128* params, const __m128& scale, const __m128& shift, __m128i upper, size_t tail)
            {
                uint8_t tmp[F];
                Term8i::Save<type>(tmp, sum, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[i] = tmp[i];
            }
        };

        template <> struct Term8i<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), norm[index]), bias[index]), params, index);
                _mm_storeu_ps((float*)dst + index*F, f32);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum,
                const __m128* params, const __m128& scale, const __m128& shift, __m128i upper)
            {
                _mm_storeu_ps((float*)dst, Sse2::Activate<type>(sum, params, 0));
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128 sum,
                const __m128* params, const __m128& scale, const __m128& shift, __m128i upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type>(tmp, sum, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term8i<Term8iInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                _mm_storeu_si128((__m128i*)buf + index, sum);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> 
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, __m128 sum, const __m128* params, 
            const __m128 & scale, const __m128 & shift, __m128i upper)
        {
            Term8i<term>::template Save<type>(dst, sum, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, __m128 sum, const __m128* params, 
            const __m128& scale, const __m128& shift, __m128i upper, size_t tail)
        {
            Term8i<term>::template Save<type>(dst, sum, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float * norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset);
        };

        template <> struct Term8iDepthwise<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_loadu_ps(norm + offset)), _mm_loadu_ps(bias + offset)), params, offset);
                __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(f32, _mm_loadu_ps(scale + offset)), _mm_loadu_ps(shift + offset)));
                ((int32_t*)(dst + offset))[0] = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
            }
        };

        template <> struct Term8iDepthwise<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_loadu_ps(norm + offset)), _mm_loadu_ps(bias + offset)), params, offset);
                _mm_storeu_ps((float*)dst + offset, f32);
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save(uint8_t* dst, __m128i sum, const float* norm, const float* bias, 
            const float* params, const float* scale, const float* shift, __m128i upper, size_t offset)
        {
            Term8iDepthwise<term>::template Save<type>(dst, sum, norm, bias, params, scale, shift, upper, offset);
        }
    }
#endif//SIMD_SSE41_ENABLE

#if defined(SIMD_AVX2_ENABLE) 
    namespace Avx2
    {
        template <Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum, 
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail);

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper);
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper, size_t tail);
        };

        template <> struct Term8i<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                __m256 f32 = Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                __m256i i32 = _mm256_cvtps_epi32(Fmadd<nofma>(f32, scale[index], shift[index]));
                ((int64_t*)dst)[index] = Extract64i<0>(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO), upper));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
            {
                uint8_t tmp[F];
                Term8i::Save<type, index, nofma>(tmp - index * F, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper)
            {
                __m256i i32 = _mm256_cvtps_epi32(Fmadd<nofma>(Activate<type>(sum, params, 0), scale, shift));
                ((int64_t*)dst)[0] = Extract64i<0>(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO), upper));
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper, size_t tail)
            {
                uint8_t tmp[F];
                Term8i::Save<type, nofma>(tmp, sum, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[i] = tmp[i];
            }
        };

        template <> struct Term8i<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                __m256 f32 = Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                _mm256_storeu_ps((float*)dst + index * F, f32);
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type, index, nofma>(tmp - index * A, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper)
            {
                _mm256_storeu_ps((float*)dst, Avx2::Activate<type>(sum, params, 0));
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256 sum,
                const __m256* params, const __m256& scale, const __m256& shift, __m256i upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type, nofma>(tmp, sum, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term8i<Term8iInterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                _mm256_storeu_si256((__m256i*)buf + index, sum);
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, const __m256* norm, 
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, __m256 sum, const __m256* params,
            const __m256& scale, const __m256& shift, __m256i upper)
        {
            Term8i<term>::template Save<type, nofma>(dst, sum, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, __m256 sum, const __m256* params,
            const __m256& scale, const __m256& shift, __m256i upper, size_t tail)
        {
            Term8i<term>::template Save<type, nofma>(dst, sum, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset);
        };

        template <> struct Term8iDepthwise<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset)
            {
                __m256 f32 = Avx2::Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), _mm256_loadu_ps(norm + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                __m256i i32 = _mm256_cvtps_epi32(Fmadd<nofma>(f32, _mm256_loadu_ps(scale + offset), _mm256_loadu_ps(shift + offset)));
                ((int64_t*)(dst + offset))[0] = Extract64i<0>(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO), upper));
            }
        };

        template <> struct Term8iDepthwise<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset)
            {
                __m256 f32 = Avx2::Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), _mm256_loadu_ps(norm + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                _mm256_storeu_ps((float*)dst + offset, f32);
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save(uint8_t* dst, __m256i sum, const float* norm, const float* bias,
            const float* params, const float* scale, const float* shift, __m256i upper, size_t offset)
        {
            Term8iDepthwise<term>::template Save<type, nofma>(dst, sum, norm, bias, params, scale, shift, upper, offset);
        }
    }
#endif//SIMD_AVX2_ENABLE

#if defined(SIMD_AVX512BW_ENABLE)  
    namespace Avx512bw
    {
        template <Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum, 
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1);

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m512 sum,
                const __m512* params, const __m512 & scale, const __m512 & shift, __m128i upper, __mmask16 tail = -1);
        };

        template <> struct Term8i<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                __m128i u8 = Cvt32fTo8u(Fmadd<nofma>(f32, scale[index], shift[index]));
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm_min_epu8(u8, upper));
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512 sum,
                const __m512* params, const __m512& scale, const __m512& shift, __m128i upper, __mmask16 tail)
            {
                __m128i u8 = Cvt32fTo8u(Fmadd<nofma>(Activate<type>(sum, params, 0), scale, shift));
                _mm_mask_storeu_epi8(dst, tail, _mm_min_epu8(u8, upper));
            }
        };

        template <> struct Term8i<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                _mm512_mask_storeu_ps((float*)dst + index * F, tail, f32);
            }

            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512 sum,
                const __m512* params, const __m512& scale, const __m512& shift, __m128i upper, __mmask16 tail)
            {
                _mm512_mask_storeu_ps((float*)dst, tail, Activate<type>(sum, params, 0));
            }
        };

        template <> struct Term8i<Term8iInterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum, const __m512* norm, const __m512* bias,
            const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1, const __m512* norm, const __m512* bias,
            const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, __m512 sum, const __m512* params, const __m512& scale, const __m512& shift, __m128i upper, __mmask16 tail = -1)
        {
            Term8i<term>::template Save<type, nofma>(dst, sum, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm, 
                const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail);
        };

        template <> struct Term8iDepthwise<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm,
                const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail)
            {
                __m512 _norm = _mm512_maskz_loadu_ps(tail, norm + offset);
                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + offset);
                __m512 f32 = Avx512f::Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), _norm, _bias), params, offset, tail);
                __m512 _scale = _mm512_maskz_loadu_ps(tail, scale + offset);
                __m512 _shift = _mm512_maskz_loadu_ps(tail, shift + offset);
                __m128i u8 = Cvt32fTo8u(Fmadd<nofma>(f32, _scale, _shift));
                _mm_mask_storeu_epi8(dst + offset, tail, _mm_min_epu8(u8, upper));
            }
        };

        template <> struct Term8iDepthwise<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm,
                const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail)
            {
                __m512 _norm = _mm512_maskz_loadu_ps(tail, norm + offset);
                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + offset);
                __m512 f32 = Avx512f::Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), _norm, _bias), params, offset, tail);
                _mm512_mask_storeu_ps((float*)dst + offset, tail, f32);
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm, const float* bias, const float* params, 
            const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail = -1)
        {
            Term8iDepthwise<term>::template Save<type, nofma>(dst, sum, norm, bias, params, scale, shift, upper, offset, tail);
        }
    }
#endif//SIMD_AVX512BW_ENABLE

#if defined(SIMD_NEON_ENABLE)
    namespace Neon
    {
        template <Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm, 
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t * scale, const float32x4_t* shift, uint8x8_t upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail);
        };

        template <> struct Term8i<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                float32x4_t f32 = Activate<type>(vaddq_f32(vmulq_f32(vcvtq_f32_s32(sum), norm[index]), bias[index]), params, index);
                int32x4_t i32 = Round(vaddq_f32(vmulq_f32(f32, scale[index]), shift[index]));
                uint8x8_t u8 = vmin_u8(vqmovun_s16(vcombine_s16(vmovn_s32(i32), vcreate_s16(0))), upper);
                ((int32_t*)dst)[index] = vget_lane_s32(vreinterpret_s32_u8(u8), 0);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
            {
                uint8_t tmp[F];
                Term8i::Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term8i<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                float32x4_t f32 = Activate<type>(vaddq_f32(vmulq_f32(vcvtq_f32_s32(sum), norm[index]), bias[index]), params, index);
                Store<false>((float*)dst + index * F, f32);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term8i<Term8iInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                Store<false>(buf + index * F, sum);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
            {
                int32_t tmp[F];
                Store<false>(tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, int32x4_t sum, 
            const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, int32x4_t sum, 
            const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, int32x4_t sum0, 
            int32x4_t sum1, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, int32x4_t sum0,
            int32x4_t sum1, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetConvolution8iCommon_h__

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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

namespace Simd
{
    enum Term8iType
    {
        Term8iSingle8u,
        Term8iSingle32f,
        Term8iFirst,
        Term8iIterim,
        Term8iLast8u,
        Term8iLast32f,
    };

    namespace Base
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE int32_t Activate(int32_t value, const int32_t* params, size_t offset);

        template<> SIMD_INLINE int32_t Activate<SimdConvolutionActivationIdentity>(int32_t value, const int32_t* params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE int32_t Activate<SimdConvolutionActivationRelu>(int32_t value, const int32_t* params, size_t offset)
        {
            return Simd::Max(0, value);
        }

        template<> SIMD_INLINE int32_t Activate<SimdConvolutionActivationRestrictRange>(int32_t value, const int32_t* params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128i Activate(__m128i value, const __m128i * params, size_t index);

        template<> SIMD_INLINE __m128i Activate<::SimdConvolutionActivationIdentity>(__m128i value, const __m128i * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m128i Activate<::SimdConvolutionActivationRelu>(__m128i value, const __m128i * params, size_t index)
        {
            return _mm_max_epi32(_mm_setzero_si128(), value);
        }

        template<> SIMD_INLINE __m128i Activate<::SimdConvolutionActivationRestrictRange>(__m128i value, const __m128i * params, size_t index)
        {
            return _mm_min_epi32(_mm_max_epi32(params[0], value), params[1]);
        }

        template <Term8iType term> struct Term
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift);
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail);
        };

        template <> struct Term<Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_slli_epi32(sum, norm), bias[index]), params, index);
                __m128 f32 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]);
                *((int32_t*)dst) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[F];
                Term::Save<type, norm, index>(tmp, buf, sum, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[i] = tmp[i];
            }
        };

        template <> struct Term<Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_slli_epi32(sum, norm), bias[index]), params, index);
                _mm_storeu_ps((float*)dst, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, norm, index>(tmp, buf, sum, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term<Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                _mm_storeu_si128((__m128i*)buf, sum);
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[i] = tmp[i];
            }
        };

        template <> struct Term<Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                _mm_storeu_si128((__m128i*)buf, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf), sum));
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[i] = tmp[i];
            }
        };

        template <> struct Term<Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf), sum);
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_slli_epi32(sum, norm), bias[index]), params, index);
                __m128 f32 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]);
                *((int32_t*)dst) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[F];
                Save<type, norm, index>(tmp, buf, sum, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[i] = tmp[i];
            }
        };

        template <> struct Term<Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf), sum);
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_slli_epi32(sum, norm), bias[index]), params, index);
                _mm_storeu_ps((float*)dst, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int norm, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, norm, index>(tmp, buf, sum, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[i] = ((float*)tmp)[i];
            }
        };
    }
#endif//SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif//SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetConvolution8iCommon_h__

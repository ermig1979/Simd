/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#if defined(SIMD_SSE41_ENABLE)   
    namespace Sse41
    {
        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf, __m128i sum, 
                const __m128 * norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf, __m128i sum, 
                const __m128 * norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail);
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
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
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
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
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                _mm_storeu_si128((__m128i*)buf + index, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum);
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
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum);
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), norm[index]), bias[index]), params, index);
                _mm_storeu_ps((float*)dst + index * F, f32);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128* norm, const __m128* bias, const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
            {
                uint8_t tmp[A];
                Term8i::Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> 
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, const __m128* norm, const __m128* bias,
            const __m128* params, const __m128* scale, const __m128* shift, __m128i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float * norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset);
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_loadu_ps(norm + offset)), _mm_loadu_ps(bias + offset)), params, offset);
                __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(f32, _mm_loadu_ps(scale + offset)), _mm_loadu_ps(shift + offset)));
                ((int32_t*)(dst + offset))[0] = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
            }
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* dst, __m128i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset)
            {
                __m128 f32 = Sse2::Activate<type>(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(sum), _mm_loadu_ps(norm + offset)), _mm_loadu_ps(bias + offset)), params, offset);
                _mm_storeu_ps((float*)dst + offset, f32);
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
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
        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum, 
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail);
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
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
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
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
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                _mm256_storeu_si256((__m256i*)buf + index, _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                sum = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum);
                __m256 f32 = Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                __m256i i32 = _mm256_cvtps_epi32(Fmadd<nofma>(f32, scale[index], shift[index]));
                ((int64_t*)dst)[index] = Extract64i<0>(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO), upper));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
            {
                uint8_t tmp[F];
                Save<type, index, nofma>(tmp - index * F, buf, sum, norm, bias, params, scale, shift, upper);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256* norm, const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
            {
                sum = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum);
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
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, const __m256* norm, 
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, const __m256* norm,
            const __m256* bias, const __m256* params, const __m256* scale, const __m256* shift, __m256i upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset);
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset)
            {
                __m256 f32 = Avx2::Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), _mm256_loadu_ps(norm + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                __m256i i32 = _mm256_cvtps_epi32(Fmadd<nofma>(f32, _mm256_loadu_ps(scale + offset), _mm256_loadu_ps(shift + offset)));
                ((int64_t*)(dst + offset))[0] = Extract64i<0>(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO), upper));
            }
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m256i sum,
                const float* norm, const float* bias, const float* params, const float* scale, const float* shift, __m256i upper, size_t offset)
            {
                __m256 f32 = Avx2::Activate<type>(Fmadd<nofma>(_mm256_cvtepi32_ps(sum), _mm256_loadu_ps(norm + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                _mm256_storeu_ps((float*)dst + offset, f32);
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
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
        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum, 
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1);
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                __m512i i32 = _mm512_cvtps_epi32(Fmadd<nofma>(f32, scale[index], shift[index]));
                __m128i u8 = _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(i32), Avx2::K_ZERO));
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm_min_epu8(u8, upper));
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                _mm512_mask_storeu_ps((float*)dst + index * F, tail, f32);
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum));
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                sum = _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum);
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                __m512i i32 = _mm512_cvtps_epi32(Fmadd<nofma>(f32, scale[index], shift[index]));
                __m128i u8 = _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(i32), Avx2::K_ZERO));
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm_min_epu8(u8, upper));
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512* norm, const __m512* bias, const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
            {
                sum = _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum);
                __m512 f32 = Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), norm[index], bias[index]), params, index);
                _mm512_mask_storeu_ps((float*)dst + index * F, tail, f32);
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum, const __m512* norm, const __m512* bias,
            const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1, const __m512* norm, const __m512* bias,
            const __m512* params, const __m512* scale, const __m512* shift, __m128i upper, __mmask16 tail = -1)
        {
            Term8i<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }

        //---------------------------------------------------------------------

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8iDepthwise
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm, 
                const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail);
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, __m512i sum, const float* norm,
                const float* bias, const float* params, const float* scale, const float* shift, __m128i upper, size_t offset, __mmask16 tail)
            {
                __m512 _norm = _mm512_maskz_loadu_ps(tail, norm + offset);
                __m512 _bias = _mm512_maskz_loadu_ps(tail, bias + offset);
                __m512 f32 = Avx512f::Activate<type>(Fmadd<nofma>(_mm512_cvtepi32_ps(sum), _norm, _bias), params, offset, tail);
                __m512 _scale = _mm512_maskz_loadu_ps(tail, scale + offset);
                __m512 _shift = _mm512_maskz_loadu_ps(tail, shift + offset);
                __m512i i32 = _mm512_cvtps_epi32(Fmadd<nofma>(f32, _scale, _shift));
                __m128i u8 = _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(i32), Avx2::K_ZERO));
                _mm_mask_storeu_epi8(dst + offset, tail, _mm_min_epu8(u8, upper));
            }
        };

        template <> struct Term8iDepthwise<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
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

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
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
        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term8i
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm, 
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t * scale, const float32x4_t* shift, uint8x8_t upper);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail);
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                Store<false>(buf + index * F, vaddq_s32(Load<false>(buf + index * F), sum));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
            {
                int32_t tmp[F];
                Store<false>(tmp, vaddq_s32(Load<false>(buf + index * F), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                sum = vaddq_s32(Load<false>(buf + index * F), sum);
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

        template <> struct Term8i<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, int32x4_t sum, const float32x4_t* norm,
                const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
            {
                sum = vaddq_s32(Load<false>(buf + index * F), sum);
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

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, int32x4_t sum, 
            const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, int32x4_t sum, 
            const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, upper, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, int32x4_t sum0, 
            int32x4_t sum1, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, int32x4_t sum0,
            int32x4_t sum1, const float32x4_t* norm, const float32x4_t* bias, const float32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, uint8x8_t upper, size_t tail)
        {
            Term8i<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift, upper);
            Term8i<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, upper, tail);
        }
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetConvolution8iCommon_h__

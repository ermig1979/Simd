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

namespace Simd
{
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

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t * dst, int32_t * buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail);
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_mullo_epi32(sum, norm), bias[index]), params, index);
                __m128 f32 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]);
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[F];
                Term::Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm_storeu_ps((float*)dst + index*F, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                _mm_storeu_si128((__m128i*)buf + index, sum);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                _mm_storeu_si128((__m128i*)buf + index, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum);
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_mullo_epi32(sum, norm), bias[index]), params, index);
                __m128 f32 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]);
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[F];
                Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
            {
                sum = _mm_add_epi32(_mm_loadu_si128((__m128i*)buf + index), sum);
                __m128i i32 = Activate<type>(_mm_add_epi32(_mm_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm_storeu_ps((float*)dst + index * F, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m128i sum, __m128i norm, const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, __m128i norm,
            const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum, __m128i norm,
            const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> 
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, __m128i norm, 
            const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1, __m128i norm,
            const __m128i* bias, const __m128i* params, const __m128* scale, const __m128* shift, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, tail);
        }
    }
#endif//SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256i Activate(__m256i value, const __m256i* params, size_t index);

        template<> SIMD_INLINE __m256i Activate<::SimdConvolutionActivationIdentity>(__m256i value, const __m256i* params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m256i Activate<::SimdConvolutionActivationRelu>(__m256i value, const __m256i* params, size_t index)
        {
            return _mm256_max_epi32(_mm256_setzero_si256(), value);
        }

        template<> SIMD_INLINE __m256i Activate<::SimdConvolutionActivationRestrictRange>(__m256i value, const __m256i* params, size_t index)
        {
            return _mm256_min_epi32(_mm256_max_epi32(params[0], value), params[1]);
        }

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail);
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                __m256i i32 = Activate<type>(_mm256_add_epi32(_mm256_mullo_epi32(sum, norm), bias[index]), params, index);
                __m256 f32 = Fmadd<nofma>(_mm256_cvtepi32_ps(i32), scale[index], shift[index]);
                ((int64_t*)dst)[index] = Extract64i<0>(PackI16ToU8(PackI32ToI16(_mm256_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                uint8_t tmp[F];
                Term::Save<type, index, nofma>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                __m256i i32 = Activate<type>(_mm256_add_epi32(_mm256_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm256_storeu_ps((float*)dst + index * F, Fmadd<nofma>(_mm256_cvtepi32_ps(i32), scale[index], shift[index]));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index, nofma>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                _mm256_storeu_si256((__m256i*)buf + index, sum);
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                _mm256_storeu_si256((__m256i*)buf + index, _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                sum = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum);
                __m256i i32 = Activate<type>(_mm256_add_epi32(_mm256_mullo_epi32(sum, norm), bias[index]), params, index);
                __m256 f32 = Fmadd<nofma>(_mm256_cvtepi32_ps(i32), scale[index], shift[index]);
                ((int64_t*)dst)[index] = Extract64i<0>(PackI16ToU8(PackI32ToI16(_mm256_cvtps_epi32(f32), K_ZERO), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                uint8_t tmp[F];
                Save<type, index, nofma>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
            {
                sum = _mm256_add_epi32(_mm256_loadu_si256((__m256i*)buf + index), sum);
                __m256i i32 = Activate<type>(_mm256_add_epi32(_mm256_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm256_storeu_ps((float*)dst + index * F, Fmadd<nofma>(_mm256_cvtepi32_ps(i32), scale[index], shift[index]));
            }

            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m256i sum, __m256i norm, const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index, nofma>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, __m256i norm,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum, __m256i norm,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, __m256i norm,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1, __m256i norm,
            const __m256i* bias, const __m256i* params, const __m256* scale, const __m256* shift, size_t tail)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, tail);
        }
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512i Activate(__m512i value, const __m512i* params, size_t index);

        template<> SIMD_INLINE __m512i Activate<::SimdConvolutionActivationIdentity>(__m512i value, const __m512i* params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m512i Activate<::SimdConvolutionActivationRelu>(__m512i value, const __m512i* params, size_t index)
        {
            return _mm512_max_epi32(_mm512_setzero_si512(), value);
        }

        template<> SIMD_INLINE __m512i Activate<::SimdConvolutionActivationRestrictRange>(__m512i value, const __m512i* params, size_t index)
        {
            return _mm512_min_epi32(_mm512_max_epi32(params[0], value), params[1]);
        }

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1);
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                __m512i i32 = Activate<type>(_mm512_add_epi32(_mm512_mullo_epi32(sum, norm), bias[index]), params, index);
                __m512 f32 = Fmadd<nofma>(_mm512_cvtepi32_ps(i32), scale[index], shift[index]);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(_mm512_cvtps_epi32(f32)), Avx2::K_ZERO)));
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                __m512i i32 = Activate<type>(_mm512_add_epi32(_mm512_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm512_mask_storeu_ps((float*)dst + index * F, tail, Fmadd<nofma>(_mm512_cvtepi32_ps(i32), scale[index], shift[index]));
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum));
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                sum = _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum);
                __m512i i32 = Activate<type>(_mm512_add_epi32(_mm512_mullo_epi32(sum, norm), bias[index]), params, index);
                __m512 f32 = Fmadd<nofma>(_mm512_cvtepi32_ps(i32), scale[index], shift[index]);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(_mm512_cvtps_epi32(f32)), Avx2::K_ZERO)));
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index, bool nofma> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                __m512i sum, __m512i norm, const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
            {
                sum = _mm512_add_epi32(_mm512_maskz_loadu_epi32(tail, buf + index * F), sum);
                __m512i i32 = Activate<type>(_mm512_add_epi32(_mm512_mullo_epi32(sum, norm), bias[index]), params, index);
                _mm512_mask_storeu_ps((float*)dst + index * F, tail, Fmadd<nofma>(_mm512_cvtepi32_ps(i32), scale[index], shift[index]));
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum, __m512i norm,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum, norm, bias, params, scale, shift, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type, bool nofma>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1, __m512i norm,
            const __m512i* bias, const __m512i* params, const __m512* scale, const __m512* shift, __mmask16 tail = -1)
        {
            Term<term>::template Save<type, 0, nofma>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1, nofma>(dst, buf, sum1, norm, bias, params, scale, shift, tail);
        }
    }
#endif//SIMD_AVX512BW_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE int32x4_t Activate(int32x4_t value, const int32x4_t* params, size_t index);

        template<> SIMD_INLINE int32x4_t Activate<::SimdConvolutionActivationIdentity>(int32x4_t value, const int32x4_t* params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE int32x4_t Activate<::SimdConvolutionActivationRelu>(int32x4_t value, const int32x4_t * params, size_t index)
        {
            return vmaxq_s32(vdupq_n_s32(0), value);
        }

        template<> SIMD_INLINE int32x4_t Activate<::SimdConvolutionActivationRestrictRange>(int32x4_t value, const int32x4_t * params, size_t index)
        {
            return vminq_s32(vmaxq_s32(params[0], value), params[1]);
        }

        template <Base::SynetConvolution8iNhwcDirect::Term8iType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t * scale, const float32x4_t* shift);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail);
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                int32x4_t i32 = Activate<type>(vaddq_s32(vmulq_s32(sum, norm), bias[index]), params, index);
                float32x4_t f32 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(i32), scale[index]), shift[index]);
                ((int32_t*)dst)[index] = vget_lane_s32(vreinterpret_s32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(Round(f32)), vcreate_s16(0)))), 0);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                uint8_t tmp[F];
                Term::Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iSingle32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                int32x4_t i32 = Activate<type>(vaddq_s32(vmulq_s32(sum, norm), bias[index]), params, index);
                Store<false>((float*)dst + index * F, vaddq_f32(vmulq_f32(vcvtq_f32_s32(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iFirst>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                Store<false>(buf + index * F, sum);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                int32_t tmp[F];
                Store<false>(tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iIterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                Store<false>(buf + index * F, vaddq_s32(Load<false>(buf + index * F), sum));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                int32_t tmp[F];
                Store<false>(tmp, vaddq_s32(Load<false>(buf + index * F), sum));
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast8u>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                sum = vaddq_s32(Load<false>(buf + index * F), sum);
                int32x4_t i32 = Activate<type>(vaddq_s32(vmulq_s32(sum, norm), bias[index]), params, index);
                float32x4_t f32 = vaddq_f32(vmulq_f32(vcvtq_f32_s32(i32), scale[index]), shift[index]);
                ((int32_t*)dst)[index] = vget_lane_s32(vreinterpret_s32_u8(vqmovun_s16(vcombine_s16(vmovn_s32(Round(f32)), vcreate_s16(0)))), 0);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                uint8_t tmp[F];
                Save<type, index>(tmp - index * F, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }
        };

        template <> struct Term<Base::SynetConvolution8iNhwcDirect::Term8iLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
            {
                sum = vaddq_s32(Load<false>(buf + index * F), sum);
                int32x4_t i32 = Activate<type>(vaddq_s32(vmulq_s32(sum, norm), bias[index]), params, index);
                Store<false>((float*)dst + index * F, vaddq_f32(vmulq_f32(vcvtq_f32_s32(i32), scale[index]), shift[index]));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf,
                int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
            {
                uint8_t tmp[A];
                Save<type, index>(tmp - index * A, buf, sum, norm, bias, params, scale, shift);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[index * F + i] = ((float*)tmp)[i];
            }
        };

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, 
            int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, 
            int32x4_t sum, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum, norm, bias, params, scale, shift, tail);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf,
            int32x4_t sum0, int32x4_t sum1, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift);
        }

        template<Base::SynetConvolution8iNhwcDirect::Term8iType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf,
            int32x4_t sum0, int32x4_t sum1, int32x4_t norm, const int32x4_t* bias, const int32x4_t* params, const float32x4_t* scale, const float32x4_t* shift, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, buf, sum0, norm, bias, params, scale, shift);
            Term<term>::template Save<type, 1>(dst, buf, sum1, norm, bias, params, scale, shift, tail);
        }
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetConvolution8iCommon_h__

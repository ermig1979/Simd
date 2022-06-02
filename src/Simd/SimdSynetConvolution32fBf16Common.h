/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#ifndef __SimdSynetConvolution32fBf16Common_h__
#define __SimdSynetConvolution32fBf16Common_h__

#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
    enum TermBf16Type
    {
        TermBf16Last16b,
        TermBf16Last32f,
        TermBf16Interim,
        TermBf16Size
    };

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template <TermBf16Type term> struct TermBf16
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail);
        };

        template <> struct TermBf16<TermBf16Last16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storel_epi64((__m128i*)ptr, Float32ToBFloat16(Sse2::Activate<type>(_mm_add_ps(value, bias[index]), params, index)));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, Float32ToBFloat16(Sse2::Activate<type>(_mm_add_ps(value, bias[index]), params, index)));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template <> struct TermBf16<TermBf16Last32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storeu_ps((float*)ptr, Sse2::Activate<type>(_mm_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, Sse2::Activate<type>(_mm_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i] = tmp[i];
            }
        };

        template <> struct TermBf16<TermBf16Interim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storeu_ps((float*)ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i] = tmp[i];
            }
        };

        template<TermBf16Type term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m128 val0, const __m128* bias, const __m128* params)
        {
            TermBf16<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<TermBf16Type term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m128 val0, const __m128* bias, const __m128* params, size_t tail)
        {
            TermBf16<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<TermBf16Type term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params)
        {
            TermBf16<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            TermBf16<term>::template Save<type, 1>(dst + F, val1, bias, params);
        }

        template<TermBf16Type term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params, size_t tail)
        {
            TermBf16<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            TermBf16<term>::template Save<type, 1>(dst + F, val1, bias, params, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif//SIMD_AVX2_ENABLE
}
#endif//__SimdSynetConvolution32fBf16Common_h__

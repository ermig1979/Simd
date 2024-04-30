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
#ifndef __SimdSynetConvolution16bCommon_h__
#define __SimdSynetConvolution16bCommon_h__

#include "Simd/SimdSynetConvolution32fCommon.h"
#include "Simd/SimdBFloat16.h"

namespace Simd
{
    enum Term16bType
    {
        Term16bLast16b,
        Term16bLast32f,
        Term16bInterim,
        Term16bSize
    };

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params, size_t tail);
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(value, bias[index]), params, index);
                _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(value, bias[index]), params, index);
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(value, bias[index]), params, index);
                _mm_storel_epi64((__m128i*)(ptr + 8 * index), _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(value, bias[index]), params, index);
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)ptr)[index * F + i] = tmp[i];
            }
        };

        template <> struct Term16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storeu_ps((float*)ptr, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storeu_ps((float*)ptr + index * F, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[index * F + i] = tmp[i];
            }
        };

        template <> struct Term16b<Term16bInterim>
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

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params)
            {
                _mm_storeu_ps(buf + index * F, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params, size_t tail)
            {
                _mm_storeu_ps(buf + index * F, value);
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m128 val0, const __m128* bias, const __m128* params)
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m128 val0, const __m128* bias, const __m128* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const __m128* bias, const __m128* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const __m128* bias, const __m128* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params)
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const __m128* bias, const __m128* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const __m128* bias, const __m128* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params, size_t tail);
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params)
            {
                __m256 f32 = Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                _mm_storeu_si128((__m128i*)ptr, _mm256_castsi256_si128(b16));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                __m256 f32 = Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                uint16_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm256_castsi256_si128(b16));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params)
            {
                __m256 f32 = Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                _mm_storeu_si128((__m128i*)ptr + index, _mm256_castsi256_si128(b16));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                __m256 f32 = Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                uint16_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm256_castsi256_si128(b16));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)ptr)[index * F + i] = tmp[i];
            }
        };

        template <> struct Term16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params)
            {
                _mm256_storeu_ps((float*)ptr, Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params)
            {
                _mm256_storeu_ps((float*)ptr + index * F, Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, Avx2::Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[index * F + i] = tmp[i];
            }
        };

        template <> struct Term16b<Term16bInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params)
            {
                _mm256_storeu_ps((float*)ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params)
            {
                _mm256_storeu_ps(buf + index * F, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params, size_t tail)
            {
                _mm256_storeu_ps(buf + index * F, value);
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m256 val0, const __m256* bias, const __m256* params)
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m256 val0, const __m256* bias, const __m256* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const __m256* bias, const __m256* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const __m256* bias, const __m256* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params)
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const __m256* bias, const __m256* params)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const __m256* bias, const __m256* params, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16(ptr, tail, _mm512_cvtepi32_epi16(Float32ToBFloat16(f32)));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16(ptr + index * DF, tail, _mm512_cvtepi32_epi16(Float32ToBFloat16(f32)));
            }
        };

        template <> struct Term16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps((float*)ptr, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps((float*)ptr + index * F, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }
        };

        template <> struct Term16b<Term16bInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps((float*)ptr, tail, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps(buf + index * F, tail, value);
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m512 val0, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m512 val0, const __m512* bias, const __m512* params, const __mmask16* tails)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tails[0]);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, const __mmask16* tails)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tails[1]);
        }
    }
#endif

#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        SIMD_INLINE void ConvertA(const float* src, uint16_t* dst)
        {
            __m512 s0 = _mm512_loadu_ps(src + 0 * 16);
            __m512 s1 = _mm512_loadu_ps(src + 1 * 16);
            _mm512_storeu_si512(dst, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
        }

        SIMD_INLINE void ConvertB(const float* src, int stride, uint16_t* dst, __mmask16 tail = __mmask16(-1))
        {
            static const __m512i PERM_IDX = _mm512_set_epi16(
                0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
                0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
            __m512 s0 = _mm512_maskz_loadu_ps(tail, src + 0 * stride);
            __m512 s1 = _mm512_maskz_loadu_ps(tail, src + 1 * stride);
            __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16(ptr, tail, (__m256i)_mm512_cvtneps_pbh(f32));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16((uint16_t*)ptr + index * F, tail, (__m256i)_mm512_cvtneps_pbh(f32));
            }
        };

        template <> struct Term16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps((float*)ptr, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                _mm512_mask_storeu_ps((float*)ptr + index * F, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }
        };

        template <> struct Term16b<Term16bInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps((float*)ptr, tail, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
            }
        };

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m512 val0, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Apply<type, 0>(ptr, buf, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1x8(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply1<term, type>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            Apply1<term, type>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Apply<type, 0>(ptr, buf, bias, params);
            Term16b<term>::template Apply<type, 1>(ptr, buf, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2x8(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply2<term, type>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            Apply2<term, type>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }
    }
#endif
}
#endif

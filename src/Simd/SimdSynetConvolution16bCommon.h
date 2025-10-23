/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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

#include "Simd/SimdSynetActivation.h"
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

    namespace Base
    {
        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, float value, const float* bias, const float* params, size_t offset);
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t * ptr, float value, const float* bias, const float* params, size_t offset)
            {
                ((uint16_t*)ptr)[offset] = Float32ToBFloat16(Activate<type>(value + bias[offset], params, offset));
            }
        };

        template <> struct Term16b<Term16bLast32f>
        {
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Save(uint8_t* ptr, float value, const float* bias, const float* params, size_t offset)
            {
                ((float*)ptr)[offset] = Activate<type>(value + bias[offset], params, offset);
            }
        };

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float val0, const float* bias, const float* params, size_t offset)
        {
            Term16b<term>::template Save<type>(ptr, val0, bias, params, offset + 0);
        }
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        template <class T> SIMD_INLINE __m128 LoadSrc(const T* src);

        template <> SIMD_INLINE __m128 LoadSrc<float>(const float* src)
        {
            return _mm_loadu_ps(src);
        }

        template <> SIMD_INLINE __m128 LoadSrc<uint16_t>(const uint16_t* src)
        {
            return  _mm_castsi128_ps(UnpackU16<0>(K_ZERO, _mm_loadl_epi64((__m128i*)src)));
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m128 value, const __m128* bias, const __m128* params, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const __m128* bias, const __m128* params, size_t tail);

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst);
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset, size_t tail);
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(_mm_loadu_ps(src + offset), _mm_loadu_ps(bias + offset)), params, offset);
                _mm_storel_epi64((__m128i*)(dst + offset * 2), _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(_mm_loadu_ps(src + offset), _mm_loadu_ps(bias + offset)), params, offset);
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)dst)[offset + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset)
            {
                __m128 f32 = ActivateNchw<type>(_mm_add_ps(value, _mm_set1_ps(bias[offset])), params, offset);
                _mm_storel_epi64((__m128i*)(ptr + index * DF), _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                __m128 f32 = ActivateNchw<type>(_mm_add_ps(value, _mm_set1_ps(bias[offset])), params, offset);
                uint16_t tmp[F];
                _mm_storel_epi64((__m128i*)tmp, _mm_packus_epi32(Float32ToBFloat16(f32), K_ZERO));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)ptr)[i + index * F] = tmp[i];
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(_mm_loadu_ps(src + offset), _mm_loadu_ps(bias + offset)), params, offset);
                _mm_storeu_ps((float*)(dst + offset * 4), f32);
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
                __m128 f32 = Activate<type>(_mm_add_ps(_mm_loadu_ps(src + offset), _mm_loadu_ps(bias + offset)), params, offset);
                float tmp[F];
                _mm_storeu_ps(tmp, f32);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[offset + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset)
            {
                _mm_storeu_ps((float*)ptr + index * F, ActivateNchw<type>(_mm_add_ps(value, _mm_set1_ps(bias[offset])), params, offset));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, ActivateNchw<type>(_mm_add_ps(value, _mm_set1_ps(bias[offset])), params, offset));
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i + index * F] = tmp[i];
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
                float tmp[F];
                _mm_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset)
            {
                _mm_storeu_ps(buf + index * F, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m128 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    buf[i + index * F] = tmp[i];
            }
        };

        //-------------------------------------------------------------------------------------------------

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

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst)
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const __m128* bias)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
        }

        template<Term16bType term> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const __m128* bias, size_t tail)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL, tail);
        }

        template<Term16bType term> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const __m128* bias)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 1>(ptr, buf, val1, bias, NULL);
        }

        template<Term16bType term> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const __m128* bias, size_t tail)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 1>(ptr, buf, val1, bias, NULL, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const float* bias, const float* params, size_t offset)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m128 val0, const float* bias, const float* params, size_t offset, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const float* bias, const float* params, size_t offset)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, offset);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m128 val0, __m128 val1, const float* bias, const float* params, size_t offset, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, offset, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <class T> SIMD_INLINE __m256 LoadSrc(const T* src);

        template <> SIMD_INLINE __m256 LoadSrc<float>(const float* src)
        {
            return _mm256_loadu_ps(src);
        }

        template <> SIMD_INLINE __m256 LoadSrc<uint16_t>(const uint16_t* src)
        {
            return Avx2::BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)src)));
        }

        //-------------------------------------------------------------------------------------------------
        
        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m256 value, const __m256* bias, const __m256* params, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const __m256* bias, const __m256* params, size_t tail);

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst);
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail);

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset, size_t tail);
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
                __m256 f32 = Activate<type>(_mm256_add_ps(_mm256_loadu_ps(src + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                _mm_storeu_si128((__m128i*)(dst + offset * 2), _mm256_castsi256_si128(b16));
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
                __m256 f32 = Activate<type>(_mm256_add_ps(_mm256_loadu_ps(src + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                uint16_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm256_castsi256_si128(b16));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)dst)[offset + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset)
            {
                __m256 f32 = ActivateNchw<type>(_mm256_add_ps(value, _mm256_set1_ps(bias[offset])), params, offset);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                _mm_storeu_si128((__m128i*)(ptr + index * DF), _mm256_castsi256_si128(b16));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                __m256 f32 = ActivateNchw<type>(_mm256_add_ps(value, _mm256_set1_ps(bias[offset])), params, offset);
                __m256i b16 = _mm256_permute4x64_epi64(_mm256_packus_epi32(Float32ToBFloat16(f32), Avx2::K_ZERO), 0xD8);
                uint16_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, _mm256_castsi256_si128(b16));
                for (size_t i = 0; i < tail; ++i)
                    ((uint16_t*)ptr)[i + index * F] = tmp[i];
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
                __m256 f32 = Activate<type>(_mm256_add_ps(_mm256_loadu_ps(src + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                _mm256_storeu_ps((float*)(dst + offset * 4), f32);
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
                __m256 f32 = Activate<type>(_mm256_add_ps(_mm256_loadu_ps(src + offset), _mm256_loadu_ps(bias + offset)), params, offset);
                float tmp[F];
                _mm256_storeu_ps(tmp, f32);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)dst)[offset + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset)
            {
                __m256 f32 = ActivateNchw<type>(_mm256_add_ps(value, _mm256_set1_ps(bias[offset])), params, offset);
                _mm256_storeu_ps((float*)ptr + index * F, f32);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                __m256 f32 = ActivateNchw<type>(_mm256_add_ps(value, _mm256_set1_ps(bias[offset])), params, offset);
                float tmp[F];
                _mm256_storeu_ps(tmp, f32);
                for (size_t i = 0; i < tail; ++i)
                    ((float*)ptr)[i + index * F] = tmp[i];
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
                float tmp[F];
                _mm256_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst)
            {
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
            {
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset)
            {
                _mm256_storeu_ps(buf + index * F, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m256 value, const float* bias, const float* params, size_t offset, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    buf[i + index * F] = tmp[i];
            }
        };

        //-------------------------------------------------------------------------------------------------

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

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst)
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst, size_t tail)
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const __m256* bias)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
        }

        template<Term16bType term> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const __m256* bias, size_t tail)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL, tail);
        }

        template<Term16bType term> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const __m256* bias)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 1>(ptr, buf, val1, bias, NULL);
        }

        template<Term16bType term> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const __m256* bias, size_t tail)
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 1>(ptr, buf, val1, bias, NULL, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const float* bias, const float* params, size_t offset)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m256 val0, const float* bias, const float* params, size_t offset, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const float* bias, const float* params, size_t offset)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, offset);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m256 val0, __m256 val1, const float* bias, const float* params, size_t offset, size_t tail)
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, offset, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <class T> SIMD_INLINE __m512 LoadSrc(const T* src, __mmask16 tail = __mmask16(-1));

        template <> SIMD_INLINE __m512 LoadSrc<float>(const float* src, __mmask16 tail)
        {
            return _mm512_maskz_loadu_ps(tail, src);
        }

        template <> SIMD_INLINE __m512 LoadSrc<uint16_t>(const uint16_t* src, __mmask16 tail)
        {
            return BFloat16ToFloat32(_mm256_maskz_loadu_epi16(tail, src));
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1));
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + offset), _mm512_loadu_ps(bias + offset)), params, offset);
                _mm256_mask_storeu_epi16(dst + offset * 2, tail, _mm512_cvtepi32_epi16(Float32ToBFloat16(f32)));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = ActivateNchw<type>(_mm512_add_ps(value, _mm512_set1_ps(bias[offset])), params, offset);
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + offset), _mm512_loadu_ps(bias + offset)), params, offset);
                _mm512_mask_storeu_ps((float*)(dst + offset * 4), tail, f32);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = ActivateNchw<type>(_mm512_add_ps(value, _mm512_set1_ps(bias[offset])), params, offset);
                _mm512_mask_storeu_ps((float*)ptr + index * F, tail, f32);
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

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps(buf + index * F, tail, value);
            }
        };

        //-------------------------------------------------------------------------------------------------

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

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m512 val0, const __m512* bias, const __m512* params, const __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, const __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m512 val0, const __m512* bias, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL, tail);
        }

        template<Term16bType term> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const __m512* bias, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 0>(ptr, buf, val0, bias, NULL);
            Term16b<term>::template Save<SimdConvolutionActivationIdentity, 1>(ptr, buf, val1, bias, NULL, tail);
        }

        //-----------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveInput1(float* dst, __m512 sum, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst, Activate<type>(_mm512_add_ps(sum, bias[0]), params, 0));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void SaveInput2(float* dst0, float* dst1, __m512 sum0, __m512 sum1, const __m512* bias, const __m512* params)
        {
            _mm512_storeu_ps(dst0, Activate<type>(_mm512_add_ps(sum0, bias[0]), params, 0));
            _mm512_storeu_ps(dst1, Activate<type>(_mm512_add_ps(sum1, bias[1]), params, 1));
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m512 val0, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, offset);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, offset, tail);
        }
    }
#endif

#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
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

        SIMD_INLINE void ConvertAq(const float* src, uint16_t* dst, __mmask8 mask = __mmask8(-1))
        {
            __m256 s0 = _mm256_maskz_loadu_ps(mask, src);
            _mm_mask_storeu_epi16(dst, mask, (__m128i)_mm256_cvtneps_pbh(s0));
        }

        SIMD_INLINE void ConvertAh(const float* src, uint16_t* dst, __mmask16 mask = __mmask16(-1))
        {
            __m512 s0 = _mm512_maskz_loadu_ps(mask, src);
            _mm256_mask_storeu_epi16(dst, mask, (__m256i)_mm512_cvtneps_pbh(s0));
        }

        SIMD_INLINE void ConvertA(const float* src, uint16_t* dst, __mmask16 srcMask0 = __mmask16(-1), __mmask16 srcMask1 = __mmask16(-1), __mmask32 dstMask = __mmask32(-1))
        {
            __m512 s0 = _mm512_maskz_loadu_ps(srcMask0, src + 0 * 16);
            __m512 s1 = _mm512_maskz_loadu_ps(srcMask1, src + 1 * 16);
            _mm512_mask_storeu_epi16(dst, dstMask, (__m512i)_mm512_cvtne2ps_pbh(s1, s0));
        }

        SIMD_INLINE void ConvertB(const float* src, int stride, uint16_t* dst, __mmask16 mask = __mmask16(-1))
        {
            static const __m512i PERM_IDX = _mm512_set_epi16(
                0x1f, 0x0f, 0x1e, 0x0e, 0x1d, 0x0d, 0x1c, 0x0c, 0x1b, 0x0b, 0x1a, 0x0a, 0x19, 0x09, 0x18, 0x08,
                0x17, 0x07, 0x16, 0x06, 0x15, 0x05, 0x14, 0x04, 0x13, 0x03, 0x12, 0x02, 0x11, 0x01, 0x10, 0x00);
            __m512 s0 = _mm512_maskz_loadu_ps(mask, src + 0 * stride);
            __m512 s1 = _mm512_maskz_loadu_ps(mask, src + 1 * stride);
            __m512i d = (__m512i)_mm512_cvtne2ps_pbh(s1, s0);
            _mm512_storeu_si512(dst, _mm512_permutexvar_epi16(PERM_IDX, d));
        }

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(const float* src, float* dst, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            _mm512_mask_storeu_ps(dst, tail, Activate<type>(_mm512_add_ps(_mm512_maskz_loadu_ps(tail, src), bias[index]), params, index));
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply1(float* dst, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply<type, 0>(dst, dst, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply1x4(float* dst, size_t stride, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply1<type>(dst + 0 * stride, bias, params, tail);
            Apply1<type>(dst + 1 * stride, bias, params, tail);
            Apply1<type>(dst + 2 * stride, bias, params, tail);
            Apply1<type>(dst + 3 * stride, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply1x8(float* dst, size_t stride, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply1<type>(dst + 0 * stride, bias, params, tail);
            Apply1<type>(dst + 1 * stride, bias, params, tail);
            Apply1<type>(dst + 2 * stride, bias, params, tail);
            Apply1<type>(dst + 3 * stride, bias, params, tail);
            Apply1<type>(dst + 4 * stride, bias, params, tail);
            Apply1<type>(dst + 5 * stride, bias, params, tail);
            Apply1<type>(dst + 6 * stride, bias, params, tail);
            Apply1<type>(dst + 7 * stride, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply2(float* dst, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply<type, 0>(dst + 0, dst + 0, bias, params);
            Apply<type, 1>(dst + F, dst + F, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply2x4(float* dst, size_t stride, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply2<type>(dst + 0 * stride, bias, params, tail);
            Apply2<type>(dst + 1 * stride, bias, params, tail);
            Apply2<type>(dst + 2 * stride, bias, params, tail);
            Apply2<type>(dst + 3 * stride, bias, params, tail);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply2x8(float* dst, size_t stride, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Apply2<type>(dst + 0 * stride, bias, params, tail);
            Apply2<type>(dst + 1 * stride, bias, params, tail);
            Apply2<type>(dst + 2 * stride, bias, params, tail);
            Apply2<type>(dst + 3 * stride, bias, params, tail);
            Apply2<type>(dst + 4 * stride, bias, params, tail);
            Apply2<type>(dst + 5 * stride, bias, params, tail);
            Apply2<type>(dst + 6 * stride, bias, params, tail);
            Apply2<type>(dst + 7 * stride, bias, params, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template <Term16bType term> struct Term16b
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1));
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1));
        };

        template <> struct Term16b<Term16bLast16b>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint16_t* ptr, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16(ptr, tail, (__m256i)_mm512_cvtneps_pbh(f32));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(uint8_t* ptr, float* buf, __m512 value, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16(ptr + index * DF, tail, (__m256i)_mm512_cvtneps_pbh(f32));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                __m512 f32 = Activate<type>(_mm512_add_ps(value, bias[index]), params, index);
                _mm256_mask_storeu_epi16((uint16_t*)ptr + index * F, tail, (__m256i)_mm512_cvtneps_pbh(f32));
                _mm_prefetch((const char*)(ptr + index * DF), _MM_HINT_NTA);
                _mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + offset), _mm512_loadu_ps(bias + offset)), params, offset);
                _mm256_mask_storeu_epi16(dst + offset * 2, tail, (__m256i)_mm512_cvtneps_pbh(f32));
                //_mm_prefetch((const char*)(src + offset), _MM_HINT_NTA);
                //_mm_prefetch((const char*)(dst + offset * 2), _MM_HINT_NTA);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                __m512 f32 = ActivateNchw<type>(_mm512_add_ps(value, _mm512_set1_ps(bias[offset])), params, offset);
                _mm256_mask_storeu_epi16((uint16_t*)ptr + index * F, tail, (__m256i)_mm512_cvtneps_pbh(f32));
                _mm_prefetch((const char*)(ptr + index * DF), _MM_HINT_NTA);
                _mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
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

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                _mm512_mask_storeu_ps((float*)ptr + index * F, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
                _mm_prefetch((const char*)(ptr + index * A), _MM_HINT_NTA);
                _mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
                __m512 f32 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(src + offset), _mm512_loadu_ps(bias + offset)), params, offset);
                _mm512_mask_storeu_ps((float*)(dst + offset * 4), tail, f32);
                //_mm_prefetch((const char*)(src + offset), _MM_HINT_NTA);
                //_mm_prefetch((const char*)(dst + offset * 4), _MM_HINT_NTA);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
                __m512 value = _mm512_maskz_loadu_ps(tail, buf + index * F);
                __m512 f32 = ActivateNchw<type>(_mm512_add_ps(value, _mm512_set1_ps(bias[offset])), params, offset);
                _mm512_mask_storeu_ps((float*)ptr + index * F, tail, f32);
                _mm_prefetch((const char*)(ptr + index * A), _MM_HINT_NTA);
                //_mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
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

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
            {
            }

            template<SimdConvolutionActivationType type> static SIMD_INLINE void Postprocess(const float* src, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
            {
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Apply(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
            {
            }
        };

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint16_t* dst, __m512 val0, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint16_t* dst, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(dst + 0 * DF, val0, bias, params);
            Term16b<term>::template Save<type, 1>(dst + 1 * DF, val1, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(uint8_t* ptr, float* buf, __m512 val0, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(uint8_t* ptr, float* buf, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Save<type, 0>(ptr, buf, val0, bias, params);
            Term16b<term>::template Save<type, 1>(ptr, buf, val1, bias, params, tail);
        }

        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply16b2(uint8_t* ptr, float* buf, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            __m512 f0 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + 0), bias[0]), params, 0);
            __m512 f1 = Activate<type>(_mm512_add_ps(_mm512_loadu_ps(buf + F), bias[1]), params, 1);
            _mm512_mask_storeu_epi16((uint16_t*)ptr, tail, (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
            _mm_prefetch((const char*)ptr, _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + 0), _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + F), _MM_HINT_NTA);
        }

        template<SimdConvolutionActivationType type> SIMD_INLINE void Apply16b2x8(uint8_t* ptr, int dP, float* buf, int dB, const __m512* bias, const __m512* params, __mmask32 tail = __mmask32(-1))
        {
            Apply16b2<type>(ptr + 0 * dP, buf + 0 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 1 * dP, buf + 1 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 2 * dP, buf + 2 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 3 * dP, buf + 3 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 4 * dP, buf + 4 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 5 * dP, buf + 5 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 6 * dP, buf + 6 * dB, bias, params, tail);
            Apply16b2<type>(ptr + 7 * dP, buf + 7 * dB, bias, params, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Postprocess(const float* sum, const float* bias, const float* params, size_t offset, uint8_t* dst, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Postprocess<type>(sum, bias, params, offset, dst, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Apply<type, 0>(ptr, buf, bias, params, offset, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply1x8(uint8_t* ptr, int dP, float* buf, int dB, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Apply1<term, type>(ptr + 0 * dP, buf + 0 * dB, bias, params, offset + 0, tail);
            Apply1<term, type>(ptr + 1 * dP, buf + 1 * dB, bias, params, offset + 1, tail);
            Apply1<term, type>(ptr + 2 * dP, buf + 2 * dB, bias, params, offset + 2, tail);
            Apply1<term, type>(ptr + 3 * dP, buf + 3 * dB, bias, params, offset + 3, tail);
            Apply1<term, type>(ptr + 4 * dP, buf + 4 * dB, bias, params, offset + 4, tail);
            Apply1<term, type>(ptr + 5 * dP, buf + 5 * dB, bias, params, offset + 5, tail);
            Apply1<term, type>(ptr + 6 * dP, buf + 6 * dB, bias, params, offset + 6, tail);
            Apply1<term, type>(ptr + 7 * dP, buf + 7 * dB, bias, params, offset + 7, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2(uint8_t* ptr, float* buf, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Term16b<term>::template Apply<type, 0>(ptr, buf, bias, params, offset);
            Term16b<term>::template Apply<type, 1>(ptr, buf, bias, params, offset, tail);
        }

        template<Term16bType term, SimdConvolutionActivationType type> SIMD_INLINE void Apply2x8(uint8_t* ptr, int dP, float* buf, int dB, const float* bias, const float* params, size_t offset, __mmask16 tail = __mmask16(-1))
        {
            Apply2<term, type>(ptr + 0 * dP, buf + 0 * dB, bias, params, offset + 0, tail);
            Apply2<term, type>(ptr + 1 * dP, buf + 1 * dB, bias, params, offset + 1, tail);
            Apply2<term, type>(ptr + 2 * dP, buf + 2 * dB, bias, params, offset + 2, tail);
            Apply2<term, type>(ptr + 3 * dP, buf + 3 * dB, bias, params, offset + 3, tail);
            Apply2<term, type>(ptr + 4 * dP, buf + 4 * dB, bias, params, offset + 4, tail);
            Apply2<term, type>(ptr + 5 * dP, buf + 5 * dB, bias, params, offset + 5, tail);
            Apply2<term, type>(ptr + 6 * dP, buf + 6 * dB, bias, params, offset + 6, tail);
            Apply2<term, type>(ptr + 7 * dP, buf + 7 * dB, bias, params, offset + 7, tail);
        }
    }
#endif
}
#endif

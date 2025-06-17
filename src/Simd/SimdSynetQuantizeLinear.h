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
#ifndef __SimdSynetQuantizeLinear_h__
#define __SimdSynetQuantizeLinear_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdSynetConvolution8iCommon.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int NearByInt(float value)
        {
            return (int)std::nearbyint(value);
        }

        SIMD_INLINE int QuantizeLinear(int value, float norm, int zero, int min, int max)
        {
            return RestrictRange(NearByInt(float(value) * norm) + zero, min, max);
        }

        SIMD_INLINE int QuantizeSumLinear(int sum, int bias, float norm, int zero, int min, int max)
        {
            return RestrictRange(NearByInt(float(sum + bias) * norm) + zero, min, max);
        }

        SIMD_INLINE void QuantizeSumLinear(const int32_t * sum, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const int32_t * bias, const float* norm, const int32_t * zero, uint8_t * dst)
        {
            constexpr int min = std::numeric_limits<uint8_t>::min();
            constexpr int max = std::numeric_limits<uint8_t>::max();
            for (size_t b = 0; b < batch; ++b)
            {
                if (format == SimdTensorFormatNchw)
                {
                    for (size_t c = 0; c < channels; ++c)
                    {
                        int32_t _bias = bias[c];
                        float _norm = norm[c];
                        int32_t _zero = zero[c];
                        for (size_t h = 0; h < height; ++h)
                        {
                            for (size_t w = 0; w < width; ++w)
                                dst[w] = (uint8_t)QuantizeSumLinear(sum[w], _bias, _norm, _zero, min, max);
                            sum += width;
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
                                dst[c] = (uint8_t)QuantizeSumLinear(sum[c], bias[c], norm[c], zero[c], min, max);
                            sum += channels;
                            dst += channels;
                        }
                    }
                }
                else
                    assert(0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE float DequantizeLinear(int value, int bias, float norm)
        {
            return float(value + bias) * norm;
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128 DequantizeLinear(__m128i value, __m128i bias, __m128 norm)
        {
            return _mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(value, bias)), norm);
        }

        SIMD_INLINE void DequantizeLinear1(const uint8_t* src, __m128i bias, __m128 norm, float* dst)
        {
            __m128i _src = _mm_set1_epi32(src[0]);
            __m128 _dst = DequantizeLinear(_src, bias, norm);
            _mm_store_ss(dst, _dst);
        }

        SIMD_INLINE void DequantizeLinear4(const uint8_t* src, __m128i bias, __m128 norm, float* dst)
        {
            __m128i _src = _mm_cvtepu8_epi32(_mm_set1_epi32(((int32_t*)src)[0]));
            __m128 _dst = DequantizeLinear(_src, bias, norm);
            _mm_storeu_ps(dst, _dst);
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i QuantizeLinear(__m128 value, __m128 norm, __m128i zero)
        {
            return _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(value, norm)), zero);
        }

        SIMD_INLINE void QuantizeLinear1(const float* src, __m128 norm, __m128i zero, uint8_t* dst)
        {
            __m128 _src = _mm_load_ss(src);
            __m128i i32 = QuantizeLinear(_src, norm, zero);
            dst[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizeLinear4(const float* src, __m128 norm, __m128i zero, uint8_t* dst)
        {
            __m128 _src = _mm_loadu_ps(src);
            __m128i i32 = QuantizeLinear(_src, norm, zero);
            ((uint32_t*)dst)[0] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO));
        }

        SIMD_INLINE void QuantizeLinear16(const float* src, __m128 norm, __m128i zero, uint8_t* dst)
        {
            __m128i i0 = QuantizeLinear(_mm_loadu_ps(src + 0 * 4), norm, zero);
            __m128i i1 = QuantizeLinear(_mm_loadu_ps(src + 1 * 4), norm, zero);
            __m128i i2 = QuantizeLinear(_mm_loadu_ps(src + 2 * 4), norm, zero);
            __m128i i3 = QuantizeLinear(_mm_loadu_ps(src + 3 * 4), norm, zero);
            _mm_storeu_si128((__m128i*)dst,  _mm_packus_epi16(_mm_packs_epi32(i0, i1), _mm_packs_epi32(i2, i3)));
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero);
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail);

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                __m128i i32 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum, bias[index])), norm[index])), zero);
                ((int32_t*)dst)[index] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO));
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
            {
                uint8_t tmp[F];
                QuntizedTerm8i::Save<index>(tmp - index * F, buf, sum, bias, norm, zero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                __m128i d0 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum0, bias[0])), norm[0])), zero);
                __m128i d1 = _mm_add_epi32(_mm_cvtps_epi32(_mm_mul_ps(_mm_cvtepi32_ps(_mm_add_epi32(sum1, bias[1])), norm[1])), zero);
                _mm_storel_epi64((__m128i*)dst, _mm_packus_epi16(_mm_packs_epi32(d0, d1), K_ZERO));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                _mm_storeu_si128((__m128i*)buf + index, sum);
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum,
                const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
            {
                int32_t tmp[F];
                _mm_storeu_si128((__m128i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
                const __m128i* bias, const __m128* norm, const __m128i& zero)
            {
                _mm_storeu_si128((__m128i*)buf + 0, sum0);
                _mm_storeu_si128((__m128i*)buf + 1, sum1);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* bias, const __m128* norm, const __m128i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m128i sum,
            const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* bias, const __m128* norm, const __m128i& zero)
        {
            QuntizedTerm8i<term>::Save(dst, buf, sum0, sum1, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m128i sum0, __m128i sum1,
            const __m128i* bias, const __m128* norm, const __m128i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256 DequantizeLinear(__m256i value, __m256i bias, __m256 norm)
        {
            return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(value, bias)), norm);
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE __m256i QuantizeLinear(__m256 value, __m256 norm, __m256i zero)
        {
            return _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(value, norm)), zero);
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero);
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail);

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                __m256i i32 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum, bias[index])), norm[index])), zero);
                _mm_storel_epi64((__m128i*)(dst + index * F), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO)));
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
            {
                uint8_t tmp[F];
                QuntizedTerm8i::Save<index>(tmp - index * F, buf, sum, bias, norm, zero);
                for (size_t i = 0; i < tail; ++i)
                    dst[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                __m256i d0 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum0, bias[0])), norm[0])), zero);
                __m256i d1 = _mm256_add_epi32(_mm256_cvtps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(sum1, bias[1])), norm[1])), zero);
                _mm_storeu_si128((__m128i*)dst, _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                _mm256_storeu_si256((__m256i*)buf + index, sum);
            }

            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum,
                const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
            {
                int32_t tmp[F];
                _mm256_storeu_si256((__m256i*)tmp, sum);
                for (size_t i = 0; i < tail; ++i)
                    buf[index * F + i] = tmp[i];
            }

            static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
                const __m256i* bias, const __m256* norm, const __m256i& zero)
            {
                _mm256_storeu_si256((__m256i*)buf + 0, sum0);
                _mm256_storeu_si256((__m256i*)buf + 1, sum1);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* bias, const __m256* norm, const __m256i& zero)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m256i sum,
            const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* bias, const __m256* norm, const __m256i& zero)
        {
            QuntizedTerm8i<term>::Save(dst, buf, sum0, sum1, bias, norm, zero);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m256i sum0, __m256i sum1,
            const __m256i* bias, const __m256* norm, const __m256i& zero, size_t tail)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE __m512 DequantizeLinear(__m512i value, __m512i bias, __m512 norm)
        {
            return _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(value, bias)), norm);
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i QuantizeLinear(__m512 value, __m512 norm, __m512i zero)
        {
            return _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(value, norm)), zero);
        }

        //--------------------------------------------------------------------------------------------------

        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
                __m512i i32 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(sum, bias[index])), norm[index])), zero);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO)));
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Save(uint8_t* dst, int32_t* buf, __m512i sum,
                const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
                _mm512_mask_storeu_epi32(buf + index * F, tail, sum);
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Save1(uint8_t* dst, int32_t* buf, __m512i sum,
            const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Save2(uint8_t* dst, int32_t* buf, __m512i sum0, __m512i sum1,
            const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Save<0>(dst, buf, sum0, bias, norm, zero);
            QuntizedTerm8i<term>::template Save<1>(dst, buf, sum1, bias, norm, zero, tail);
        }
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE))    
    namespace AmxBf16
    {
        template <Term8iType term> struct QuntizedTerm8i
        {
            template<int index> static SIMD_INLINE void Apply(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1);
        };

        template <> struct QuntizedTerm8i<Term8iLast8u>
        {
            template<int index> static SIMD_INLINE void Apply(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
                __m512i i32 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(buf + index * F), bias[index])), norm[index])), zero);
                _mm_mask_storeu_epi8(dst + index * F, tail, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(i32, K_ZERO), K_ZERO)));
                _mm_prefetch((const char*)(dst + index * A), _MM_HINT_NTA);
                _mm_prefetch((const char*)(buf + index * F), _MM_HINT_NTA);
            }
        };

        template <> struct QuntizedTerm8i<Term8iInterim>
        {
            template<int index> static SIMD_INLINE void Apply(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
            {
            }
        };

        template<Term8iType term>
        SIMD_INLINE void Apply1(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Apply<0>(dst, buf, bias, norm, zero, tail);
        }

        template<Term8iType term> SIMD_INLINE void Apply1x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            Apply1<term>(ptr + 0 * dP, buf + 0 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 1 * dP, buf + 1 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 2 * dP, buf + 2 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 3 * dP, buf + 3 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 4 * dP, buf + 4 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 5 * dP, buf + 5 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 6 * dP, buf + 6 * dB, bias, norm, zero, tail);
            Apply1<term>(ptr + 7 * dP, buf + 7 * dB, bias, norm, zero, tail);
        }

        template<Term8iType term>
        SIMD_INLINE void Apply2(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            QuntizedTerm8i<term>::template Apply<0>(dst, buf, bias, norm, zero);
            QuntizedTerm8i<term>::template Apply<1>(dst, buf, bias, norm, zero, tail);
        }

        template<Term8iType term> SIMD_INLINE void Apply2x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask16 tail = -1)
        {
            Apply2<term>(ptr + 0 * dP, buf + 0 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 1 * dP, buf + 1 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 2 * dP, buf + 2 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 3 * dP, buf + 3 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 4 * dP, buf + 4 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 5 * dP, buf + 5 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 6 * dP, buf + 6 * dB, bias, norm, zero, tail);
            Apply2<term>(ptr + 7 * dP, buf + 7 * dB, bias, norm, zero, tail);
        }

        SIMD_INLINE void Apply8u2(uint8_t* dst, int32_t* buf, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask32 tail = -1)
        {
            __m512i d0 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(buf + 0), bias[0])), norm[0])), zero);
            __m512i d1 = _mm512_add_epi32(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_loadu_si512(buf + F), bias[1])), norm[1])), zero);
            _mm256_mask_storeu_epi8(dst, tail, _mm512_castsi512_si256(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO)));
            _mm_prefetch((const char*)(dst), _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + 0), _MM_HINT_NTA);
            _mm_prefetch((const char*)(buf + F), _MM_HINT_NTA);
        }

        SIMD_INLINE void Apply8u2x8(uint8_t* ptr, int dP, int32_t* buf, int dB, const __m512i* bias, const __m512* norm, const __m512i& zero, __mmask32 tail = -1)
        {
            Apply8u2(ptr + 0 * dP, buf + 0 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 1 * dP, buf + 1 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 2 * dP, buf + 2 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 3 * dP, buf + 3 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 4 * dP, buf + 4 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 5 * dP, buf + 5 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 6 * dP, buf + 6 * dB, bias, norm, zero, tail);
            Apply8u2(ptr + 7 * dP, buf + 7 * dB, bias, norm, zero, tail);
        }
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif
}

#endif

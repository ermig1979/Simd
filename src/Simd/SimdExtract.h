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
#ifndef __SimdExtract_h__
#define __SimdExtract_h__

#include "Simd/SimdConst.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE float ExtractValue(__m128 a, int i)
        {
            float SIMD_ALIGNED(16) _a[4];
            _mm_store_ps(_a, a);
            return _a[i];
        }

        SIMD_INLINE float ExtractSum(__m128 a)
        {
            float SIMD_ALIGNED(16) _a[4];
            _mm_store_ps(_a, a);
            return _a[0] + _a[1] + _a[2] + _a[3];
        }        
        
        template <int index> SIMD_INLINE int ExtractInt8(__m128i a)
        {
            return _mm_extract_epi16(_mm_srli_si128(a, index & 0x1), index >> 1) & 0xFF;
        }

        template <int index> SIMD_INLINE int ExtractInt16(__m128i a)
        {
            return _mm_extract_epi16(a, index);
        }

        template <int index> SIMD_INLINE int ExtractInt32(__m128i a)
        {
            return _mm_cvtsi128_si32(_mm_srli_si128(a, 4 * index));
        }

        SIMD_INLINE int ExtractInt32Sum(__m128i a)
        {
            int SIMD_ALIGNED(16) _a[4];
            _mm_store_si128((__m128i*)_a, a);
            return _a[0] + _a[1] + _a[2] + _a[3];
        }

        template <int index> SIMD_INLINE int64_t ExtractInt64(__m128i a)
        {
#if defined(SIMD_X64_ENABLE) && (!defined(_MSC_VER) || (defined(_MSC_VER) && _MSC_VER >= 1600))
            return _mm_cvtsi128_si64(_mm_srli_si128(a, 8 * index));
#else
            return (int64_t)ExtractInt32<2 * index + 1>(a) * 0x100000000 + (uint32_t)ExtractInt32<2 * index>(a);
#endif
        }

        SIMD_INLINE int64_t ExtractInt64Sum(__m128i a)
        {
            int64_t SIMD_ALIGNED(16) _a[2];
            _mm_store_si128((__m128i*)_a, a);
            return _a[0] + _a[1];
        }
    }
#endif// SIMD_SSE2_ENABLE

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        SIMD_INLINE float ExtractSum(__m128 a)
        {
            return _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(a, _mm_setzero_ps()), _mm_setzero_ps()));
        }

        SIMD_INLINE __m128 Extract4Sums(const __m128 a[4])
        {
            return _mm_hadd_ps(_mm_hadd_ps(a[0], a[1]), _mm_hadd_ps(a[2], a[3]));
        }
    }
#endif//SIMD_SSE41_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        SIMD_INLINE float ExtractValue(__m256 a, int i)
        {
            float SIMD_ALIGNED(32) _a[8];
            _mm256_store_ps(_a, a);
            return _a[i];
        }

        SIMD_INLINE float ExtractSum(__m256 a)
        {
            float SIMD_ALIGNED(32) _a[8];
            _mm256_store_ps(_a, _mm256_hadd_ps(_mm256_hadd_ps(a, _mm256_setzero_ps()), _mm256_setzero_ps()));
            return _a[0] + _a[4];
        }

        SIMD_INLINE __m128 Extract4Sums(const __m256 a[4])
        {
            __m256 b = _mm256_hadd_ps(_mm256_hadd_ps(a[0], a[1]), _mm256_hadd_ps(a[2], a[3]));
            return _mm_add_ps(_mm256_castps256_ps128(b), _mm256_extractf128_ps(b, 1));
        }

        SIMD_INLINE __m128 Extract4Sums(const __m256 & a0, const __m256 & a1, const __m256 & a2, const __m256 & a3)
        {
            __m256 b = _mm256_hadd_ps(_mm256_hadd_ps(a0, a1), _mm256_hadd_ps(a2, a3));
            return _mm_add_ps(_mm256_castps256_ps128(b), _mm256_extractf128_ps(b, 1));
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        template <class T> SIMD_INLINE T Extract(__m256i a, size_t index)
        {
            const size_t size = A / sizeof(T);
            assert(index < size);
            T buffer[size];
            _mm256_storeu_si256((__m256i*)buffer, a);
            return buffer[index];
        }

        template <class T> SIMD_INLINE T ExtractSum(__m256i a)
        {
            const size_t size = A / sizeof(T);
            T buffer[size];
            _mm256_storeu_si256((__m256i*)buffer, a);
            T sum = 0;
            for (size_t i = 0; i < size; ++i)
                sum += buffer[i];
            return sum;
        }

        template <> SIMD_INLINE uint32_t ExtractSum<uint32_t>(__m256i a)
        {
            __m128i b = _mm_add_epi32(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(a, 1));
            return _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(b, _mm_setzero_si128()), _mm_setzero_si128()), 0);
        }

#if defined(SIMD_X64_ENABLE)
        template <> SIMD_INLINE uint64_t ExtractSum<uint64_t>(__m256i a)
        {
            __m128i b = _mm_add_epi64(_mm256_extractf128_si256(a, 0), _mm256_extractf128_si256(a, 1));
            return _mm_extract_epi64(b, 0) + _mm_extract_epi64(b, 1);
        }
#endif

        template <int index> SIMD_INLINE int64_t Extract64i(__m256i value)
        {
            assert(index >= 0 && index < 4);
#if defined(SIMD_X64_ENABLE)
#if (defined(_MSC_VER) && (_MSC_VER <= 1900))
            return _mm_extract_epi64(_mm256_extractf128_si256(value, index / 2), index % 2);
#else
            return _mm256_extract_epi64(value, index);
#endif
#else
            SIMD_ALIGNED(32) int64_t buffer[4];
            _mm256_store_si256((__m256i*)buffer, value);
            return buffer[index];
#endif
        }
    }
#endif// SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
        SIMD_INLINE float Extract(const __m512 & a, size_t index)
        {
            float buffer[F];
            _mm512_storeu_ps(buffer, a);
            return buffer[index];
        }

        SIMD_INLINE float ExtractSum(const __m512 & a)
        {
            __m128 lo = _mm_add_ps(_mm512_extractf32x4_ps(a, 0), _mm512_extractf32x4_ps(a, 1));
            __m128 hi = _mm_add_ps(_mm512_extractf32x4_ps(a, 2), _mm512_extractf32x4_ps(a, 3));
            return _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(_mm_add_ps(lo, hi), _mm_setzero_ps()), _mm_setzero_ps()));
        }

        SIMD_INLINE __m128 Extract4Sums(const __m512 a[4])
        {
            __m256 b0 = _mm512_castps512_ps256(_mm512_add_ps(a[0], Alignr<8>(a[0], a[0])));
            __m256 b1 = _mm512_castps512_ps256(_mm512_add_ps(a[1], Alignr<8>(a[1], a[1])));
            __m256 b2 = _mm512_castps512_ps256(_mm512_add_ps(a[2], Alignr<8>(a[2], a[2])));
            __m256 b3 = _mm512_castps512_ps256(_mm512_add_ps(a[3], Alignr<8>(a[3], a[3])));
            __m256 c = _mm256_hadd_ps(_mm256_hadd_ps(b0, b1), _mm256_hadd_ps(b2, b3));
            return _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
        }

        SIMD_INLINE __m128 Extract4Sums(const __m512 & a0, const __m512 & a1, const __m512 & a2, const __m512 & a3)
        {
            __m256 b0 = _mm512_castps512_ps256(_mm512_add_ps(a0, Alignr<8>(a0, a0)));
            __m256 b1 = _mm512_castps512_ps256(_mm512_add_ps(a1, Alignr<8>(a1, a1)));
            __m256 b2 = _mm512_castps512_ps256(_mm512_add_ps(a2, Alignr<8>(a2, a2)));
            __m256 b3 = _mm512_castps512_ps256(_mm512_add_ps(a3, Alignr<8>(a3, a3)));
            __m256 c = _mm256_hadd_ps(_mm256_hadd_ps(b0, b1), _mm256_hadd_ps(b2, b3));
            return _mm_add_ps(_mm256_castps256_ps128(c), _mm256_extractf128_ps(c, 1));
        }
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        template <class T> SIMD_INLINE T ExtractSum(__m512i a)
        {
            const size_t size = A / sizeof(T);
            T buffer[size];
            _mm512_storeu_si512(buffer, a);
            T sum = 0;
            for (size_t i = 0; i < size; ++i)
                sum += buffer[i];
            return sum;
        }

        template <> SIMD_INLINE uint32_t ExtractSum<uint32_t>(__m512i a)
        {
            __m256i b = _mm256_add_epi32(_mm512_extracti64x4_epi64(a, 0), _mm512_extracti64x4_epi64(a, 1));
            __m128i c = _mm_add_epi32(_mm256_extractf128_si256(b, 0), _mm256_extractf128_si256(b, 1));
            return _mm_extract_epi32(_mm_hadd_epi32(_mm_hadd_epi32(c, _mm_setzero_si128()), _mm_setzero_si128()), 0);
        }

#if defined(SIMD_X64_ENABLE)
        template <> SIMD_INLINE uint64_t ExtractSum<uint64_t>(__m512i a)
        {
            __m256i b = _mm256_add_epi64(_mm512_extracti64x4_epi64(a, 0), _mm512_extracti64x4_epi64(a, 1));
            __m128i c = _mm_add_epi64(_mm256_extractf128_si256(b, 0), _mm256_extractf128_si256(b, 1));
            return _mm_extract_epi64(c, 0) + _mm_extract_epi64(c, 1);
        }
#endif

        template <> SIMD_INLINE __m128i ExtractSum<__m128i>(__m512i a)
        {
            __m256i b = _mm256_add_epi64(_mm512_extracti64x4_epi64(a, 0), _mm512_extracti64x4_epi64(a, 1));
            return _mm_add_epi64(_mm256_extractf128_si256(b, 0), _mm256_extractf128_si256(b, 1));
        }
    }
#endif//SIMD_AVX512BW_ENABLE

#ifdef SIMD_VMX_ENABLE
    namespace Vmx
    {
        SIMD_INLINE uint32_t ExtractSum(v128_u32 a)
        {
            return vec_extract(a, 0) + vec_extract(a, 1) + vec_extract(a, 2) + vec_extract(a, 3);
        }

        SIMD_INLINE float ExtractSum(v128_f32 a)
        {
            return vec_extract(a, 0) + vec_extract(a, 1) + vec_extract(a, 2) + vec_extract(a, 3);
        }
    }
#endif// SIMD_VMX_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE uint32_t ExtractSum32u(const uint32x4_t & a)
        {
            return vgetq_lane_u32(a, 0) + vgetq_lane_u32(a, 1) + vgetq_lane_u32(a, 2) + vgetq_lane_u32(a, 3);
        }

        SIMD_INLINE int32_t ExtractSum32s(const int32x4_t& a)
        {
            return vgetq_lane_s32(a, 0) + vgetq_lane_s32(a, 1) + vgetq_lane_s32(a, 2) + vgetq_lane_s32(a, 3);
        }

        SIMD_INLINE uint64_t ExtractSum64u(const uint64x2_t & a)
        {
            return vgetq_lane_u64(a, 0) + vgetq_lane_u64(a, 1);
        }

        SIMD_INLINE int64_t ExtractSum64i(const int64x2_t & a)
        {
            return vgetq_lane_s64(a, 0) + vgetq_lane_s64(a, 1);
        }

        SIMD_INLINE float ExtractSum32f(const float32x4_t & a)
        {
            return vgetq_lane_f32(a, 0) + vgetq_lane_f32(a, 1) + vgetq_lane_f32(a, 2) + vgetq_lane_f32(a, 3);
        }

        SIMD_INLINE float32x4_t Extract4Sums(const float32x4_t a[4])
        {
            float32x4x2_t b0 = vzipq_f32(a[0], a[2]);
            float32x4x2_t b1 = vzipq_f32(a[1], a[3]);
            float32x4x2_t c0 = vzipq_f32(b0.val[0], b1.val[0]);
            float32x4x2_t c1 = vzipq_f32(b0.val[1], b1.val[1]);
            return vaddq_f32(vaddq_f32(c0.val[0], c0.val[1]), vaddq_f32(c1.val[0], c1.val[1]));
        }

        SIMD_INLINE float32x4_t Extract4Sums(const float32x4_t & a0, const float32x4_t & a1, const float32x4_t & a2, const float32x4_t & a3)
        {
            float32x4x2_t b0 = vzipq_f32(a0, a2);
            float32x4x2_t b1 = vzipq_f32(a1, a3);
            float32x4x2_t c0 = vzipq_f32(b0.val[0], b1.val[0]);
            float32x4x2_t c1 = vzipq_f32(b0.val[1], b1.val[1]);
            return vaddq_f32(vaddq_f32(c0.val[0], c0.val[1]), vaddq_f32(c1.val[0], c1.val[1]));
        }
    }
#endif// SIMD_NEON_ENABLE
}

#endif//__SimdExtract_h__

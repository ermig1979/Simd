/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        SIMD_INLINE __m256i Encode32f(__m256 src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i value = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(src, min), scale));
            sum = _mm256_add_epi32(value, sum);
            sqsum = _mm256_add_epi32(_mm256_madd_epi16(value, value), sqsum);
            return value;
        }

        SIMD_INLINE __m256i Encode32f(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            return Encode32f(_mm256_loadu_ps(src), scale, min, sum, sqsum);
        }

        static SIMD_INLINE __m128i Encode32f4x8(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0 * 8, scale, min, sum, sqsum);
            __m128i s0 = _mm_srli_epi32(_mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E4_MULLO), 12);
            return _mm_packus_epi16(_mm_packus_epi32(s0, Sse41::K_ZERO), Sse41::K_ZERO);
        }

        static SIMD_INLINE __m128i Encode32f4x32(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0 * 8, scale, min, sum, sqsum);
            __m256i i1 = Encode32f(src + 1 * 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_srli_epi32(_mm256_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            __m256i i2 = Encode32f(src + 2 * 8, scale, min, sum, sqsum);
            __m256i i3 = Encode32f(src + 3 * 8, scale, min, sum, sqsum);
            __m256i s1 = _mm256_srli_epi32(_mm256_mullo_epi16(PackU32ToI16(i2, i3), E4_MULLO), 12);
            return _mm_packus_epi16(_mm_packus_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1)), 
                _mm_packus_epi32(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1)));
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size32 = AlignLo(size, 32);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < size32; i += 32, src += 32, dst += 16)
                _mm_storeu_si128((__m128i*)dst, Encode32f4x32(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 4)
                *(uint32_t*)(dst) = _mm_extract_epi32(Encode32f4x8(src, _scale, _min, _sum, _sqsum), 0);
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f5x1(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E5_MULLO);
            return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E5_SHFL0), _mm_shuffle_epi8(s0, Sse41::E5_SHFL1)), _mm_shuffle_epi8(s0, Sse41::E5_SHFL2));
        }

        static SIMD_INLINE __m128i Encode32f5x2(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m256i i8 = Encode32f(src + 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E5_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(s0, E5_SHFL0), _mm256_shuffle_epi8(s0, E5_SHFL1)), _mm256_shuffle_epi8(s0, E5_SHFL2));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 10)
                _mm_storeu_si128((__m128i*)dst, Encode32f5x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 5)
                _mm_storel_epi64((__m128i*)dst, Encode32f5x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                __m128i d0 = Encode32f5x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint8_t*)(dst + 4) = _mm_extract_epi8(d0, 4);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f6x1(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E6_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E6_SHFL0), _mm_shuffle_epi8(s0, Sse41::E6_SHFL1));
        }

        static SIMD_INLINE __m128i Encode32f6x2(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m256i i8 = Encode32f(src + 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E6_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, E6_SHFL0), _mm256_shuffle_epi8(s0, E6_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 12)
                _mm_storeu_si128((__m128i*)dst, Encode32f6x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode32f6x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode32f6x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f7x1(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E7_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E7_SHFL0), _mm_shuffle_epi8(s0, Sse41::E7_SHFL1));
        }

        static SIMD_INLINE __m128i Encode32f7x2(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m256i i8 = Encode32f(src + 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E7_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, E7_SHFL0), _mm256_shuffle_epi8(s0, E7_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 14)
                _mm_storeu_si128((__m128i*)dst, Encode32f7x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode32f7x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode32f7x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i d0 = Encode32f(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m256i d1 = Encode32f(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m256i d2 = Encode32f(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m256i d3 = Encode32f(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm256_storeu_si256((__m256i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m256i d0 = Encode32f(src + i, _scale, _min, _sum, _sqsum);
                _mm_storel_epi64((__m128i*)(dst + i), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, _mm256_setzero_si256()), _mm256_setzero_si256())));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE __m128i Encode16f4x8(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src)), scale, min, sum, sqsum);
            __m128i s0 = _mm_srli_epi32(_mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E4_MULLO), 12);
            return _mm_packus_epi16(_mm_packus_epi32(s0, Sse41::K_ZERO), Sse41::K_ZERO);
        }

        static SIMD_INLINE __m128i Encode16f4x32(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 0)), scale, min, sum, sqsum);
            __m256i i1 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 1)), scale, min, sum, sqsum);
            __m256i s0 = _mm256_srli_epi32(_mm256_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            __m256i i2 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 2)), scale, min, sum, sqsum);
            __m256i i3 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 3)), scale, min, sum, sqsum);
            __m256i s1 = _mm256_srli_epi32(_mm256_mullo_epi16(PackU32ToI16(i2, i3), E4_MULLO), 12);
            return _mm_packus_epi16(_mm_packus_epi32(_mm256_castsi256_si128(s0), _mm256_extracti128_si256(s0, 1)),
                _mm_packus_epi32(_mm256_castsi256_si128(s1), _mm256_extracti128_si256(s1, 1)));
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size32 = AlignLo(size, 32);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < size32; i += 32, src += 32, dst += 16)
                _mm_storeu_si128((__m128i*)dst, Encode16f4x32(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 4)
                *(uint32_t*)(dst) = _mm_extract_epi32(Encode16f4x8(src, _scale, _min, _sum, _sqsum), 0);
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f5x1(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E5_MULLO);
            return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E5_SHFL0), _mm_shuffle_epi8(s0, Sse41::E5_SHFL1)), _mm_shuffle_epi8(s0, Sse41::E5_SHFL2));
        }

        static SIMD_INLINE __m128i Encode16f5x2(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 0)), scale, min, sum, sqsum);
            __m256i i8 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 1)), scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E5_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(s0, E5_SHFL0), _mm256_shuffle_epi8(s0, E5_SHFL1)), _mm256_shuffle_epi8(s0, E5_SHFL2));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode16f5(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 10)
                _mm_storeu_si128((__m128i*)dst, Encode16f5x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 5)
                _mm_storel_epi64((__m128i*)dst, Encode16f5x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                __m128i d0 = Encode16f5x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint8_t*)(dst + 4) = _mm_extract_epi8(d0, 4);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f6x1(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E6_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E6_SHFL0), _mm_shuffle_epi8(s0, Sse41::E6_SHFL1));
        }

        static SIMD_INLINE __m128i Encode16f6x2(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 0)), scale, min, sum, sqsum);
            __m256i i8 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 1)), scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E6_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, E6_SHFL0), _mm256_shuffle_epi8(s0, E6_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode16f6(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 12)
                _mm_storeu_si128((__m128i*)dst, Encode16f6x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode16f6x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode16f6x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f7x1(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), Sse41::E7_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, Sse41::E7_SHFL0), _mm_shuffle_epi8(s0, Sse41::E7_SHFL1));
        }

        static SIMD_INLINE __m128i Encode16f7x2(const uint16_t* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i i0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 0)), scale, min, sum, sqsum);
            __m256i i8 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)src + 1)), scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), E7_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, E7_SHFL0), _mm256_shuffle_epi8(s0, E7_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode16f7(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 14)
                _mm_storeu_si128((__m128i*)dst, Encode16f7x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode16f7x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode16f7x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i d0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i) + 0)), _scale, _min, _sum, _sqsum);
                __m256i d1 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i) + 1)), _scale, _min, _sum, _sqsum);
                __m256i d2 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i) + 2)), _scale, _min, _sum, _sqsum);
                __m256i d3 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i) + 3)), _scale, _min, _sum, _sqsum);
                _mm256_storeu_si256((__m256i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m256i d0 = Encode32f(_mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i))), _scale, _min, _sum, _sqsum);
                _mm_storel_epi64((__m128i*)(dst + i), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, _mm256_setzero_si256()), _mm256_setzero_si256())));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Encode32f4;
            case 5: return Encode32f5;
            case 6: return Encode32f6;
            case 7: return Encode32f7;
            case 8: return Encode32f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Encode16f4;
            case 5: return Encode16f5;
            case 6: return Encode16f6;
            case 7: return Encode16f7;
            case 8: return Encode16f8;
            default: assert(0); return NULL;
            }
        }
    }
#endif
}

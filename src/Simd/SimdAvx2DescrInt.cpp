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
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m256 _min = _mm256_set1_ps(FLT_MAX);
            __m256 _max = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_loadu_ps(src + i);
                _min = _mm256_min_ps(_src, _min);
                _max = _mm256_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m256 _min = _mm256_set1_ps(FLT_MAX);
            __m256 _max = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_cvtph_ps(_mm_loadu_si128((__m128i*)(src + i)));
                _min = _mm256_min_ps(_src, _min);
                _max = _mm256_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

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

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, C6_SHFL), C6_MULLO), 10);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift));
                _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift));
                src += 12;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift));
                src += 6;
                dst += 8;
            }
        }

        static void Decode32f7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, C7_SHFL), C7_MULLO), 9);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift));
                _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift));
                src += 14;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift));
                src += 7;
                dst += 8;
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
            {
                __m128i u8 = _mm_loadu_si128((__m128i*)(src + i));
                _mm256_storeu_ps(dst + i + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8)), _scale, _shift));
                _mm256_storeu_ps(dst + i + F, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u8, 8))), _scale, _shift));
            }
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + i))));
                _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_src, _scale, _shift));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i _ab = _mm256_setzero_si256();
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16, a += 12, b += 12)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)a)), C6_SHFL), C6_MULLO), 10);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)b)), C6_SHFL), C6_MULLO), 10);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i _ab = _mm256_setzero_si256();
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16, a += 14, b += 14)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)a)), C7_SHFL), C7_MULLO), 9);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)b)), C7_SHFL), C7_MULLO), 9);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            size_t i = 0, size16 = AlignLo(size, 16);
            __m256i _ab = _mm256_setzero_si256();
            for (; i < size16; i += 16)
            {
                __m256i _a = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(a + i)));
                __m256i _b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(b + i)));
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8)
            {
                __m256i _a = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(a + i)));
                __m256i _b = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(b + i)));
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, (float)size, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistances2x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances2x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[1] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 6)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[1] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

        template<> void MicroCosineDistances2x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[1] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 7)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[1] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

        template<> void MicroCosineDistances2x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 16)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)));
                a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[1] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 8)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));
                a1 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[1] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

        template<int bits> void MicroCosineDistances1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 6)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 7)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 16)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 8)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistances(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances2x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                {
                    CosineDistance<bits>(A[i + 0], B[j], size, distances + j + 0 * stride);
                    CosineDistance<bits>(A[i + 1], B[j], size, distances + j + 1 * stride);
                }
                distances += 2 * stride;
            }
            for (; i < M; i++)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances1x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Sse41::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;
            switch (depth)
            {
            case 6:
            {
                _encode32f = Encode32f6;
                _encode16f = Encode16f6;
                _decode32f = Decode32f6;
                _cosineDistance = Avx2::CosineDistance<6>;
                _macroCosineDistances = Avx2::MacroCosineDistances<6>;
                break;
            }
            case 7:
            {
                _encode32f = Encode32f7;
                _encode16f = Encode16f7;
                _decode32f = Decode32f7;
                _cosineDistance = Avx2::CosineDistance<7>;
                _macroCosineDistances = Avx2::MacroCosineDistances<7>;
                break;
            }
            case 8:
            {
                _encode32f = Encode32f8;
                _encode16f = Encode16f8;
                _decode32f = Decode32f8;
                _cosineDistance = Avx2::CosineDistance<8>;
                _macroCosineDistances = Avx2::MacroCosineDistances<8>;
                break;
            }
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Avx2::DescrInt(size, depth);
        }
    }
#endif
}

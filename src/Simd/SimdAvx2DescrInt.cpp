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
        static void MinMax(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m256 _min256 = _mm256_set1_ps(FLT_MAX);
            __m256 _max256 = _mm256_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_loadu_ps(src + i);
                _min256 = _mm256_min_ps(_src, _min256);
                _max256 = _mm256_max_ps(_src, _max256);
            }
            __m128 _min = _mm_min_ps(_mm256_castps256_ps128(_min256), _mm256_extractf128_ps(_min256, 1));
            __m128 _max = _mm_max_ps(_mm256_castps256_ps128(_max256), _mm256_extractf128_ps(_max256, 1));
            _min = _mm_min_ps(_min, Sse41::Shuffle32f<0x0E>(_min));
            _max = _mm_max_ps(_max, Sse41::Shuffle32f<0x0E>(_max));
            _min = _mm_min_ss(_min, Sse41::Shuffle32f<0x01>(_min));
            _max = _mm_max_ss(_max, Sse41::Shuffle32f<0x01>(_max));
            _mm_store_ss(&min, _min);
            _mm_store_ss(&max, _max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m256i Encode(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            __m256i value = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(src), min), scale));
            sum = _mm256_add_epi32(value, sum);
            sqsum = _mm256_add_epi32(_mm256_madd_epi16(value, value), sqsum);
            return value;
        }

        static SIMD_INLINE __m128i Encode6x1(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            static const __m128i SHIFT = SIMD_MM_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
            static const __m128i SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), SHIFT);
            return _mm_or_si128(_mm_shuffle_epi8(s0, SHFL0), _mm_shuffle_epi8(s0, SHFL1));
        }

        static SIMD_INLINE __m128i Encode6x2(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);
            __m256i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m256i i8 = Encode(src + 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 12)
                _mm_storeu_si128((__m128i*)dst, Encode6x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode6x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode6x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode7x1(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            static const __m128i SHIFT = SIMD_MM_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
            static const __m128i SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m256i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm256_castsi256_si128(PackU32ToI16(i0, _mm256_setzero_si256())), SHIFT);
            return _mm_or_si128(_mm_shuffle_epi8(s0, SHFL0), _mm_shuffle_epi8(s0, SHFL1));
        }

        static SIMD_INLINE __m128i Encode7x2(const float* src, __m256 scale, __m256 min, __m256i& sum, __m256i& sqsum)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);
            __m256i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m256i i8 = Encode(src + 8, scale, min, sum, sqsum);
            __m256i s0 = _mm256_mullo_epi16(PackU32ToI16(i0, i8), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static void Encode7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8, main16 = AlignLo(main, 16);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < main16; i += 16, src += 16, dst += 14)
                _mm_storeu_si128((__m128i*)dst, Encode7x2(src, _scale, _min, _sum, _sqsum));
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode7x1(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode7x1(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _min = _mm256_set1_ps(min);
            __m256i _sum = _mm256_setzero_si256();
            __m256i _sqsum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i d0 = Encode(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m256i d1 = Encode(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m256i d2 = Encode(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m256i d3 = Encode(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm256_storeu_si256((__m256i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m256i d0 = Encode(src + i, _scale, _min, _sum, _sqsum);
                _mm_storel_epi64((__m128i*)(dst + i), _mm256_castsi256_si128(PackI16ToU8(PackI32ToI16(d0, _mm256_setzero_si256()), _mm256_setzero_si256())));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        const __m256i C6_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m256i C6_MULLO = SIMD_MM256_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m256i C7_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m256i C7_MULLO = SIMD_MM256_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        static void Decode6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
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

        static void Decode7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
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

        static void Decode8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
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
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_broadcastsi128_si256(_a), _mm256_broadcastsi128_si256(_b)), _ab);
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
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_broadcastsi128_si256(_a), _mm256_broadcastsi128_si256(_b)), _ab);
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
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[1] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
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
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[1] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
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
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
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
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadl_epi64((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
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
            _minMax = MinMax;
            switch (depth)
            {
            case 6:
            {
                _encode = Encode6;
                _decode = Decode6;
                _cosineDistance = Avx2::CosineDistance<6>;
                _macroCosineDistances = Avx2::MacroCosineDistances<6>;
                break;
            }
            case 7:
            {
                _encode = Encode7;
                _decode = Decode7;
                _cosineDistance = Avx2::CosineDistance<7>;
                _macroCosineDistances = Avx2::MacroCosineDistances<7>;
                break;
            }
            case 8:
            {
                _encode = Encode8;
                _decode = Decode8;
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

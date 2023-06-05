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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m512 _min = _mm512_set1_ps(FLT_MAX);
            __m512 _max = _mm512_set1_ps(-FLT_MAX);
            size_t i = 0, sizeF = AlignLo(size, F);
            for (; i < sizeF; i += F)
            {
                __m512 _src = _mm512_loadu_ps(src + i);
                _min = _mm512_min_ps(_src, _min);
                _max = _mm512_max_ps(_src, _max);
            }
            for (; i < size; i += 8)
            {
                __m512 _src = _mm512_maskz_loadu_ps(0xFF, src + i);
                _min = _mm512_mask_min_ps(_min, 0xFF, _src, _min);
                _max = _mm512_mask_max_ps(_max, 0xFF, _src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i Encode32f(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask, src), min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
        }

        static SIMD_INLINE __m128i Encode32f6x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode32f6x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            static const __m512i SHIFT = SIMD_MM512_SETR_EPI16(
                256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4,
                256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
            static const __m512i SHFL0 = SIMD_MM512_SETR_EPI8(
                -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1,
                0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD,
                -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
            static const __m512i SHFL1 = SIMD_MM512_SETR_EPI8(
                -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1,
                0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE,
                -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);
            static const __m512i PERM = SIMD_MM512_SETR_EPI64(0, 2, 1, 3, 4, 6, 5, 7);
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(PERM, _mm512_packus_epi32(i0, i1)), SHIFT);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, SHFL0), _mm512_shuffle_epi8(s0, SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 24)
                _mm256_mask_storeu_epi8(dst - 4, 0x0FFFFFF0, Encode32f6x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 12)
                _mm_mask_storeu_epi8(dst, 0x0FFF, Encode32f6x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x003F, Encode32f6x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f7x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode32f7x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            static const __m512i SHIFT = SIMD_MM512_SETR_EPI16(
                256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2,
                256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
            static const __m512i SHFL0 = SIMD_MM512_SETR_EPI8(
                -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1,
                0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 
                -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
            static const __m512i SHFL1 = SIMD_MM512_SETR_EPI8(
                -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1,
                0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE,
                -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);
            static const __m512i PERM = SIMD_MM512_SETR_EPI64(0, 2, 1, 3, 4, 6, 5, 7);
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(PERM, _mm512_packus_epi32(i0, i1)), SHIFT);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, SHFL0), _mm512_shuffle_epi8(s0, SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 28)
                _mm256_mask_storeu_epi8(dst - 2, 0x3FFFFFFC, Encode32f7x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 14)
                _mm_mask_storeu_epi8(dst, 0x3FFF, Encode32f7x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x007F, Encode32f7x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeF = AlignLo(size, F), sizeA = AlignLo(size, A), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i d0 = Encode32f(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m512i d1 = Encode32f(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m512i d2 = Encode32f(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m512i d3 = Encode32f(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm512_storeu_si512((__m512i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < sizeF; i += F)
            {
                __m512i d0 = Encode32f(src + i, _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }            
            if (i < size)
            {
                __m512i d0 = Encode32f(src + i, _scale, _min, _sum, _sqsum, 0xFF);
                _mm_mask_storeu_epi8(dst + i, 0xFF, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------


        const __m512i C6_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x3, 0x4, 0x0, 0x0, 0x4, 0x5, 0x0, 0x0);
        const __m512i C6_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C6_MULLO = SIMD_MM512_SETR_EPI16(
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256,
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m512i C7_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x3, 0x0, 0x3, 0x4, 0x5, 0x0, 0x5, 0x6, 0x0, 0x0);
        const __m512i C7_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8,
            0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C7_MULLO = SIMD_MM512_SETR_EPI16(
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256,
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, Avx2::C6_SHFL), Avx2::C6_MULLO), 10);
                _mm512_storeu_ps(dst + 0, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift));
                src += 12;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)));
                src += 6;
                dst += 8;
            }
        }

        static void Decode32f7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, Avx2::C7_SHFL), Avx2::C7_MULLO), 9);
                _mm512_storeu_ps(dst + 0, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift));
                src += 14;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)));
                src += 7;
                dst += 8;
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size64 = AlignLo(size, 64);
            for (; i < size64; i += 64)
            {
                __m512i u8 = _mm512_loadu_si512((__m512i*)(src + i));
                _mm512_storeu_ps(dst + i + 0 * F, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 0))), _scale, _shift));
                _mm512_storeu_ps(dst + i + 1 * F, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 1))), _scale, _shift));
                _mm512_storeu_ps(dst + i + 2 * F, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 2))), _scale, _shift));
                _mm512_storeu_ps(dst + i + 3 * F, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 3))), _scale, _shift));
            }
            for (; i < size16; i += 16)
            {
                __m128i u8 = _mm_loadu_si128((__m128i*)(src + i));
                _mm512_storeu_ps(dst + i, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8)), _scale, _shift));
            }
            if (i < size)
            {
                __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + i))));
                _mm256_storeu_ps(dst + i, _mm256_fmadd_ps(_src, _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        SIMD_INLINE __m512i Load6(const uint8_t* ptr, __mmask32 mask = 0x00FFFFFF)
        {
            return _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(C6_PERM, _mm512_castsi256_si512(_mm256_maskz_loadu_epi8(mask, ptr))), C6_SHFL), C6_MULLO), 10);
        }

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i _ab = _mm512_setzero_si512();
            size_t i = 0, size32 = AlignLo(size, 32);
            for (; i < size32; i += 32, a += 24, b += 24)
            {
                __m512i _a = Load6(a);
                __m512i _b = Load6(b);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                __m512i _a = Load6(a, mask);
                __m512i _b = Load6(b, mask);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        SIMD_INLINE __m512i Load7(const uint8_t* ptr, __mmask32 mask = 0x0FFFFFFF)
        {
            return _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(C7_PERM, _mm512_castsi256_si512(_mm256_maskz_loadu_epi8(mask, ptr))), C7_SHFL), C7_MULLO), 9);
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i _ab = _mm512_setzero_si512();
            size_t i = 0, size32 = AlignLo(size, 32);
            for (; i < size32; i += 32, a += 28, b += 28)
            {
                __m512i _a = Load7(a);
                __m512i _b = Load7(b);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                __m512i _a = Load7(a, mask);
                __m512i _b = Load7(b, mask);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            size_t i = 0, size32 = AlignLo(size, 32);
            __m512i _ab = _mm512_setzero_si512();
            for (; i < size32; i += 32)
            {
                __m512i _a = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(a + i)));
                __m512i _b = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(b + i)));
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            if ( i < size)
            {
                __mmask32 mask = TailMask32(size - i);
                __m512i _a = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, a + i));
                __m512i _b = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, b + i));
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, (float)size, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistances4x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances4x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, a1, a2, a3, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            __m512i ab10 = _mm512_setzero_si512();
            __m512i ab11 = _mm512_setzero_si512();
            __m512i ab12 = _mm512_setzero_si512();
            __m512i ab13 = _mm512_setzero_si512();
            __m512i ab20 = _mm512_setzero_si512();
            __m512i ab21 = _mm512_setzero_si512();
            __m512i ab22 = _mm512_setzero_si512();
            __m512i ab23 = _mm512_setzero_si512();
            __m512i ab30 = _mm512_setzero_si512();
            __m512i ab31 = _mm512_setzero_si512();
            __m512i ab32 = _mm512_setzero_si512();
            __m512i ab33 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 24)
            {
                a0 = Load6(A[0] + o);
                a1 = Load6(A[1] + o);
                a2 = Load6(A[2] + o);
                a3 = Load6(A[3] + o);

                b0 = Load6(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load6(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load6(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load6(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                a0 = Load6(A[0] + o, mask);
                a1 = Load6(A[1] + o, mask);
                a2 = Load6(A[2] + o, mask);
                a3 = Load6(A[3] + o, mask);

                b0 = Load6(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load6(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load6(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load6(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
            Sse41::DecodeCosineDistances(A[2], B, ab2, _size, distances + 2 * stride);
            Sse41::DecodeCosineDistances(A[3], B, ab3, _size, distances + 3 * stride);
        }

        template<> void MicroCosineDistances4x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, a1, a2, a3, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            __m512i ab10 = _mm512_setzero_si512();
            __m512i ab11 = _mm512_setzero_si512();
            __m512i ab12 = _mm512_setzero_si512();
            __m512i ab13 = _mm512_setzero_si512();
            __m512i ab20 = _mm512_setzero_si512();
            __m512i ab21 = _mm512_setzero_si512();
            __m512i ab22 = _mm512_setzero_si512();
            __m512i ab23 = _mm512_setzero_si512();
            __m512i ab30 = _mm512_setzero_si512();
            __m512i ab31 = _mm512_setzero_si512();
            __m512i ab32 = _mm512_setzero_si512();
            __m512i ab33 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 28)
            {
                a0 = Load7(A[0] + o);
                a1 = Load7(A[1] + o);
                a2 = Load7(A[2] + o);
                a3 = Load7(A[3] + o);

                b0 = Load7(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load7(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load7(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load7(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                a0 = Load7(A[0] + o, mask);
                a1 = Load7(A[1] + o, mask);
                a2 = Load7(A[2] + o, mask);
                a3 = Load7(A[3] + o, mask);

                b0 = Load7(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load7(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load7(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load7(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
            Sse41::DecodeCosineDistances(A[2], B, ab2, _size, distances + 2 * stride);
            Sse41::DecodeCosineDistances(A[3], B, ab3, _size, distances + 3 * stride);
        }

        template<> void MicroCosineDistances4x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, a1, a2, a3, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            __m512i ab10 = _mm512_setzero_si512();
            __m512i ab11 = _mm512_setzero_si512();
            __m512i ab12 = _mm512_setzero_si512();
            __m512i ab13 = _mm512_setzero_si512();
            __m512i ab20 = _mm512_setzero_si512();
            __m512i ab21 = _mm512_setzero_si512();
            __m512i ab22 = _mm512_setzero_si512();
            __m512i ab23 = _mm512_setzero_si512();
            __m512i ab30 = _mm512_setzero_si512();
            __m512i ab31 = _mm512_setzero_si512();
            __m512i ab32 = _mm512_setzero_si512();
            __m512i ab33 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 32)
            {
                a0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(A[0] + o)));
                a1 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(A[1] + o)));
                a2 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(A[2] + o)));
                a3 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(A[3] + o)));

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[0] + o)));
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)));
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)));
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)));
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32(size - i);
                a0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[0] + o));
                a1 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[1] + o));
                a2 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[2] + o));
                a3 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[3] + o));

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[0] + o));
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[1] + o));
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[2] + o));
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[3] + o));
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            Sse41::DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
            Sse41::DecodeCosineDistances(A[2], B, ab2, _size, distances + 2 * stride);
            Sse41::DecodeCosineDistances(A[3], B, ab3, _size, distances + 3 * stride);
        }

        template<int bits> void MicroCosineDistances1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 24)
            {
                a0 = Load6(A[0] + o);

                b0 = Load6(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load6(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load6(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load6(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                a0 = Load6(A[0] + o, mask);

                b0 = Load6(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load6(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load6(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load6(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 28)
            {
                a0 = Load7(A[0] + o);

                b0 = Load7(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load7(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load7(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load7(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                a0 = Load7(A[0] + o, mask);

                b0 = Load7(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load7(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load7(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load7(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 32)
            {
                a0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(A[0] + o)));

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[0] + o)));
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)));
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)));
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)));
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32(size - i);
                a0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[0] + o));

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[0] + o));
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[1] + o));
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[2] + o));
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[3] + o));
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            Sse41::DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistances(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistances4x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                {
                    CosineDistance<bits>(A[i + 0], B[j], size, distances + j + 0 * stride);
                    CosineDistance<bits>(A[i + 1], B[j], size, distances + j + 1 * stride);
                    CosineDistance<bits>(A[i + 2], B[j], size, distances + j + 2 * stride);
                    CosineDistance<bits>(A[i + 3], B[j], size, distances + j + 3 * stride);
                }
                distances += 4 * stride;
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
            : Avx2::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            switch (depth)
            {
            case 6:
            {
                _encode32f = Encode32f6;
                _decode32f = Decode32f6;
                _cosineDistance = Avx512bw::CosineDistance<6>;
                _macroCosineDistances = Avx512bw::MacroCosineDistances<6>;
                break;
            }
            case 7:
            {
                _encode32f = Encode32f7;
                _decode32f = Decode32f7;
                _cosineDistance = Avx512bw::CosineDistance<7>;
                _macroCosineDistances = Avx512bw::MacroCosineDistances<7>;
                break;
            }
            case 8:
            {
                _encode32f = Encode32f8;
                _decode32f = Decode32f8;
                _cosineDistance = Avx512bw::CosineDistance<8>;
                _macroCosineDistances = Avx512bw::MacroCosineDistances<8>;
                _microM = 4;
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
            return new Avx512bw::DescrInt(size, depth);
        }
    }
#endif
}

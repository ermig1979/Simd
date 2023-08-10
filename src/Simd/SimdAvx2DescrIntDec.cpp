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
        static void Decode32f4(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s4, C4_SHFL), C4_MULLO), 12);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift));
                _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<4>(src), Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift));
                src += 4;
                dst += 8;
            }
        }

        static void Decode32f5(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s5 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s5, C5_SHFL), C5_MULLO), 11);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift));
                _mm256_storeu_ps(dst + 8, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift));
                src += 10;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<5>(src), Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift));
                src += 5;
                dst += 8;
            }
        }

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
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
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<6>(src), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
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
            size_t i = 0, size16 = AlignLo(size - 1, 16);
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
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<7>(src), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
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

        static void Decode16f4(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s4, C4_SHFL), C4_MULLO), 12);
                _mm_storeu_si128((__m128i*)dst + 0, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift), 0));
                _mm_storeu_si128((__m128i*)dst + 1, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift), 0));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<4>(src), Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 4;
                dst += 8;
            }
        }

        static void Decode16f5(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s5 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s5, C5_SHFL), C5_MULLO), 11);
                _mm_storeu_si128((__m128i*)dst + 0, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift), 0));
                _mm_storeu_si128((__m128i*)dst + 1, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift), 0));
                src += 10;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<5>(src), Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 5;
                dst += 8;
            }
        }

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, C6_SHFL), C6_MULLO), 10);
                _mm_storeu_si128((__m128i*)dst + 0, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift), 0));
                _mm_storeu_si128((__m128i*)dst + 1, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift), 0));
                src += 12;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<6>(src), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 6;
                dst += 8;
            }
        }

        static void Decode16f7(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size - 1, 16);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, C7_SHFL), C7_MULLO), 9);
                _mm_storeu_si128((__m128i*)dst + 0, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 0))), _scale, _shift), 0));
                _mm_storeu_si128((__m128i*)dst + 1, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(s16, 1))), _scale, _shift), 0));
                src += 14;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<7>(src), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 7;
                dst += 8;
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m256 _scale = _mm256_set1_ps(scale);
            __m256 _shift = _mm256_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16);
            for (; i < size16; i += 16)
            {
                __m128i u8 = _mm_loadu_si128((__m128i*)(src + i));
                _mm_storeu_si128((__m128i*)(dst + i) + 0, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(u8)), _scale, _shift), 0));
                _mm_storeu_si128((__m128i*)(dst + i) + 1, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(u8, 8))), _scale, _shift), 0));
            }
            for (; i < size; i += 8)
            {
                __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + i))));
                _mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(_mm256_fmadd_ps(_src, _scale, _shift), 0));
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode32f4;
            case 5: return Decode32f5;
            case 6: return Decode32f6;
            case 7: return Decode32f7;
            case 8: return Decode32f8;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth)
        {
            switch (depth)
            {
            case 4: return Decode16f4;
            case 5: return Decode16f5;
            case 6: return Decode16f6;
            case 7: return Decode16f7;
            case 8: return Decode16f8;
            default: assert(0); return NULL;
            }
        }
    }
#endif
}

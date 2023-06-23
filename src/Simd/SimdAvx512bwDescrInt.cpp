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

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m512 _min = _mm512_set1_ps(FLT_MAX);
            __m512 _max = _mm512_set1_ps(-FLT_MAX);
            size_t i = 0, sizeF = AlignLo(size, F);
            for (; i < sizeF; i += F)
            {
                __m512 _src = _mm512_cvtph_ps(_mm256_loadu_si256((__m256i*)(src + i)));
                _min = _mm512_min_ps(_src, _min);
                _max = _mm512_max_ps(_src, _max);
            }
            for (; i < size; i += 8)
            {
                __m512 _src = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(0xFF, src + i));
                _min = _mm512_mask_min_ps(_min, 0xFF, _src, _min);
                _max = _mm512_mask_max_ps(_max, 0xFF, _src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m512i Encode32f(__m512 src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(src, min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
        }

        SIMD_INLINE __m512i Encode32f(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            return Encode32f(_mm512_maskz_loadu_ps(mask, src), scale, min, sum, sqsum);
        }

        static SIMD_INLINE __m128i Encode32f4x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 m0, __mmask16 m1)
        {
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum, m0);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum, m1);
            __m512i s0 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            return _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(s0), Avx2::K_ZERO));
        }

        static SIMD_INLINE __m256i Encode32f4x8(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i i2 = Encode32f(src + 2 * F, scale, min, sum, sqsum);
            __m512i i3 = Encode32f(src + 3 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            __m512i s1 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i2, i3), E4_MULLO), 12);
            return Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(s0), _mm512_cvtepi32_epi16(s1));
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size32 = AlignLo(size, 32), size64 = AlignLo(size, 64);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size64; i += 64, src += 64, dst += 32)
                _mm256_storeu_si256((__m256i*)dst, Encode32f4x8(src, _scale, _min, _sum, _sqsum));
            for (; i < size32; i += 32, src += 32, dst += 16)
                _mm_mask_storeu_epi8(dst, -1, Encode32f4x4(src, _scale, _min, _sum, _sqsum, -1, -1));
            if (i < size)
            {
                __mmask16 ms0 = TailMask16(size - size32 - 0 * F);
                __mmask16 ms1 = TailMask16(size - size32 - 1 * F);
                __mmask16 md= TailMask16((size - size32) / 2);
                _mm_mask_storeu_epi8(dst, md, Encode32f4x4(src, _scale, _min, _sum, _sqsum, ms0, ms1));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f5x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E5_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E5_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode32f5x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E5_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_or_si512(_mm512_shuffle_epi8(s0, E5_SHFL0), _mm512_shuffle_epi8(s0, E5_SHFL1)), _mm512_shuffle_epi8(s0, E5_SHFL2));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 20)
                _mm256_mask_storeu_epi8(dst - 6, 0x03FFFFC0, Encode32f5x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 10)
                _mm_mask_storeu_epi8(dst, 0x03FF, Encode32f5x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x001F, Encode32f5x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f6x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E6_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E6_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E6_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode32f6x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E6_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, E6_SHFL0), _mm512_shuffle_epi8(s0, E6_SHFL1));
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
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E7_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E7_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E7_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode32f7x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode32f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode32f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E7_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, E7_SHFL0), _mm512_shuffle_epi8(s0, E7_SHFL1));
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

        SIMD_INLINE __m512i Encode16f(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            return Encode32f(_mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, src)), scale, min, sum, sqsum);
        }

        static SIMD_INLINE __m128i Encode16f4x4(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 m0, __mmask16 m1)
        {
            __m512i i0 = Encode16f(src + 0 * F, scale, min, sum, sqsum, m0);
            __m512i i1 = Encode16f(src + 1 * F, scale, min, sum, sqsum, m1);
            __m512i s0 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            return _mm256_castsi256_si128(Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(s0), Avx2::K_ZERO));
        }

        static SIMD_INLINE __m256i Encode16f4x8(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode16f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode16f(src + 1 * F, scale, min, sum, sqsum);
            __m512i i2 = Encode16f(src + 2 * F, scale, min, sum, sqsum);
            __m512i i3 = Encode16f(src + 3 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i0, i1), E4_MULLO), 12);
            __m512i s1 = _mm512_srli_epi32(_mm512_mullo_epi16(PackU32ToI16(i2, i3), E4_MULLO), 12);
            return Avx2::PackI16ToU8(_mm512_cvtepi32_epi16(s0), _mm512_cvtepi32_epi16(s1));
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size32 = AlignLo(size, 32), size64 = AlignLo(size, 64);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size64; i += 64, src += 64, dst += 32)
                _mm256_storeu_si256((__m256i*)dst, Encode16f4x8(src, _scale, _min, _sum, _sqsum));
            for (; i < size32; i += 32, src += 32, dst += 16)
                _mm_mask_storeu_epi8(dst, -1, Encode16f4x4(src, _scale, _min, _sum, _sqsum, -1, -1));
            if (i < size)
            {
                __mmask16 ms0 = TailMask16(size - size32 - 0 * F);
                __mmask16 ms1 = TailMask16(size - size32 - 1 * F);
                __mmask16 md = TailMask16((size - size32) / 2);
                _mm_mask_storeu_epi8(dst, md, Encode16f4x4(src, _scale, _min, _sum, _sqsum, ms0, ms1));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f5x2(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode16f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E5_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E5_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode16f5x4(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode16f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode16f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E5_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_or_si512(_mm512_shuffle_epi8(s0, E5_SHFL0), _mm512_shuffle_epi8(s0, E5_SHFL1)), _mm512_shuffle_epi8(s0, E5_SHFL2));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode16f5(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 20)
                _mm256_mask_storeu_epi8(dst - 6, 0x03FFFFC0, Encode16f5x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 10)
                _mm_mask_storeu_epi8(dst, 0x03FF, Encode16f5x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x001F, Encode16f5x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f6x2(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode16f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E6_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E6_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E6_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode16f6x4(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode16f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode16f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E6_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, E6_SHFL0), _mm512_shuffle_epi8(s0, E6_SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode16f6(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 24)
                _mm256_mask_storeu_epi8(dst - 4, 0x0FFFFFF0, Encode16f6x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 12)
                _mm_mask_storeu_epi8(dst, 0x0FFF, Encode16f6x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x003F, Encode16f6x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f7x2(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode16f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E7_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E7_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E7_SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode16f7x4(const uint16_t* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i i0 = Encode16f(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode16f(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(EX_PERM, _mm512_packus_epi32(i0, i1)), E7_MULLO);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, E7_SHFL0), _mm512_shuffle_epi8(s0, E7_SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode16f7(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 28)
                _mm256_mask_storeu_epi8(dst - 2, 0x3FFFFFFC, Encode16f7x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 14)
                _mm_mask_storeu_epi8(dst, 0x3FFF, Encode16f7x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x007F, Encode16f7x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeF = AlignLo(size, F), sizeA = AlignLo(size, A), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i d0 = Encode16f(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m512i d1 = Encode16f(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m512i d2 = Encode16f(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m512i d3 = Encode16f(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm512_storeu_si512((__m512i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < sizeF; i += F)
            {
                __m512i d0 = Encode16f(src + i, _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }
            if (i < size)
            {
                __m512i d0 = Encode16f(src + i, _scale, _min, _sum, _sqsum, 0xFF);
                _mm_mask_storeu_epi8(dst + i, 0xFF, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode32f4(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s4, Avx2::C4_SHFL), Avx2::C4_MULLO), 12);
                _mm512_storeu_ps(dst + 0, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s4 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)));
                src += 4;
                dst += 8;
            }
        }

        static void Decode32f5(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, Avx2::C5_SHFL), Avx2::C5_MULLO), 11);
                _mm512_storeu_ps(dst + 0, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift));
                src += 10;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s5 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s5, Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                _mm256_storeu_ps(dst + 0, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)));
                src += 5;
                dst += 8;
            }
        }

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

        static void Decode16f4(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s4 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s4, Avx2::C4_SHFL), Avx2::C4_MULLO), 12);
                _mm256_storeu_si256((__m256i*)dst, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s4 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)), 0));
                src += 4;
                dst += 8;
            }
        }

        static void Decode16f5(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s5 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s5, Avx2::C5_SHFL), Avx2::C5_MULLO), 11);
                _mm256_storeu_si256((__m256i*)dst, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 10;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s5 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s5, Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)), 0));
                src += 5;
                dst += 8;
            }
        }

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, Avx2::C6_SHFL), Avx2::C6_MULLO), 10);
                _mm256_storeu_si256((__m256i*)dst, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 12;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)), 0));
                src += 6;
                dst += 8;
            }
        }

        static void Decode16f7(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size32 = AlignLo(size, 32);
            for (; i < size16; i += 16)
            {
                __m256i s6 = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)src));
                __m256i s16 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(s6, Avx2::C7_SHFL), Avx2::C7_MULLO), 9);
                _mm256_storeu_si256((__m256i*)dst, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(s16)), _scale, _shift), 0));
                src += 14;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _mm_storeu_si128((__m128i*)dst, _mm256_cvtps_ph(_mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(s16)), _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)), 0));
                src += 7;
                dst += 8;
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _shift = _mm512_set1_ps(shift);
            size_t i = 0, size16 = AlignLo(size, 16), size64 = AlignLo(size, 64);
            for (; i < size64; i += 64)
            {
                __m512i u8 = _mm512_loadu_si512((__m512i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i) + 0, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 0))), _scale, _shift), 0));
                _mm256_storeu_si256((__m256i*)(dst + i) + 1, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 1))), _scale, _shift), 0));
                _mm256_storeu_si256((__m256i*)(dst + i) + 2, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 2))), _scale, _shift), 0));
                _mm256_storeu_si256((__m256i*)(dst + i) + 3, _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u8, 3))), _scale, _shift), 0));
            }
            for (; i < size16; i += 16)
            {
                __m128i u8 = _mm_loadu_si128((__m128i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i), _mm512_cvtps_ph(_mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(u8)), _scale, _shift), 0));
            }
            if (i < size)
            {
                __m256 _src = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(src + i))));
                _mm_storeu_si128((__m128i*)(dst + i), _mm256_cvtps_ph(_mm256_fmadd_ps(_src, _mm512_castps512_ps256(_scale), _mm512_castps512_ps256(_shift)), 0));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i ab32 = _mm512_setzero_si512();
            size_t i = 0, size128 = AlignLo(size, 128);
            for (; i < size128; i += 128, a += 64, b += 64)
            {
                __m512i _a = _mm512_loadu_si512((__m512i*)a);
                __m512i _b = _mm512_loadu_si512((__m512i*)b);
                __m512i ab16 = _mm512_maddubs_epi16(_mm512_and_si512(_a, K8_0F), _mm512_and_si512(_b, K8_0F));
                ab16 = _mm512_add_epi16(ab16, _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(_a, 4), K8_0F), _mm512_and_si512(_mm512_srli_epi16(_b, 4), K8_0F)));
                ab32 = _mm512_add_epi32(ab32, _mm512_madd_epi16(ab16, K16_0001));
            }
            if(i < size)
            {
                __mmask16 mask = TailMask16((size - i) / 8);
                __m512i _a = _mm512_maskz_loadu_epi32(mask, a);
                __m512i _b = _mm512_maskz_loadu_epi32(mask, b);
                __m512i ab16 = _mm512_maddubs_epi16(_mm512_and_si512(_a, K8_0F), _mm512_and_si512(_b, K8_0F));
                ab16 = _mm512_add_epi16(ab16, _mm512_maddubs_epi16(_mm512_and_si512(_mm512_srli_epi16(_a, 4), K8_0F), _mm512_and_si512(_mm512_srli_epi16(_b, 4), K8_0F)));
                ab32 = _mm512_add_epi32(ab32, _mm512_madd_epi16(ab16, K16_0001));
            }
            return ExtractSum<uint32_t>(ab32);
        }

        SIMD_INLINE __m512i Load5(const uint8_t* ptr, __mmask32 mask = 0x000FFFFF)
        {
            return _mm512_srli_epi16(_mm512_mullo_epi16(_mm512_shuffle_epi8(_mm512_permutexvar_epi32(C5_PERM, _mm512_castsi256_si512(_mm256_maskz_loadu_epi8(mask, ptr))), C5_SHFL), C5_MULLO), 11);
        }

        template<> int32_t Correlation<5>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i _ab = _mm512_setzero_si512();
            size_t i = 0, size32 = AlignLo(size, 32);
            for (; i < size32; i += 32, a += 20, b += 20)
            {
                __m512i _a = Load5(a);
                __m512i _b = Load5(b);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                __m512i _a = Load5(a, mask);
                __m512i _b = Load5(b, mask);
                _ab = _mm512_add_epi32(_mm512_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

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
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistancesDirect4x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistancesDirect4x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size128 = AlignLo(size, 128), o = 16;
            __m512i a00, a10, a20, a30, a01, a11, a21, a31, b00, b01;
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
            for (; i < size128; i += 128, o += 64)
            {
                a01 = _mm512_loadu_si512((__m512i*)(A[0] + o));
                a00 = _mm512_and_si512(a01, K8_0F);
                a01 = _mm512_and_si512(_mm512_srli_epi16(a01, 4), K8_0F);
                a11 = _mm512_loadu_si512((__m512i*)(A[1] + o));
                a10 = _mm512_and_si512(a11, K8_0F);
                a11 = _mm512_and_si512(_mm512_srli_epi16(a11, 4), K8_0F);
                a21 = _mm512_loadu_si512((__m512i*)(A[2] + o));
                a20 = _mm512_and_si512(a21, K8_0F);
                a21 = _mm512_and_si512(_mm512_srli_epi16(a21, 4), K8_0F);
                a31 = _mm512_loadu_si512((__m512i*)(A[3] + o));
                a30 = _mm512_and_si512(a31, K8_0F);
                a31 = _mm512_and_si512(_mm512_srli_epi16(a31, 4), K8_0F);

                b01 = _mm512_loadu_si512((__m512i*)(B[0] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab00 = _mm512_add_epi32(ab00, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab10 = _mm512_add_epi32(ab10, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab20 = _mm512_add_epi32(ab20, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab30 = _mm512_add_epi32(ab30, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[1] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_add_epi32(ab01, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab11 = _mm512_add_epi32(ab11, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab21 = _mm512_add_epi32(ab21, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab31 = _mm512_add_epi32(ab31, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[2] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_add_epi32(ab02, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab12 = _mm512_add_epi32(ab12, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab22 = _mm512_add_epi32(ab22, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab32 = _mm512_add_epi32(ab32, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[3] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_add_epi32(ab03, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab13 = _mm512_add_epi32(ab13, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab23 = _mm512_add_epi32(ab23, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab33 = _mm512_add_epi32(ab33, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));
            }
            if (i < size)
            {
                __mmask16 mask = TailMask32((size - i) / 8);
                a01 = _mm512_maskz_loadu_epi32(mask, A[0] + o);
                a00 = _mm512_and_si512(a01, K8_0F);
                a01 = _mm512_and_si512(_mm512_srli_epi16(a01, 4), K8_0F);
                a11 = _mm512_maskz_loadu_epi32(mask, A[1] + o);
                a10 = _mm512_and_si512(a11, K8_0F);
                a11 = _mm512_and_si512(_mm512_srli_epi16(a11, 4), K8_0F);
                a21 = _mm512_maskz_loadu_epi32(mask, A[2] + o);
                a20 = _mm512_and_si512(a21, K8_0F);
                a21 = _mm512_and_si512(_mm512_srli_epi16(a21, 4), K8_0F);
                a31 = _mm512_maskz_loadu_epi32(mask, A[3] + o);
                a30 = _mm512_and_si512(a31, K8_0F);
                a31 = _mm512_and_si512(_mm512_srli_epi16(a31, 4), K8_0F);

                b01 = _mm512_maskz_loadu_epi32(mask, B[0] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab00 = _mm512_add_epi32(ab00, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab10 = _mm512_add_epi32(ab10, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab20 = _mm512_add_epi32(ab20, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab30 = _mm512_add_epi32(ab30, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_maskz_loadu_epi32(mask, B[1] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_add_epi32(ab01, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab11 = _mm512_add_epi32(ab11, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab21 = _mm512_add_epi32(ab21, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab31 = _mm512_add_epi32(ab31, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_maskz_loadu_epi32(mask, B[2] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_add_epi32(ab02, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab12 = _mm512_add_epi32(ab12, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab22 = _mm512_add_epi32(ab22, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab32 = _mm512_add_epi32(ab32, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));

                b01 = _mm512_maskz_loadu_epi32(mask, B[3] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_add_epi32(ab03, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                ab13 = _mm512_add_epi32(ab13, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a10, b00), _mm512_maddubs_epi16(a11, b01)), K16_0001));
                ab23 = _mm512_add_epi32(ab23, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a20, b00), _mm512_maddubs_epi16(a21, b01)), K16_0001));
                ab33 = _mm512_add_epi32(ab33, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a30, b00), _mm512_maddubs_epi16(a31, b01)), K16_0001));
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            for (; i < size32; i += 32, o += 20)
            {
                a0 = Load5(A[0] + o);
                a1 = Load5(A[1] + o);
                a2 = Load5(A[2] + o);
                a3 = Load5(A[3] + o);

                b0 = Load5(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load5(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load5(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load5(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                a0 = Load5(A[0] + o, mask);
                a1 = Load5(A[1] + o, mask);
                a2 = Load5(A[2] + o, mask);
                a3 = Load5(A[3] + o, mask);

                b0 = Load5(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);
                ab10 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab10);
                ab20 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab20);
                ab30 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab30);

                b0 = Load5(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);
                ab11 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab11);
                ab21 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab21);
                ab31 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab31);

                b0 = Load5(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);
                ab12 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab12);
                ab22 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab22);
                ab32 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab32);

                b0 = Load5(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
                ab13 = _mm512_add_epi32(_mm512_madd_epi16(a1, b0), ab13);
                ab23 = _mm512_add_epi32(_mm512_madd_epi16(a2, b0), ab23);
                ab33 = _mm512_add_epi32(_mm512_madd_epi16(a3, b0), ab33);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 ab2 = _mm_cvtepi32_ps(Extract4Sums(ab20, ab21, ab22, ab23));
            __m128 ab3 = _mm_cvtepi32_ps(Extract4Sums(ab30, ab31, ab32, ab33));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<> void MicroCosineDistancesDirect4x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            Sse41::DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
            Sse41::DecodeCosineDistances1x4(A[2], B, ab2, distances + 2 * stride);
            Sse41::DecodeCosineDistances1x4(A[3], B, ab3, distances + 3 * stride);
        }

        template<int bits> void MicroCosineDistancesDirect1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistancesDirect1x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size128 = AlignLo(size, 128), o = 16;
            __m512i a00, a01, b00, b01;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            for (; i < size128; i += 128, o += 64)
            {
                a01 = _mm512_loadu_si512((__m512i*)(A[0] + o));
                a00 = _mm512_and_si512(a01, K8_0F);
                a01 = _mm512_and_si512(_mm512_srli_epi16(a01, 4), K8_0F);

                b01 = _mm512_loadu_si512((__m512i*)(B[0] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab00 = _mm512_add_epi32(ab00, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[1] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_add_epi32(ab01, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[2] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_add_epi32(ab02, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));

                b01 = _mm512_loadu_si512((__m512i*)(B[3] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_add_epi32(ab03, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
            }
            if (i < size)
            {
                __mmask16 mask = TailMask32((size - i) / 8);
                a01 = _mm512_maskz_loadu_epi32(mask, A[0] + o);
                a00 = _mm512_and_si512(a01, K8_0F);
                a01 = _mm512_and_si512(_mm512_srli_epi16(a01, 4), K8_0F);

                b01 = _mm512_maskz_loadu_epi32(mask, B[0] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab00 = _mm512_add_epi32(ab00, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));

                b01 = _mm512_maskz_loadu_epi32(mask, B[1] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_add_epi32(ab01, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
                b01 = _mm512_maskz_loadu_epi32(mask, B[2] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_add_epi32(ab02, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));

                b01 = _mm512_maskz_loadu_epi32(mask, B[3] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_add_epi32(ab03, _mm512_madd_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(a00, b00), _mm512_maddubs_epi16(a01, b01)), K16_0001));
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m512i a0, b0;
            __m512i ab00 = _mm512_setzero_si512();
            __m512i ab01 = _mm512_setzero_si512();
            __m512i ab02 = _mm512_setzero_si512();
            __m512i ab03 = _mm512_setzero_si512();
            for (; i < size32; i += 32, o += 20)
            {
                a0 = Load5(A[0] + o);

                b0 = Load5(B[0] + o);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load5(B[1] + o);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load5(B[2] + o);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load5(B[3] + o);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                a0 = Load5(A[0] + o, mask);

                b0 = Load5(B[0] + o, mask);
                ab00 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab00);

                b0 = Load5(B[1] + o, mask);
                ab01 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab01);

                b0 = Load5(B[2] + o, mask);
                ab02 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab02);

                b0 = Load5(B[3] + o, mask);
                ab03 = _mm512_add_epi32(_mm512_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect4x4<bits>(A + i, B + j, size, distances + j, stride);
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
                    MicroCosineDistancesDirect1x4<bits>(A + i, B + j, size, distances + j, stride);
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
            _minMax16f = MinMax16f;
            switch (depth)
            {
            case 4:
            {
                _encode32f = Encode32f4;
                _encode16f = Encode16f4;
                _decode32f = Decode32f4;
                _decode16f = Decode16f4;
                _cosineDistance = Avx512bw::CosineDistance<4>;
                _macroCosineDistancesDirect = Avx512bw::MacroCosineDistancesDirect<4>;
                break;
            }
            case 5:
            {
                _encode32f = Encode32f5;
                _encode16f = Encode16f5;
                _decode32f = Decode32f5;
                _decode16f = Decode16f5;
                _cosineDistance = Avx512bw::CosineDistance<5>;
                _macroCosineDistancesDirect = Avx512bw::MacroCosineDistancesDirect<5>;
                break;
            }
            case 6:
            {
                _encode32f = Encode32f6;
                _encode16f = Encode16f6;
                _decode32f = Decode32f6;
                _decode16f = Decode16f6;
                _cosineDistance = Avx512bw::CosineDistance<6>;
                _macroCosineDistancesDirect = Avx512bw::MacroCosineDistancesDirect<6>;
                break;
            }
            case 7:
            {
                _encode32f = Encode32f7;
                _encode16f = Encode16f7;
                _decode32f = Decode32f7;
                _decode16f = Decode16f7;
                _cosineDistance = Avx512bw::CosineDistance<7>;
                _macroCosineDistancesDirect = Avx512bw::MacroCosineDistancesDirect<7>;
                break;
            }
            case 8:
            {
                _encode32f = Encode32f8;
                _encode16f = Encode16f8;
                _decode32f = Decode32f8;
                _decode16f = Decode16f8;
                _cosineDistance = Avx512bw::CosineDistance<8>;
                _macroCosineDistancesDirect = Avx512bw::MacroCosineDistancesDirect<8>;
                _microMd = 4;
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

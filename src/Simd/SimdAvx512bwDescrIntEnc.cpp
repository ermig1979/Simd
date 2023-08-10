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
        SIMD_INLINE __m512i Encode32f(__m512 src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
        {
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(src, min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
        }

        SIMD_INLINE __m512i Encode32f(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512 _src = _mm512_maskz_loadu_ps(mask, src);
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_maskz_sub_ps(mask, _src, min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
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
                __mmask16 md = TailMask16((size - size32) / 2);
                _mm_mask_storeu_epi8(dst, md, Encode32f4x4(src, _scale, _min, _sum, _sqsum, ms0, ms1));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f5x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i i0 = Encode32f(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), Avx2::E5_MULLO);
            __m256i e0 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E5_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL1)), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL2));
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
            __m512 _src = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(mask, src));
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_maskz_sub_ps(mask, _src, min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
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
            __m256i e0 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(s0, Avx2::E5_SHFL0), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL1)), _mm256_shuffle_epi8(s0, Avx2::E5_SHFL2));
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

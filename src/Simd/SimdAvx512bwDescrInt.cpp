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
        static void MinMax(const float* src, size_t size, float& min, float& max)
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

        SIMD_INLINE __m512i Encode(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            __m512i value = _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(_mm512_maskz_loadu_ps(mask, src), min), scale));
            sum = _mm512_add_epi32(value, sum);
            sqsum = _mm512_add_epi32(_mm512_madd_epi16(value, value), sqsum);
            return value;
        }

        static SIMD_INLINE __m128i Encode6x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);
            __m512i i0 = Encode(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode6x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
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
            __m512i i0 = Encode(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(PERM, _mm512_packus_epi32(i0, i1)), SHIFT);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, SHFL0), _mm512_shuffle_epi8(s0, SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 24)
                _mm256_mask_storeu_epi8(dst - 4, 0x0FFFFFF0, Encode6x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 12)
                _mm_mask_storeu_epi8(dst, 0x0FFF, Encode6x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x003F, Encode6x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static SIMD_INLINE __m128i Encode7x2(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum, __mmask16 mask = -1)
        {
            static const __m256i SHIFT = SIMD_MM256_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
            static const __m256i SHFL0 = SIMD_MM256_SETR_EPI8(
                0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
            static const __m256i SHFL1 = SIMD_MM256_SETR_EPI8(
                0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);
            __m512i i0 = Encode(src, scale, min, sum, sqsum, mask);
            __m256i s0 = _mm256_mullo_epi16(_mm512_cvtepi32_epi16(i0), SHIFT);
            __m256i e0 = _mm256_or_si256(_mm256_shuffle_epi8(s0, SHFL0), _mm256_shuffle_epi8(s0, SHFL1));
            return _mm_or_si128(_mm256_castsi256_si128(e0), _mm256_extracti128_si256(e0, 1));
        }

        static SIMD_INLINE __m256i Encode7x4(const float* src, __m512 scale, __m512 min, __m512i& sum, __m512i& sqsum)
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
            __m512i i0 = Encode(src + 0 * F, scale, min, sum, sqsum);
            __m512i i1 = Encode(src + 1 * F, scale, min, sum, sqsum);
            __m512i s0 = _mm512_mullo_epi16(_mm512_permutexvar_epi64(PERM, _mm512_packus_epi32(i0, i1)), SHIFT);
            __m512i e0 = _mm512_or_si512(_mm512_shuffle_epi8(s0, SHFL0), _mm512_shuffle_epi8(s0, SHFL1));
            return _mm256_or_si256(_mm512_castsi512_si256(e0), _mm512_extracti32x8_epi32(e0, 1));
        }

        static void Encode7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t size16 = AlignLo(size, 16), size32 = AlignLo(size, 32), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < size32; i += 32, src += 32, dst += 28)
                _mm256_mask_storeu_epi8(dst - 2, 0x3FFFFFFC, Encode7x4(src, _scale, _min, _sum, _sqsum));
            for (; i < size16; i += 16, src += 16, dst += 14)
                _mm_mask_storeu_epi8(dst, 0x3FFF, Encode7x2(src, _scale, _min, _sum, _sqsum));
            if (i < size)
                _mm_mask_storeu_epi8(dst, 0x007F, Encode7x2(src, _scale, _min, _sum, _sqsum, 0x00FF));
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        static void Encode8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeF = AlignLo(size, F), sizeA = AlignLo(size, A), i = 0;
            __m512 _scale = _mm512_set1_ps(scale);
            __m512 _min = _mm512_set1_ps(min);
            __m512i _sum = _mm512_setzero_si512();
            __m512i _sqsum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i d0 = Encode(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m512i d1 = Encode(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m512i d2 = Encode(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m512i d3 = Encode(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm512_storeu_si512((__m512i*)(dst + i), PackI16ToU8(PackI32ToI16(d0, d1), PackI32ToI16(d2, d3)));
            }
            for (; i < sizeF; i += F)
            {
                __m512i d0 = Encode(src + i, _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }            
            if (i < size)
            {
                __m512i d0 = Encode(src + i, _scale, _min, _sum, _sqsum, 0xFF);
                _mm_mask_storeu_epi8(dst + i, 0xFF, _mm512_castsi512_si128(PackI16ToU8(PackI32ToI16(d0))));
            }
            sum = ExtractSum<uint32_t>(_sum);
            sqsum = ExtractSum<uint32_t>(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Avx2::DescrInt(size, depth)
        {
            _minMax = MinMax;
            switch (depth)
            {
            case 6:
            {
                _encode = Encode6;
            //    _decode = Decode6;
            //    _cosineDistance = Avx2::CosineDistance<6>;
            //    _macroCosineDistances = Avx2::MacroCosineDistances<6>;
                break;
            }
            case 7:
            {
                _encode = Encode7;
            //    _decode = Decode7;
            //    _cosineDistance = Avx2::CosineDistance<7>;
            //    _macroCosineDistances = Avx2::MacroCosineDistances<7>;
                break;
            }
            case 8:
            {
                _encode = Encode8;
            //    _decode = Decode8;
            //    _cosineDistance = Avx2::CosineDistance<8>;
            //    _macroCosineDistances = Avx2::MacroCosineDistances<8>;
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

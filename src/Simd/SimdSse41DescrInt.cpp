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

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        static void MinMax(const float* src, size_t size, float& min, float& max)
        {
            size_t sizeF = AlignLo(size, F);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < sizeF; i += F)
            {
                __m128 _src = _mm_loadu_ps(src + i);
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            for (; i < size; i += 1)
            {
                __m128 _src = _mm_load_ss(src + i);
                _min = _mm_min_ss(_src, _min);
                _max = _mm_max_ss(_src, _max);
            }
            _min = _mm_min_ps(_min, Shuffle32f<0x22>(_min));
            _max = _mm_max_ps(_max, Shuffle32f<0x22>(_max));
            _min = _mm_min_ss(_min, Shuffle32f<0x11>(_min));
            _max = _mm_max_ss(_max, Shuffle32f<0x11>(_max));
            _mm_store_ss(&min, _min);
            _mm_store_ss(&max, _max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Encode(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i value = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(src), min), scale));
            sum = _mm_add_epi32(value, sum);
            sqsum = _mm_add_epi32(_mm_madd_epi16(value, value), sqsum);
            return value;
        }

        static SIMD_INLINE __m128i Encode6(const float* src, __m128 scale, __m128 min, __m128i & sum, __m128i & sqsum)
        {
            static const __m128i SHIFT = SIMD_MM_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
            static const __m128i SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m128i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode(src + 4, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), SHIFT);
            return _mm_or_si128(_mm_shuffle_epi8(s0, SHFL0), _mm_shuffle_epi8(s0, SHFL1));
        }

        static void Encode6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode6(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode6(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static SIMD_INLINE __m128i Encode7(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            static const __m128i SHIFT = SIMD_MM_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
            static const __m128i SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            __m128i i0 = Encode(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode(src + 4, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), SHIFT);
            return _mm_or_si128(_mm_shuffle_epi8(s0, SHFL0), _mm_shuffle_epi8(s0, SHFL1));
        }

        static void Encode7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode7(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode7(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static void Encode8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i d0 = Encode(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m128i d1 = Encode(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m128i d2 = Encode(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m128i d3 = Encode(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m128i d0 = Encode(src + i, _scale, _min, _sum, _sqsum);
                _mm_storeu_si32((uint32_t*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(d0, _mm_setzero_si128()), _mm_setzero_si128()));
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            static const __m128i SHFL = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
            static const __m128i MUL = SIMD_MM_SETR_EPI16(256, 1024, 4096, 16384, 256, 1024, 4096, 16384);
            static const __m128i MASK = SIMD_MM_SET1_EPI16(0x3F);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(s6, SHFL), MUL), MASK);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 6;
                dst += 8;
            }
        }

        static void Decode7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            static const __m128i SHFL = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
            static const __m128i MUL = SIMD_MM_SETR_EPI16(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
            static const __m128i MASK = SIMD_MM_SET1_EPI16(0x7F);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(s7, SHFL), MUL), MASK);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 7;
                dst += 8;
            }
        }

        static void Decode8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t i = 0, sizeF = AlignLo(size, F);
            for (; i < sizeF; i += F)
            {
                __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(src + i)));
                _mm_storeu_ps(dst + i, _mm_add_ps(_mm_mul_ps(_src, _scale), _shift));
            }
            for (; i < size; i += 1)
            {
                __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_loadu_si32(src + i)));
                _mm_store_ss(dst + i, _mm_add_ss(_mm_mul_ss(_src, _scale), _shift));
            }
        }

        //-------------------------------------------------------------------------------------------------

        static int32_t Correlation6(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            static const __m128i SHFL = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
            static const __m128i MUL = SIMD_MM_SETR_EPI16(256, 1024, 4096, 16384, 256, 1024, 4096, 16384);
            static const __m128i MASK = SIMD_MM_SET1_EPI16(0x3F);
            __m128i _ab = _mm_setzero_si128();            
            for (size_t i = 0; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), SHFL), MUL), MASK);
                __m128i _b = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), SHFL), MUL), MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        static int32_t Correlation7(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            static const __m128i SHFL = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
            static const __m128i MUL = SIMD_MM_SETR_EPI16(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
            static const __m128i MASK = SIMD_MM_SET1_EPI16(0x7F);
            __m128i _ab = _mm_setzero_si128();
            for (size_t i = 0; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), SHFL), MUL), MASK);
                __m128i _b = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), SHFL), MUL), MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        static int32_t Correlation8(const uint8_t* a, const uint8_t* b, size_t size)
        {
            size_t i = 0, sizeA = AlignLo(size, A);
            __m128i _ab = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _a = _mm_loadu_si128((__m128i*)(a + i));
                __m128i _b = _mm_loadu_si128((__m128i*)(b + i));
                _ab = _mm_add_epi32(_mm_madd_epi16(UnpackU8<0>(_a), UnpackU8<0>(_b)), _ab);
                _ab = _mm_add_epi32(_mm_madd_epi16(UnpackU8<1>(_a), UnpackU8<1>(_b)), _ab);
            }
            for (; i < size; i += 8)
            {
                __m128i _a = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(a + i)));
                __m128i _b = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(b + i)));
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Base::DescrInt(size, depth)
        {
            _minMax = MinMax;
            switch (depth)
            {
            case 6:
            {
                _encode = Encode6;
                _decode = Decode6;
                _correlation = Correlation6;
                break;
            }
            case 7:
            {
                _encode = Encode7;
                _decode = Decode7;
                _correlation = Correlation7;
                break;
            }
            case 8:
            {
                _encode = Encode8;
                _decode = Decode8;
                _correlation = Correlation8;
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
            return new Sse41::DescrInt(size, depth);
        }
    }
#endif
}

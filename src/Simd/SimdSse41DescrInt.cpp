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

        const __m128i C6_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m128i C6_SHFL1 = SIMD_MM_SETR_EPI8(0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m128i C6_MUL = SIMD_MM_SETR_EPI16(256, 1024, 4096, 16384, 256, 1024, 4096, 16384);
        const __m128i C6_MASK = SIMD_MM_SET1_EPI16(0x3F);

        const __m128i C7_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m128i C7_SHFL1 = SIMD_MM_SETR_EPI8(0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m128i C7_MUL = SIMD_MM_SETR_EPI16(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
        const __m128i C7_MASK = SIMD_MM_SET1_EPI16(0x7F);

        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();  
            size_t i = 0, sizeA = AlignLo(size, A);
            for (; i < sizeA; i += A, a += 12, b += 12)
            {
                __m128i _a = _mm_loadu_si128((__m128i*)a);
                __m128i _b = _mm_loadu_si128((__m128i*)b);
                __m128i a0 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_a, C6_SHFL0), C6_MUL), C6_MASK);
                __m128i b0 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_b, C6_SHFL0), C6_MUL), C6_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_a, C6_SHFL1), C6_MUL), C6_MASK);
                __m128i b1 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_b, C6_SHFL1), C6_MUL), C6_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), C6_SHFL0), C6_MUL), C6_MASK);
                __m128i _b = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), C6_SHFL0), C6_MUL), C6_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();
            size_t i = 0, sizeA = AlignLo(size, A);
            for (; i < sizeA; i += A, a += 14, b += 14)
            {
                __m128i _a = _mm_loadu_si128((__m128i*)a);
                __m128i _b = _mm_loadu_si128((__m128i*)b);
                __m128i a0 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_a, C7_SHFL0), C7_MUL), C7_MASK);
                __m128i b0 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_b, C7_SHFL0), C7_MUL), C7_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_a, C7_SHFL1), C7_MUL), C7_MASK);
                __m128i b1 = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_b, C7_SHFL1), C7_MUL), C7_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), C7_SHFL0), C7_MUL), C7_MASK);
                __m128i _b = _mm_and_si128(_mm_mulhi_epu16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), C7_SHFL0), C7_MUL), C7_MASK);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
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

        template<int bits> void CosineDistance(const uint8_t* a, const uint8_t* b, size_t size, float* distance)
        {
            int abSum = Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, (int32_t)size, distance);
        }


        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistances2x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances2x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m128i a00, a01, a10, a11, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            __m128i ab10 = _mm_setzero_si128();
            __m128i ab11 = _mm_setzero_si128();
            __m128i ab12 = _mm_setzero_si128();
            __m128i ab13 = _mm_setzero_si128();
            for (; i < size16; i += 16, o += 16)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = UnpackU8<0>(a01);
                a01 = UnpackU8<1>(a01);
                a11 = _mm_loadu_si128((__m128i*)(A[1] + o));
                a10 = UnpackU8<0>(a11);
                a11 = UnpackU8<1>(a11);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = UnpackU8<0>(b01);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = UnpackU8<1>(b01);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = UnpackU8<0>(b01);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = UnpackU8<1>(b01);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = UnpackU8<0>(b01);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = UnpackU8<1>(b01);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = UnpackU8<0>(b01);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = UnpackU8<1>(b01);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 8)
            {
                a00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));
                a10 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[1] + o)));

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

        template<int bits> void MacroCosineDistances(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLo(N, 4);
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
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Base::DescrInt(size, depth)
        {
            _minMax = MinMax;
            _microM = 2;
            _microN = 4;
            switch (depth)
            {
            case 6:
            {
                _encode = Encode6;
                _decode = Decode6;
                _cosineDistance = Sse41::CosineDistance<6>;
                break;
            }
            case 7:
            {
                _encode = Encode7;
                _decode = Decode7;
                _cosineDistance = Sse41::CosineDistance<7>;
                break;
            }
            case 8:
            {
                _encode = Encode8;
                _decode = Decode8;
                _cosineDistance = Sse41::CosineDistance<8>;
                _macroCosineDistances = Sse41::MacroCosineDistances<8>;
                break;
            }
            default:
                assert(0);
            }
        }

        void DescrInt::CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            if (_depth != 8)
            {
                Base::DescrInt::CosineDistancesMxNa(M, N, A, B, distances);
                return;
            }
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / _encSize, _microN);
            size_t mM = AlignLoAny(L2 / _encSize, _microM);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    _macroCosineDistances(dM, dN, A + i, B + j, _size, distances + i * N + j, N);
                }
            }
        }

        void DescrInt::CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const
        {
            if (_depth != 8)
            {
                Base::DescrInt::CosineDistancesMxNp(M, N, A, B, distances);
                return;
            }
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / _encSize, _microN);
            size_t mM = AlignLoAny(L2 / _encSize, _microM);
            Array8ucp ap(mM), bp(N);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t k = 0; k < dM; ++k)
                    ap[k] = A + k * _encSize;
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    if (i == 0)
                    {
                        for (size_t k = j, n = j + dN; k < n; ++k)
                            bp[k] = B + k * _encSize;
                    }
                    _macroCosineDistances(dM, dN, ap.data, bp.data + j, _size, distances + i * N + j, N);
                }
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

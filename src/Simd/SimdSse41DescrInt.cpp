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
#include "Simd/SimdFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 4)
            {
                __m128 _src = _mm_loadu_ps(src + i);
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            __m128 _min = _mm_set1_ps(FLT_MAX);
            __m128 _max = _mm_set1_ps(-FLT_MAX);
            size_t i = 0;
            for (; i < size; i += 4)
            {
                __m128i f16 = _mm_loadl_epi64((__m128i*)(src + i));
                __m128 _src = Float16ToFloat32(UnpackU16<0>(f16));
                _min = _mm_min_ps(_src, _min);
                _max = _mm_max_ps(_src, _max);
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Encode32f(__m128 src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i value = _mm_cvtps_epi32(_mm_mul_ps(_mm_sub_ps(src, min), scale));
            sum = _mm_add_epi32(value, sum);
            sqsum = _mm_add_epi32(_mm_madd_epi16(value, value), sqsum);
            return value;
        }

        SIMD_INLINE __m128i Encode32f(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            return Encode32f(_mm_loadu_ps(src), scale, min, sum, sqsum);
        }

        static SIMD_INLINE __m128i Encode32f6(const float* src, __m128 scale, __m128 min, __m128i & sum, __m128i & sqsum)
        {
            __m128i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode32f(src + 4, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E6_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, E6_SHFL0), _mm_shuffle_epi8(s0, E6_SHFL1));
        }

        static void Encode32f6(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode32f6(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode32f6(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f7(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode32f(src + 4, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E7_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, E7_SHFL0), _mm_shuffle_epi8(s0, E7_SHFL1));
        }

        static void Encode32f7(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode32f7(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode32f7(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static void Encode32f8(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i d0 = Encode32f(src + i + 0 * F, _scale, _min, _sum, _sqsum);
                __m128i d1 = Encode32f(src + i + 1 * F, _scale, _min, _sum, _sqsum);
                __m128i d2 = Encode32f(src + i + 2 * F, _scale, _min, _sum, _sqsum);
                __m128i d3 = Encode32f(src + i + 3 * F, _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m128i d0 = Encode32f(src + i, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + i) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(d0, _mm_setzero_si128()), _mm_setzero_si128()));
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static SIMD_INLINE __m128i Encode16f6(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i u0 = _mm_loadu_si128((__m128i*)(src));
            __m128i i0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), scale, min, sum, sqsum);
            __m128i i4 = Encode32f(Float16ToFloat32(UnpackU16<1>(u0)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E6_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, E6_SHFL0), _mm_shuffle_epi8(s0, E6_SHFL1));
        }

        static void Encode16f6(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 6)
                _mm_storel_epi64((__m128i*)dst, Encode16f6(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 6)
            {
                __m128i d0 = Encode16f6(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f7(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i u0 = _mm_loadu_si128((__m128i*)(src));
            __m128i i0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), scale, min, sum, sqsum);
            __m128i i4 = Encode32f(Float16ToFloat32(UnpackU16<1>(u0)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E7_MULLO);
            return _mm_or_si128(_mm_shuffle_epi8(s0, E7_SHFL0), _mm_shuffle_epi8(s0, E7_SHFL1));
        }

        static void Encode16f7(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 7)
                _mm_storel_epi64((__m128i*)dst, Encode16f7(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 7)
            {
                __m128i d0 = Encode16f7(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint16_t*)(dst + 4) = _mm_extract_epi16(d0, 2);
                *(uint8_t*)(dst + 6) = _mm_extract_epi8(d0, 6);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static void Encode16f8(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t sizeA = AlignLo(size, A), i = 0;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i u0 = _mm_loadu_si128((__m128i*)(src + i + 0 * F));
                __m128i d0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), _scale, _min, _sum, _sqsum);
                __m128i d1 = Encode32f(Float16ToFloat32(UnpackU16<1>(u0)), _scale, _min, _sum, _sqsum);
                __m128i u2 = _mm_loadu_si128((__m128i*)(src + i + 2 * F));
                __m128i d2 = Encode32f(Float16ToFloat32(UnpackU16<0>(u2)), _scale, _min, _sum, _sqsum);
                __m128i d3 = Encode32f(Float16ToFloat32(UnpackU16<1>(u2)), _scale, _min, _sum, _sqsum);
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi16(_mm_packus_epi32(d0, d1), _mm_packus_epi32(d2, d3)));
            }
            for (; i < size; i += F)
            {
                __m128i u0 = _mm_loadl_epi64((__m128i*)(src + i));
                __m128i d0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + i) = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(d0, _mm_setzero_si128()), _mm_setzero_si128()));
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, C6_SHFL0), C6_MULLO), 10);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 6;
                dst += 8;
            }
        }

        static void Decode32f7(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, C7_SHFL0), C7_MULLO), 9);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 7;
                dst += 8;
            }
        }

        static void Decode32f8(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t i = 0;
            for (; i < size; i += 4)
            {
                __m128 _src = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(uint32_t*)(src + i))));
                _mm_storeu_ps(dst + i, _mm_add_ps(_mm_mul_ps(_src, _scale), _shift));
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, C6_SHFL0), C6_MULLO), 10);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 6;
                dst += 8;
            }
        }

        static void Decode16f7(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            for (size_t i = 0; i < size; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, C7_SHFL0), C7_MULLO), 9);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 7;
                dst += 8;
            }
        }

        static void Decode16f8(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t i = 0;
            for (; i < size; i += 8)
            {
                __m128i s8 = _mm_loadl_epi64((__m128i*)(src + i));
                __m128 s0 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(s8, 0)));
                __m128 s4 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(s8, 4)));
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(s0, _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(s4, _scale), _shift));
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packus_epi32(d0, d4));
            }
        }

        //-------------------------------------------------------------------------------------------------

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
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C6_SHFL0), C6_MULLO), 10);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C6_SHFL0), C6_MULLO), 10);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C6_SHFL1), C6_MULLO), 10);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C6_SHFL1), C6_MULLO), 10);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), C6_SHFL0), C6_MULLO), 10);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), C6_SHFL0), C6_MULLO), 10);
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
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C7_SHFL0), C7_MULLO), 9);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C7_SHFL0), C7_MULLO), 9);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C7_SHFL1), C7_MULLO), 9);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C7_SHFL1), C7_MULLO), 9);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)a), C7_SHFL0), C7_MULLO), 9);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)b), C7_SHFL0), C7_MULLO), 9);
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
            float abSum = (float)Correlation<bits>(a + 16, b + 16, size);
            Base::DecodeCosineDistance(a, b, abSum, (float)size, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistances2x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances2x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            for (; i < size16; i += 16, o += 12)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C6_SHFL0), C6_MULLO), 10);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 6), C6_SHFL0), C6_MULLO), 10);
                a11 = _mm_loadu_si128((__m128i*)(A[1] + o));
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C6_SHFL0), C6_MULLO), 10);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 6), C6_SHFL0), C6_MULLO), 10);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 6)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[0] + o)), C6_SHFL0), C6_MULLO), 10);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[1] + o)), C6_SHFL0), C6_MULLO), 10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[0] + o)), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[1] + o)), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[2] + o)), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[3] + o)), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

        template<> void MicroCosineDistances2x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            for (; i < size16; i += 16, o += 14)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C7_SHFL0), C7_MULLO), 9);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 7), C7_SHFL0), C7_MULLO), 9);
                a11 = _mm_loadu_si128((__m128i*)(A[1] + o));
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C7_SHFL0), C7_MULLO), 9);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 7), C7_SHFL0), C7_MULLO), 9);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 7)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[0] + o)), C7_SHFL0), C7_MULLO), 9);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[1] + o)), C7_SHFL0), C7_MULLO), 9);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[0] + o)), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[1] + o)), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[2] + o)), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[3] + o)), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
            DecodeCosineDistances(A[1], B, ab1, _size, distances + 1 * stride);
        }

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

        template<int bits> void MicroCosineDistances1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistances1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16; i += 16, o += 12)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C6_SHFL0), C6_MULLO), 10);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 6), C6_SHFL0), C6_MULLO), 10);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 6)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[0] + o)), C6_SHFL0), C6_MULLO), 10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[0] + o)), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[1] + o)), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[2] + o)), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[3] + o)), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16; i += 16, o += 14)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C7_SHFL0), C7_MULLO), 9);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 7), C7_SHFL0), C7_MULLO), 9);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 7)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(A[0] + o)), C7_SHFL0), C7_MULLO), 9);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[0] + o)), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[1] + o)), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[2] + o)), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_loadl_epi64((__m128i*)(B[3] + o)), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
        }

        template<> void MicroCosineDistances1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16; i += 16, o += 16)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = UnpackU8<0>(a01);
                a01 = UnpackU8<1>(a01);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = UnpackU8<0>(b01);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = UnpackU8<1>(b01);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = UnpackU8<0>(b01);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = UnpackU8<1>(b01);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = UnpackU8<0>(b01);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = UnpackU8<1>(b01);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = UnpackU8<0>(b01);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = UnpackU8<1>(b01);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 8)
            {
                a00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 _size = _mm_set1_ps(float(size));
            DecodeCosineDistances(A[0], B, ab0, _size, distances + 0 * stride);
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
                for (; j < N4; j += 4)
                    MicroCosineDistances1x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Base::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;
            _microM = 2;
            _microN = 4;
            switch (depth)
            {
            case 6:
            {
                _encode32f = Encode32f6;
                _encode16f = Encode16f6;
                _decode32f = Decode32f6;
                _decode16f = Decode16f6;
                _cosineDistance = Sse41::CosineDistance<6>;
                _macroCosineDistances = Sse41::MacroCosineDistances<6>;
                break;
            }
            case 7:
            {
                _encode32f = Encode32f7;
                _encode16f = Encode16f7;
                _decode32f = Decode32f7;
                _decode16f = Decode16f7;
                _cosineDistance = Sse41::CosineDistance<7>;
                _macroCosineDistances = Sse41::MacroCosineDistances<7>;
                break;
            }
            case 8:
            {
                _encode32f = Encode32f8;
                _encode16f = Encode16f8;
                _decode32f = Decode32f8;
                _decode16f = Decode16f8;
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
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / _encSize, _microN);
            size_t mM = AlignLoAny(L2 / _encSize, _microM);
            Array8ucp ap(mM), bp(N);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t k = 0; k < dM; ++k)
                    ap[k] = A + (i + k) * _encSize;
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

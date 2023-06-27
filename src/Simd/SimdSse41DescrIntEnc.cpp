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
#include "Simd/SimdFloat16.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
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

        static SIMD_INLINE __m128i Encode32f4(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode32f(src + 4, scale, min, sum, sqsum);
            return _mm_srli_epi32(_mm_mullo_epi16(_mm_packus_epi32(i0, i4), E4_MULLO), 12);
        }

        static SIMD_INLINE __m128i Encode32f4x8(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i s0 = Encode32f4(src + 0 * 8, scale, min, sum, sqsum);
            return _mm_packus_epi16(_mm_packus_epi32(s0, K_ZERO), K_ZERO);
        }

        static SIMD_INLINE __m128i Encode32f4x16(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i s0 = Encode32f4(src + 0 * 8, scale, min, sum, sqsum);
            __m128i s1 = Encode32f4(src + 1 * 8, scale, min, sum, sqsum);
            return _mm_packus_epi16(_mm_packus_epi32(s0, s1), K_ZERO);
        }

        static void Encode32f4(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size16 = AlignLo(size, 16);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < size16; i += 16, src += 16, dst += 8)
                _mm_storel_epi64((__m128i*)dst, Encode32f4x16(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 4)
                *(uint32_t*)(dst) = _mm_extract_epi32(Encode32f4x8(src, _scale, _min, _sum, _sqsum), 0);
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static SIMD_INLINE __m128i Encode32f5(const float* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i i0 = Encode32f(src + 0, scale, min, sum, sqsum);
            __m128i i4 = Encode32f(src + 4, scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E5_MULLO);
            return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s0, E5_SHFL0), _mm_shuffle_epi8(s0, E5_SHFL1)), _mm_shuffle_epi8(s0, E5_SHFL2));
        }

        static void Encode32f5(const float* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 5)
                _mm_storel_epi64((__m128i*)dst, Encode32f5(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                __m128i d0 = Encode32f5(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint8_t*)(dst + 4) = _mm_extract_epi8(d0, 4);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
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

        static SIMD_INLINE __m128i Encode16f4(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i u0 = _mm_loadu_si128((__m128i*)(src));
            __m128i i0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), scale, min, sum, sqsum);
            __m128i i4 = Encode32f(Float16ToFloat32(UnpackU16<1>(u0)), scale, min, sum, sqsum);
            return _mm_srli_epi32(_mm_mullo_epi16(_mm_packus_epi32(i0, i4), E4_MULLO), 12);
        }

        static SIMD_INLINE __m128i Encode16f4x8(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i s0 = Encode16f4(src + 0 * 8, scale, min, sum, sqsum);
            return _mm_packus_epi16(_mm_packus_epi32(s0, K_ZERO), K_ZERO);
        }

        static SIMD_INLINE __m128i Encode16f4x16(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i s0 = Encode16f4(src + 0 * 8, scale, min, sum, sqsum);
            __m128i s1 = Encode16f4(src + 1 * 8, scale, min, sum, sqsum);
            return _mm_packus_epi16(_mm_packus_epi32(s0, s1), K_ZERO);
        }

        static void Encode16f4(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, size16 = AlignLo(size, 16);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < size16; i += 16, src += 16, dst += 8)
                _mm_storel_epi64((__m128i*)dst, Encode16f4x16(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 4)
                *(uint32_t*)(dst) = _mm_extract_epi32(Encode16f4x8(src, _scale, _min, _sum, _sqsum), 0);
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

        static SIMD_INLINE __m128i Encode16f5(const uint16_t* src, __m128 scale, __m128 min, __m128i& sum, __m128i& sqsum)
        {
            __m128i u0 = _mm_loadu_si128((__m128i*)(src));
            __m128i i0 = Encode32f(Float16ToFloat32(UnpackU16<0>(u0)), scale, min, sum, sqsum);
            __m128i i4 = Encode32f(Float16ToFloat32(UnpackU16<1>(u0)), scale, min, sum, sqsum);
            __m128i s0 = _mm_mullo_epi16(_mm_packus_epi32(i0, i4), E5_MULLO);
            return _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s0, E5_SHFL0), _mm_shuffle_epi8(s0, E5_SHFL1)), _mm_shuffle_epi8(s0, E5_SHFL2));
        }

        static void Encode16f5(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst)
        {
            assert(size % 8 == 0);
            size_t i = 0, main = size - 8;
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _min = _mm_set1_ps(min);
            __m128i _sum = _mm_setzero_si128();
            __m128i _sqsum = _mm_setzero_si128();
            for (; i < main; i += 8, src += 8, dst += 5)
                _mm_storel_epi64((__m128i*)dst, Encode16f5(src, _scale, _min, _sum, _sqsum));
            for (; i < size; i += 8, src += 8, dst += 5)
            {
                __m128i d0 = Encode16f5(src, _scale, _min, _sum, _sqsum);
                *(uint32_t*)(dst + 0) = _mm_extract_epi32(d0, 0);
                *(uint8_t*)(dst + 4) = _mm_extract_epi8(d0, 4);
            }
            sum = ExtractInt32Sum(_sum);
            sqsum = ExtractInt32Sum(_sqsum);
        }

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

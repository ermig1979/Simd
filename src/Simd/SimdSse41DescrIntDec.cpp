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
        static void Decode32f4(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size16 = AlignLo(size, 16), i = 0;
            for (; i < size16; i += 16)
            {
                __m128i s4 = _mm_loadl_epi64((__m128i*)src);
                __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, C4_SHFL0), C4_MULLO), 12);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(lo)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(lo)), _scale), _shift));
                __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, C4_SHFL1), C4_MULLO), 12);
                _mm_storeu_ps(dst + 8, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(hi)), _scale), _shift));
                _mm_storeu_ps(dst + 12, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(hi)), _scale), _shift));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(src), C4_SHFL0), C4_MULLO), 12);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(lo)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(lo)), _scale), _shift));
                src += 4;
                dst += 8;
            }
        }

        static void Decode32f5(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s5 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s5, C5_SHFL0), C5_MULLO), 11);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 5;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(src), C5_SHFL0), C5_MULLO), 11);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 5;
                dst += 8;
            }
        }

        static void Decode32f6(const uint8_t* src, float scale, float shift, size_t size, float* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, C6_SHFL0), C6_MULLO), 10);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 6;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(src), C6_SHFL0), C6_MULLO), 10);
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
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, C7_SHFL0), C7_MULLO), 9);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                src += 7;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(src), C7_SHFL0), C7_MULLO), 9);
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
            for (; i < size; i += 8)
            {
                __m128i s8 = _mm_loadl_epi64((__m128i*)src);
                _mm_storeu_ps(dst + 0, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(s8)), _scale), _shift));
                _mm_storeu_ps(dst + 4, _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(s8, 4))), _scale), _shift));
                src += 8;
                dst += 8;
            }
        }

        //-------------------------------------------------------------------------------------------------

        static void Decode16f4(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size16 = AlignLo(size, 16), i = 0;
            for (; i < size16; i += 16)
            {
                __m128i s4 = _mm_loadl_epi64((__m128i*)src);
                __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, C4_SHFL0), C4_MULLO), 12);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(lo)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(lo)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst + 0, _mm_packus_epi32(d0, d4));
                __m128i hi = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s4, C4_SHFL1), C4_MULLO), 12);
                __m128i d8 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(hi)), _scale), _shift));
                __m128i d12 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(hi)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst + 1, _mm_packus_epi32(d8, d12));
                src += 8;
                dst += 16;
            }
            for (; i < size; i += 8)
            {
                __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(src), C4_SHFL0), C4_MULLO), 12);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(lo)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(lo)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 4;
                dst += 8;
            }
        }

        static void Decode16f5(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s5 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s5, C5_SHFL0), C5_MULLO), 11);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 5;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(src), C5_SHFL0), C5_MULLO), 11);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 5;
                dst += 8;
            }
        }

        static void Decode16f6(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst)
        {
            assert(size % 8 == 0);
            __m128 _scale = _mm_set1_ps(scale);
            __m128 _shift = _mm_set1_ps(shift);
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s6 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s6, C6_SHFL0), C6_MULLO), 10);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 6;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(src), C6_SHFL0), C6_MULLO), 10);
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
            size_t size8 = AlignLo(size - 1, 8), i = 0;
            for (; i < size8; i += 8)
            {
                __m128i s7 = _mm_loadl_epi64((__m128i*)src);
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(s7, C7_SHFL0), C7_MULLO), 9);
                __m128i d0 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<0>(s16)), _scale), _shift));
                __m128i d4 = Float32ToFloat16(_mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(UnpackU16<1>(s16)), _scale), _shift));
                _mm_storeu_si128((__m128i*)dst, _mm_packus_epi32(d0, d4));
                src += 7;
                dst += 8;
            }
            for (; i < size; i += 8)
            {
                __m128i s16 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(src), C7_SHFL0), C7_MULLO), 9);
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

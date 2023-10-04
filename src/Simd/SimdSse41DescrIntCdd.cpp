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
        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i ab32 = _mm_setzero_si128();
            size_t i = 0, size32 = AlignLo(size, 32);
            for (; i < size32; i += 32, a += 16, b += 16)
            {
                __m128i _a = _mm_loadu_si128((__m128i*)a);
                __m128i _b = _mm_loadu_si128((__m128i*)b);
                __m128i ab16 = _mm_maddubs_epi16(_mm_and_si128(_a, K8_0F), _mm_and_si128(_b, K8_0F));
                ab16 = _mm_add_epi16(ab16, _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(_a, 4), K8_0F), _mm_and_si128(_mm_srli_epi16(_b, 4), K8_0F)));
                ab32 = _mm_add_epi32(ab32, _mm_madd_epi16(ab16, K16_0001));
            }
            for (; i < size; i += 8, a += 4, b += 4)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(a), C4_SHFL0), C4_MULLO), 12);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(b), C4_SHFL0), C4_MULLO), 12);
                ab32 = _mm_add_epi32(_mm_madd_epi16(_a, _b), ab32);
            }
            return ExtractInt32Sum(ab32);
        }

        template<> int32_t Correlation<5>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();
            size_t i = 0, sizeA1 = AlignLo(size - 1, A), sizeA = AlignLo(size, A);
            for (; i < sizeA1; i += A, a += 10, b += 10)
            {
                __m128i _a = _mm_loadu_si128((__m128i*)a);
                __m128i _b = _mm_loadu_si128((__m128i*)b);
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C5_SHFL0), C5_MULLO), 11);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C5_SHFL0), C5_MULLO), 11);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C5_SHFL1), C5_MULLO), 11);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C5_SHFL1), C5_MULLO), 11);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < sizeA; i += A, a += 10, b += 10)
            {
                __m128i _a = LoadLast16<5>(a);
                __m128i _b = LoadLast16<5>(b);
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C5_SHFL0), C5_MULLO), 11);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C5_SHFL0), C5_MULLO), 11);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C5_SHFL1), C5_MULLO), 11);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C5_SHFL1), C5_MULLO), 11);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 5, b += 5)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(a), C5_SHFL0), C5_MULLO), 11);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(b), C5_SHFL0), C5_MULLO), 11);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();  
            size_t i = 0, sizeA1 = AlignLo(size - 1, A), sizeA = AlignLo(size, A);
            for (; i < sizeA1; i += A, a += 12, b += 12)
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
            for (; i < sizeA; i += A, a += 12, b += 12)
            {
                __m128i _a = LoadLast16<6>(a);
                __m128i _b = LoadLast16<6>(b);
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C6_SHFL0), C6_MULLO), 10);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C6_SHFL0), C6_MULLO), 10);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C6_SHFL1), C6_MULLO), 10);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C6_SHFL1), C6_MULLO), 10);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(a), C6_SHFL0), C6_MULLO), 10);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(b), C6_SHFL0), C6_MULLO), 10);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();
            size_t i = 0, sizeA1 = AlignLo(size - 1, A), sizeA = AlignLo(size, A);
            for (; i < sizeA1; i += A, a += 14, b += 14)
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
            for (; i < sizeA; i += A, a += 14, b += 14)
            {
                __m128i _a = LoadLast16<7>(a);
                __m128i _b = LoadLast16<7>(b);
                __m128i a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C7_SHFL0), C7_MULLO), 9);
                __m128i b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C7_SHFL0), C7_MULLO), 9);
                _ab = _mm_add_epi32(_mm_madd_epi16(a0, b0), _ab);
                __m128i a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_a, C7_SHFL1), C7_MULLO), 9);
                __m128i b1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_b, C7_SHFL1), C7_MULLO), 9);
                _ab = _mm_add_epi32(_mm_madd_epi16(a1, b1), _ab);
            }
            for (; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(a), C7_SHFL0), C7_MULLO), 9);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(b), C7_SHFL0), C7_MULLO), 9);
                _ab = _mm_add_epi32(_mm_madd_epi16(_a, _b), _ab);
            }
            return ExtractInt32Sum(_ab);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m128i _ab = _mm_setzero_si128();
            size_t i = 0, sizeA = AlignLo(size, A);
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
            Base::DecodeCosineDistance(a, b, abSum, distance);
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void MicroCosineDistancesDirect2x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistancesDirect2x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m128i a0, a1, b0;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            __m128i ab10 = _mm_setzero_si128();
            __m128i ab11 = _mm_setzero_si128();
            __m128i ab12 = _mm_setzero_si128();
            __m128i ab13 = _mm_setzero_si128();
            for (; i < size32; i += 32, o += 16)
            {
                a0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(A[0] + o)), K8_0F);
                a1 = _mm_and_si128(_mm_loadu_si128((__m128i*)(A[1] + o)), K8_0F);

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[0] + o)), K8_0F);
                ab00 = _mm_add_epi32(ab00, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab10 = _mm_add_epi32(ab10, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[1] + o)), K8_0F);
                ab01 = _mm_add_epi32(ab01, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab11 = _mm_add_epi32(ab11, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[2] + o)), K8_0F);
                ab02 = _mm_add_epi32(ab02, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab12 = _mm_add_epi32(ab12, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[3] + o)), K8_0F);
                ab03 = _mm_add_epi32(ab03, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab13 = _mm_add_epi32(ab13, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                a0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)), 4), K8_0F);
                a1 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(A[1] + o)), 4), K8_0F);

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)), 4), K8_0F);
                ab00 = _mm_add_epi32(ab00, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab10 = _mm_add_epi32(ab10, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)), 4), K8_0F);
                ab01 = _mm_add_epi32(ab01, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab11 = _mm_add_epi32(ab11, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)), 4), K8_0F);
                ab02 = _mm_add_epi32(ab02, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab12 = _mm_add_epi32(ab12, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)), 4), K8_0F);
                ab03 = _mm_add_epi32(ab03, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
                ab13 = _mm_add_epi32(ab13, _mm_madd_epi16(_mm_maddubs_epi16(a1, b0), K16_0001));
            }
            for (; i < size; i += 8, o += 4)
            {
                a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(A[0] + o), C4_SHFL0), C4_MULLO), 12);
                a1 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(A[1] + o), C4_SHFL0), C4_MULLO), 12);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[0] + o), C4_SHFL0), C4_MULLO), 12);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a1, b0), ab10);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[1] + o), C4_SHFL0), C4_MULLO), 12);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a1, b0), ab11);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[2] + o), C4_SHFL0), C4_MULLO), 12);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a1, b0), ab12);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[3] + o), C4_SHFL0), C4_MULLO), 12);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a1, b0), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, a10, a11, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            __m128i ab10 = _mm_setzero_si128();
            __m128i ab11 = _mm_setzero_si128();
            __m128i ab12 = _mm_setzero_si128();
            __m128i ab13 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 10)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C5_SHFL0), C5_MULLO), 11);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 5), C5_SHFL0), C5_MULLO), 11);
                a11 = _mm_loadu_si128((__m128i*)(A[1] + o));
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C5_SHFL0), C5_MULLO), 11);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 5), C5_SHFL0), C5_MULLO), 11);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size16; i += 16, o += 10)
            {
                a01 = LoadLast16<5>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C5_SHFL0), C5_MULLO), 11);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 5), C5_SHFL0), C5_MULLO), 11);
                a11 = LoadLast16<5>(A[1] + o);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C5_SHFL0), C5_MULLO), 11);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 5), C5_SHFL0), C5_MULLO), 11);

                b01 = LoadLast16<5>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = LoadLast16<5>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = LoadLast16<5>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = LoadLast16<5>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 5)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(A[0] + o), C5_SHFL0), C5_MULLO), 11);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(A[1] + o), C5_SHFL0), C5_MULLO), 11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[0] + o), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[1] + o), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[2] + o), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[3] + o), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, a10, a11, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            __m128i ab10 = _mm_setzero_si128();
            __m128i ab11 = _mm_setzero_si128();
            __m128i ab12 = _mm_setzero_si128();
            __m128i ab13 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 12)
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
            for (; i < size16; i += 16, o += 12)
            {
                a01 = LoadLast16<6>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C6_SHFL0), C6_MULLO), 10);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 6), C6_SHFL0), C6_MULLO), 10);
                a11 = LoadLast16<6>(A[1] + o);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C6_SHFL0), C6_MULLO), 10);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 6), C6_SHFL0), C6_MULLO), 10);

                b01 = LoadLast16<6>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = LoadLast16<6>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = LoadLast16<6>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = LoadLast16<6>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 6)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(A[0] + o), C6_SHFL0), C6_MULLO), 10);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(A[1] + o), C6_SHFL0), C6_MULLO), 10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[0] + o), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[1] + o), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[2] + o), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[3] + o), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, a10, a11, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            __m128i ab10 = _mm_setzero_si128();
            __m128i ab11 = _mm_setzero_si128();
            __m128i ab12 = _mm_setzero_si128();
            __m128i ab13 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 14)
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
            for (; i < size16; i += 16, o += 14)
            {
                a01 = LoadLast16<7>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C7_SHFL0), C7_MULLO), 9);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 7), C7_SHFL0), C7_MULLO), 9);
                a11 = LoadLast16<7>(A[1] + o);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a11, C7_SHFL0), C7_MULLO), 9);
                a11 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a11, 7), C7_SHFL0), C7_MULLO), 9);

                b01 = LoadLast16<7>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab10);

                b01 = LoadLast16<7>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab11);

                b01 = LoadLast16<7>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab12);

                b01 = LoadLast16<7>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a11, b00), ab13);
            }
            for (; i < size; i += 8, o += 7)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(A[0] + o), C7_SHFL0), C7_MULLO), 9);
                a10 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(A[1] + o), C7_SHFL0), C7_MULLO), 9);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[0] + o), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                ab10 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[1] + o), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                ab11 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[2] + o), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                ab12 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab12);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[3] + o), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                ab13 = _mm_add_epi32(_mm_madd_epi16(a10, b00), ab13);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            __m128 ab1 = _mm_cvtepi32_ps(Extract4Sums(ab10, ab11, ab12, ab13));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<> void MicroCosineDistancesDirect2x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
            DecodeCosineDistances1x4(A[1], B, ab1, distances + 1 * stride);
        }

        template<int bits> void MicroCosineDistancesDirect1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistancesDirect1x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size32 = AlignLo(size, 32), o = 16;
            __m128i a0, b0;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size32; i += 32, o += 16)
            {
                a0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(A[0] + o)), K8_0F);

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[0] + o)), K8_0F);
                ab00 = _mm_add_epi32(ab00, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[1] + o)), K8_0F);
                ab01 = _mm_add_epi32(ab01, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[2] + o)), K8_0F);
                ab02 = _mm_add_epi32(ab02, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_loadu_si128((__m128i*)(B[3] + o)), K8_0F);
                ab03 = _mm_add_epi32(ab03, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                a0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)), 4), K8_0F);

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)), 4), K8_0F);
                ab00 = _mm_add_epi32(ab00, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)), 4), K8_0F);
                ab01 = _mm_add_epi32(ab01, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)), 4), K8_0F);
                ab02 = _mm_add_epi32(ab02, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)), 4), K8_0F);
                ab03 = _mm_add_epi32(ab03, _mm_madd_epi16(_mm_maddubs_epi16(a0, b0), K16_0001));
            }
            for (; i < size; i += 8, o += 4)
            {
                a0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(A[0] + o), C4_SHFL0), C4_MULLO), 12);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[0] + o), C4_SHFL0), C4_MULLO), 12);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab00);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[1] + o), C4_SHFL0), C4_MULLO), 12);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab01);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[2] + o), C4_SHFL0), C4_MULLO), 12);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab02);

                b0 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<4>(B[3] + o), C4_SHFL0), C4_MULLO), 12);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 10)
            {
                a01 = _mm_loadu_si128((__m128i*)(A[0] + o));
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C5_SHFL0), C5_MULLO), 11);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 5), C5_SHFL0), C5_MULLO), 11);

                b01 = _mm_loadu_si128((__m128i*)(B[0] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = _mm_loadu_si128((__m128i*)(B[1] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = _mm_loadu_si128((__m128i*)(B[2] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = _mm_loadu_si128((__m128i*)(B[3] + o));
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size16; i += 16, o += 10)
            {
                a01 = LoadLast16<5>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C5_SHFL0), C5_MULLO), 11);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 5), C5_SHFL0), C5_MULLO), 11);

                b01 = LoadLast16<5>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = LoadLast16<5>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = LoadLast16<5>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = LoadLast16<5>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 5), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 5)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(A[0] + o), C5_SHFL0), C5_MULLO), 11);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[0] + o), C5_SHFL0), C5_MULLO), 11);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[1] + o), C5_SHFL0), C5_MULLO), 11);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[2] + o), C5_SHFL0), C5_MULLO), 11);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<5>(B[3] + o), C5_SHFL0), C5_MULLO), 11);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 12)
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
            for (; i < size16; i += 16, o += 12)
            {
                a01 = LoadLast16<6>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C6_SHFL0), C6_MULLO), 10);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 6), C6_SHFL0), C6_MULLO), 10);

                b01 = LoadLast16<6>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = LoadLast16<6>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = LoadLast16<6>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = LoadLast16<6>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 6), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 6)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(A[0] + o), C6_SHFL0), C6_MULLO), 10);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[0] + o), C6_SHFL0), C6_MULLO), 10);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[1] + o), C6_SHFL0), C6_MULLO), 10);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[2] + o), C6_SHFL0), C6_MULLO), 10);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<6>(B[3] + o), C6_SHFL0), C6_MULLO), 10);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m128i a00, a01, b00, b01;
            __m128i ab00 = _mm_setzero_si128();
            __m128i ab01 = _mm_setzero_si128();
            __m128i ab02 = _mm_setzero_si128();
            __m128i ab03 = _mm_setzero_si128();
            for (; i < size16a; i += 16, o += 14)
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
            for (; i < size16; i += 16, o += 14)
            {
                a01 = LoadLast16<7>(A[0] + o);
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(a01, C7_SHFL0), C7_MULLO), 9);
                a01 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(a01, 7), C7_SHFL0), C7_MULLO), 9);

                b01 = LoadLast16<7>(B[0] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab00);

                b01 = LoadLast16<7>(B[1] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab01);

                b01 = LoadLast16<7>(B[2] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab02);

                b01 = LoadLast16<7>(B[3] + o);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(b01, C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_mm_srli_si128(b01, 7), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a01, b00), ab03);
            }
            for (; i < size; i += 8, o += 7)
            {
                a00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(A[0] + o), C7_SHFL0), C7_MULLO), 9);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[0] + o), C7_SHFL0), C7_MULLO), 9);
                ab00 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab00);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[1] + o), C7_SHFL0), C7_MULLO), 9);
                ab01 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab01);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[2] + o), C7_SHFL0), C7_MULLO), 9);
                ab02 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab02);

                b00 = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(LoadLast8<7>(B[3] + o), C7_SHFL0), C7_MULLO), 9);
                ab03 = _mm_add_epi32(_mm_madd_epi16(a00, b00), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
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
            DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLo(N, 4);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    MicroCosineDistancesDirect2x4<bits>(A + i, B + j, size, distances + j, stride);
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
                    MicroCosineDistancesDirect1x4<bits>(A + i, B + j, size, distances + j, stride);
                for (; j < N; j += 1)
                    CosineDistance<bits>(A[i], B[j], size, distances + j);
                distances += 1 * stride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth)
        {
            switch (depth)
            {
            case 4: return CosineDistance<4>;
            case 5: return CosineDistance<5>;
            case 6: return CosineDistance<6>;
            case 7: return CosineDistance<7>;
            case 8: return CosineDistance<8>;
            default: assert(0); return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth)
        {
            switch (depth)
            {
            case 4: return MacroCosineDistancesDirect<4>;
            case 5: return MacroCosineDistancesDirect<5>;
            case 6: return MacroCosineDistancesDirect<6>;
            case 7: return MacroCosineDistancesDirect<7>;
            case 8: return MacroCosineDistancesDirect<8>;
            default: assert(0); return NULL;
            }
        }
    }
#endif
}

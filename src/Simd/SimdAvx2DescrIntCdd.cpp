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
        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i ab32 = _mm256_setzero_si256();
            size_t i = 0, size64 = AlignLo(size, 64);
            for (; i < size64; i += 64, a += 32, b += 32)
            {
                __m256i _a = _mm256_loadu_si256((__m256i*)a);
                __m256i _b = _mm256_loadu_si256((__m256i*)b);
                __m256i ab16 = _mm256_maddubs_epi16(_mm256_and_si256(_a, K8_0F), _mm256_and_si256(_b, K8_0F));
                ab16 = _mm256_add_epi16(ab16, _mm256_maddubs_epi16(_mm256_and_si256(_mm256_srli_epi16(_a, 4), K8_0F), _mm256_and_si256(_mm256_srli_epi16(_b, 4), K8_0F)));
                ab32 = _mm256_add_epi32(ab32, _mm256_madd_epi16(ab16, K16_0001));
            }
            for (; i < size; i += 8, a += 4, b += 4)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<4>(a), Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<4>(b), Sse41::C4_SHFL0), Sse41::C4_MULLO), 12);
                ab32 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), ab32);
            }
            return ExtractSum<uint32_t>(ab32);
        }

        template<> int32_t Correlation<5>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i _ab = _mm256_setzero_si256();
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16);
            for (; i < size16a; i += 16, a += 10, b += 10)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)a)), C5_SHFL), C5_MULLO), 11);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)b)), C5_SHFL), C5_MULLO), 11);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size16; i += 16, a += 10, b += 10)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(a)), C5_SHFL), C5_MULLO), 11);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(b)), C5_SHFL), C5_MULLO), 11);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8, a += 5, b += 5)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<5>(a), Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<5>(b), Sse41::C5_SHFL0), Sse41::C5_MULLO), 11);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<6>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i _ab = _mm256_setzero_si256();
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16);
            for (; i < size16; i += 16, a += 12, b += 12)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)a)), C6_SHFL), C6_MULLO), 10);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)b)), C6_SHFL), C6_MULLO), 10);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size16a; i += 16, a += 12, b += 12)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(a)), C6_SHFL), C6_MULLO), 10);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(b)), C6_SHFL), C6_MULLO), 10);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8, a += 6, b += 6)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<6>(a), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<6>(b), Sse41::C6_SHFL0), Sse41::C6_MULLO), 10);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<7>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m256i _ab = _mm256_setzero_si256();
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16);
            for (; i < size16a; i += 16, a += 14, b += 14)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)a)), C7_SHFL), C7_MULLO), 9);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)b)), C7_SHFL), C7_MULLO), 9);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size16; i += 16, a += 14, b += 14)
            {
                __m256i _a = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(a)), C7_SHFL), C7_MULLO), 9);
                __m256i _b = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(b)), C7_SHFL), C7_MULLO), 9);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8, a += 7, b += 7)
            {
                __m128i _a = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<7>(a), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                __m128i _b = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(Sse41::LoadLast8<7>(b), Sse41::C7_SHFL0), Sse41::C7_MULLO), 9);
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_mm256_castsi128_si256(_a), _mm256_castsi128_si256(_b)), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
        }

        template<> int32_t Correlation<8>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            size_t i = 0, size16 = AlignLo(size, 16);
            __m256i _ab = _mm256_setzero_si256();
            for (; i < size16; i += 16)
            {
                __m256i _a = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(a + i)));
                __m256i _b = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(b + i)));
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            for (; i < size; i += 8)
            {
                __m256i _a = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(a + i)));
                __m256i _b = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(b + i)));
                _ab = _mm256_add_epi32(_mm256_madd_epi16(_a, _b), _ab);
            }
            return ExtractSum<uint32_t>(_ab);
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
            size_t i = 0, size64 = AlignLo(size, 64), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size64; i += 64, o += 32)
            {
                a0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(A[0] + o)), K8_0F);
                a1 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(A[1] + o)), K8_0F);

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[0] + o)), K8_0F);
                ab00 = _mm256_add_epi32(ab00, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab10 = _mm256_add_epi32(ab10, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[1] + o)), K8_0F);
                ab01 = _mm256_add_epi32(ab01, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab11 = _mm256_add_epi32(ab11, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[2] + o)), K8_0F);
                ab02 = _mm256_add_epi32(ab02, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab12 = _mm256_add_epi32(ab12, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[3] + o)), K8_0F);
                ab03 = _mm256_add_epi32(ab03, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab13 = _mm256_add_epi32(ab13, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                a0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(A[0] + o)), 4), K8_0F);
                a1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(A[1] + o)), 4), K8_0F);

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[0] + o)), 4), K8_0F);
                ab00 = _mm256_add_epi32(ab00, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab10 = _mm256_add_epi32(ab10, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)), 4), K8_0F);
                ab01 = _mm256_add_epi32(ab01, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab11 = _mm256_add_epi32(ab11, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)), 4), K8_0F);
                ab02 = _mm256_add_epi32(ab02, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab12 = _mm256_add_epi32(ab12, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)), 4), K8_0F);
                ab03 = _mm256_add_epi32(ab03, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
                ab13 = _mm256_add_epi32(ab13, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, b0), K16_0001));
            }
            for (; i < size; i += 8, o += 4)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(A[0] + o)), C4_SHFL), C4_MULLO), 12);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(A[1] + o)), C4_SHFL), C4_MULLO), 12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[0] + o)), C4_SHFL), C4_MULLO), 12);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[1] + o)), C4_SHFL), C4_MULLO), 12);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[2] + o)), C4_SHFL), C4_MULLO), 12);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[3] + o)), C4_SHFL), C4_MULLO), 12);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m256 ab = _mm256_cvtepi32_ps(Extract8Sums(ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13));
            DecodeCosineDistances2x4(A, B, ab, distances, stride);
        }

        template<> void MicroCosineDistancesDirect2x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 10)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C5_SHFL), C5_MULLO), 11);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[1] + o))), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size16; i += 16, o += 10)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(A[0] + o)), C5_SHFL), C5_MULLO), 11);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(A[1] + o)), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[0] + o)), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[1] + o)), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[2] + o)), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[3] + o)), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 5)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(A[0] + o)), C5_SHFL), C5_MULLO), 11);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(A[1] + o)), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[0] + o)), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[1] + o)), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[2] + o)), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[3] + o)), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m256 ab = _mm256_cvtepi32_ps(Extract8Sums(ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13));
            DecodeCosineDistances2x4(A, B, ab, distances, stride);
        }

        template<> void MicroCosineDistancesDirect2x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[1] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size16; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(A[0] + o)), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(A[1] + o)), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[0] + o)), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[1] + o)), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[2] + o)), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[3] + o)), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 6)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(A[0] + o)), C6_SHFL), C6_MULLO), 10);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(A[1] + o)), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[0] + o)), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[1] + o)), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[2] + o)), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[3] + o)), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m256 ab = _mm256_cvtepi32_ps(Extract8Sums(ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13));
            DecodeCosineDistances2x4(A, B, ab, distances, stride);
        }

        template<> void MicroCosineDistancesDirect2x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[1] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size16; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(A[0] + o)), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(A[1] + o)), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[0] + o)), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[1] + o)), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[2] + o)), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[3] + o)), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 7)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(A[0] + o)), C7_SHFL), C7_MULLO), 9);
                a1 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(A[1] + o)), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[0] + o)), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[1] + o)), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[2] + o)), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[3] + o)), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m256 ab = _mm256_cvtepi32_ps(Extract8Sums(ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13));
            DecodeCosineDistances2x4(A, B, ab, distances, stride);
        }

        template<> void MicroCosineDistancesDirect2x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, a1, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            __m256i ab10 = _mm256_setzero_si256();
            __m256i ab11 = _mm256_setzero_si256();
            __m256i ab12 = _mm256_setzero_si256();
            __m256i ab13 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 16)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)));
                a1 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[1] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            for (; i < size; i += 8, o += 8)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));
                a1 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[1] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);
                ab10 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab10);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);
                ab11 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab11);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);
                ab12 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab12);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
                ab13 = _mm256_add_epi32(_mm256_madd_epi16(a1, b0), ab13);
            }
            __m256 ab = _mm256_cvtepi32_ps(Extract8Sums(ab00, ab01, ab02, ab03, ab10, ab11, ab12, ab13));
            DecodeCosineDistances2x4(A, B, ab, distances, stride);
        }

        template<int bits> void MicroCosineDistancesDirect1x4(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);

        template<> void MicroCosineDistancesDirect1x4<4>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size64 = AlignLo(size, 64), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size64; i += 64, o += 32)
            {
                a0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(A[0] + o)), K8_0F);

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[0] + o)), K8_0F);
                ab00 = _mm256_add_epi32(ab00, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[1] + o)), K8_0F);
                ab01 = _mm256_add_epi32(ab01, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[2] + o)), K8_0F);
                ab02 = _mm256_add_epi32(ab02, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_loadu_si256((__m256i*)(B[3] + o)), K8_0F);
                ab03 = _mm256_add_epi32(ab03, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                a0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(A[0] + o)), 4), K8_0F);

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[0] + o)), 4), K8_0F);
                ab00 = _mm256_add_epi32(ab00, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)), 4), K8_0F);
                ab01 = _mm256_add_epi32(ab01, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)), 4), K8_0F);
                ab02 = _mm256_add_epi32(ab02, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));

                b0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)), 4), K8_0F);
                ab03 = _mm256_add_epi32(ab03, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, b0), K16_0001));
            }
            for (; i < size; i += 8, o += 4)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(A[0] + o)), C4_SHFL), C4_MULLO), 12);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[0] + o)), C4_SHFL), C4_MULLO), 12);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[1] + o)), C4_SHFL), C4_MULLO), 12);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[2] + o)), C4_SHFL), C4_MULLO), 12);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<4>(B[3] + o)), C4_SHFL), C4_MULLO), 12);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<5>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 10)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size16; i += 16, o += 10)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(A[0] + o)), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[0] + o)), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[1] + o)), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[2] + o)), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<5>(B[3] + o)), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 5)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(A[0] + o)), C5_SHFL), C5_MULLO), 11);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[0] + o)), C5_SHFL), C5_MULLO), 11);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[1] + o)), C5_SHFL), C5_MULLO), 11);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[2] + o)), C5_SHFL), C5_MULLO), 11);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<5>(B[3] + o)), C5_SHFL), C5_MULLO), 11);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<6>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size16; i += 16, o += 12)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(A[0] + o)), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[0] + o)), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[1] + o)), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[2] + o)), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<6>(B[3] + o)), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 6)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(A[0] + o)), C6_SHFL), C6_MULLO), 10);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[0] + o)), C6_SHFL), C6_MULLO), 10);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[1] + o)), C6_SHFL), C6_MULLO), 10);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[2] + o)), C6_SHFL), C6_MULLO), 10);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<6>(B[3] + o)), C6_SHFL), C6_MULLO), 10);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<7>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), size16a = AlignLo(size - 1, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16a; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(A[0] + o))), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[0] + o))), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[1] + o))), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[2] + o))), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i*)(B[3] + o))), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size16; i += 16, o += 14)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(A[0] + o)), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[0] + o)), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[1] + o)), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[2] + o)), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_broadcastsi128_si256(Sse41::LoadLast16<7>(B[3] + o)), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 7)
            {
                a0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(A[0] + o)), C7_SHFL), C7_MULLO), 9);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[0] + o)), C7_SHFL), C7_MULLO), 9);
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[1] + o)), C7_SHFL), C7_MULLO), 9);
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[2] + o)), C7_SHFL), C7_MULLO), 9);
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_srli_epi16(_mm256_mullo_epi16(_mm256_shuffle_epi8(_mm256_castsi128_si256(Sse41::LoadLast8<7>(B[3] + o)), C7_SHFL), C7_MULLO), 9);
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<> void MicroCosineDistancesDirect1x4<8>(const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t i = 0, size16 = AlignLo(size, 16), o = 16;
            __m256i a0, b0;
            __m256i ab00 = _mm256_setzero_si256();
            __m256i ab01 = _mm256_setzero_si256();
            __m256i ab02 = _mm256_setzero_si256();
            __m256i ab03 = _mm256_setzero_si256();
            for (; i < size16; i += 16, o += 16)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(A[0] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_cvtepu8_epi16(_mm_loadu_si128((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            for (; i < size; i += 8, o += 8)
            {
                a0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(A[0] + o)));

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[0] + o)));
                ab00 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab00);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[1] + o)));
                ab01 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab01);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[2] + o)));
                ab02 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab02);

                b0 = _mm256_cvtepu8_epi16(_mm_loadl_epi64((__m128i*)(B[3] + o)));
                ab03 = _mm256_add_epi32(_mm256_madd_epi16(a0, b0), ab03);
            }
            __m128 ab0 = _mm_cvtepi32_ps(Extract4Sums(ab00, ab01, ab02, ab03));
            Sse41::DecodeCosineDistances1x4(A[0], B, ab0, distances + 0 * stride);
        }

        template<int bits> void MacroCosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLoAny(N, 4);
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

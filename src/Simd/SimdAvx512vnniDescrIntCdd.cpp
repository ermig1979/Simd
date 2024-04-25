/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifdef SIMD_AVX512VNNI_ENABLE    
    namespace Avx512vnni
    {
        template<int bits> int32_t Correlation(const uint8_t* a, const uint8_t* b, size_t size);

        template<> int32_t Correlation<4>(const uint8_t* a, const uint8_t* b, size_t size)
        {
            assert(size % 8 == 0);
            __m512i ab0 = _mm512_setzero_si512();
            __m512i ab1 = _mm512_setzero_si512();
            size_t i = 0, size128 = AlignLo(size, 128);
            for (; i < size128; i += 128, a += 64, b += 64)
            {
                __m512i _a = _mm512_loadu_si512((__m512i*)a);
                __m512i _b = _mm512_loadu_si512((__m512i*)b);
                ab0 = _mm512_dpbusd_epi32(ab0, _mm512_and_si512(_a, K8_0F), _mm512_and_si512(_b, K8_0F));
                ab1 = _mm512_dpbusd_epi32(ab1, _mm512_and_si512(_mm512_srli_epi16(_a, 4), K8_0F), _mm512_and_si512(_mm512_srli_epi16(_b, 4), K8_0F));
            }
            if (i < size)
            {
                __mmask16 mask = TailMask16((size - i) / 8);
                __m512i _a = _mm512_maskz_loadu_epi32(mask, a);
                __m512i _b = _mm512_maskz_loadu_epi32(mask, b);
                ab0 = _mm512_dpbusd_epi32(ab0, _mm512_and_si512(_a, K8_0F), _mm512_and_si512(_b, K8_0F));
                ab1 = _mm512_dpbusd_epi32(ab1, _mm512_and_si512(_mm512_srli_epi16(_a, 4), K8_0F), _mm512_and_si512(_mm512_srli_epi16(_b, 4), K8_0F));
            }
            return ExtractSum<uint32_t>(_mm512_add_epi32(ab0, ab1));
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
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                __m512i _a = Load5(a, mask);
                __m512i _b = Load5(b, mask);
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
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
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                __m512i _a = Load6(a, mask);
                __m512i _b = Load6(b, mask);
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
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
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                __m512i _a = Load7(a, mask);
                __m512i _b = Load7(b, mask);
                _ab = _mm512_dpwssd_epi32(_ab, _a, _b);
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
            if (i < size)
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
                ab00 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab00, a01, b01), a00, b00);
                ab10 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab10, a11, b01), a10, b00);
                ab20 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab20, a21, b01), a20, b00);
                ab30 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab30, a31, b01), a30, b00);

                b01 = _mm512_loadu_si512((__m512i*)(B[1] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab01, a01, b01), a00, b00);
                ab11 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab11, a11, b01), a10, b00);
                ab21 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab21, a21, b01), a20, b00);
                ab31 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab31, a31, b01), a30, b00);

                b01 = _mm512_loadu_si512((__m512i*)(B[2] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab02, a01, b01), a00, b00);
                ab12 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab12, a11, b01), a10, b00);
                ab22 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab22, a21, b01), a20, b00);
                ab32 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab32, a31, b01), a30, b00);

                b01 = _mm512_loadu_si512((__m512i*)(B[3] + o));
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab03, a01, b01), a00, b00);
                ab13 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab13, a11, b01), a10, b00);
                ab23 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab23, a21, b01), a20, b00);
                ab33 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab33, a31, b01), a30, b00);
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
                ab00 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab00, a01, b01), a00, b00);
                ab10 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab10, a11, b01), a10, b00);
                ab20 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab20, a21, b01), a20, b00);
                ab30 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab30, a31, b01), a30, b00);

                b01 = _mm512_maskz_loadu_epi32(mask, B[1] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab01 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab01, a01, b01), a00, b00);
                ab11 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab11, a11, b01), a10, b00);
                ab21 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab21, a21, b01), a20, b00);
                ab31 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab31, a31, b01), a30, b00);

                b01 = _mm512_maskz_loadu_epi32(mask, B[2] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab02 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab02, a01, b01), a00, b00);
                ab12 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab12, a11, b01), a10, b00);
                ab22 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab22, a21, b01), a20, b00);
                ab32 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab32, a31, b01), a30, b00);

                b01 = _mm512_maskz_loadu_epi32(mask, B[3] + o);
                b00 = _mm512_and_si512(b01, K8_0F);
                b01 = _mm512_and_si512(_mm512_srli_epi16(b01, 4), K8_0F);
                ab03 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab03, a01, b01), a00, b00);
                ab13 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab13, a11, b01), a10, b00);
                ab23 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab23, a21, b01), a20, b00);
                ab33 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(ab33, a31, b01), a30, b00);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load5(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load5(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load5(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                a0 = Load5(A[0] + o, mask);
                a1 = Load5(A[1] + o, mask);
                a2 = Load5(A[2] + o, mask);
                a3 = Load5(A[3] + o, mask);

                b0 = Load5(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load5(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load5(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load5(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load6(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load6(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load6(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                a0 = Load6(A[0] + o, mask);
                a1 = Load6(A[1] + o, mask);
                a2 = Load6(A[2] + o, mask);
                a3 = Load6(A[3] + o, mask);

                b0 = Load6(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load6(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load6(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load6(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load7(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load7(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load7(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                a0 = Load7(A[0] + o, mask);
                a1 = Load7(A[1] + o, mask);
                a2 = Load7(A[2] + o, mask);
                a3 = Load7(A[3] + o, mask);

                b0 = Load7(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = Load7(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = Load7(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = Load7(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)));
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)));
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)));
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32(size - i);
                a0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[0] + o));
                a1 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[1] + o));
                a2 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[2] + o));
                a3 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[3] + o));

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[0] + o));
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);
                ab10 = _mm512_dpwssd_epi32(ab10, a1, b0);
                ab20 = _mm512_dpwssd_epi32(ab20, a2, b0);
                ab30 = _mm512_dpwssd_epi32(ab30, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[1] + o));
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);
                ab11 = _mm512_dpwssd_epi32(ab11, a1, b0);
                ab21 = _mm512_dpwssd_epi32(ab21, a2, b0);
                ab31 = _mm512_dpwssd_epi32(ab31, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[2] + o));
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);
                ab12 = _mm512_dpwssd_epi32(ab12, a1, b0);
                ab22 = _mm512_dpwssd_epi32(ab22, a2, b0);
                ab32 = _mm512_dpwssd_epi32(ab32, a3, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[3] + o));
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
                ab13 = _mm512_dpwssd_epi32(ab13, a1, b0);
                ab23 = _mm512_dpwssd_epi32(ab23, a2, b0);
                ab33 = _mm512_dpwssd_epi32(ab33, a3, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load5(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load5(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load5(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 5);
                a0 = Load5(A[0] + o, mask);

                b0 = Load5(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load5(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load5(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load5(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load6(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load6(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load6(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 6);
                a0 = Load6(A[0] + o, mask);

                b0 = Load6(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load6(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load6(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load6(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load7(B[1] + o);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load7(B[2] + o);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load7(B[3] + o);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32((size - i) / 8 * 7);
                a0 = Load7(A[0] + o, mask);

                b0 = Load7(B[0] + o, mask);
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = Load7(B[1] + o, mask);
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = Load7(B[2] + o, mask);
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = Load7(B[3] + o, mask);
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
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
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[1] + o)));
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[2] + o)));
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)(B[3] + o)));
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
            }
            if (i < size)
            {
                __mmask32 mask = TailMask32(size - i);
                a0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, A[0] + o));

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[0] + o));
                ab00 = _mm512_dpwssd_epi32(ab00, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[1] + o));
                ab01 = _mm512_dpwssd_epi32(ab01, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[2] + o));
                ab02 = _mm512_dpwssd_epi32(ab02, a0, b0);

                b0 = _mm512_cvtepu8_epi16(_mm256_maskz_loadu_epi8(mask, B[3] + o));
                ab03 = _mm512_dpwssd_epi32(ab03, a0, b0);
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

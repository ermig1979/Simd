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
#ifndef __SimdDescrIntCommon_h__
#define __SimdDescrIntCommon_h__

#include "Simd/SimdDefs.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdStore.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void DecodeCosineDistance(const uint8_t* a, const uint8_t* b, float abSum, float* distance)
        {
            float aScale = ((float*)a)[0];
            float aShift = ((float*)a)[1];
            float aMean = ((float*)a)[2];
            float aNorm = ((float*)a)[3];
            float bScale = ((float*)b)[0];
            float bShift = ((float*)b)[1];
            float bMean = ((float*)b)[2];
            float bNorm = ((float*)b)[3];
            float ab = abSum * aScale * bScale + aMean * bShift + bMean * aShift;
            distance[0] = Simd::RestrictRange(1.0f - ab / (aNorm * bNorm), 0.0f, 2.0f);
        }
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i E4_MULLO = SIMD_MM_SETR_EPI16(4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m128i E5_MULLO = SIMD_MM_SETR_EPI16(256, 32, 4, 128, 16, 2, 64, 8);
        const __m128i E5_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E5_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E5_SHFL2 = SIMD_MM_SETR_EPI8( -1, 0x6,  -1, 0xC,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i E6_MULLO = SIMD_MM_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
        const __m128i E6_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E6_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i E7_MULLO = SIMD_MM_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
        const __m128i E7_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E7_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i C4_MULLO = SIMD_MM_SETR_EPI16(4096, 256, 4096, 256, 4096, 256, 4096, 256);
        const __m128i C4_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3);
        const __m128i C4_SHFL1 = SIMD_MM_SETR_EPI8(0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);

        const __m128i C5_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4);
        const __m128i C5_SHFL1 = SIMD_MM_SETR_EPI8(0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9);
        const __m128i C5_MULLO = SIMD_MM_SETR_EPI16(8, 64, 2, 16, 128, 4, 32, 256);

        const __m128i C6_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m128i C6_SHFL1 = SIMD_MM_SETR_EPI8(0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m128i C6_MULLO = SIMD_MM_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256);

        const __m128i C7_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m128i C7_SHFL1 = SIMD_MM_SETR_EPI8(0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m128i C7_MULLO = SIMD_MM_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        template<int bits> __m128i UnpackData8(const uint8_t* src);

        template<> SIMD_INLINE __m128i UnpackData8<4>(const uint8_t* src)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C4_SHFL0), C4_MULLO), 12);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<5>(const uint8_t* src)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C5_SHFL0), C5_MULLO), 11);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<6>(const uint8_t* src)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C6_SHFL0), C6_MULLO), 10);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        template<> SIMD_INLINE __m128i UnpackData8<7>(const uint8_t* src)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)src);
            __m128i lo = _mm_srli_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(_src, C7_SHFL0), C7_MULLO), 9);
            return _mm_packus_epi16(lo, K_ZERO);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1x4(const uint8_t* a, const uint8_t* const* B, __m128 abSum, float* distances)
        {
            __m128 aScale, aShift, aMean, aNorm, bScale, bShift, bMean, bNorm;
            bScale = _mm_loadu_ps((float*)B[0]);
            bShift = _mm_loadu_ps((float*)B[1]);
            bMean = _mm_loadu_ps((float*)B[2]);
            bNorm = _mm_loadu_ps((float*)B[3]);
            aScale = _mm_unpacklo_ps(bScale, bMean);
            aShift = _mm_unpacklo_ps(bShift, bNorm);
            aMean = _mm_unpackhi_ps(bScale, bMean);
            aNorm = _mm_unpackhi_ps(bShift, bNorm);
            bScale = _mm_unpacklo_ps(aScale, aShift);
            bShift = _mm_unpackhi_ps(aScale, aShift);
            bMean = _mm_unpacklo_ps(aMean, aNorm);
            bNorm = _mm_unpackhi_ps(aMean, aNorm);

            aScale = _mm_set1_ps(((float*)a)[0]);
            aShift = _mm_set1_ps(((float*)a)[1]);
            aMean = _mm_set1_ps(((float*)a)[2]);
            aNorm = _mm_set1_ps(((float*)a)[3]);

            __m128 ab = _mm_mul_ps(abSum, _mm_mul_ps(aScale, bScale));
            ab = _mm_add_ps(_mm_mul_ps(aMean, bShift), ab);
            ab = _mm_add_ps(_mm_mul_ps(bMean, aShift), ab);

            _mm_storeu_ps(distances, _mm_min_ps(_mm_max_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_div_ps(ab, _mm_mul_ps(aNorm, bNorm))), _mm_setzero_ps()), _mm_set1_ps(2.0f)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float *b, size_t stride, __m128i abSum, float* distances)
        {
            __m128 aScale = _mm_set1_ps(a[0]);
            __m128 aShift = _mm_set1_ps(a[1]);
            __m128 aMean = _mm_set1_ps(a[2]);
            __m128 aNorm = _mm_set1_ps(a[3]);
            __m128 bScale = _mm_loadu_ps(b + 0 * stride);
            __m128 bShift = _mm_loadu_ps(b + 1 * stride);
            __m128 bMean = _mm_loadu_ps(b + 2 * stride);
            __m128 bNorm = _mm_loadu_ps(b + 3 * stride);
            __m128 ab = _mm_mul_ps(_mm_cvtepi32_ps(abSum), _mm_mul_ps(aScale, bScale));
            ab = _mm_add_ps(_mm_mul_ps(aMean, bShift), ab);
            ab = _mm_add_ps(_mm_mul_ps(bMean, aShift), ab);
            _mm_storeu_ps(distances, _mm_min_ps(_mm_max_ps(_mm_sub_ps(_mm_set1_ps(1.0f), _mm_div_ps(ab, _mm_mul_ps(aNorm, bNorm))), _mm_setzero_ps()), _mm_set1_ps(2.0f)));
        }

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m128i abSum, float* distances, size_t N)
        {
            float d[F];
            DecodeCosineDistances1xF(a, b, stride, abSum, d);
            for (size_t i = 0; i < N; ++i)
                distances[i] = d[i];
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        const __m256i E4_MULLO = SIMD_MM256_SETR_EPI16(4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m256i E5_MULLO = SIMD_MM256_SETR_EPI16(256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8);
        const __m256i E5_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1);
        const __m256i E5_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1);
        const __m256i E5_SHFL2 = SIMD_MM256_SETR_EPI8(
            -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1);

        const __m256i E6_MULLO = SIMD_MM256_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
        const __m256i E6_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
        const __m256i E6_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);

        const __m256i E7_MULLO = SIMD_MM256_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
        const __m256i E7_SHFL0 = SIMD_MM256_SETR_EPI8(
            0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
        const __m256i E7_SHFL1 = SIMD_MM256_SETR_EPI8(
            0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);

        const __m256i C4_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3,
            0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7, 0x7);
        const __m256i C4_MULLO = SIMD_MM256_SETR_EPI16(4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256);

        const __m256i C5_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x5, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9);
        const __m256i C5_MULLO = SIMD_MM256_SETR_EPI16(8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256);

        const __m256i C6_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m256i C6_MULLO = SIMD_MM256_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m256i C7_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m256i C7_MULLO = SIMD_MM256_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances2x4(const uint8_t* const* A, const uint8_t* const* B, __m256 abSum, float* distances, size_t stride)
        {
            __m256 aScale, aShift, aMean, aNorm, bScale, bShift, bMean, bNorm;
            bScale = _mm256_broadcast_ps((__m128*)B[0]);
            bShift = _mm256_broadcast_ps((__m128*)B[1]);
            bMean = _mm256_broadcast_ps((__m128*)B[2]);
            bNorm = _mm256_broadcast_ps((__m128*)B[3]);
            aScale = _mm256_unpacklo_ps(bScale, bMean);
            aShift = _mm256_unpacklo_ps(bShift, bNorm);
            aMean = _mm256_unpackhi_ps(bScale, bMean);
            aNorm = _mm256_unpackhi_ps(bShift, bNorm);
            bScale = _mm256_unpacklo_ps(aScale, aShift);
            bShift = _mm256_unpackhi_ps(aScale, aShift);
            bMean = _mm256_unpacklo_ps(aMean, aNorm);
            bNorm = _mm256_unpackhi_ps(aMean, aNorm);

            aNorm = Avx::Load<false>((float*)A[0], (float*)A[1]);
            aScale = Broadcast<0>(aNorm);
            aShift = Broadcast<1>(aNorm);
            aMean = Broadcast<2>(aNorm);
            aNorm = Broadcast<3>(aNorm);

            __m256 ab = _mm256_mul_ps(abSum, _mm256_mul_ps(aScale, bScale));
            ab = _mm256_fmadd_ps(aMean, bShift, ab);
            ab = _mm256_fmadd_ps(bMean, aShift, ab);

            Avx::Store<false>(distances + 0 * stride, distances + 1 * stride,
                _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(ab, _mm256_mul_ps(aNorm, bNorm))), _mm256_setzero_ps()), _mm256_set1_ps(2.0f)));
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m256i abSum, float* distances)
        {
            __m256 aScale = _mm256_set1_ps(a[0]);
            __m256 aShift = _mm256_set1_ps(a[1]);
            __m256 aMean = _mm256_set1_ps(a[2]);
            __m256 aNorm = _mm256_set1_ps(a[3]);
            __m256 bScale = _mm256_loadu_ps(b + 0 * stride);
            __m256 bShift = _mm256_loadu_ps(b + 1 * stride);
            __m256 bMean = _mm256_loadu_ps(b + 2 * stride);
            __m256 bNorm = _mm256_loadu_ps(b + 3 * stride);
            __m256 ab = _mm256_mul_ps(_mm256_cvtepi32_ps(abSum), _mm256_mul_ps(aScale, bScale));
            ab = _mm256_add_ps(_mm256_mul_ps(aMean, bShift), ab);
            ab = _mm256_add_ps(_mm256_mul_ps(bMean, aShift), ab);
            _mm256_storeu_ps(distances, _mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_div_ps(ab, _mm256_mul_ps(aNorm, bNorm))), _mm256_setzero_ps()), _mm256_set1_ps(2.0f)));
        }

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m256i abSum, float* distances, size_t N)
        {
            float d[F];
            DecodeCosineDistances1xF(a, b, stride, abSum, d);
            for (size_t i = 0; i < N; ++i)
                distances[i] = d[i];
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        const __m512i EX_PERM = SIMD_MM512_SETR_EPI64(0, 2, 1, 3, 4, 6, 5, 7);

        const __m512i E4_MULLO = SIMD_MM512_SETR_EPI16(
            4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1,
            4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1, 4096, 1);

        const __m512i E5_MULLO = SIMD_MM512_SETR_EPI16(
            256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8,
            256, 32, 4, 128, 16, 2, 64, 8, 256, 32, 4, 128, 16, 2, 64, 8);
        const __m512i E5_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD,
            -1, -1, -1, -1, -1, 0x1, 0x3, 0x7, 0x9, 0xD, -1, -1, -1, -1, -1, -1);
        const __m512i E5_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE,
            -1, -1, -1, -1, -1, 0x2, 0x4, 0x8, 0xA, 0xE, -1, -1, -1, -1, -1, -1);
        const __m512i E5_SHFL2 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1,
            -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1,
            -1, -1, -1, -1, -1, -1, 0x6, -1, 0xC, -1, -1, -1, -1, -1, -1, -1);

        const __m512i E6_MULLO = SIMD_MM512_SETR_EPI16(
            256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4,
            256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4, 256, 64, 16, 4);
        const __m512i E6_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD,
            -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1);
        const __m512i E6_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE,
            -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1);

        const __m512i E7_MULLO = SIMD_MM512_SETR_EPI16(
            256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2,
            256, 128, 64, 32, 16, 8, 4, 2, 256, 128, 64, 32, 16, 8, 4, 2);
        const __m512i E7_SHFL0 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1,
            0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD,
            -1, -1, -1, -1, -1, -1, -1, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1);
        const __m512i E7_SHFL1 = SIMD_MM512_SETR_EPI8(
            -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1,
            0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE,
            -1, -1, -1, -1, -1, -1, -1, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1);

        const __m512i C4_MULLO = SIMD_MM512_SETR_EPI16(
            4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256,
            4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256, 4096, 256);

        const __m512i C5_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x2, 0x3, 0x0, 0x0, 0x3, 0x4, 0x0, 0x0);
        const __m512i C5_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x4,
            0x1, 0x1, 0x1, 0x2, 0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x3, 0x3, 0x3, 0x4, 0x4, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C5_MULLO = SIMD_MM512_SETR_EPI16(
            8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256,
            8, 64, 2, 16, 128, 4, 32, 256, 8, 64, 2, 16, 128, 4, 32, 256);

        const __m512i C6_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x0, 0x0, 0x3, 0x4, 0x0, 0x0, 0x4, 0x5, 0x0, 0x0);
        const __m512i C6_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7,
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x4, 0x5, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C6_MULLO = SIMD_MM512_SETR_EPI16(
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256,
            4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m512i C7_PERM = SIMD_MM512_SETR_EPI32(
            0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x3, 0x0, 0x3, 0x4, 0x5, 0x0, 0x5, 0x6, 0x0, 0x0);
        const __m512i C7_SHFL = SIMD_MM512_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0x9,
            0x2, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8,
            0x1, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x7, 0x7, 0x7);
        const __m512i C7_MULLO = SIMD_MM512_SETR_EPI16(
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256,
            2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances1xF(const float* a, const float* b, size_t stride, __m512i abSum, float* distances, __mmask16 mask = -1)
        {
            __m512 aScale = _mm512_set1_ps(a[0]);
            __m512 aShift = _mm512_set1_ps(a[1]);
            __m512 aMean = _mm512_set1_ps(a[2]);
            __m512 aNorm = _mm512_set1_ps(a[3]);
            __m512 bScale = _mm512_maskz_loadu_ps(mask, b + 0 * stride);
            __m512 bShift = _mm512_maskz_loadu_ps(mask, b + 1 * stride);
            __m512 bMean = _mm512_maskz_loadu_ps(mask, b + 2 * stride);
            __m512 bNorm = _mm512_maskz_loadu_ps(mask, b + 3 * stride);
            __m512 ab = _mm512_mul_ps(_mm512_cvtepi32_ps(abSum), _mm512_mul_ps(aScale, bScale));
            ab = _mm512_add_ps(_mm512_mul_ps(aMean, bShift), ab);
            ab = _mm512_add_ps(_mm512_mul_ps(bMean, aShift), ab);
            _mm512_mask_storeu_ps(distances, mask, _mm512_min_ps(_mm512_max_ps(_mm512_sub_ps(_mm512_set1_ps(1.0f), _mm512_div_ps(ab, _mm512_mul_ps(aNorm, bNorm))), _mm512_setzero_ps()), _mm512_set1_ps(2.0f)));
        }
    }
#endif
}
#endif//__SimdDescrIntCommon_h__

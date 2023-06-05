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

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void DecodeCosineDistance(const uint8_t* a, const uint8_t* b, float abSum, float size, float* distance)
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
        const __m128i E6_MULLO = SIMD_MM_SETR_EPI16(256, 64, 16, 4, 256, 64, 16, 4);
        const __m128i E6_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E6_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i E7_MULLO = SIMD_MM_SETR_EPI16(256, 128, 64, 32, 16, 8, 4, 2);
        const __m128i E7_SHFL0 = SIMD_MM_SETR_EPI8(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i E7_SHFL1 = SIMD_MM_SETR_EPI8(0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m128i C6_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m128i C6_SHFL1 = SIMD_MM_SETR_EPI8(0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m128i C6_MULLO = SIMD_MM_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256);

        const __m128i C7_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m128i C7_SHFL1 = SIMD_MM_SETR_EPI8(0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m128i C7_MULLO = SIMD_MM_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances(const uint8_t* a, const uint8_t* const* B, __m128 abSum, __m128 size, float* distances)
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
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
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

        const __m256i C6_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5,
            0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m256i C6_MULLO = SIMD_MM256_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256, 4, 16, 64, 256);

        const __m256i C7_SHFL = SIMD_MM256_SETR_EPI8(
            0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6,
            0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m256i C7_MULLO = SIMD_MM256_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
    }
#endif
}
#endif//__SimdDescrIntCommon_h__

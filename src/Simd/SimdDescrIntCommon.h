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

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE void DecodeCosineDistance(const uint8_t* a, const uint8_t* b, float abSum, float size, float* distance)
        {
            float aScale = ((float*)a)[0];
            float aShift = ((float*)a)[1];
            float aSum = ((float*)a)[2];
            float aaSum = ((float*)a)[3];
            float bScale = ((float*)b)[0];
            float bShift = ((float*)b)[1];
            float bSum = ((float*)b)[2];
            float bbSum = ((float*)b)[3];
            float aa = aaSum * aScale * aScale + aSum * aScale * aShift * 2.0f + size * aShift * aShift;
            float ab = abSum * aScale * bScale + aSum * aScale * bShift + bSum * bScale * aShift + size * aShift * bShift;
            float bb = bbSum * bScale * bScale + bSum * bScale * bShift * 2.0f + size * bShift * bShift;
            distance[0] = 1.0f - ab / ::sqrt(aa * bb);
        }
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        const __m128i C6_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x2, 0x3, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x5);
        const __m128i C6_SHFL1 = SIMD_MM_SETR_EPI8(0x6, 0x6, 0x6, 0x7, 0x7, 0x8, 0x8, 0x8, 0x9, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xB);
        const __m128i C6_MULLO = SIMD_MM_SETR_EPI16(4, 16, 64, 256, 4, 16, 64, 256);

        const __m128i C7_SHFL0 = SIMD_MM_SETR_EPI8(0x0, 0x0, 0x0, 0x1, 0x1, 0x2, 0x2, 0x3, 0x3, 0x4, 0x4, 0x5, 0x5, 0x6, 0x6, 0x6);
        const __m128i C7_SHFL1 = SIMD_MM_SETR_EPI8(0x7, 0x7, 0x7, 0x8, 0x8, 0x9, 0x9, 0xA, 0xA, 0xB, 0xB, 0xC, 0xC, 0xD, 0xD, 0xD);
        const __m128i C7_MULLO = SIMD_MM_SETR_EPI16(2, 4, 8, 16, 32, 64, 128, 256);

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void DecodeCosineDistances(const uint8_t* a, const uint8_t* const* B, __m128 abSum, __m128 size, float* distances)
        {
            __m128 aScale, aShift, aSum, aaSum, bScale, bShift, bSum, bbSum;
            bScale = _mm_loadu_ps((float*)B[0]);
            bShift = _mm_loadu_ps((float*)B[1]);
            bSum = _mm_loadu_ps((float*)B[2]);
            bbSum = _mm_loadu_ps((float*)B[3]);
            aScale = _mm_unpacklo_ps(bScale, bSum);
            aShift = _mm_unpacklo_ps(bShift, bbSum);
            aSum = _mm_unpackhi_ps(bScale, bSum);
            aaSum = _mm_unpackhi_ps(bShift, bbSum);
            bScale = _mm_unpacklo_ps(aScale, aShift);
            bShift = _mm_unpackhi_ps(aScale, aShift);
            bSum = _mm_unpacklo_ps(aSum, aaSum);
            bbSum = _mm_unpackhi_ps(aSum, aaSum);

            aScale = _mm_set1_ps(((float*)a)[0]);
            aShift = _mm_set1_ps(((float*)a)[1]);
            aSum = _mm_set1_ps(((float*)a)[2]);
            aaSum = _mm_set1_ps(((float*)a)[3]);

            __m128 _2 = _mm_set1_ps(2.0);
            __m128 aa = _mm_mul_ps(aaSum, _mm_mul_ps(aScale, aScale));
            aa = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_2, aSum), _mm_mul_ps(aScale, aShift)), aa);
            aa = _mm_add_ps(_mm_mul_ps(size, _mm_mul_ps(aShift, aShift)), aa);

            __m128 ab = _mm_mul_ps(abSum, _mm_mul_ps(aScale, bScale));
            ab = _mm_add_ps(_mm_mul_ps(aSum, _mm_mul_ps(aScale, bShift)), ab);
            ab = _mm_add_ps(_mm_mul_ps(bSum, _mm_mul_ps(bScale, aShift)), ab);
            ab = _mm_add_ps(_mm_mul_ps(size, _mm_mul_ps(aShift, bShift)), ab);

            __m128 bb = _mm_mul_ps(bbSum, _mm_mul_ps(bScale, bScale));
            bb = _mm_add_ps(_mm_mul_ps(_mm_mul_ps(_2, bSum), _mm_mul_ps(bScale, bShift)), bb);
            bb = _mm_add_ps(_mm_mul_ps(size, _mm_mul_ps(bShift, bShift)), bb);

            _mm_storeu_ps(distances, _mm_sub_ps(_mm_set1_ps(1.0), _mm_div_ps(ab, _mm_sqrt_ps(_mm_mul_ps(bb, aa)))));
        }
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
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

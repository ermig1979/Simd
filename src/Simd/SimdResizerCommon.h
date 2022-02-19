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
#ifndef __SimdResizerCommon_h__
#define __SimdResizerCommon_h__

#include "Simd/SimdLoad.h"

namespace Simd
{
    namespace Base
    {
        template<int N, int F, int L> SIMD_INLINE int32_t CubicSumX(const uint8_t* src, const int8_t* ax)
        {
            return (int)ax[0] * src[F * N] + (int)ax[1] * src[0 * N] + (int)ax[2] * src[1 * N] + (int)ax[3] * src[L * N];
        }

        template<int N, int F, int L> SIMD_INLINE void BicubicInt(const uint8_t* src0, const uint8_t* src1,
            const uint8_t* src2, const uint8_t* src3, size_t sx, const int8_t* ax, const int32_t* ay, uint8_t* dst)
        {
            for (size_t c = 0; c < N; ++c)
            {
                int32_t rs0 = CubicSumX<N, F, L>(src0 + sx + c, ax);
                int32_t rs1 = CubicSumX<N, F, L>(src1 + sx + c, ax);
                int32_t rs2 = CubicSumX<N, F, L>(src2 + sx + c, ax);
                int32_t rs3 = CubicSumX<N, F, L>(src3 + sx + c, ax);
                int32_t sum = ay[0] * rs0 + ay[1] * rs1 + ay[2] * rs2 + ay[3] * rs3;
                dst[c] = Base::RestrictRange((sum + Base::BICUBIC_ROUND) >> Base::BICUBIC_SHIFT, 0, 255);
            }
        }

        template<int N, int F, int L> SIMD_INLINE void PixelCubicSumX(const uint8_t* src, const int8_t* ax, int32_t* dst)
        {
            for (size_t c = 0; c < N; ++c)
                dst[c] = CubicSumX<N, F, L>(src + c, ax);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i RSB_1_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, -1, -1, 0x4, 0x5, -1, -1, 0x8, 0x9, -1, -1, 0xC, 0xD, -1, -1);
        const __m128i RSB_1_1 = SIMD_MM_SETR_EPI8(0x2, 0x3, -1, -1, 0x6, 0x7, -1, -1, 0xA, 0xB, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m128 BilColS1(const uint16_t* src, const int32_t* idx, __m128 fx0, __m128 fx1)
        {
            __m128i s = _mm_setr_epi32(
                *(uint32_t*)(src + idx[0]), *(uint32_t*)(src + idx[1]),
                *(uint32_t*)(src + idx[2]), *(uint32_t*)(src + idx[3]));
            __m128 m0 = _mm_mul_ps(fx0, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_1_0)));
            __m128 m1 = _mm_mul_ps(fx1, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_1_1)));
            return _mm_add_ps(m0, m1);
        }

        const __m128i RSB_2_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1);
        const __m128i RSB_2_1 = SIMD_MM_SETR_EPI8(0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m128 BilColS2(const uint16_t* src, const int32_t* idx, __m128 fx0, __m128 fx1)
        {
            __m128i s = Sse2::Load((__m128i*)(src + idx[0]), (__m128i*)(src + idx[2]));
            __m128 m0 = _mm_mul_ps(fx0, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_2_0)));
            __m128 m1 = _mm_mul_ps(fx1, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_2_1)));
            return _mm_add_ps(m0, m1);
        }

        const __m128i RSB_3_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, -1, -1, -1, -1);
        const __m128i RSB_3_1 = SIMD_MM_SETR_EPI8(0x6, 0x7, -1, -1, 0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1);

        SIMD_INLINE __m128 BilColS3(const uint16_t* src, __m128 fx0, __m128 fx1)
        {
            __m128i s = _mm_loadu_si128((__m128i*)src);
            __m128 m0 = _mm_mul_ps(fx0, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_3_0)));
            __m128 m1 = _mm_mul_ps(fx1, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_3_1)));
            return _mm_add_ps(m0, m1);
        }

        const __m128i RSB_4_0 = SIMD_MM_SETR_EPI8(0x0, 0x1, -1, -1, 0x2, 0x3, -1, -1, 0x4, 0x5, -1, -1, 0x6, 0x7, -1, -1);
        const __m128i RSB_4_1 = SIMD_MM_SETR_EPI8(0x8, 0x9, -1, -1, 0xA, 0xB, -1, -1, 0xC, 0xD, -1, -1, 0xE, 0xF, -1, -1);

        SIMD_INLINE __m128 BilColS4(const uint16_t* src, __m128 fx0, __m128 fx1)
        {
            __m128i s = _mm_loadu_si128((__m128i*)src);
            __m128 m0 = _mm_mul_ps(fx0, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_4_0)));
            __m128 m1 = _mm_mul_ps(fx1, _mm_cvtepi32_ps(_mm_shuffle_epi8(s, RSB_4_1)));
            return _mm_add_ps(m0, m1);
        }

        const __m128i RSB_3_P = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x8, 0x9, 0xA, 0xB, 0xC, 0xD, -1, -1, -1, -1);
    }
#endif //SIMD_SSE41_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <class Idx> SIMD_INLINE void ResizerByteBilinearLoadGrayInterpolated(const uint8_t * src, const Idx & index, const uint8_t * alpha, uint8_t * dst)
        {
            __m256i _src = _mm256_loadu_si256((__m256i*)(src + index.src));
            __m256i _shuffle = _mm256_loadu_si256((__m256i*)&index.shuffle);
            __m256i _alpha = _mm256_loadu_si256((__m256i*)(alpha + index.dst));
            _mm256_storeu_si256((__m256i*)(dst + index.dst), _mm256_maddubs_epi16(Avx2::Shuffle(_src, _shuffle), _alpha));
        }
    }
#endif //SIMD_AVX2_ENABLE 
}
#endif//__SimdResizerCommon_h__

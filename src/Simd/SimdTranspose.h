/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#ifndef __SimdTranspose_h__
#define __SimdTranspose_h__

#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE_ENABLE
    namespace Sse
    {
        template <bool align> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            Store<align>(dst, Load<align>(src));
        }

        template<bool align> SIMD_INLINE void Transpose4x4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m128 s0 = Load<align>(src + 0 * srcStride);
            __m128 s1 = Load<align>(src + 1 * srcStride);
            __m128 s2 = Load<align>(src + 2 * srcStride);
            __m128 s3 = Load<align>(src + 3 * srcStride);
            __m128 s00 = _mm_unpacklo_ps(s0, s2);
            __m128 s01 = _mm_unpacklo_ps(s1, s3);
            __m128 s10 = _mm_unpackhi_ps(s0, s2);
            __m128 s11 = _mm_unpackhi_ps(s1, s3);
            Store<align>(dst + 0 * dstStride, _mm_unpacklo_ps(s00, s01));
            Store<align>(dst + 1 * dstStride, _mm_unpackhi_ps(s00, s01));
            Store<align>(dst + 2 * dstStride, _mm_unpacklo_ps(s10, s11));
            Store<align>(dst + 3 * dstStride, _mm_unpackhi_ps(s10, s11));
        }

        template<bool align> SIMD_INLINE void Transpose4x4xF(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m128 buf00 = Load<align>(src + 0 * F);
            __m128 buf01 = Load<align>(src + 1 * F);
            __m128 buf02 = Load<align>(src + 2 * F);
            __m128 buf03 = Load<align>(src + 3 * F);
            src += srcStride;
            __m128 buf10 = Load<align>(src + 0 * F);
            __m128 buf11 = Load<align>(src + 1 * F);
            __m128 buf12 = Load<align>(src + 2 * F);
            __m128 buf13 = Load<align>(src + 3 * F);
            src += srcStride;
            __m128 buf20 = Load<align>(src + 0 * F);
            __m128 buf21 = Load<align>(src + 1 * F);
            __m128 buf22 = Load<align>(src + 2 * F);
            __m128 buf23 = Load<align>(src + 3 * F);
            src += srcStride;
            __m128 buf30 = Load<align>(src + 0 * F);
            __m128 buf31 = Load<align>(src + 1 * F);
            __m128 buf32 = Load<align>(src + 2 * F);
            __m128 buf33 = Load<align>(src + 3 * F);
            Store<align>(dst + 0 * F, buf00);
            Store<align>(dst + 1 * F, buf10);
            Store<align>(dst + 2 * F, buf20);
            Store<align>(dst + 3 * F, buf30);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf01);
            Store<align>(dst + 1 * F, buf11);
            Store<align>(dst + 2 * F, buf21);
            Store<align>(dst + 3 * F, buf31);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf02);
            Store<align>(dst + 1 * F, buf12);
            Store<align>(dst + 2 * F, buf22);
            Store<align>(dst + 3 * F, buf32);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf03);
            Store<align>(dst + 1 * F, buf13);
            Store<align>(dst + 2 * F, buf23);
            Store<align>(dst + 3 * F, buf33);
        }
    }
#endif//SIMD_SSE_ENABLE

#ifdef SIMD_AVX_ENABLE
    namespace Avx
    {
        template <bool align> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            Store<align>(dst, Load<align>(src));
        }

        template<bool align> SIMD_INLINE void Transpose8x8(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = Load<align>(src + 0 * srcStride);
            a1 = Load<align>(src + 1 * srcStride);
            a2 = Load<align>(src + 2 * srcStride);
            a3 = Load<align>(src + 3 * srcStride);
            a4 = Load<align>(src + 4 * srcStride);
            a5 = Load<align>(src + 5 * srcStride);
            a6 = Load<align>(src + 6 * srcStride);
            a7 = Load<align>(src + 7 * srcStride);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);
            b4 = _mm256_unpacklo_ps(a4, a6);
            b5 = _mm256_unpacklo_ps(a5, a7);
            b6 = _mm256_unpackhi_ps(a4, a6);
            b7 = _mm256_unpackhi_ps(a5, a7);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);
            a4 = _mm256_unpacklo_ps(b4, b5);
            a5 = _mm256_unpackhi_ps(b4, b5);
            a6 = _mm256_unpacklo_ps(b6, b7);
            a7 = _mm256_unpackhi_ps(b6, b7);

            b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
            b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
            b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
            b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
            b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
            b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
            b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
            b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

            Store<align>(dst + 0 * dstStride, b0);
            Store<align>(dst + 1 * dstStride, b1);
            Store<align>(dst + 2 * dstStride, b2);
            Store<align>(dst + 3 * dstStride, b3);
            Store<align>(dst + 4 * dstStride, b4);
            Store<align>(dst + 5 * dstStride, b5);
            Store<align>(dst + 6 * dstStride, b6);
            Store<align>(dst + 7 * dstStride, b7);
        }

        template<bool align> SIMD_INLINE void Transpose4x4xF(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 buf00 = Load<align>(src + 0 * F);
            __m256 buf01 = Load<align>(src + 1 * F);
            __m256 buf02 = Load<align>(src + 2 * F);
            __m256 buf03 = Load<align>(src + 3 * F);
            src += srcStride;
            __m256 buf10 = Load<align>(src + 0 * F);
            __m256 buf11 = Load<align>(src + 1 * F);
            __m256 buf12 = Load<align>(src + 2 * F);
            __m256 buf13 = Load<align>(src + 3 * F);
            src += srcStride;
            __m256 buf20 = Load<align>(src + 0 * F);
            __m256 buf21 = Load<align>(src + 1 * F);
            __m256 buf22 = Load<align>(src + 2 * F);
            __m256 buf23 = Load<align>(src + 3 * F);
            src += srcStride;
            __m256 buf30 = Load<align>(src + 0 * F);
            __m256 buf31 = Load<align>(src + 1 * F);
            __m256 buf32 = Load<align>(src + 2 * F);
            __m256 buf33 = Load<align>(src + 3 * F);
            Store<align>(dst + 0 * F, buf00);
            Store<align>(dst + 1 * F, buf10);
            Store<align>(dst + 2 * F, buf20);
            Store<align>(dst + 3 * F, buf30);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf01);
            Store<align>(dst + 1 * F, buf11);
            Store<align>(dst + 2 * F, buf21);
            Store<align>(dst + 3 * F, buf31);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf02);
            Store<align>(dst + 1 * F, buf12);
            Store<align>(dst + 2 * F, buf22);
            Store<align>(dst + 3 * F, buf32);
            dst += dstStride;
            Store<align>(dst + 0 * F, buf03);
            Store<align>(dst + 1 * F, buf13);
            Store<align>(dst + 2 * F, buf23);
            Store<align>(dst + 3 * F, buf33);
        }
    }
#endif

#ifdef SIMD_AVX512F_ENABLE
    namespace Avx512f
    {
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdTranspose_h__

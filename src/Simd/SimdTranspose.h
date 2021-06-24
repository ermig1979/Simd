/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
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
#endif//SIMD_SSE2_ENABLE

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

        template<bool align> SIMD_INLINE void Transpose8x4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = Load<align>(src + 0 * srcStride);
            a1 = Load<align>(src + 1 * srcStride);
            a2 = Load<align>(src + 2 * srcStride);
            a3 = Load<align>(src + 3 * srcStride);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);

            Store<align>(dst + 0 * dstStride, dst + 4 * dstStride, a0);
            Store<align>(dst + 1 * dstStride, dst + 5 * dstStride, a1);
            Store<align>(dst + 2 * dstStride, dst + 6 * dstStride, a2);
            Store<align>(dst + 3 * dstStride, dst + 7 * dstStride, a3);
        }

        template<bool align> SIMD_INLINE void Transpose4x8(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m256 a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = Load<align>(src + 0 * srcStride, src + 4 * srcStride);
            a1 = Load<align>(src + 1 * srcStride, src + 5 * srcStride);
            a2 = Load<align>(src + 2 * srcStride, src + 6 * srcStride);
            a3 = Load<align>(src + 3 * srcStride, src + 7 * srcStride);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);

            Store<align>(dst + 0 * dstStride, a0);
            Store<align>(dst + 1 * dstStride, a1);
            Store<align>(dst + 2 * dstStride, a2);
            Store<align>(dst + 3 * dstStride, a3);
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
        template <bool align> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            Store<align>(dst, Load<align>(src));
        }

        template <bool align, bool mask> SIMD_INLINE void Copy(const float * src, float * dst, __mmask16 tail = -1)
        {
            Store<align, mask>(dst, Load<align, mask>(src, tail), tail);
        }

        template <bool align> SIMD_INLINE void CopyZP(const float * src, float * dst, __mmask16 tail)
        {
            Store<align>(dst, Load<align, true>(src, tail));
        }

        template<bool align> SIMD_INLINE void Transpose16x16(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA, aB, aC, aD, aE, aF, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, bA, bB, bC, bD, bE, bF;

            a0 = Load<align>(src + 0x0 * srcStride);
            a1 = Load<align>(src + 0x1 * srcStride);
            a2 = Load<align>(src + 0x2 * srcStride);
            a3 = Load<align>(src + 0x3 * srcStride);
            a4 = Load<align>(src + 0x4 * srcStride);
            a5 = Load<align>(src + 0x5 * srcStride);
            a6 = Load<align>(src + 0x6 * srcStride);
            a7 = Load<align>(src + 0x7 * srcStride);
            a8 = Load<align>(src + 0x8 * srcStride);
            a9 = Load<align>(src + 0x9 * srcStride);
            aA = Load<align>(src + 0xA * srcStride);
            aB = Load<align>(src + 0xB * srcStride);
            aC = Load<align>(src + 0xC * srcStride);
            aD = Load<align>(src + 0xD * srcStride);
            aE = Load<align>(src + 0xE * srcStride);
            aF = Load<align>(src + 0xF * srcStride);

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);
            b4 = _mm512_unpacklo_ps(a4, a6);
            b5 = _mm512_unpacklo_ps(a5, a7);
            b6 = _mm512_unpackhi_ps(a4, a6);
            b7 = _mm512_unpackhi_ps(a5, a7);
            b8 = _mm512_unpacklo_ps(a8, aA);
            b9 = _mm512_unpacklo_ps(a9, aB);
            bA = _mm512_unpackhi_ps(a8, aA);
            bB = _mm512_unpackhi_ps(a9, aB);
            bC = _mm512_unpacklo_ps(aC, aE);
            bD = _mm512_unpacklo_ps(aD, aF);
            bE = _mm512_unpackhi_ps(aC, aE);
            bF = _mm512_unpackhi_ps(aD, aF);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);
            a4 = _mm512_unpacklo_ps(b4, b5);
            a5 = _mm512_unpackhi_ps(b4, b5);
            a6 = _mm512_unpacklo_ps(b6, b7);
            a7 = _mm512_unpackhi_ps(b6, b7);
            a8 = _mm512_unpacklo_ps(b8, b9);
            a9 = _mm512_unpackhi_ps(b8, b9);
            aA = _mm512_unpacklo_ps(bA, bB);
            aB = _mm512_unpackhi_ps(bA, bB);
            aC = _mm512_unpacklo_ps(bC, bD);
            aD = _mm512_unpackhi_ps(bC, bD);
            aE = _mm512_unpacklo_ps(bE, bF);
            aF = _mm512_unpackhi_ps(bE, bF);

            b0 = _mm512_shuffle_f32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_f32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_f32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_f32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_f32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_f32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_f32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_f32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_f32x4(a8, aC, 0x44);
            b9 = _mm512_shuffle_f32x4(a9, aD, 0x44);
            bA = _mm512_shuffle_f32x4(aA, aE, 0x44);
            bB = _mm512_shuffle_f32x4(aB, aF, 0x44);
            bC = _mm512_shuffle_f32x4(a8, aC, 0xEE);
            bD = _mm512_shuffle_f32x4(a9, aD, 0xEE);
            bE = _mm512_shuffle_f32x4(aA, aE, 0xEE);
            bF = _mm512_shuffle_f32x4(aB, aF, 0xEE);

            a0 = _mm512_shuffle_f32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_f32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_f32x4(b2, bA, 0x88);
            a3 = _mm512_shuffle_f32x4(b3, bB, 0x88);
            a4 = _mm512_shuffle_f32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_f32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_f32x4(b2, bA, 0xDD);
            a7 = _mm512_shuffle_f32x4(b3, bB, 0xDD);
            a8 = _mm512_shuffle_f32x4(b4, bC, 0x88);
            a9 = _mm512_shuffle_f32x4(b5, bD, 0x88);
            aA = _mm512_shuffle_f32x4(b6, bE, 0x88);
            aB = _mm512_shuffle_f32x4(b7, bF, 0x88);
            aC = _mm512_shuffle_f32x4(b4, bC, 0xDD);
            aD = _mm512_shuffle_f32x4(b5, bD, 0xDD);
            aE = _mm512_shuffle_f32x4(b6, bE, 0xDD);
            aF = _mm512_shuffle_f32x4(b7, bF, 0xDD);

            Store<align>(dst + 0x0 * dstStride, a0);
            Store<align>(dst + 0x1 * dstStride, a1);
            Store<align>(dst + 0x2 * dstStride, a2);
            Store<align>(dst + 0x3 * dstStride, a3);
            Store<align>(dst + 0x4 * dstStride, a4);
            Store<align>(dst + 0x5 * dstStride, a5);
            Store<align>(dst + 0x6 * dstStride, a6);
            Store<align>(dst + 0x7 * dstStride, a7);
            Store<align>(dst + 0x8 * dstStride, a8);
            Store<align>(dst + 0x9 * dstStride, a9);
            Store<align>(dst + 0xA * dstStride, aA);
            Store<align>(dst + 0xB * dstStride, aB);
            Store<align>(dst + 0xC * dstStride, aC);
            Store<align>(dst + 0xD * dstStride, aD);
            Store<align>(dst + 0xE * dstStride, aE);
            Store<align>(dst + 0xF * dstStride, aF);
        }

        template<bool align> SIMD_INLINE void Transpose8x16(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = Load<align>(src + 0x0 * srcStride, src + 0x4 * srcStride);
            a1 = Load<align>(src + 0x1 * srcStride, src + 0x5 * srcStride);
            a2 = Load<align>(src + 0x2 * srcStride, src + 0x6 * srcStride);
            a3 = Load<align>(src + 0x3 * srcStride, src + 0x7 * srcStride);
            a4 = Load<align>(src + 0x8 * srcStride, src + 0xC * srcStride);
            a5 = Load<align>(src + 0x9 * srcStride, src + 0xD * srcStride);
            a6 = Load<align>(src + 0xA * srcStride, src + 0xE * srcStride);
            a7 = Load<align>(src + 0xB * srcStride, src + 0xF * srcStride);

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);
            b4 = _mm512_unpacklo_ps(a4, a6);
            b5 = _mm512_unpacklo_ps(a5, a7);
            b6 = _mm512_unpackhi_ps(a4, a6);
            b7 = _mm512_unpackhi_ps(a5, a7);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);
            a4 = _mm512_unpacklo_ps(b4, b5);
            a5 = _mm512_unpackhi_ps(b4, b5);
            a6 = _mm512_unpacklo_ps(b6, b7);
            a7 = _mm512_unpackhi_ps(b6, b7);

            b0 = _mm512_shuffle_f32x4(a0, a4, 0x88);
            b1 = _mm512_shuffle_f32x4(a1, a5, 0x88);
            b2 = _mm512_shuffle_f32x4(a2, a6, 0x88);
            b3 = _mm512_shuffle_f32x4(a3, a7, 0x88);
            b4 = _mm512_shuffle_f32x4(a0, a4, 0xDD);
            b5 = _mm512_shuffle_f32x4(a1, a5, 0xDD);
            b6 = _mm512_shuffle_f32x4(a2, a6, 0xDD);
            b7 = _mm512_shuffle_f32x4(a3, a7, 0xDD);

            Store<align>(dst + 0x0 * dstStride, b0);
            Store<align>(dst + 0x1 * dstStride, b1);
            Store<align>(dst + 0x2 * dstStride, b2);
            Store<align>(dst + 0x3 * dstStride, b3);
            Store<align>(dst + 0x4 * dstStride, b4);
            Store<align>(dst + 0x5 * dstStride, b5);
            Store<align>(dst + 0x6 * dstStride, b6);
            Store<align>(dst + 0x7 * dstStride, b7);
        }

        template<bool align> SIMD_INLINE void Transpose4x16(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = Load<align>(src + 0x0 * srcStride, src + 0x4 * srcStride, src + 0x8 * srcStride, src + 0xC * srcStride);
            a1 = Load<align>(src + 0x1 * srcStride, src + 0x5 * srcStride, src + 0x9 * srcStride, src + 0xD * srcStride);
            a2 = Load<align>(src + 0x2 * srcStride, src + 0x6 * srcStride, src + 0xA * srcStride, src + 0xE * srcStride);
            a3 = Load<align>(src + 0x3 * srcStride, src + 0x7 * srcStride, src + 0xB * srcStride, src + 0xF * srcStride);

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);

            Store<align>(dst + 0x0 * dstStride, a0);
            Store<align>(dst + 0x1 * dstStride, a1);
            Store<align>(dst + 0x2 * dstStride, a2);
            Store<align>(dst + 0x3 * dstStride, a3);
        }

        template<bool align> SIMD_INLINE void Transpose16x8(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = Load<align>(src + 0x0 * srcStride);
            a1 = Load<align>(src + 0x1 * srcStride);
            a2 = Load<align>(src + 0x2 * srcStride);
            a3 = Load<align>(src + 0x3 * srcStride);
            a4 = Load<align>(src + 0x4 * srcStride);
            a5 = Load<align>(src + 0x5 * srcStride);
            a6 = Load<align>(src + 0x6 * srcStride);
            a7 = Load<align>(src + 0x7 * srcStride);

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);
            b4 = _mm512_unpacklo_ps(a4, a6);
            b5 = _mm512_unpacklo_ps(a5, a7);
            b6 = _mm512_unpackhi_ps(a4, a6);
            b7 = _mm512_unpackhi_ps(a5, a7);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);
            a4 = _mm512_unpacklo_ps(b4, b5);
            a5 = _mm512_unpackhi_ps(b4, b5);
            a6 = _mm512_unpacklo_ps(b6, b7);
            a7 = _mm512_unpackhi_ps(b6, b7);

            b0 = _mm512_shuffle_f32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_f32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_f32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_f32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_f32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_f32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_f32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_f32x4(a3, a7, 0xEE);

            a0 = _mm512_shuffle_f32x4(b0, b0, 0xD8);
            a1 = _mm512_shuffle_f32x4(b1, b1, 0xD8);
            a2 = _mm512_shuffle_f32x4(b2, b2, 0xD8);
            a3 = _mm512_shuffle_f32x4(b3, b3, 0xD8);
            a4 = _mm512_shuffle_f32x4(b4, b4, 0xD8);
            a5 = _mm512_shuffle_f32x4(b5, b5, 0xD8);
            a6 = _mm512_shuffle_f32x4(b6, b6, 0xD8);
            a7 = _mm512_shuffle_f32x4(b7, b7, 0xD8);

            Store<align>(dst + 0x0 * dstStride, dst + 0x4 * dstStride, a0);
            Store<align>(dst + 0x1 * dstStride, dst + 0x5 * dstStride, a1);
            Store<align>(dst + 0x2 * dstStride, dst + 0x6 * dstStride, a2);
            Store<align>(dst + 0x3 * dstStride, dst + 0x7 * dstStride, a3);
            Store<align>(dst + 0x8 * dstStride, dst + 0xC * dstStride, a4);
            Store<align>(dst + 0x9 * dstStride, dst + 0xD * dstStride, a5);
            Store<align>(dst + 0xA * dstStride, dst + 0xE * dstStride, a6);
            Store<align>(dst + 0xB * dstStride, dst + 0xF * dstStride, a7);
        }

        template<bool align> SIMD_INLINE void Transpose16x4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = Load<align>(src + 0x0 * srcStride);
            a1 = Load<align>(src + 0x1 * srcStride);
            a2 = Load<align>(src + 0x2 * srcStride);
            a3 = Load<align>(src + 0x3 * srcStride);

            b0 = _mm512_unpacklo_ps(a0, a2);
            b1 = _mm512_unpacklo_ps(a1, a3);
            b2 = _mm512_unpackhi_ps(a0, a2);
            b3 = _mm512_unpackhi_ps(a1, a3);

            a0 = _mm512_unpacklo_ps(b0, b1);
            a1 = _mm512_unpackhi_ps(b0, b1);
            a2 = _mm512_unpacklo_ps(b2, b3);
            a3 = _mm512_unpackhi_ps(b2, b3);

            Store<align>(dst + 0x0 * dstStride, dst + 0x4 * dstStride, dst + 0x8 * dstStride, dst + 0xC * dstStride, a0);
            Store<align>(dst + 0x1 * dstStride, dst + 0x5 * dstStride, dst + 0x9 * dstStride, dst + 0xD * dstStride, a1);
            Store<align>(dst + 0x2 * dstStride, dst + 0x6 * dstStride, dst + 0xA * dstStride, dst + 0xE * dstStride, a2);
            Store<align>(dst + 0x3 * dstStride, dst + 0x7 * dstStride, dst + 0xB * dstStride, dst + 0xF * dstStride, a3);
        }

        template<bool align> SIMD_INLINE void Transpose4x4xF(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            __m512 buf00 = Load<align>(src + 0 * F);
            __m512 buf01 = Load<align>(src + 1 * F);
            __m512 buf02 = Load<align>(src + 2 * F);
            __m512 buf03 = Load<align>(src + 3 * F);
            src += srcStride;
            __m512 buf10 = Load<align>(src + 0 * F);
            __m512 buf11 = Load<align>(src + 1 * F);
            __m512 buf12 = Load<align>(src + 2 * F);
            __m512 buf13 = Load<align>(src + 3 * F);
            src += srcStride;
            __m512 buf20 = Load<align>(src + 0 * F);
            __m512 buf21 = Load<align>(src + 1 * F);
            __m512 buf22 = Load<align>(src + 2 * F);
            __m512 buf23 = Load<align>(src + 3 * F);
            src += srcStride;
            __m512 buf30 = Load<align>(src + 0 * F);
            __m512 buf31 = Load<align>(src + 1 * F);
            __m512 buf32 = Load<align>(src + 2 * F);
            __m512 buf33 = Load<align>(src + 3 * F);
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
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        template <bool align> SIMD_INLINE void Copy(const float * src, float * dst)
        {
            Store<align>(dst, Load<align>(src));
        }

        template<bool align> SIMD_INLINE void Transpose4x4(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float32x4x2_t a0, a1, b0, b1;
            a0.val[0] = Load<align>(src + 0 * srcStride);
            a0.val[1] = Load<align>(src + 1 * srcStride);
            a1.val[0] = Load<align>(src + 2 * srcStride);
            a1.val[1] = Load<align>(src + 3 * srcStride);
            b0 = vzipq_f32(a0.val[0], a1.val[0]);
            b1 = vzipq_f32(a0.val[1], a1.val[1]);
            a0 = vzipq_f32(b0.val[0], b1.val[0]);
            a1 = vzipq_f32(b0.val[1], b1.val[1]);
            Store<align>(dst + 0 * dstStride, a0.val[0]);
            Store<align>(dst + 1 * dstStride, a0.val[1]);
            Store<align>(dst + 2 * dstStride, a1.val[0]);
            Store<align>(dst + 3 * dstStride, a1.val[1]);
        }

        template<bool align> SIMD_INLINE void Transpose4x4xF(const float * src, size_t srcStride, float * dst, size_t dstStride)
        {
            float32x4_t buf00 = Load<align>(src + 0 * F);
            float32x4_t buf01 = Load<align>(src + 1 * F);
            float32x4_t buf02 = Load<align>(src + 2 * F);
            float32x4_t buf03 = Load<align>(src + 3 * F);
            src += srcStride;
            float32x4_t buf10 = Load<align>(src + 0 * F);
            float32x4_t buf11 = Load<align>(src + 1 * F);
            float32x4_t buf12 = Load<align>(src + 2 * F);
            float32x4_t buf13 = Load<align>(src + 3 * F);
            src += srcStride;
            float32x4_t buf20 = Load<align>(src + 0 * F);
            float32x4_t buf21 = Load<align>(src + 1 * F);
            float32x4_t buf22 = Load<align>(src + 2 * F);
            float32x4_t buf23 = Load<align>(src + 3 * F);
            src += srcStride;
            float32x4_t buf30 = Load<align>(src + 0 * F);
            float32x4_t buf31 = Load<align>(src + 1 * F);
            float32x4_t buf32 = Load<align>(src + 2 * F);
            float32x4_t buf33 = Load<align>(src + 3 * F);
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
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdTranspose_h__

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
#ifndef __SimdTransform_h__
#define __SimdTransform_h__

#include "Simd/SimdCopyPixel.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdStore.h"

namespace Simd
{
    namespace Base
    {
        struct ImageTransforms
        {
            typedef void(*TransformPtr)(const uint8_t* src, ptrdiff_t srcStride, size_t width, size_t height, uint8_t* dst, ptrdiff_t dstStride);

            TransformPtr transforms[4][8];

            SIMD_INLINE void TransformImage(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t* dst, size_t dstStride)
            {
                assert(pixelSize >= 1 && pixelSize <= 4 && transform >= SimdTransformRotate0 && transform <= SimdTransformTransposeRotate270);

                transforms[pixelSize - 1][transform](src, srcStride, width, height, dst, dstStride);
            }

            ImageTransforms();
        };
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        const __m128i K8_MIRROR_1 = SIMD_MM_SETR_EPI8(0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0);
        const __m128i K8_MIRROR_2 = SIMD_MM_SETR_EPI8(0xE, 0xF, 0xC, 0xD, 0xA, 0xB, 0x8, 0x9, 0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);
        const __m128i K8_MIRROR_3_02 = SIMD_MM_SETR_EPI8(0xD, 0xE, 0xF, 0xA, 0xB, 0xC, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1);
        const __m128i K8_MIRROR_3_01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xE);
        const __m128i K8_MIRROR_3_12 = SIMD_MM_SETR_EPI8(-1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_MIRROR_3_11 = SIMD_MM_SETR_EPI8(0xF, -1, 0xB, 0xC, 0xD, 0x8, 0x9, 0xA, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0);
        const __m128i K8_MIRROR_3_10 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xF, -1);
        const __m128i K8_MIRROR_3_21 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_MIRROR_3_20 = SIMD_MM_SETR_EPI8(-1, 0xC, 0xD, 0xE, 0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2);
        const __m128i K8_MIRROR_4 = SIMD_MM_SETR_EPI8(0xC, 0xD, 0xE, 0xF, 0x8, 0x9, 0xA, 0xB, 0x4, 0x5, 0x6, 0x7, 0x0, 0x1, 0x2, 0x3);

        template<size_t N> SIMD_INLINE void TransformImageMirror16(const uint8_t* src, uint8_t* dst)
        {
            dst += (16 - 1) * N;
            for (size_t i = 0; i < 16; ++i)
                Base::CopyPixel<N>(src + i * N, dst - i * N);
        }

        template<> SIMD_INLINE void TransformImageMirror16<1>(const uint8_t* src, uint8_t* dst)
        {
            _mm_storeu_si128((__m128i*)dst, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src), K8_MIRROR_1));
        }

        template<> SIMD_INLINE void TransformImageMirror16<2>(const uint8_t* src, uint8_t* dst)
        {
            _mm_storeu_si128((__m128i*)dst + 1, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 0), K8_MIRROR_2));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 1), K8_MIRROR_2));
        }

        template<> SIMD_INLINE void TransformImageMirror16<3>(const uint8_t* src, uint8_t* dst)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)src + 0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src + 1);
            __m128i s2 = _mm_loadu_si128((__m128i*)src + 2);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_or_si128(_mm_shuffle_epi8(s2, K8_MIRROR_3_02), _mm_shuffle_epi8(s1, K8_MIRROR_3_01)));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s2, K8_MIRROR_3_12), _mm_shuffle_epi8(s1, K8_MIRROR_3_11)), _mm_shuffle_epi8(s0, K8_MIRROR_3_10)));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_or_si128(_mm_shuffle_epi8(s1, K8_MIRROR_3_21), _mm_shuffle_epi8(s0, K8_MIRROR_3_20)));
        }

        template<> SIMD_INLINE void TransformImageMirror16<4>(const uint8_t* src, uint8_t* dst)
        {
            _mm_storeu_si128((__m128i*)dst + 3, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 0), K8_MIRROR_4));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 1), K8_MIRROR_4));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 2), K8_MIRROR_4));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 3), K8_MIRROR_4));
        }

        template<size_t N> SIMD_INLINE void TransformImageMirror64(const uint8_t* src, uint8_t* dst)
        {
            TransformImageMirror16<N>(src + 0 * N * 16, dst - 0 * N * 16);
            TransformImageMirror16<N>(src + 1 * N * 16, dst - 1 * N * 16);
            TransformImageMirror16<N>(src + 2 * N * 16, dst - 2 * N * 16);
            TransformImageMirror16<N>(src + 3 * N * 16, dst - 3 * N * 16);
        }

        //-----------------------------------------------------------------------------------------

        const __m128i K8_SHUFFLE_BGR_TO_BGRA = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);
        const __m128i K8_SHUFFLE_BGRA_TO_BGR = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        SIMD_INLINE void TransformImageTranspose_1x8x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m128i a0 = _mm_loadu_si128((__m128i*)(src + 0 * srcStride));
            __m128i a1 = _mm_loadu_si128((__m128i*)(src + 1 * srcStride));
            __m128i a2 = _mm_loadu_si128((__m128i*)(src + 2 * srcStride));
            __m128i a3 = _mm_loadu_si128((__m128i*)(src + 3 * srcStride));
            __m128i a4 = _mm_loadu_si128((__m128i*)(src + 4 * srcStride));
            __m128i a5 = _mm_loadu_si128((__m128i*)(src + 5 * srcStride));
            __m128i a6 = _mm_loadu_si128((__m128i*)(src + 6 * srcStride));
            __m128i a7 = _mm_loadu_si128((__m128i*)(src + 7 * srcStride));
            __m128i b0 = _mm_unpacklo_epi8(a0, a4);
            __m128i b1 = _mm_unpackhi_epi8(a0, a4);
            __m128i b2 = _mm_unpacklo_epi8(a1, a5);
            __m128i b3 = _mm_unpackhi_epi8(a1, a5);
            __m128i b4 = _mm_unpacklo_epi8(a2, a6);
            __m128i b5 = _mm_unpackhi_epi8(a2, a6);
            __m128i b6 = _mm_unpacklo_epi8(a3, a7);
            __m128i b7 = _mm_unpackhi_epi8(a3, a7);
            a0 = _mm_unpacklo_epi8(b0, b4);
            a1 = _mm_unpackhi_epi8(b0, b4);
            a2 = _mm_unpacklo_epi8(b1, b5);
            a3 = _mm_unpackhi_epi8(b1, b5);
            a4 = _mm_unpacklo_epi8(b2, b6);
            a5 = _mm_unpackhi_epi8(b2, b6);
            a6 = _mm_unpacklo_epi8(b3, b7);
            a7 = _mm_unpackhi_epi8(b3, b7);
            b0 = _mm_unpacklo_epi8(a0, a4);
            b1 = _mm_unpackhi_epi8(a0, a4);
            b2 = _mm_unpacklo_epi8(a1, a5);
            b3 = _mm_unpackhi_epi8(a1, a5);
            b4 = _mm_unpacklo_epi8(a2, a6);
            b5 = _mm_unpackhi_epi8(a2, a6);
            b6 = _mm_unpacklo_epi8(a3, a7);
            b7 = _mm_unpackhi_epi8(a3, a7);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0x0 * dstStride), b0);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0x1 * dstStride), b0);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0x2 * dstStride), b1);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0x3 * dstStride), b1);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0x4 * dstStride), b2);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0x5 * dstStride), b2);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0x6 * dstStride), b3);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0x7 * dstStride), b3);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0x8 * dstStride), b4);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0x9 * dstStride), b4);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0xa * dstStride), b5);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0xb * dstStride), b5);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0xc * dstStride), b6);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0xd * dstStride), b6);
            Sse2::StoreHalf<0>((__m128i*)(dst + 0xe * dstStride), b7);
            Sse2::StoreHalf<1>((__m128i*)(dst + 0xf * dstStride), b7);
        }

        SIMD_INLINE void TransformImageTranspose_2x8x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m128i a0 = _mm_loadu_si128((__m128i*)(src + 0 * srcStride));
            __m128i a1 = _mm_loadu_si128((__m128i*)(src + 1 * srcStride));
            __m128i a2 = _mm_loadu_si128((__m128i*)(src + 2 * srcStride));
            __m128i a3 = _mm_loadu_si128((__m128i*)(src + 3 * srcStride));
            __m128i a4 = _mm_loadu_si128((__m128i*)(src + 4 * srcStride));
            __m128i a5 = _mm_loadu_si128((__m128i*)(src + 5 * srcStride));
            __m128i a6 = _mm_loadu_si128((__m128i*)(src + 6 * srcStride));
            __m128i a7 = _mm_loadu_si128((__m128i*)(src + 7 * srcStride));
            __m128i b0 = _mm_unpacklo_epi16(a0, a4);
            __m128i b1 = _mm_unpackhi_epi16(a0, a4);
            __m128i b2 = _mm_unpacklo_epi16(a1, a5);
            __m128i b3 = _mm_unpackhi_epi16(a1, a5);
            __m128i b4 = _mm_unpacklo_epi16(a2, a6);
            __m128i b5 = _mm_unpackhi_epi16(a2, a6);
            __m128i b6 = _mm_unpacklo_epi16(a3, a7);
            __m128i b7 = _mm_unpackhi_epi16(a3, a7);
            a0 = _mm_unpacklo_epi16(b0, b4);
            a1 = _mm_unpackhi_epi16(b0, b4);
            a2 = _mm_unpacklo_epi16(b1, b5);
            a3 = _mm_unpackhi_epi16(b1, b5);
            a4 = _mm_unpacklo_epi16(b2, b6);
            a5 = _mm_unpackhi_epi16(b2, b6);
            a6 = _mm_unpacklo_epi16(b3, b7);
            a7 = _mm_unpackhi_epi16(b3, b7);
            _mm_storeu_si128((__m128i*)(dst + 0 * dstStride), _mm_unpacklo_epi16(a0, a4));
            _mm_storeu_si128((__m128i*)(dst + 1 * dstStride), _mm_unpackhi_epi16(a0, a4));
            _mm_storeu_si128((__m128i*)(dst + 2 * dstStride), _mm_unpacklo_epi16(a1, a5));
            _mm_storeu_si128((__m128i*)(dst + 3 * dstStride), _mm_unpackhi_epi16(a1, a5));
            _mm_storeu_si128((__m128i*)(dst + 4 * dstStride), _mm_unpacklo_epi16(a2, a6));
            _mm_storeu_si128((__m128i*)(dst + 5 * dstStride), _mm_unpackhi_epi16(a2, a6));
            _mm_storeu_si128((__m128i*)(dst + 6 * dstStride), _mm_unpacklo_epi16(a3, a7));
            _mm_storeu_si128((__m128i*)(dst + 7 * dstStride), _mm_unpackhi_epi16(a3, a7));
        }

        SIMD_INLINE void TransformImageTranspose_3x4x4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m128i a0 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + 0 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m128i a1 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + 1 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m128i a2 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + 2 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m128i a3 = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(src + 3 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpackhi_epi32(a0, a2);
            __m128i b2 = _mm_unpacklo_epi32(a1, a3);
            __m128i b3 = _mm_unpackhi_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)(dst + 0 * dstStride), _mm_shuffle_epi8(_mm_unpacklo_epi32(b0, b2), K8_SHUFFLE_BGRA_TO_BGR));
            _mm_storeu_si128((__m128i*)(dst + 1 * dstStride), _mm_shuffle_epi8(_mm_unpackhi_epi32(b0, b2), K8_SHUFFLE_BGRA_TO_BGR));
            _mm_storeu_si128((__m128i*)(dst + 2 * dstStride), _mm_shuffle_epi8(_mm_unpacklo_epi32(b1, b3), K8_SHUFFLE_BGRA_TO_BGR));
            _mm_storeu_si128((__m128i*)(dst + 3 * dstStride), _mm_shuffle_epi8(_mm_unpackhi_epi32(b1, b3), K8_SHUFFLE_BGRA_TO_BGR));
        }

        SIMD_INLINE void TransformImageTranspose_4x4x4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m128i a0 = _mm_loadu_si128((__m128i*)(src + 0 * srcStride));
            __m128i a1 = _mm_loadu_si128((__m128i*)(src + 1 * srcStride));
            __m128i a2 = _mm_loadu_si128((__m128i*)(src + 2 * srcStride));
            __m128i a3 = _mm_loadu_si128((__m128i*)(src + 3 * srcStride));
            __m128i b0 = _mm_unpacklo_epi32(a0, a2);
            __m128i b1 = _mm_unpackhi_epi32(a0, a2);
            __m128i b2 = _mm_unpacklo_epi32(a1, a3);
            __m128i b3 = _mm_unpackhi_epi32(a1, a3);
            _mm_storeu_si128((__m128i*)(dst + 0 * dstStride), _mm_unpacklo_epi32(b0, b2));
            _mm_storeu_si128((__m128i*)(dst + 1 * dstStride), _mm_unpackhi_epi32(b0, b2));
            _mm_storeu_si128((__m128i*)(dst + 2 * dstStride), _mm_unpacklo_epi32(b1, b3));
            _mm_storeu_si128((__m128i*)(dst + 3 * dstStride), _mm_unpackhi_epi32(b1, b3));
        }

        //-----------------------------------------------------------------------------------------

        struct ImageTransforms : public Base::ImageTransforms
        {
            ImageTransforms();
        };
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const __m256i K8_MIRROR_1 = SIMD_MM256_SETR_EPI8(
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0);
        const __m256i K8_MIRROR_2 = SIMD_MM256_SETR_EPI8(
            0xE, 0xF, 0xC, 0xD, 0xA, 0xB, 0x8, 0x9, 0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1,
            0xE, 0xF, 0xC, 0xD, 0xA, 0xB, 0x8, 0x9, 0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);

        const __m256i K8_MIRROR_3_02 = SIMD_MM256_SETR_EPI8(
            0xD, 0xE, 0xF, 0xA, 0xB, 0xC, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1,
            0xD, 0xE, 0xF, 0xA, 0xB, 0xC, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1);
        const __m256i K8_MIRROR_3_01 = SIMD_MM256_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xE,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xE);
        const __m256i K8_MIRROR_3_12 = SIMD_MM256_SETR_EPI8(
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_MIRROR_3_11 = SIMD_MM256_SETR_EPI8(
            0xF, -1, 0xB, 0xC, 0xD, 0x8, 0x9, 0xA, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0,
            0xF, -1, 0xB, 0xC, 0xD, 0x8, 0x9, 0xA, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0);
        const __m256i K8_MIRROR_3_10 = SIMD_MM256_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xF, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xF, -1);
        const __m256i K8_MIRROR_3_21 = SIMD_MM256_SETR_EPI8(
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m256i K8_MIRROR_3_20 = SIMD_MM256_SETR_EPI8(
            -1, 0xC, 0xD, 0xE, 0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2,
            -1, 0xC, 0xD, 0xE, 0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2);

        const __m256i K32_MIRROR_4 = SIMD_MM256_SETR_EPI32(7, 6, 5, 4, 3, 2, 1, 0);

        template<size_t N> SIMD_INLINE void TransformImageMirror32(const uint8_t* src, uint8_t* dst)
        {
            dst += (32 - 1) * N;
            for (size_t i = 0; i < 32; ++i)
                Base::CopyPixel<N>(src + i * N, dst - i * N);
        }

        template<> SIMD_INLINE void TransformImageMirror32<1>(const uint8_t* src, uint8_t* dst)
        {
            _mm256_storeu_si256((__m256i*)dst, _mm256_shuffle_epi8(Load<false>((__m128i*)src + 1, (__m128i*)src + 0), K8_MIRROR_1));
        }

        template<> SIMD_INLINE void TransformImageMirror32<2>(const uint8_t* src, uint8_t* dst)
        {
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_shuffle_epi8(Load<false>((__m128i*)src + 1, (__m128i*)src + 0), K8_MIRROR_2));
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_shuffle_epi8(Load<false>((__m128i*)src + 3, (__m128i*)src + 2), K8_MIRROR_2));
        }

        template<> SIMD_INLINE void TransformImageMirror32<3>(const uint8_t* src, uint8_t* dst)
        {
            __m256i s0 = _mm256_loadu_si256((__m256i*)src + 0);
            __m256i s1 = _mm256_loadu_si256((__m256i*)src + 1);
            __m256i s2 = _mm256_loadu_si256((__m256i*)src + 2);
            __m256i d0 = _mm256_permute2f128_si256(s0, s1, 0x30);
            __m256i d1 = _mm256_permute2f128_si256(s0, s2, 0x21);
            __m256i d2 = _mm256_permute2f128_si256(s1, s2, 0x30);
            s0 = _mm256_or_si256(_mm256_shuffle_epi8(d2, K8_MIRROR_3_02), _mm256_shuffle_epi8(d1, K8_MIRROR_3_01));
            s1 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(d2, K8_MIRROR_3_12), _mm256_shuffle_epi8(d1, K8_MIRROR_3_11)), _mm256_shuffle_epi8(d0, K8_MIRROR_3_10));
            s2 = _mm256_or_si256(_mm256_shuffle_epi8(d1, K8_MIRROR_3_21), _mm256_shuffle_epi8(d0, K8_MIRROR_3_20));
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permute2f128_si256(s0, s1, 0x31));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permute2f128_si256(s2, s0, 0x21));
            _mm256_storeu_si256((__m256i*)dst + 2, _mm256_permute2f128_si256(s1, s2, 0x20));
        }

        template<> SIMD_INLINE void TransformImageMirror32<4>(const uint8_t* src, uint8_t* dst)
        {
            _mm256_storeu_si256((__m256i*)dst + 3, _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)src + 0), K32_MIRROR_4));
            _mm256_storeu_si256((__m256i*)dst + 2, _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)src + 1), K32_MIRROR_4));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)src + 2), K32_MIRROR_4));
            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_permutevar8x32_epi32(_mm256_loadu_si256((__m256i*)src + 3), K32_MIRROR_4));
        }

        template<size_t N> SIMD_INLINE void TransformImageMirror64(const uint8_t* src, uint8_t* dst)
        {
            TransformImageMirror32<N>(src + 0 * N * 32, dst - 0 * N * 32);
            TransformImageMirror32<N>(src + 1 * N * 32, dst - 1 * N * 32);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TransformImageTranspose_1x16x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = Load<false>((__m128i*)(src + 0x0 * srcStride), (__m128i*)(src + 0x8 * srcStride));
            __m256i a1 = Load<false>((__m128i*)(src + 0x1 * srcStride), (__m128i*)(src + 0x9 * srcStride));
            __m256i a2 = Load<false>((__m128i*)(src + 0x2 * srcStride), (__m128i*)(src + 0xa * srcStride));
            __m256i a3 = Load<false>((__m128i*)(src + 0x3 * srcStride), (__m128i*)(src + 0xb * srcStride));
            __m256i a4 = Load<false>((__m128i*)(src + 0x4 * srcStride), (__m128i*)(src + 0xc * srcStride));
            __m256i a5 = Load<false>((__m128i*)(src + 0x5 * srcStride), (__m128i*)(src + 0xd * srcStride));
            __m256i a6 = Load<false>((__m128i*)(src + 0x6 * srcStride), (__m128i*)(src + 0xe * srcStride));
            __m256i a7 = Load<false>((__m128i*)(src + 0x7 * srcStride), (__m128i*)(src + 0xf * srcStride));
            __m256i b0 = _mm256_unpacklo_epi8(a0, a4);
            __m256i b1 = _mm256_unpackhi_epi8(a0, a4);
            __m256i b2 = _mm256_unpacklo_epi8(a1, a5);
            __m256i b3 = _mm256_unpackhi_epi8(a1, a5);
            __m256i b4 = _mm256_unpacklo_epi8(a2, a6);
            __m256i b5 = _mm256_unpackhi_epi8(a2, a6);
            __m256i b6 = _mm256_unpacklo_epi8(a3, a7);
            __m256i b7 = _mm256_unpackhi_epi8(a3, a7);
            a0 = _mm256_unpacklo_epi8(b0, b4);
            a1 = _mm256_unpackhi_epi8(b0, b4);
            a2 = _mm256_unpacklo_epi8(b1, b5);
            a3 = _mm256_unpackhi_epi8(b1, b5);
            a4 = _mm256_unpacklo_epi8(b2, b6);
            a5 = _mm256_unpackhi_epi8(b2, b6);
            a6 = _mm256_unpacklo_epi8(b3, b7);
            a7 = _mm256_unpackhi_epi8(b3, b7);
            b0 = _mm256_unpacklo_epi8(a0, a4);
            b1 = _mm256_unpackhi_epi8(a0, a4);
            b2 = _mm256_unpacklo_epi8(a1, a5);
            b3 = _mm256_unpackhi_epi8(a1, a5);
            b4 = _mm256_unpacklo_epi8(a2, a6);
            b5 = _mm256_unpackhi_epi8(a2, a6);
            b6 = _mm256_unpacklo_epi8(a3, a7);
            b7 = _mm256_unpackhi_epi8(a3, a7);
            Avx2::Store<false>((__m128i*)(dst + 0x0 * dstStride), (__m128i*)(dst + 0x1 * dstStride), _mm256_permute4x64_epi64(b0, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0x2 * dstStride), (__m128i*)(dst + 0x3 * dstStride), _mm256_permute4x64_epi64(b1, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0x4 * dstStride), (__m128i*)(dst + 0x5 * dstStride), _mm256_permute4x64_epi64(b2, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0x6 * dstStride), (__m128i*)(dst + 0x7 * dstStride), _mm256_permute4x64_epi64(b3, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0x8 * dstStride), (__m128i*)(dst + 0x9 * dstStride), _mm256_permute4x64_epi64(b4, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0xa * dstStride), (__m128i*)(dst + 0xb * dstStride), _mm256_permute4x64_epi64(b5, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0xc * dstStride), (__m128i*)(dst + 0xd * dstStride), _mm256_permute4x64_epi64(b6, 0xD8));
            Avx2::Store<false>((__m128i*)(dst + 0xe * dstStride), (__m128i*)(dst + 0xf * dstStride), _mm256_permute4x64_epi64(b7, 0xD8));
        }
        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TransformImageTranspose_2x16x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = Load<false>((__m128i*)(src + 0x0 * srcStride), (__m128i*)(src + 0x8 * srcStride));
            __m256i a1 = Load<false>((__m128i*)(src + 0x1 * srcStride), (__m128i*)(src + 0x9 * srcStride));
            __m256i a2 = Load<false>((__m128i*)(src + 0x2 * srcStride), (__m128i*)(src + 0xa * srcStride));
            __m256i a3 = Load<false>((__m128i*)(src + 0x3 * srcStride), (__m128i*)(src + 0xb * srcStride));
            __m256i a4 = Load<false>((__m128i*)(src + 0x4 * srcStride), (__m128i*)(src + 0xc * srcStride));
            __m256i a5 = Load<false>((__m128i*)(src + 0x5 * srcStride), (__m128i*)(src + 0xd * srcStride));
            __m256i a6 = Load<false>((__m128i*)(src + 0x6 * srcStride), (__m128i*)(src + 0xe * srcStride));
            __m256i a7 = Load<false>((__m128i*)(src + 0x7 * srcStride), (__m128i*)(src + 0xf * srcStride));
            __m256i b0 = _mm256_unpacklo_epi16(a0, a4);
            __m256i b1 = _mm256_unpackhi_epi16(a0, a4);
            __m256i b2 = _mm256_unpacklo_epi16(a1, a5);
            __m256i b3 = _mm256_unpackhi_epi16(a1, a5);
            __m256i b4 = _mm256_unpacklo_epi16(a2, a6);
            __m256i b5 = _mm256_unpackhi_epi16(a2, a6);
            __m256i b6 = _mm256_unpacklo_epi16(a3, a7);
            __m256i b7 = _mm256_unpackhi_epi16(a3, a7);
            a0 = _mm256_unpacklo_epi16(b0, b4);
            a1 = _mm256_unpackhi_epi16(b0, b4);
            a2 = _mm256_unpacklo_epi16(b1, b5);
            a3 = _mm256_unpackhi_epi16(b1, b5);
            a4 = _mm256_unpacklo_epi16(b2, b6);
            a5 = _mm256_unpackhi_epi16(b2, b6);
            a6 = _mm256_unpacklo_epi16(b3, b7);
            a7 = _mm256_unpackhi_epi16(b3, b7);
            _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_unpacklo_epi16(a0, a4));
            _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_unpackhi_epi16(a0, a4));
            _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_unpacklo_epi16(a1, a5));
            _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_unpackhi_epi16(a1, a5));
            _mm256_storeu_si256((__m256i*)(dst + 4 * dstStride), _mm256_unpacklo_epi16(a2, a6));
            _mm256_storeu_si256((__m256i*)(dst + 5 * dstStride), _mm256_unpackhi_epi16(a2, a6));
            _mm256_storeu_si256((__m256i*)(dst + 6 * dstStride), _mm256_unpacklo_epi16(a3, a7));
            _mm256_storeu_si256((__m256i*)(dst + 7 * dstStride), _mm256_unpackhi_epi16(a3, a7));
        }

        SIMD_INLINE void TransformImageTranspose_2x8x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = _mm256_loadu_si256((__m256i*)(src + 0 * srcStride));
            __m256i a1 = _mm256_loadu_si256((__m256i*)(src + 1 * srcStride));
            __m256i a2 = _mm256_loadu_si256((__m256i*)(src + 2 * srcStride));
            __m256i a3 = _mm256_loadu_si256((__m256i*)(src + 3 * srcStride));
            __m256i a4 = _mm256_loadu_si256((__m256i*)(src + 4 * srcStride));
            __m256i a5 = _mm256_loadu_si256((__m256i*)(src + 5 * srcStride));
            __m256i a6 = _mm256_loadu_si256((__m256i*)(src + 6 * srcStride));
            __m256i a7 = _mm256_loadu_si256((__m256i*)(src + 7 * srcStride));
            __m256i b0 = _mm256_unpacklo_epi16(a0, a4);
            __m256i b1 = _mm256_unpackhi_epi16(a0, a4);
            __m256i b2 = _mm256_unpacklo_epi16(a1, a5);
            __m256i b3 = _mm256_unpackhi_epi16(a1, a5);
            __m256i b4 = _mm256_unpacklo_epi16(a2, a6);
            __m256i b5 = _mm256_unpackhi_epi16(a2, a6);
            __m256i b6 = _mm256_unpacklo_epi16(a3, a7);
            __m256i b7 = _mm256_unpackhi_epi16(a3, a7);
            a0 = _mm256_unpacklo_epi16(b0, b4);
            a1 = _mm256_unpackhi_epi16(b0, b4);
            a2 = _mm256_unpacklo_epi16(b1, b5);
            a3 = _mm256_unpackhi_epi16(b1, b5);
            a4 = _mm256_unpacklo_epi16(b2, b6);
            a5 = _mm256_unpackhi_epi16(b2, b6);
            a6 = _mm256_unpacklo_epi16(b3, b7);
            a7 = _mm256_unpackhi_epi16(b3, b7);
            Store<false>((__m128i*)(dst + 0x0 * dstStride), (__m128i*)(dst + 0x8 * dstStride), _mm256_unpacklo_epi16(a0, a4));
            Store<false>((__m128i*)(dst + 0x1 * dstStride), (__m128i*)(dst + 0x9 * dstStride), _mm256_unpackhi_epi16(a0, a4));
            Store<false>((__m128i*)(dst + 0x2 * dstStride), (__m128i*)(dst + 0xa * dstStride), _mm256_unpacklo_epi16(a1, a5));
            Store<false>((__m128i*)(dst + 0x3 * dstStride), (__m128i*)(dst + 0xb * dstStride), _mm256_unpackhi_epi16(a1, a5));
            Store<false>((__m128i*)(dst + 0x4 * dstStride), (__m128i*)(dst + 0xc * dstStride), _mm256_unpacklo_epi16(a2, a6));
            Store<false>((__m128i*)(dst + 0x5 * dstStride), (__m128i*)(dst + 0xd * dstStride), _mm256_unpackhi_epi16(a2, a6));
            Store<false>((__m128i*)(dst + 0x6 * dstStride), (__m128i*)(dst + 0xe * dstStride), _mm256_unpacklo_epi16(a3, a7));
            Store<false>((__m128i*)(dst + 0x7 * dstStride), (__m128i*)(dst + 0xf * dstStride), _mm256_unpackhi_epi16(a3, a7));
        }

        //-----------------------------------------------------------------------------------------

        const __m256i K8_SHUFFLE_BGR_TO_BGRA = SIMD_MM256_SETR_EPI8(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        SIMD_INLINE void TransformImageTranspose_3x4x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)(src + 0 * srcStride)), 0x94), K8_BGR_TO_BGRA_SHUFFLE);
            __m256i a1 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)(src + 1 * srcStride)), 0x94), K8_BGR_TO_BGRA_SHUFFLE);
            __m256i a2 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)(src + 2 * srcStride)), 0x94), K8_BGR_TO_BGRA_SHUFFLE);
            __m256i a3 = _mm256_shuffle_epi8(_mm256_permute4x64_epi64(_mm256_loadu_si256((__m256i*)(src + 3 * srcStride)), 0x94), K8_BGR_TO_BGRA_SHUFFLE);
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b2 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            Avx2::Store<false>((__m128i*)(dst + 0x0 * dstStride), (__m128i*)(dst + 0x4 * dstStride), _mm256_shuffle_epi8(_mm256_unpacklo_epi32(b0, b2), K8_SHUFFLE_BGRA_TO_BGR));
            Avx2::Store<false>((__m128i*)(dst + 0x1 * dstStride), (__m128i*)(dst + 0x5 * dstStride), _mm256_shuffle_epi8(_mm256_unpackhi_epi32(b0, b2), K8_SHUFFLE_BGRA_TO_BGR));
            Avx2::Store<false>((__m128i*)(dst + 0x2 * dstStride), (__m128i*)(dst + 0x6 * dstStride), _mm256_shuffle_epi8(_mm256_unpacklo_epi32(b1, b3), K8_SHUFFLE_BGRA_TO_BGR));
            Avx2::Store<false>((__m128i*)(dst + 0x3 * dstStride), (__m128i*)(dst + 0x7 * dstStride), _mm256_shuffle_epi8(_mm256_unpackhi_epi32(b1, b3), K8_SHUFFLE_BGRA_TO_BGR));
        }

        SIMD_INLINE void TransformImageTranspose_3x8x4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + 0 * srcStride), (__m128i*)(src + 4 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m256i a1 = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + 1 * srcStride), (__m128i*)(src + 5 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m256i a2 = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + 2 * srcStride), (__m128i*)(src + 6 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m256i a3 = _mm256_shuffle_epi8(Load<false>((__m128i*)(src + 3 * srcStride), (__m128i*)(src + 7 * srcStride)), K8_SHUFFLE_BGR_TO_BGRA);
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b2 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_unpacklo_epi32(b0, b1), K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR));
            _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_unpackhi_epi32(b0, b1), K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR));
            _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_unpacklo_epi32(b2, b3), K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR));
            _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(_mm256_unpackhi_epi32(b2, b3), K8_SHUFFLE_BGRA_TO_BGR), K32_PERMUTE_BGRA_TO_BGR));
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TransformImageTranspose_4x4x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = _mm256_loadu_si256((__m256i*)(src + 0 * srcStride));
            __m256i a1 = _mm256_loadu_si256((__m256i*)(src + 1 * srcStride));
            __m256i a2 = _mm256_loadu_si256((__m256i*)(src + 2 * srcStride));
            __m256i a3 = _mm256_loadu_si256((__m256i*)(src + 3 * srcStride));
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b2 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            Avx2::Store<false>((__m128i*)(dst + 0x0 * dstStride), (__m128i*)(dst + 0x4 * dstStride), _mm256_unpacklo_epi32(b0, b1));
            Avx2::Store<false>((__m128i*)(dst + 0x1 * dstStride), (__m128i*)(dst + 0x5 * dstStride), _mm256_unpackhi_epi32(b0, b1));
            Avx2::Store<false>((__m128i*)(dst + 0x2 * dstStride), (__m128i*)(dst + 0x6 * dstStride), _mm256_unpacklo_epi32(b2, b3));
            Avx2::Store<false>((__m128i*)(dst + 0x3 * dstStride), (__m128i*)(dst + 0x7 * dstStride), _mm256_unpackhi_epi32(b2, b3));
        }

        SIMD_INLINE void TransformImageTranspose_4x8x4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = Load<false>((__m128i*)(src + 0 * srcStride), (__m128i*)(src + 4 * srcStride));
            __m256i a1 = Load<false>((__m128i*)(src + 1 * srcStride), (__m128i*)(src + 5 * srcStride));
            __m256i a2 = Load<false>((__m128i*)(src + 2 * srcStride), (__m128i*)(src + 6 * srcStride));
            __m256i a3 = Load<false>((__m128i*)(src + 3 * srcStride), (__m128i*)(src + 7 * srcStride));
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b2 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_unpacklo_epi32(b0, b1));
            _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_unpackhi_epi32(b0, b1));
            _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_unpacklo_epi32(b2, b3));
            _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_unpackhi_epi32(b2, b3));
        }

        SIMD_INLINE void TransformImageTranspose_4x8x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m256i a0 = _mm256_loadu_si256((__m256i*)(src + 0 * srcStride));
            __m256i a1 = _mm256_loadu_si256((__m256i*)(src + 1 * srcStride));
            __m256i a2 = _mm256_loadu_si256((__m256i*)(src + 2 * srcStride));
            __m256i a3 = _mm256_loadu_si256((__m256i*)(src + 3 * srcStride));
            __m256i a4 = _mm256_loadu_si256((__m256i*)(src + 4 * srcStride));
            __m256i a5 = _mm256_loadu_si256((__m256i*)(src + 5 * srcStride));
            __m256i a6 = _mm256_loadu_si256((__m256i*)(src + 6 * srcStride));
            __m256i a7 = _mm256_loadu_si256((__m256i*)(src + 7 * srcStride));
            __m256i b0 = _mm256_unpacklo_epi32(a0, a2);
            __m256i b1 = _mm256_unpacklo_epi32(a1, a3);
            __m256i b2 = _mm256_unpackhi_epi32(a0, a2);
            __m256i b3 = _mm256_unpackhi_epi32(a1, a3);
            __m256i b4 = _mm256_unpacklo_epi32(a4, a6);
            __m256i b5 = _mm256_unpacklo_epi32(a5, a7);
            __m256i b6 = _mm256_unpackhi_epi32(a4, a6);
            __m256i b7 = _mm256_unpackhi_epi32(a5, a7);
            a0 = _mm256_unpacklo_epi32(b0, b1);
            a1 = _mm256_unpackhi_epi32(b0, b1);
            a2 = _mm256_unpacklo_epi32(b2, b3);
            a3 = _mm256_unpackhi_epi32(b2, b3);
            a4 = _mm256_unpacklo_epi32(b4, b5);
            a5 = _mm256_unpackhi_epi32(b4, b5);
            a6 = _mm256_unpacklo_epi32(b6, b7);
            a7 = _mm256_unpackhi_epi32(b6, b7);
            _mm256_storeu_si256((__m256i*)(dst + 0 * dstStride), _mm256_permute2f128_si256(a0, a4, 0x20));
            _mm256_storeu_si256((__m256i*)(dst + 1 * dstStride), _mm256_permute2f128_si256(a1, a5, 0x20));
            _mm256_storeu_si256((__m256i*)(dst + 2 * dstStride), _mm256_permute2f128_si256(a2, a6, 0x20));
            _mm256_storeu_si256((__m256i*)(dst + 3 * dstStride), _mm256_permute2f128_si256(a3, a7, 0x20));
            _mm256_storeu_si256((__m256i*)(dst + 4 * dstStride), _mm256_permute2f128_si256(a0, a4, 0x31));
            _mm256_storeu_si256((__m256i*)(dst + 5 * dstStride), _mm256_permute2f128_si256(a1, a5, 0x31));
            _mm256_storeu_si256((__m256i*)(dst + 6 * dstStride), _mm256_permute2f128_si256(a2, a6, 0x31));
            _mm256_storeu_si256((__m256i*)(dst + 7 * dstStride), _mm256_permute2f128_si256(a3, a7, 0x31));
        }

        //-----------------------------------------------------------------------------------------

        struct ImageTransforms : public Sse41::ImageTransforms
        {
            ImageTransforms();
        };
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K8_MIRROR_1 = SIMD_MM512_SETR_EPI8(
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0,
            0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0);

        const __m512i K64_MIRROR_1 = SIMD_MM512_SETR_EPI64(6, 7, 4, 5, 2, 3, 0, 1);

        SIMD_INLINE void TransformImageMirror1x64(const uint8_t* src, uint8_t* dst, __mmask64 tail = -1, __mmask64 nose = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi8(tail, src);
            __m512i _dst = _mm512_permutexvar_epi64(K64_MIRROR_1, _mm512_shuffle_epi8(_src, K8_MIRROR_1));
            _mm512_mask_storeu_epi8(dst, nose, _dst);
        }

        SIMD_INLINE void TransformImageMirror1x256(const uint8_t* src, uint8_t* dst)
        {
            _mm512_storeu_si512(dst - 0 * A, _mm512_permutexvar_epi64(K64_MIRROR_1, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 0 * A), K8_MIRROR_1)));
            _mm512_storeu_si512(dst - 1 * A, _mm512_permutexvar_epi64(K64_MIRROR_1, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 1 * A), K8_MIRROR_1)));
            _mm512_storeu_si512(dst - 2 * A, _mm512_permutexvar_epi64(K64_MIRROR_1, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 2 * A), K8_MIRROR_1)));
            _mm512_storeu_si512(dst - 3 * A, _mm512_permutexvar_epi64(K64_MIRROR_1, _mm512_shuffle_epi8(_mm512_loadu_si512(src + 3 * A), K8_MIRROR_1)));
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K16_MIRROR_2 = SIMD_MM512_SETR_EPI16(
            0x1f, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x18, 0x17, 0x16, 0x15, 0x14, 0x13, 0x12, 0x11, 0x10,
            0x0f, 0x0e, 0x0d, 0x0c, 0x0b, 0x0a, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00);

        SIMD_INLINE void TransformImageMirror2x32(const uint8_t* src, uint8_t* dst, __mmask32 tail = -1, __mmask32 nose = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512i _dst = _mm512_permutexvar_epi16(K16_MIRROR_2, _src);
            _mm512_mask_storeu_epi16(dst, nose, _dst);
        }

        SIMD_INLINE void TransformImageMirror2x128(const uint8_t* src, uint8_t* dst)
        {
            _mm512_storeu_si512(dst - 0 * A, _mm512_permutexvar_epi16(K16_MIRROR_2, _mm512_loadu_si512(src + 0 * A)));
            _mm512_storeu_si512(dst - 1 * A, _mm512_permutexvar_epi16(K16_MIRROR_2, _mm512_loadu_si512(src + 1 * A)));
            _mm512_storeu_si512(dst - 2 * A, _mm512_permutexvar_epi16(K16_MIRROR_2, _mm512_loadu_si512(src + 2 * A)));
            _mm512_storeu_si512(dst - 3 * A, _mm512_permutexvar_epi16(K16_MIRROR_2, _mm512_loadu_si512(src + 3 * A)));
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K32_MIRROR_3_1U = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x0, 0x3, 0x4, 0x5, 0x0, 0x6, 0x7, 0x8, 0x0, 0x9, 0xa, 0xb, 0x0);
        const __m512i K8_MIRROR_3_1R = SIMD_MM512_SETR_EPI8(
            0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2, -1, -1, -1, -1,
            0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2, -1, -1, -1, -1,
            0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2, -1, -1, -1, -1,
            0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2, -1, -1, -1, -1);
        const __m512i K32_MIRROR_3_1P = SIMD_MM512_SETR_EPI32(0xc, 0xd, 0xe, 0x8, 0x9, 0xa, 0x4, 0x5, 0x6, 0x0, 0x1, 0x2, 0x0, 0x0, 0x0, 0x0);

        SIMD_INLINE void TransformImageMirror3x16(const uint8_t* src, uint8_t* dst, __mmask64 tail = 0x0000FFFFFFFFFFFF, __mmask64 nose = 0x0000FFFFFFFFFFFF)
        {
            __m512i _src = _mm512_maskz_loadu_epi8(tail, src);
            __m512i perm = _mm512_permutexvar_epi32(K32_MIRROR_3_1U, _src);
            __m512i shfl = _mm512_shuffle_epi8(perm, K8_MIRROR_3_1R);
            __m512i _dst = _mm512_permutexvar_epi32(K32_MIRROR_3_1P, shfl);
            _mm512_mask_storeu_epi8(dst, nose, _dst);
        }

        const __m512i K64_MIRROR_3_4P00 = SIMD_MM512_SETR_EPI64(0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);
        const __m512i K64_MIRROR_3_4P01 = SIMD_MM512_SETR_EPI64(0x0, 0x0, 0x6, 0xb, 0xc, 0x9, 0xa, 0x0);
        const __m512i K8_MIRROR_3_4S00 = SIMD_MM512_SETR_EPI8(
            -1, 0xc, 0xd, 0xe, 0x9, 0xa, 0xb, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2,
            0xd, 0xe, 0xf, 0xa, 0xb, 0xc, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1,
            0xf, -1, 0xb, 0xc, 0xd, 0x8, 0x9, 0xa, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0,
            -1, 0xc, 0xd, 0xe, 0x9, 0xa, 0xb, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2);
        const __m512i K8_MIRROR_3_4S01 = SIMD_MM512_SETR_EPI8(
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xe,
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xf, -1,
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

        const __m512i K64_MIRROR_3_4P10 = SIMD_MM512_SETR_EPI64(0x0, 0xd, 0xc, 0xd, 0xa, 0xb, 0x8, 0x9);
        const __m512i K64_MIRROR_3_4P11 = SIMD_MM512_SETR_EPI64(0x6, 0x7, 0x6, 0x3, 0x4, 0x1, 0x2, 0xf);
        const __m512i K8_MIRROR_3_4S10 = SIMD_MM512_SETR_EPI8(
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xf, -1,
            -1, 0xc, 0xd, 0xe, 0x9, 0xa, 0xb, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2,
            0xd, 0xe, 0xf, 0xa, 0xb, 0xc, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1,
            0xf, -1, 0xb, 0xc, 0xd, 0x8, 0x9, 0xa, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0);
        const __m512i K8_MIRROR_3_4S11 = SIMD_MM512_SETR_EPI8(
            0xf, -1, 0xb, 0xc, 0xd, 0x8, 0x9, 0xa, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0,
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xe,
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xf, -1);

        const __m512i K64_MIRROR_3_4P20 = SIMD_MM512_SETR_EPI64(0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);
        const __m512i K64_MIRROR_3_4P21 = SIMD_MM512_SETR_EPI64(0x0, 0x5, 0x6, 0x3, 0x4, 0x1, 0x0, 0xf);
        const __m512i K8_MIRROR_3_4S20 = SIMD_MM512_SETR_EPI8(
            0xd, 0xe, 0xf, 0xa, 0xb, 0xc, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1,
            0xf, -1, 0xb, 0xc, 0xd, 0x8, 0x9, 0xa, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0,
            -1, 0xc, 0xd, 0xe, 0x9, 0xa, 0xb, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2,
            0xd, 0xe, 0xf, 0xa, 0xb, 0xc, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1);
        const __m512i K8_MIRROR_3_4S21 = SIMD_MM512_SETR_EPI8(
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xe,
            -1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xf, -1,
            0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xe);

        SIMD_INLINE void TransformImageMirror3x64(const uint8_t* src, uint8_t* dst)
        {
            __m512i s0 = _mm512_loadu_si512(src + 0 * A);
            __m512i s1 = _mm512_loadu_si512(src + 1 * A);
            __m512i s2 = _mm512_loadu_si512(src + 2 * A);

            __m512i p00 = _mm512_permutexvar_epi64(K64_MIRROR_3_4P00, s0);
            __m512i p01 = _mm512_permutex2var_epi64(s1, K64_MIRROR_3_4P01, s0);
            __m512i d0 = _mm512_or_si512(_mm512_shuffle_epi8(p00, K8_MIRROR_3_4S00), _mm512_shuffle_epi8(p01, K8_MIRROR_3_4S01));

            __m512i p10 = _mm512_permutex2var_epi64(s2, K64_MIRROR_3_4P10, s1);
            __m512i p11 = _mm512_permutex2var_epi64(s1, K64_MIRROR_3_4P11, s0);
            __m512i d1 = _mm512_or_si512(_mm512_shuffle_epi8(p10, K8_MIRROR_3_4S10), _mm512_shuffle_epi8(p11, K8_MIRROR_3_4S11));

            __m512i p20 = _mm512_permutexvar_epi64(K64_MIRROR_3_4P20, s2);
            __m512i p21 = _mm512_permutex2var_epi64(s2, K64_MIRROR_3_4P21, s1);
            __m512i d2 = _mm512_or_si512(_mm512_shuffle_epi8(p20, K8_MIRROR_3_4S20), _mm512_shuffle_epi8(p21, K8_MIRROR_3_4S21));

            _mm512_storeu_si512(dst - 0 * A, d0);
            _mm512_storeu_si512(dst - 1 * A, d1);
            _mm512_storeu_si512(dst - 2 * A, d2);
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K32_MIRROR_4 = SIMD_MM512_SETR_EPI32(0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0);

        SIMD_INLINE void TransformImageMirror4x16(const uint8_t* src, uint8_t* dst, __mmask16 tail = -1, __mmask16 nose = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi32(tail, src);
            __m512i _dst = _mm512_permutexvar_epi32(K32_MIRROR_4, _src);
            _mm512_mask_storeu_epi32(dst, nose, _dst);
        }

        SIMD_INLINE void TransformImageMirror4x64(const uint8_t* src, uint8_t* dst)
        {
            _mm512_storeu_si512(dst - 0 * A, _mm512_permutexvar_epi32(K32_MIRROR_4, _mm512_loadu_si512(src + 0 * A)));
            _mm512_storeu_si512(dst - 1 * A, _mm512_permutexvar_epi32(K32_MIRROR_4, _mm512_loadu_si512(src + 1 * A)));
            _mm512_storeu_si512(dst - 2 * A, _mm512_permutexvar_epi32(K32_MIRROR_4, _mm512_loadu_si512(src + 2 * A)));
            _mm512_storeu_si512(dst - 3 * A, _mm512_permutexvar_epi32(K32_MIRROR_4, _mm512_loadu_si512(src + 3 * A)));
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K16_TRANSPOSE_1x64x16 = SIMD_MM512_SETR_EPI16(
            0x00, 0x08, 0x10, 0x18, 0x01, 0x09, 0x11, 0x19, 0x02, 0x0a, 0x12, 0x1a, 0x03, 0x0b, 0x13, 0x1b,
            0x04, 0x0c, 0x14, 0x1c, 0x05, 0x0d, 0x15, 0x1d, 0x06, 0x0e, 0x16, 0x1e, 0x07, 0x0f, 0x17, 0x1f);

        SIMD_INLINE void TransformImageTranspose_1x64x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf;

            a0 = Load<false>((__m128i*)(src + 0x00 * srcStride), (__m128i*)(src + 0x10 * srcStride), (__m128i*)(src + 0x20 * srcStride), (__m128i*)(src + 0x30 * srcStride));
            a1 = Load<false>((__m128i*)(src + 0x01 * srcStride), (__m128i*)(src + 0x11 * srcStride), (__m128i*)(src + 0x21 * srcStride), (__m128i*)(src + 0x31 * srcStride));
            a2 = Load<false>((__m128i*)(src + 0x02 * srcStride), (__m128i*)(src + 0x12 * srcStride), (__m128i*)(src + 0x22 * srcStride), (__m128i*)(src + 0x32 * srcStride));
            a3 = Load<false>((__m128i*)(src + 0x03 * srcStride), (__m128i*)(src + 0x13 * srcStride), (__m128i*)(src + 0x23 * srcStride), (__m128i*)(src + 0x33 * srcStride));
            a4 = Load<false>((__m128i*)(src + 0x04 * srcStride), (__m128i*)(src + 0x14 * srcStride), (__m128i*)(src + 0x24 * srcStride), (__m128i*)(src + 0x34 * srcStride));
            a5 = Load<false>((__m128i*)(src + 0x05 * srcStride), (__m128i*)(src + 0x15 * srcStride), (__m128i*)(src + 0x25 * srcStride), (__m128i*)(src + 0x35 * srcStride));
            a6 = Load<false>((__m128i*)(src + 0x06 * srcStride), (__m128i*)(src + 0x16 * srcStride), (__m128i*)(src + 0x26 * srcStride), (__m128i*)(src + 0x36 * srcStride));
            a7 = Load<false>((__m128i*)(src + 0x07 * srcStride), (__m128i*)(src + 0x17 * srcStride), (__m128i*)(src + 0x27 * srcStride), (__m128i*)(src + 0x37 * srcStride));
            a8 = Load<false>((__m128i*)(src + 0x08 * srcStride), (__m128i*)(src + 0x18 * srcStride), (__m128i*)(src + 0x28 * srcStride), (__m128i*)(src + 0x38 * srcStride));
            a9 = Load<false>((__m128i*)(src + 0x09 * srcStride), (__m128i*)(src + 0x19 * srcStride), (__m128i*)(src + 0x29 * srcStride), (__m128i*)(src + 0x39 * srcStride));
            aa = Load<false>((__m128i*)(src + 0x0a * srcStride), (__m128i*)(src + 0x1a * srcStride), (__m128i*)(src + 0x2a * srcStride), (__m128i*)(src + 0x3a * srcStride));
            ab = Load<false>((__m128i*)(src + 0x0b * srcStride), (__m128i*)(src + 0x1b * srcStride), (__m128i*)(src + 0x2b * srcStride), (__m128i*)(src + 0x3b * srcStride));
            ac = Load<false>((__m128i*)(src + 0x0c * srcStride), (__m128i*)(src + 0x1c * srcStride), (__m128i*)(src + 0x2c * srcStride), (__m128i*)(src + 0x3c * srcStride));
            ad = Load<false>((__m128i*)(src + 0x0d * srcStride), (__m128i*)(src + 0x1d * srcStride), (__m128i*)(src + 0x2d * srcStride), (__m128i*)(src + 0x3d * srcStride));
            ae = Load<false>((__m128i*)(src + 0x0e * srcStride), (__m128i*)(src + 0x1e * srcStride), (__m128i*)(src + 0x2e * srcStride), (__m128i*)(src + 0x3e * srcStride));
            af = Load<false>((__m128i*)(src + 0x0f * srcStride), (__m128i*)(src + 0x1f * srcStride), (__m128i*)(src + 0x2f * srcStride), (__m128i*)(src + 0x3f * srcStride));

            b0 = _mm512_unpacklo_epi8(a0, a1);
            b1 = _mm512_unpackhi_epi8(a0, a1);
            b2 = _mm512_unpacklo_epi8(a2, a3);
            b3 = _mm512_unpackhi_epi8(a2, a3);
            b4 = _mm512_unpacklo_epi8(a4, a5);
            b5 = _mm512_unpackhi_epi8(a4, a5);
            b6 = _mm512_unpacklo_epi8(a6, a7);
            b7 = _mm512_unpackhi_epi8(a6, a7);
            b8 = _mm512_unpacklo_epi8(a8, a9);
            b9 = _mm512_unpackhi_epi8(a8, a9);
            ba = _mm512_unpacklo_epi8(aa, ab);
            bb = _mm512_unpackhi_epi8(aa, ab);
            bc = _mm512_unpacklo_epi8(ac, ad);
            bd = _mm512_unpackhi_epi8(ac, ad);
            be = _mm512_unpacklo_epi8(ae, af);
            bf = _mm512_unpackhi_epi8(ae, af);

            a0 = _mm512_unpacklo_epi16(b0, b2);
            a1 = _mm512_unpackhi_epi16(b0, b2);
            a2 = _mm512_unpacklo_epi16(b1, b3);
            a3 = _mm512_unpackhi_epi16(b1, b3);
            a4 = _mm512_unpacklo_epi16(b4, b6);
            a5 = _mm512_unpackhi_epi16(b4, b6);
            a6 = _mm512_unpacklo_epi16(b5, b7);
            a7 = _mm512_unpackhi_epi16(b5, b7);
            a8 = _mm512_unpacklo_epi16(b8, ba);
            a9 = _mm512_unpackhi_epi16(b8, ba);
            aa = _mm512_unpacklo_epi16(b9, bb);
            ab = _mm512_unpackhi_epi16(b9, bb);
            ac = _mm512_unpacklo_epi16(bc, be);
            ad = _mm512_unpackhi_epi16(bc, be);
            ae = _mm512_unpacklo_epi16(bd, bf);
            af = _mm512_unpackhi_epi16(bd, bf);

            b0 = _mm512_unpacklo_epi32(a0, a4);
            b1 = _mm512_unpackhi_epi32(a0, a4);
            b2 = _mm512_unpacklo_epi32(a1, a5);
            b3 = _mm512_unpackhi_epi32(a1, a5);
            b4 = _mm512_unpacklo_epi32(a2, a6);
            b5 = _mm512_unpackhi_epi32(a2, a6);
            b6 = _mm512_unpacklo_epi32(a3, a7);
            b7 = _mm512_unpackhi_epi32(a3, a7);
            b8 = _mm512_unpacklo_epi32(a8, ac);
            b9 = _mm512_unpackhi_epi32(a8, ac);
            ba = _mm512_unpacklo_epi32(a9, ad);
            bb = _mm512_unpackhi_epi32(a9, ad);
            bc = _mm512_unpacklo_epi32(aa, ae);
            bd = _mm512_unpackhi_epi32(aa, ae);
            be = _mm512_unpacklo_epi32(ab, af);
            bf = _mm512_unpackhi_epi32(ab, af);

            a0 = _mm512_unpacklo_epi64(b0, b8);
            a1 = _mm512_unpackhi_epi64(b0, b8);
            a2 = _mm512_unpacklo_epi64(b1, b9);
            a3 = _mm512_unpackhi_epi64(b1, b9);
            a4 = _mm512_unpacklo_epi64(b2, ba);
            a5 = _mm512_unpackhi_epi64(b2, ba);
            a6 = _mm512_unpacklo_epi64(b3, bb);
            a7 = _mm512_unpackhi_epi64(b3, bb);
            a8 = _mm512_unpacklo_epi64(b4, bc);
            a9 = _mm512_unpackhi_epi64(b4, bc);
            aa = _mm512_unpacklo_epi64(b5, bd);
            ab = _mm512_unpackhi_epi64(b5, bd);
            ac = _mm512_unpacklo_epi64(b6, be);
            ad = _mm512_unpackhi_epi64(b6, be);
            ae = _mm512_unpacklo_epi64(b7, bf);
            af = _mm512_unpackhi_epi64(b7, bf);

            _mm512_storeu_si512(dst + 0x0 * dstStride, a0);
            _mm512_storeu_si512(dst + 0x1 * dstStride, a1);
            _mm512_storeu_si512(dst + 0x2 * dstStride, a2);
            _mm512_storeu_si512(dst + 0x3 * dstStride, a3);
            _mm512_storeu_si512(dst + 0x4 * dstStride, a4);
            _mm512_storeu_si512(dst + 0x5 * dstStride, a5);
            _mm512_storeu_si512(dst + 0x6 * dstStride, a6);
            _mm512_storeu_si512(dst + 0x7 * dstStride, a7);
            _mm512_storeu_si512(dst + 0x8 * dstStride, a8);
            _mm512_storeu_si512(dst + 0x9 * dstStride, a9);
            _mm512_storeu_si512(dst + 0xa * dstStride, aa);
            _mm512_storeu_si512(dst + 0xb * dstStride, ab);
            _mm512_storeu_si512(dst + 0xc * dstStride, ac);
            _mm512_storeu_si512(dst + 0xd * dstStride, ad);
            _mm512_storeu_si512(dst + 0xe * dstStride, ae);
            _mm512_storeu_si512(dst + 0xf * dstStride, af);
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K16_TRANSPOSE_2x32x8 = SIMD_MM512_SETR_EPI16(
            0x00, 0x08, 0x10, 0x18, 0x01, 0x09, 0x11, 0x19, 0x02, 0x0a, 0x12, 0x1a, 0x03, 0x0b, 0x13, 0x1b,
            0x04, 0x0c, 0x14, 0x1c, 0x05, 0x0d, 0x15, 0x1d, 0x06, 0x0e, 0x16, 0x1e, 0x07, 0x0f, 0x17, 0x1f);

        SIMD_INLINE void TransformImageTranspose_2x32x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = Load<false>((__m128i*)(src + 0x00 * srcStride), (__m128i*)(src + 0x01 * srcStride), (__m128i*)(src + 0x02 * srcStride), (__m128i*)(src + 0x03 * srcStride));
            a1 = Load<false>((__m128i*)(src + 0x04 * srcStride), (__m128i*)(src + 0x05 * srcStride), (__m128i*)(src + 0x06 * srcStride), (__m128i*)(src + 0x07 * srcStride));
            a2 = Load<false>((__m128i*)(src + 0x08 * srcStride), (__m128i*)(src + 0x09 * srcStride), (__m128i*)(src + 0x0a * srcStride), (__m128i*)(src + 0x0b * srcStride));
            a3 = Load<false>((__m128i*)(src + 0x0c * srcStride), (__m128i*)(src + 0x0d * srcStride), (__m128i*)(src + 0x0e * srcStride), (__m128i*)(src + 0x0f * srcStride));
            a4 = Load<false>((__m128i*)(src + 0x10 * srcStride), (__m128i*)(src + 0x11 * srcStride), (__m128i*)(src + 0x12 * srcStride), (__m128i*)(src + 0x13 * srcStride));
            a5 = Load<false>((__m128i*)(src + 0x14 * srcStride), (__m128i*)(src + 0x15 * srcStride), (__m128i*)(src + 0x16 * srcStride), (__m128i*)(src + 0x17 * srcStride));
            a6 = Load<false>((__m128i*)(src + 0x18 * srcStride), (__m128i*)(src + 0x19 * srcStride), (__m128i*)(src + 0x1a * srcStride), (__m128i*)(src + 0x1b * srcStride));
            a7 = Load<false>((__m128i*)(src + 0x1c * srcStride), (__m128i*)(src + 0x1d * srcStride), (__m128i*)(src + 0x1e * srcStride), (__m128i*)(src + 0x1f * srcStride));

            a0 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a0);
            a1 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a1);
            a2 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a2);
            a3 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a3);
            a4 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a4);
            a5 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a5);
            a6 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a6);
            a7 = _mm512_permutexvar_epi16(K16_TRANSPOSE_2x32x8, a7);

            b0 = _mm512_permutex2var_epi64(a0, K64_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi64(a0, K64_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi64(a1, K64_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi64(a1, K64_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi64(a2, K64_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi64(a2, K64_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi64(a3, K64_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi64(a3, K64_INTERLEAVE_1, a7);

            a0 = _mm512_permutex2var_epi64(b0, K64_INTERLEAVE_0, b4);
            a1 = _mm512_permutex2var_epi64(b0, K64_INTERLEAVE_1, b4);
            a2 = _mm512_permutex2var_epi64(b1, K64_INTERLEAVE_0, b5);
            a3 = _mm512_permutex2var_epi64(b1, K64_INTERLEAVE_1, b5);
            a4 = _mm512_permutex2var_epi64(b2, K64_INTERLEAVE_0, b6);
            a5 = _mm512_permutex2var_epi64(b2, K64_INTERLEAVE_1, b6);
            a6 = _mm512_permutex2var_epi64(b3, K64_INTERLEAVE_0, b7);
            a7 = _mm512_permutex2var_epi64(b3, K64_INTERLEAVE_1, b7);

            b0 = _mm512_permutex2var_epi64(a0, K64_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi64(a0, K64_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi64(a1, K64_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi64(a1, K64_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi64(a2, K64_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi64(a2, K64_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi64(a3, K64_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi64(a3, K64_INTERLEAVE_1, a7);

            _mm512_storeu_si512(dst + 0x0 * dstStride, b0);
            _mm512_storeu_si512(dst + 0x1 * dstStride, b1);
            _mm512_storeu_si512(dst + 0x2 * dstStride, b2);
            _mm512_storeu_si512(dst + 0x3 * dstStride, b3);
            _mm512_storeu_si512(dst + 0x4 * dstStride, b4);
            _mm512_storeu_si512(dst + 0x5 * dstStride, b5);
            _mm512_storeu_si512(dst + 0x6 * dstStride, b6);
            _mm512_storeu_si512(dst + 0x7 * dstStride, b7);
        }

        SIMD_INLINE void TransformImageTranspose_2x32x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf;

            a0 = Load<false>((__m256i*)(src + 0x00 * srcStride), (__m256i*)(src + 0x08 * srcStride));
            a1 = Load<false>((__m256i*)(src + 0x01 * srcStride), (__m256i*)(src + 0x09 * srcStride));
            a2 = Load<false>((__m256i*)(src + 0x02 * srcStride), (__m256i*)(src + 0x0a * srcStride));
            a3 = Load<false>((__m256i*)(src + 0x03 * srcStride), (__m256i*)(src + 0x0b * srcStride));
            a4 = Load<false>((__m256i*)(src + 0x04 * srcStride), (__m256i*)(src + 0x0c * srcStride));
            a5 = Load<false>((__m256i*)(src + 0x05 * srcStride), (__m256i*)(src + 0x0d * srcStride));
            a6 = Load<false>((__m256i*)(src + 0x06 * srcStride), (__m256i*)(src + 0x0e * srcStride));
            a7 = Load<false>((__m256i*)(src + 0x07 * srcStride), (__m256i*)(src + 0x0f * srcStride));
            a8 = Load<false>((__m256i*)(src + 0x10 * srcStride), (__m256i*)(src + 0x18 * srcStride));
            a9 = Load<false>((__m256i*)(src + 0x11 * srcStride), (__m256i*)(src + 0x19 * srcStride));
            aa = Load<false>((__m256i*)(src + 0x12 * srcStride), (__m256i*)(src + 0x1a * srcStride));
            ab = Load<false>((__m256i*)(src + 0x13 * srcStride), (__m256i*)(src + 0x1b * srcStride));
            ac = Load<false>((__m256i*)(src + 0x14 * srcStride), (__m256i*)(src + 0x1c * srcStride));
            ad = Load<false>((__m256i*)(src + 0x15 * srcStride), (__m256i*)(src + 0x1d * srcStride));
            ae = Load<false>((__m256i*)(src + 0x16 * srcStride), (__m256i*)(src + 0x1e * srcStride));
            af = Load<false>((__m256i*)(src + 0x17 * srcStride), (__m256i*)(src + 0x1f * srcStride));

            b0 = _mm512_shuffle_i32x4(a0, a8, 0x88);
            b1 = _mm512_shuffle_i32x4(a1, a9, 0x88);
            b2 = _mm512_shuffle_i32x4(a2, aa, 0x88);
            b3 = _mm512_shuffle_i32x4(a3, ab, 0x88);
            b4 = _mm512_shuffle_i32x4(a4, ac, 0x88);
            b5 = _mm512_shuffle_i32x4(a5, ad, 0x88);
            b6 = _mm512_shuffle_i32x4(a6, ae, 0x88);
            b7 = _mm512_shuffle_i32x4(a7, af, 0x88);
            b8 = _mm512_shuffle_i32x4(a0, a8, 0xDD);
            b9 = _mm512_shuffle_i32x4(a1, a9, 0xDD);
            ba = _mm512_shuffle_i32x4(a2, aa, 0xDD);
            bb = _mm512_shuffle_i32x4(a3, ab, 0xDD);
            bc = _mm512_shuffle_i32x4(a4, ac, 0xDD);
            bd = _mm512_shuffle_i32x4(a5, ad, 0xDD);
            be = _mm512_shuffle_i32x4(a6, ae, 0xDD);
            bf = _mm512_shuffle_i32x4(a7, af, 0xDD);

            a0 = _mm512_unpacklo_epi16(b0, b1);
            a1 = _mm512_unpackhi_epi16(b0, b1);
            a2 = _mm512_unpacklo_epi16(b2, b3);
            a3 = _mm512_unpackhi_epi16(b2, b3);
            a4 = _mm512_unpacklo_epi16(b4, b5);
            a5 = _mm512_unpackhi_epi16(b4, b5);
            a6 = _mm512_unpacklo_epi16(b6, b7);
            a7 = _mm512_unpackhi_epi16(b6, b7);
            a8 = _mm512_unpacklo_epi16(b8, b9);
            a9 = _mm512_unpackhi_epi16(b8, b9);
            aa = _mm512_unpacklo_epi16(ba, bb);
            ab = _mm512_unpackhi_epi16(ba, bb);
            ac = _mm512_unpacklo_epi16(bc, bd);
            ad = _mm512_unpackhi_epi16(bc, bd);
            ae = _mm512_unpacklo_epi16(be, bf);
            af = _mm512_unpackhi_epi16(be, bf);

            b0 = _mm512_unpacklo_epi32(a0, a2);
            b1 = _mm512_unpackhi_epi32(a0, a2);
            b2 = _mm512_unpacklo_epi32(a1, a3);
            b3 = _mm512_unpackhi_epi32(a1, a3);
            b4 = _mm512_unpacklo_epi32(a4, a6);
            b5 = _mm512_unpackhi_epi32(a4, a6);
            b6 = _mm512_unpacklo_epi32(a5, a7);
            b7 = _mm512_unpackhi_epi32(a5, a7);
            b8 = _mm512_unpacklo_epi32(a8, aa);
            b9 = _mm512_unpackhi_epi32(a8, aa);
            ba = _mm512_unpacklo_epi32(a9, ab);
            bb = _mm512_unpackhi_epi32(a9, ab);
            bc = _mm512_unpacklo_epi32(ac, ae);
            bd = _mm512_unpackhi_epi32(ac, ae);
            be = _mm512_unpacklo_epi32(ad, af);
            bf = _mm512_unpackhi_epi32(ad, af);

            a0 = _mm512_unpacklo_epi64(b0, b4);
            a1 = _mm512_unpackhi_epi64(b0, b4);
            a2 = _mm512_unpacklo_epi64(b1, b5);
            a3 = _mm512_unpackhi_epi64(b1, b5);
            a4 = _mm512_unpacklo_epi64(b2, b6);
            a5 = _mm512_unpackhi_epi64(b2, b6);
            a6 = _mm512_unpacklo_epi64(b3, b7);
            a7 = _mm512_unpackhi_epi64(b3, b7);
            a8 = _mm512_unpacklo_epi64(b8, bc);
            a9 = _mm512_unpackhi_epi64(b8, bc);
            aa = _mm512_unpacklo_epi64(b9, bd);
            ab = _mm512_unpackhi_epi64(b9, bd);
            ac = _mm512_unpacklo_epi64(ba, be);
            ad = _mm512_unpackhi_epi64(ba, be);
            ae = _mm512_unpacklo_epi64(bb, bf);
            af = _mm512_unpackhi_epi64(bb, bf);

            _mm512_storeu_si512(dst + 0x0 * dstStride, a0);
            _mm512_storeu_si512(dst + 0x1 * dstStride, a1);
            _mm512_storeu_si512(dst + 0x2 * dstStride, a2);
            _mm512_storeu_si512(dst + 0x3 * dstStride, a3);
            _mm512_storeu_si512(dst + 0x4 * dstStride, a4);
            _mm512_storeu_si512(dst + 0x5 * dstStride, a5);
            _mm512_storeu_si512(dst + 0x6 * dstStride, a6);
            _mm512_storeu_si512(dst + 0x7 * dstStride, a7);
            _mm512_storeu_si512(dst + 0x8 * dstStride, a8);
            _mm512_storeu_si512(dst + 0x9 * dstStride, a9);
            _mm512_storeu_si512(dst + 0xa * dstStride, aa);
            _mm512_storeu_si512(dst + 0xb * dstStride, ab);
            _mm512_storeu_si512(dst + 0xc * dstStride, ac);
            _mm512_storeu_si512(dst + 0xd * dstStride, ad);
            _mm512_storeu_si512(dst + 0xe * dstStride, ae);
            _mm512_storeu_si512(dst + 0xf * dstStride, af);
        }

        //-----------------------------------------------------------------------------------------

        const __m512i K32_3x16x16_I_0 = SIMD_MM512_SETR_EPI32(0x00, 0x01, 0x02, 0x00, 0x03, 0x04, 0x05, 0x00, 0x10, 0x11, 0x12, 0x00, 0x13, 0x14, 0x15, 0x00);
        const __m512i K32_3x16x16_I_1 = SIMD_MM512_SETR_EPI32(0x06, 0x07, 0x08, 0x00, 0x09, 0x0a, 0x0b, 0x00, 0x16, 0x17, 0x18, 0x10, 0x19, 0x1a, 0x1b, 0x00);

        const __m512i K8_3x16x16_I = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1,
            0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);

        const __m512i K8_3x16x16_O = SIMD_MM512_SETR_EPI8(
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xa, 0xc, 0xd, 0xe, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xa, 0xc, 0xd, 0xe, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xa, 0xc, 0xd, 0xe, -1, -1, -1, -1,
            0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xa, 0xc, 0xd, 0xe, -1, -1, -1, -1);

        const __m512i K32_3x16x16_O = SIMD_MM512_SETR_EPI32(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xa, 0xc, 0xd, 0xe, 0x0, 0x0, 0x0, 0x0);

        SIMD_INLINE void TransformImageTranspose_3x16x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf;

            a0 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x0 * srcStride);
            a1 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x1 * srcStride);
            a2 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x2 * srcStride);
            a3 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x3 * srcStride);
            a4 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x4 * srcStride);
            a5 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x5 * srcStride);
            a6 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x6 * srcStride);
            a7 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x7 * srcStride);
            a8 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x8 * srcStride);
            a9 = _mm512_maskz_loadu_epi32(0xFFF, src + 0x9 * srcStride);
            aa = _mm512_maskz_loadu_epi32(0xFFF, src + 0xa * srcStride);
            ab = _mm512_maskz_loadu_epi32(0xFFF, src + 0xb * srcStride);
            ac = _mm512_maskz_loadu_epi32(0xFFF, src + 0xc * srcStride);
            ad = _mm512_maskz_loadu_epi32(0xFFF, src + 0xd * srcStride);
            ae = _mm512_maskz_loadu_epi32(0xFFF, src + 0xe * srcStride);
            af = _mm512_maskz_loadu_epi32(0xFFF, src + 0xf * srcStride);

            b0 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a0, K32_3x16x16_I_0, a4), K8_3x16x16_I);
            b1 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a1, K32_3x16x16_I_0, a5), K8_3x16x16_I);
            b2 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a2, K32_3x16x16_I_0, a6), K8_3x16x16_I);
            b3 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a3, K32_3x16x16_I_0, a7), K8_3x16x16_I);
            b4 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a0, K32_3x16x16_I_1, a4), K8_3x16x16_I);
            b5 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a1, K32_3x16x16_I_1, a5), K8_3x16x16_I);
            b6 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a2, K32_3x16x16_I_1, a6), K8_3x16x16_I);
            b7 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a3, K32_3x16x16_I_1, a7), K8_3x16x16_I);
            b8 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a8, K32_3x16x16_I_0, ac), K8_3x16x16_I);
            b9 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a9, K32_3x16x16_I_0, ad), K8_3x16x16_I);
            ba = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(aa, K32_3x16x16_I_0, ae), K8_3x16x16_I);
            bb = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(ab, K32_3x16x16_I_0, af), K8_3x16x16_I);
            bc = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a8, K32_3x16x16_I_1, ac), K8_3x16x16_I);
            bd = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(a9, K32_3x16x16_I_1, ad), K8_3x16x16_I);
            be = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(aa, K32_3x16x16_I_1, ae), K8_3x16x16_I);
            bf = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(ab, K32_3x16x16_I_1, af), K8_3x16x16_I);

            a0 = _mm512_shuffle_i32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_i32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_i32x4(b2, ba, 0x88);
            a3 = _mm512_shuffle_i32x4(b3, bb, 0x88);
            a4 = _mm512_shuffle_i32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_i32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_i32x4(b2, ba, 0xDD);
            a7 = _mm512_shuffle_i32x4(b3, bb, 0xDD);
            a8 = _mm512_shuffle_i32x4(b4, bc, 0x88);
            a9 = _mm512_shuffle_i32x4(b5, bd, 0x88);
            aa = _mm512_shuffle_i32x4(b6, be, 0x88);
            ab = _mm512_shuffle_i32x4(b7, bf, 0x88);
            ac = _mm512_shuffle_i32x4(b4, bc, 0xDD);
            ad = _mm512_shuffle_i32x4(b5, bd, 0xDD);
            ae = _mm512_shuffle_i32x4(b6, be, 0xDD);
            af = _mm512_shuffle_i32x4(b7, bf, 0xDD);

            b0 = _mm512_unpacklo_epi32(a0, a1);
            b1 = _mm512_unpackhi_epi32(a0, a1);
            b2 = _mm512_unpacklo_epi32(a2, a3);
            b3 = _mm512_unpackhi_epi32(a2, a3);
            b4 = _mm512_unpacklo_epi32(a4, a5);
            b5 = _mm512_unpackhi_epi32(a4, a5);
            b6 = _mm512_unpacklo_epi32(a6, a7);
            b7 = _mm512_unpackhi_epi32(a6, a7);
            b8 = _mm512_unpacklo_epi32(a8, a9);
            b9 = _mm512_unpackhi_epi32(a8, a9);
            ba = _mm512_unpacklo_epi32(aa, ab);
            bb = _mm512_unpackhi_epi32(aa, ab);
            bc = _mm512_unpacklo_epi32(ac, ad);
            bd = _mm512_unpackhi_epi32(ac, ad);
            be = _mm512_unpacklo_epi32(ae, af);
            bf = _mm512_unpackhi_epi32(ae, af);

            a0 = _mm512_unpacklo_epi64(b0, b2);
            a1 = _mm512_unpackhi_epi64(b0, b2);
            a2 = _mm512_unpacklo_epi64(b1, b3);
            a3 = _mm512_unpackhi_epi64(b1, b3);
            a4 = _mm512_unpacklo_epi64(b4, b6);
            a5 = _mm512_unpackhi_epi64(b4, b6);
            a6 = _mm512_unpacklo_epi64(b5, b7);
            a7 = _mm512_unpackhi_epi64(b5, b7);
            a8 = _mm512_unpacklo_epi64(b8, ba);
            a9 = _mm512_unpackhi_epi64(b8, ba);
            aa = _mm512_unpacklo_epi64(b9, bb);
            ab = _mm512_unpackhi_epi64(b9, bb);
            ac = _mm512_unpacklo_epi64(bc, be);
            ad = _mm512_unpackhi_epi64(bc, be);
            ae = _mm512_unpacklo_epi64(bd, bf);
            af = _mm512_unpackhi_epi64(bd, bf);

            a0 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a0, K8_3x16x16_O));
            a1 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a1, K8_3x16x16_O));
            a2 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a2, K8_3x16x16_O));
            a3 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a3, K8_3x16x16_O));
            a4 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a4, K8_3x16x16_O));
            a5 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a5, K8_3x16x16_O));
            a6 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a6, K8_3x16x16_O));
            a7 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a7, K8_3x16x16_O));
            a8 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a8, K8_3x16x16_O));
            a9 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a9, K8_3x16x16_O));
            aa = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(aa, K8_3x16x16_O));
            ab = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(ab, K8_3x16x16_O));
            ac = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(ac, K8_3x16x16_O));
            ad = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(ad, K8_3x16x16_O));
            ae = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(ae, K8_3x16x16_O));
            af = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(af, K8_3x16x16_O));

            _mm512_mask_storeu_epi32(dst + 0x0 * dstStride, 0xFFF, a0);
            _mm512_mask_storeu_epi32(dst + 0x1 * dstStride, 0xFFF, a1);
            _mm512_mask_storeu_epi32(dst + 0x2 * dstStride, 0xFFF, a2);
            _mm512_mask_storeu_epi32(dst + 0x3 * dstStride, 0xFFF, a3);
            _mm512_mask_storeu_epi32(dst + 0x4 * dstStride, 0xFFF, a4);
            _mm512_mask_storeu_epi32(dst + 0x5 * dstStride, 0xFFF, a5);
            _mm512_mask_storeu_epi32(dst + 0x6 * dstStride, 0xFFF, a6);
            _mm512_mask_storeu_epi32(dst + 0x7 * dstStride, 0xFFF, a7);
            _mm512_mask_storeu_epi32(dst + 0x8 * dstStride, 0xFFF, a8);
            _mm512_mask_storeu_epi32(dst + 0x9 * dstStride, 0xFFF, a9);
            _mm512_mask_storeu_epi32(dst + 0xa * dstStride, 0xFFF, aa);
            _mm512_mask_storeu_epi32(dst + 0xb * dstStride, 0xFFF, ab);
            _mm512_mask_storeu_epi32(dst + 0xc * dstStride, 0xFFF, ac);
            _mm512_mask_storeu_epi32(dst + 0xd * dstStride, 0xFFF, ad);
            _mm512_mask_storeu_epi32(dst + 0xe * dstStride, 0xFFF, ae);
            _mm512_mask_storeu_epi32(dst + 0xf * dstStride, 0xFFF, af);
        }

        SIMD_INLINE void TransformImageTranspose_3x16x4(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = Load<false>((__m128i*)(src + 0x00 * srcStride), (__m128i*)(src + 0x04 * srcStride), (__m128i*)(src + 0x08 * srcStride), (__m128i*)(src + 0x0c * srcStride));
            a1 = Load<false>((__m128i*)(src + 0x01 * srcStride), (__m128i*)(src + 0x05 * srcStride), (__m128i*)(src + 0x09 * srcStride), (__m128i*)(src + 0x0d * srcStride));
            a2 = Load<false>((__m128i*)(src + 0x02 * srcStride), (__m128i*)(src + 0x06 * srcStride), (__m128i*)(src + 0x0a * srcStride), (__m128i*)(src + 0x0e * srcStride));
            a3 = Load<false>((__m128i*)(src + 0x03 * srcStride), (__m128i*)(src + 0x07 * srcStride), (__m128i*)(src + 0x0b * srcStride), (__m128i*)(src + 0x0f * srcStride));

            a0 = _mm512_shuffle_epi8(a0, K8_3x16x16_I);
            a1 = _mm512_shuffle_epi8(a1, K8_3x16x16_I);
            a2 = _mm512_shuffle_epi8(a2, K8_3x16x16_I);
            a3 = _mm512_shuffle_epi8(a3, K8_3x16x16_I);

            b0 = _mm512_unpacklo_epi32(a0, a1);
            b1 = _mm512_unpackhi_epi32(a0, a1);
            b2 = _mm512_unpacklo_epi32(a2, a3);
            b3 = _mm512_unpackhi_epi32(a2, a3);

            a0 = _mm512_unpacklo_epi64(b0, b2);
            a1 = _mm512_unpackhi_epi64(b0, b2);
            a2 = _mm512_unpacklo_epi64(b1, b3);
            a3 = _mm512_unpackhi_epi64(b1, b3);

            a0 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a0, K8_3x16x16_O));
            a1 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a1, K8_3x16x16_O));
            a2 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a2, K8_3x16x16_O));
            a3 = _mm512_permutexvar_epi32(K32_3x16x16_O, _mm512_shuffle_epi8(a3, K8_3x16x16_O));

            _mm512_mask_storeu_epi32(dst + 0x0 * dstStride, 0xFFF, a0);
            _mm512_mask_storeu_epi32(dst + 0x1 * dstStride, 0xFFF, a1);
            _mm512_mask_storeu_epi32(dst + 0x2 * dstStride, 0xFFF, a2);
            _mm512_mask_storeu_epi32(dst + 0x3 * dstStride, 0xFFF, a3);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TransformImageTranspose_4x4x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, b0, b1, b2, b3;

            a0 = _mm512_loadu_si512(src + 0x0 * srcStride);
            a1 = _mm512_loadu_si512(src + 0x1 * srcStride);
            a2 = _mm512_loadu_si512(src + 0x2 * srcStride);
            a3 = _mm512_loadu_si512(src + 0x3 * srcStride);

            b0 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_0, a2);
            b1 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_1, a2);
            b2 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_0, a3);
            b3 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_1, a3);

            a0 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_0, b2);
            a1 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_1, b2);
            a2 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_0, b3);
            a3 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_1, b3);

            Store<false>((__m128i*)(dst + 0x0 * dstStride), (__m128i*)(dst + 0x1 * dstStride), (__m128i*)(dst + 0x2 * dstStride), (__m128i*)(dst + 0x3 * dstStride), a0);
            Store<false>((__m128i*)(dst + 0x4 * dstStride), (__m128i*)(dst + 0x5 * dstStride), (__m128i*)(dst + 0x6 * dstStride), (__m128i*)(dst + 0x7 * dstStride), a1);
            Store<false>((__m128i*)(dst + 0x8 * dstStride), (__m128i*)(dst + 0x9 * dstStride), (__m128i*)(dst + 0xa * dstStride), (__m128i*)(dst + 0xb * dstStride), a2);
            Store<false>((__m128i*)(dst + 0xc * dstStride), (__m128i*)(dst + 0xd * dstStride), (__m128i*)(dst + 0xe * dstStride), (__m128i*)(dst + 0xf * dstStride), a3);
        }

        SIMD_INLINE void TransformImageTranspose_4x8x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm512_loadu_si512(src + 0x0 * srcStride);
            a1 = _mm512_loadu_si512(src + 0x1 * srcStride);
            a2 = _mm512_loadu_si512(src + 0x2 * srcStride);
            a3 = _mm512_loadu_si512(src + 0x3 * srcStride);
            a4 = _mm512_loadu_si512(src + 0x4 * srcStride);
            a5 = _mm512_loadu_si512(src + 0x5 * srcStride);
            a6 = _mm512_loadu_si512(src + 0x6 * srcStride);
            a7 = _mm512_loadu_si512(src + 0x7 * srcStride);

            b0 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_1, a7);

            a0 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_0, b4);
            a1 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_1, b4);
            a2 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_0, b5);
            a3 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_1, b5);
            a4 = _mm512_permutex2var_epi32(b2, K32_INTERLEAVE_0, b6);
            a5 = _mm512_permutex2var_epi32(b2, K32_INTERLEAVE_1, b6);
            a6 = _mm512_permutex2var_epi32(b3, K32_INTERLEAVE_0, b7);
            a7 = _mm512_permutex2var_epi32(b3, K32_INTERLEAVE_1, b7);

            b0 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_1, a7);

            Store<false>((__m256i*)(dst + 0x0 * dstStride), (__m256i*)(dst + 0x8 * dstStride), b0);
            Store<false>((__m256i*)(dst + 0x1 * dstStride), (__m256i*)(dst + 0x9 * dstStride), b1);
            Store<false>((__m256i*)(dst + 0x2 * dstStride), (__m256i*)(dst + 0xa * dstStride), b2);
            Store<false>((__m256i*)(dst + 0x3 * dstStride), (__m256i*)(dst + 0xb * dstStride), b3);
            Store<false>((__m256i*)(dst + 0x4 * dstStride), (__m256i*)(dst + 0xc * dstStride), b4);
            Store<false>((__m256i*)(dst + 0x5 * dstStride), (__m256i*)(dst + 0xd * dstStride), b5);
            Store<false>((__m256i*)(dst + 0x6 * dstStride), (__m256i*)(dst + 0xe * dstStride), b6);
            Store<false>((__m256i*)(dst + 0x7 * dstStride), (__m256i*)(dst + 0xf * dstStride), b7);
        }

        const __m512i K32_PERM_4_16_8 = SIMD_MM512_SETR_EPI32(0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb, 0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf);

        SIMD_INLINE void TransformImageTranspose_4x16x8(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x0 * srcStride), (__m256i*)(src + 0x8 * srcStride)));
            a1 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x1 * srcStride), (__m256i*)(src + 0x9 * srcStride)));
            a2 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x2 * srcStride), (__m256i*)(src + 0xa * srcStride)));
            a3 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x3 * srcStride), (__m256i*)(src + 0xb * srcStride)));
            a4 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x4 * srcStride), (__m256i*)(src + 0xc * srcStride)));
            a5 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x5 * srcStride), (__m256i*)(src + 0xd * srcStride)));
            a6 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x6 * srcStride), (__m256i*)(src + 0xe * srcStride)));
            a7 = _mm512_permutexvar_epi32(K32_PERM_4_16_8, Load<false>((__m256i*)(src + 0x7 * srcStride), (__m256i*)(src + 0xf * srcStride)));

            b0 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_1, a7);

            a0 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_0, b4);
            a1 = _mm512_permutex2var_epi32(b0, K32_INTERLEAVE_1, b4);
            a2 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_0, b5);
            a3 = _mm512_permutex2var_epi32(b1, K32_INTERLEAVE_1, b5);
            a4 = _mm512_permutex2var_epi32(b2, K32_INTERLEAVE_0, b6);
            a5 = _mm512_permutex2var_epi32(b2, K32_INTERLEAVE_1, b6);
            a6 = _mm512_permutex2var_epi32(b3, K32_INTERLEAVE_0, b7);
            a7 = _mm512_permutex2var_epi32(b3, K32_INTERLEAVE_1, b7);

            b0 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_0, a4);
            b1 = _mm512_permutex2var_epi32(a0, K32_INTERLEAVE_1, a4);
            b2 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_0, a5);
            b3 = _mm512_permutex2var_epi32(a1, K32_INTERLEAVE_1, a5);
            b4 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_0, a6);
            b5 = _mm512_permutex2var_epi32(a2, K32_INTERLEAVE_1, a6);
            b6 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_0, a7);
            b7 = _mm512_permutex2var_epi32(a3, K32_INTERLEAVE_1, a7);

            _mm512_storeu_si512(dst + 0x0 * dstStride, b0);
            _mm512_storeu_si512(dst + 0x1 * dstStride, b1);
            _mm512_storeu_si512(dst + 0x2 * dstStride, b2);
            _mm512_storeu_si512(dst + 0x3 * dstStride, b3);
            _mm512_storeu_si512(dst + 0x4 * dstStride, b4);
            _mm512_storeu_si512(dst + 0x5 * dstStride, b5);
            _mm512_storeu_si512(dst + 0x6 * dstStride, b6);
            _mm512_storeu_si512(dst + 0x7 * dstStride, b7);
        }

        SIMD_INLINE void TransformImageTranspose_4x16x16(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aa, ab, ac, ad, ae, af, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, ba, bb, bc, bd, be, bf;

            a0 = _mm512_loadu_si512(src + 0x0 * srcStride);
            a1 = _mm512_loadu_si512(src + 0x1 * srcStride);
            a2 = _mm512_loadu_si512(src + 0x2 * srcStride);
            a3 = _mm512_loadu_si512(src + 0x3 * srcStride);
            a4 = _mm512_loadu_si512(src + 0x4 * srcStride);
            a5 = _mm512_loadu_si512(src + 0x5 * srcStride);
            a6 = _mm512_loadu_si512(src + 0x6 * srcStride);
            a7 = _mm512_loadu_si512(src + 0x7 * srcStride);
            a8 = _mm512_loadu_si512(src + 0x8 * srcStride);
            a9 = _mm512_loadu_si512(src + 0x9 * srcStride);
            aa = _mm512_loadu_si512(src + 0xa * srcStride);
            ab = _mm512_loadu_si512(src + 0xb * srcStride);
            ac = _mm512_loadu_si512(src + 0xc * srcStride);
            ad = _mm512_loadu_si512(src + 0xd * srcStride);
            ae = _mm512_loadu_si512(src + 0xe * srcStride);
            af = _mm512_loadu_si512(src + 0xf * srcStride);

            b0 = _mm512_shuffle_i32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_i32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_i32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_i32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_i32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_i32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_i32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_i32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_i32x4(a8, ac, 0x44);
            b9 = _mm512_shuffle_i32x4(a9, ad, 0x44);
            ba = _mm512_shuffle_i32x4(aa, ae, 0x44);
            bb = _mm512_shuffle_i32x4(ab, af, 0x44);
            bc = _mm512_shuffle_i32x4(a8, ac, 0xEE);
            bd = _mm512_shuffle_i32x4(a9, ad, 0xEE);
            be = _mm512_shuffle_i32x4(aa, ae, 0xEE);
            bf = _mm512_shuffle_i32x4(ab, af, 0xEE);

            a0 = _mm512_shuffle_i32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_i32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_i32x4(b2, ba, 0x88);
            a3 = _mm512_shuffle_i32x4(b3, bb, 0x88);
            a4 = _mm512_shuffle_i32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_i32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_i32x4(b2, ba, 0xDD);
            a7 = _mm512_shuffle_i32x4(b3, bb, 0xDD);
            a8 = _mm512_shuffle_i32x4(b4, bc, 0x88);
            a9 = _mm512_shuffle_i32x4(b5, bd, 0x88);
            aa = _mm512_shuffle_i32x4(b6, be, 0x88);
            ab = _mm512_shuffle_i32x4(b7, bf, 0x88);
            ac = _mm512_shuffle_i32x4(b4, bc, 0xDD);
            ad = _mm512_shuffle_i32x4(b5, bd, 0xDD);
            ae = _mm512_shuffle_i32x4(b6, be, 0xDD);
            af = _mm512_shuffle_i32x4(b7, bf, 0xDD);

            b0 = _mm512_unpacklo_epi32(a0, a1);
            b1 = _mm512_unpackhi_epi32(a0, a1);
            b2 = _mm512_unpacklo_epi32(a2, a3);
            b3 = _mm512_unpackhi_epi32(a2, a3);
            b4 = _mm512_unpacklo_epi32(a4, a5);
            b5 = _mm512_unpackhi_epi32(a4, a5);
            b6 = _mm512_unpacklo_epi32(a6, a7);
            b7 = _mm512_unpackhi_epi32(a6, a7);
            b8 = _mm512_unpacklo_epi32(a8, a9);
            b9 = _mm512_unpackhi_epi32(a8, a9);
            ba = _mm512_unpacklo_epi32(aa, ab);
            bb = _mm512_unpackhi_epi32(aa, ab);
            bc = _mm512_unpacklo_epi32(ac, ad);
            bd = _mm512_unpackhi_epi32(ac, ad);
            be = _mm512_unpacklo_epi32(ae, af);
            bf = _mm512_unpackhi_epi32(ae, af);

            a0 = _mm512_unpacklo_epi64(b0, b2);
            a1 = _mm512_unpackhi_epi64(b0, b2);
            a2 = _mm512_unpacklo_epi64(b1, b3);
            a3 = _mm512_unpackhi_epi64(b1, b3);
            a4 = _mm512_unpacklo_epi64(b4, b6);
            a5 = _mm512_unpackhi_epi64(b4, b6);
            a6 = _mm512_unpacklo_epi64(b5, b7);
            a7 = _mm512_unpackhi_epi64(b5, b7);
            a8 = _mm512_unpacklo_epi64(b8, ba);
            a9 = _mm512_unpackhi_epi64(b8, ba);
            aa = _mm512_unpacklo_epi64(b9, bb);
            ab = _mm512_unpackhi_epi64(b9, bb);
            ac = _mm512_unpacklo_epi64(bc, be);
            ad = _mm512_unpackhi_epi64(bc, be);
            ae = _mm512_unpacklo_epi64(bd, bf);
            af = _mm512_unpackhi_epi64(bd, bf);

            _mm512_storeu_si512(dst + 0x0 * dstStride, a0);
            _mm512_storeu_si512(dst + 0x1 * dstStride, a1);
            _mm512_storeu_si512(dst + 0x2 * dstStride, a2);
            _mm512_storeu_si512(dst + 0x3 * dstStride, a3);
            _mm512_storeu_si512(dst + 0x4 * dstStride, a4);
            _mm512_storeu_si512(dst + 0x5 * dstStride, a5);
            _mm512_storeu_si512(dst + 0x6 * dstStride, a6);
            _mm512_storeu_si512(dst + 0x7 * dstStride, a7);
            _mm512_storeu_si512(dst + 0x8 * dstStride, a8);
            _mm512_storeu_si512(dst + 0x9 * dstStride, a9);
            _mm512_storeu_si512(dst + 0xa * dstStride, aa);
            _mm512_storeu_si512(dst + 0xb * dstStride, ab);
            _mm512_storeu_si512(dst + 0xc * dstStride, ac);
            _mm512_storeu_si512(dst + 0xd * dstStride, ad);
            _mm512_storeu_si512(dst + 0xe * dstStride, ae);
            _mm512_storeu_si512(dst + 0xf * dstStride, af);
        }

        //-----------------------------------------------------------------------------------------

        struct ImageTransforms : public Avx2::ImageTransforms
        {
            ImageTransforms();
        };
    }
#endif
}

#endif//__SimdTransform_h__

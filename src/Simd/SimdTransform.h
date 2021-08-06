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
            typedef void(*TransformPtr)(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride);

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
}

#endif//__SimdTransform_h__

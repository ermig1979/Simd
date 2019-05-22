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
#include "Simd/SimdDefs.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
#ifdef SIMD_SSSE3_ENABLE    
    namespace Ssse3
    {
        const __m128i K8_TURN_H1 = SIMD_MM_SETR_EPI8(0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8, 0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0);
        const __m128i K8_TURN_H2 = SIMD_MM_SETR_EPI8(0xE, 0xF, 0xC, 0xD, 0xA, 0xB, 0x8, 0x9, 0x6, 0x7, 0x4, 0x5, 0x2, 0x3, 0x0, 0x1);
        const __m128i K8_TURN_H3_02 = SIMD_MM_SETR_EPI8(0xD, 0xE, 0xF, 0xA, 0xB, 0xC, 0x7, 0x8, 0x9, 0x4, 0x5, 0x6, 0x1, 0x2, 0x3, -1);
        const __m128i K8_TURN_H3_01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xE);
        const __m128i K8_TURN_H3_12 = SIMD_MM_SETR_EPI8(-1, 0x0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_TURN_H3_11 = SIMD_MM_SETR_EPI8(0xF, -1, 0xB, 0xC, 0xD, 0x8, 0x9, 0xA, 0x5, 0x6, 0x7, 0x2, 0x3, 0x4, -1, 0x0);
        const __m128i K8_TURN_H3_10 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xF, -1);
        const __m128i K8_TURN_H3_21 = SIMD_MM_SETR_EPI8(0x1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i K8_TURN_H3_20 = SIMD_MM_SETR_EPI8(-1, 0xC, 0xD, 0xE, 0x9, 0xA, 0xB, 0x6, 0x7, 0x8, 0x3, 0x4, 0x5, 0x0, 0x1, 0x2);
        const __m128i K8_TURN_H4 = SIMD_MM_SETR_EPI8(0xC, 0xD, 0xE, 0xF, 0x8, 0x9, 0xA, 0xB, 0x4, 0x5, 0x6, 0x7, 0x0, 0x1, 0x2, 0x3);

        template<size_t N> SIMD_INLINE void CopyPixel(const uint8_t * src, uint8_t * dst)
        {
            for (size_t i = 0; i < N; ++i)
                dst[i] = src[i];
        }

        template<> SIMD_INLINE void CopyPixel<1>(const uint8_t * src, uint8_t * dst)
        {
            dst[0] = src[0];
        }

        template<> SIMD_INLINE void CopyPixel<2>(const uint8_t * src, uint8_t * dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
        }

        template<> SIMD_INLINE void CopyPixel<3>(const uint8_t * src, uint8_t * dst)
        {
            ((uint16_t*)dst)[0] = ((uint16_t*)src)[0];
            dst[2] = src[2];
        }

        template<> SIMD_INLINE void CopyPixel<4>(const uint8_t * src, uint8_t * dst)
        {
            ((uint32_t*)dst)[0] = ((uint32_t*)src)[0];
        }

        template<size_t N> void TransformImageRotate0(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t rowSize = width * N;
            for (size_t row = 0; row < height; ++row)
            {
                memcpy(dst, src, rowSize);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageRotate90(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (width - 1)*dstStride;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * dstStride);
                src += srcStride;
                dst += N;
            }
        }

        template<size_t N> SIMD_INLINE void TransformImageRotate180A(const uint8_t * src, uint8_t * dst)
        {
            dst += (A - 1)*N;
            for (size_t i = 0; i < A; ++i)
                CopyPixel<N>(src + i * N, dst - i * N);
        }

        template<> SIMD_INLINE void TransformImageRotate180A<1>(const uint8_t * src, uint8_t * dst)
        {
            _mm_storeu_si128((__m128i*)dst, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src), K8_TURN_H1));
        }

        template<> SIMD_INLINE void TransformImageRotate180A<2>(const uint8_t * src, uint8_t * dst)
        {
            _mm_storeu_si128((__m128i*)dst + 1, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 0), K8_TURN_H2));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 1), K8_TURN_H2));
        }

        template<> SIMD_INLINE void TransformImageRotate180A<3>(const uint8_t * src, uint8_t * dst)
        {
            __m128i s0 = _mm_loadu_si128((__m128i*)src + 0);
            __m128i s1 = _mm_loadu_si128((__m128i*)src + 1);
            __m128i s2 = _mm_loadu_si128((__m128i*)src + 2);
            _mm_storeu_si128((__m128i*)dst + 0, _mm_or_si128(_mm_shuffle_epi8(s2, K8_TURN_H3_02), _mm_shuffle_epi8(s1, K8_TURN_H3_01)));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(s2, K8_TURN_H3_12), _mm_shuffle_epi8(s1, K8_TURN_H3_11)), _mm_shuffle_epi8(s0, K8_TURN_H3_10)));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_or_si128(_mm_shuffle_epi8(s1, K8_TURN_H3_21), _mm_shuffle_epi8(s0, K8_TURN_H3_20)));
        }

        template<> SIMD_INLINE void TransformImageRotate180A<4>(const uint8_t * src, uint8_t * dst)
        {
            _mm_storeu_si128((__m128i*)dst + 3, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 0), K8_TURN_H4));
            _mm_storeu_si128((__m128i*)dst + 2, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 1), K8_TURN_H4));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 2), K8_TURN_H4));
            _mm_storeu_si128((__m128i*)dst + 0, _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)src + 3), K8_TURN_H4));
        }

        template<size_t N> SIMD_INLINE void TransformImageRotate180QA(const uint8_t * src, uint8_t * dst)
        {
            TransformImageRotate180A<N>(src + 0 * N * A, dst - 0 * N * A);
            TransformImageRotate180A<N>(src + 1 * N * A, dst - 1 * N * A);
            TransformImageRotate180A<N>(src + 2 * N * A, dst - 2 * N * A);
            TransformImageRotate180A<N>(src + 3 * N * A, dst - 3 * N * A);
        }

        template<size_t N> void TransformImageRotate180(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (height - 1)*dstStride + (width - A)*N;
            size_t widthA = AlignLo(width, A);
            size_t widthQA = AlignLo(width, QA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthQA; col += QA)
                    TransformImageRotate180QA<N>(src + col * N, dst - col * N);
                for (; col < widthA; col += A)
                    TransformImageRotate180A<N>(src + col * N, dst - col * N);
                if(col < width)
                    TransformImageRotate180A<N>(src + (width - A) * N, dst - (width - A) * N);
                src += srcStride;
                dst -= dstStride;
            }
        }

        template<size_t N> void TransformImageRotate270(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (height - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst + col * dstStride);
                src += srcStride;
                dst -= N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate0(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst + col * dstStride);
                src += srcStride;
                dst += N;
            }
        }

        __m128i K8_SHUFFLE_BGR_TO_BGRA = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, -1, 0x3, 0x4, 0x5, -1, 0x6, 0x7, 0x8, -1, 0x9, 0xA, 0xB, -1);
        __m128i K8_SHUFFLE_BGRA_TO_BGR = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x2, 0x4, 0x5, 0x6, 0x8, 0x9, 0xA, 0xC, 0xD, 0xE, -1, -1, -1, -1);

        SIMD_INLINE void TransformImageTransposeRotate0_3x4x4(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
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

        template<> void TransformImageTransposeRotate0<3>(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t width4 = AlignLo(width - 2, 4);
            size_t height4 = AlignLo(height, 4);
            size_t row = 0;
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    TransformImageTransposeRotate0_3x4x4(src + col * 3, srcStride, dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        CopyPixel<3>(src + col * 3 + i * srcStride, dst + col * dstStride + i * 3);
                src += 4 * srcStride;
                dst += 12;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<3>(src + col * 3, dst + col * dstStride);
                src += srcStride;
                dst += 3;
            }
        }

        SIMD_INLINE void TransformImageTransposeRotate0_4x4x4(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride)
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

        template<> void TransformImageTransposeRotate0<4>(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t width4 = AlignLo(width, 4);
            size_t height4 = AlignLo(height, 4);
            size_t row = 0;
            for (; row < height4; row += 4)
            {
                size_t col = 0;
                for (; col < width4; col += 4)
                    TransformImageTransposeRotate0_4x4x4(src + col * 4, srcStride,  dst + col * dstStride, dstStride);
                for (; col < width; ++col)
                    for (size_t i = 0; i < 4; ++i)
                        CopyPixel<4>(src + col * 4 + i*srcStride, dst + col * dstStride + i*4);
                src += 4*srcStride;
                dst += 16;
            }
            for (; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<4>(src + col * 4, dst + col * dstStride);
                src += srcStride;
                dst += 4;
            }
        }

        template<size_t N> void TransformImageTransposeRotate90(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (width - A)*N;
            size_t widthA = AlignLo(width, A);
            size_t widthQA = AlignLo(width, QA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthQA; col += QA)
                    TransformImageRotate180QA<N>(src + col * N, dst - col * N);
                for (; col < widthA; col += A)
                    TransformImageRotate180A<N>(src + col * N, dst - col * N);
                if (col < width)
                    TransformImageRotate180A<N>(src + (width - A) * N, dst - (width - A) * N);
                src += srcStride;
                dst += dstStride;
            }
        }

        template<size_t N> void TransformImageTransposeRotate180(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            dst += (width - 1)*dstStride + (height - 1)*N;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < width; ++col)
                    CopyPixel<N>(src + col * N, dst - col * dstStride);
                src += srcStride;
                dst -= N;
            }
        }

        template<size_t N> void TransformImageTransposeRotate270(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            size_t rowSize = width * N;
            dst += (height - 1)*dstStride;
            for (size_t row = 0; row < height; ++row)
            {
                memcpy(dst, src, rowSize);
                src += srcStride;
                dst -= dstStride;
            }
        }

        template<size_t N> void TransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, SimdTransformType transform, uint8_t * dst, size_t dstStride)
        {
            typedef void(*TransformImagePtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride);
            static const TransformImagePtr transformImage[8] = { TransformImageRotate0<N>, TransformImageRotate90<N>, TransformImageRotate180<N>, TransformImageRotate270<N>,
                TransformImageTransposeRotate0<N>, TransformImageTransposeRotate90<N>, TransformImageTransposeRotate180<N>, TransformImageTransposeRotate270<N> };
            transformImage[(int)transform](src, srcStride, width, height, dst, dstStride);
        };

        void TransformImage(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride)
        {
            switch (pixelSize)
            {
            case 1: TransformImage<1>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 2: TransformImage<2>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 3: TransformImage<3>(src, srcStride, width, height, transform, dst, dstStride); break;
            case 4: TransformImage<4>(src, srcStride, width, height, transform, dst, dstStride); break;
            default: assert(0);
           }
        }
    }
#endif// SIMD_SSSE3_ENABLE
}

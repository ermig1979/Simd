/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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
#ifndef __SimdBayer_h__
#define __SimdBayer_h__

#include "Simd/SimdConst.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdLoad.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE uint8_t BayerToGreen(uint8_t greenLeft, uint8_t greenTop, uint8_t greenRight, uint8_t greenBottom,
            uint8_t blueOrRedLeft, uint8_t blueOrRedTop, uint8_t blueOrRedRight, uint8_t blueOrRedBottom)
        {
            int verticalAbsDifference = AbsDifference(blueOrRedTop, blueOrRedBottom);
            int horizontalAbsDifference = AbsDifference(blueOrRedLeft, blueOrRedRight);
            if (verticalAbsDifference < horizontalAbsDifference)
                return Average(greenTop, greenBottom);
            else if (verticalAbsDifference > horizontalAbsDifference)
                return Average(greenRight, greenLeft);
            else
                return Average(greenLeft, greenTop, greenRight, greenBottom);
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const uint8_t * src[6],
            size_t col0, size_t col1, size_t col2, size_t col3, size_t col4, size_t col5,
            uint8_t * dst00, uint8_t * dst01, uint8_t * dst10, uint8_t * dst11);

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const uint8_t * src[6],
            size_t col0, size_t col1, size_t col2, size_t col3, size_t col4, size_t col5,
            uint8_t * dst00, uint8_t * dst01, uint8_t * dst10, uint8_t * dst11)
        {
            dst00[0] = Average(src[1][col2], src[3][col2]);
            dst00[1] = src[2][col2];
            dst00[2] = Average(src[2][col1], src[2][col3]);

            dst01[0] = Average(src[1][col2], src[1][col4], src[3][col2], src[3][col4]);
            dst01[1] = BayerToGreen(src[2][col2], src[1][col3], src[2][col4], src[3][col3], src[2][col1], src[0][col3], src[2][col5], src[4][col3]);
            dst01[2] = src[2][col3];

            dst10[0] = src[3][col2];
            dst10[1] = BayerToGreen(src[3][col1], src[2][col2], src[3][col3], src[4][col2], src[3][col0], src[1][col2], src[3][col4], src[5][col2]);
            dst10[2] = Average(src[2][col1], src[2][col3], src[4][col1], src[4][col3]);

            dst11[0] = Average(src[3][col2], src[3][col4]);
            dst11[1] = src[3][col3];
            dst11[2] = Average(src[2][col3], src[4][col3]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const uint8_t * src[6],
            size_t col0, size_t col1, size_t col2, size_t col3, size_t col4, size_t col5,
            uint8_t * dst00, uint8_t * dst01, uint8_t * dst10, uint8_t * dst11)
        {
            dst00[0] = Average(src[2][col1], src[2][col3]);
            dst00[1] = src[2][col2];
            dst00[2] = Average(src[1][col2], src[3][col2]);

            dst01[0] = src[2][col3];
            dst01[1] = BayerToGreen(src[2][col2], src[1][col3], src[2][col4], src[3][col3], src[2][col1], src[0][col3], src[2][col5], src[4][col3]);
            dst01[2] = Average(src[1][col2], src[1][col4], src[3][col2], src[3][col4]);

            dst10[0] = Average(src[2][col1], src[2][col3], src[4][col1], src[4][col3]);
            dst10[1] = BayerToGreen(src[3][col1], src[2][col2], src[3][col3], src[4][col2], src[3][col0], src[1][col2], src[3][col4], src[5][col2]);
            dst10[2] = src[3][col2];

            dst11[0] = Average(src[2][col3], src[4][col3]);
            dst11[1] = src[3][col3];
            dst11[2] = Average(src[3][col2], src[3][col4]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const uint8_t * src[6],
            size_t col0, size_t col1, size_t col2, size_t col3, size_t col4, size_t col5,
            uint8_t * dst00, uint8_t * dst01, uint8_t * dst10, uint8_t * dst11)
        {
            dst00[0] = Average(src[1][col1], src[1][col3], src[3][col1], src[3][col3]);
            dst00[1] = BayerToGreen(src[2][col1], src[1][col2], src[2][col3], src[3][col2], src[2][col0], src[0][col2], src[2][col4], src[4][col2]);
            dst00[2] = src[2][col2];

            dst01[0] = Average(src[1][col3], src[3][col3]);
            dst01[1] = src[2][col3];
            dst01[2] = Average(src[2][col2], src[2][col4]);

            dst10[0] = Average(src[3][col1], src[3][col3]);
            dst10[1] = src[3][col2];
            dst10[2] = Average(src[2][col2], src[4][col2]);

            dst11[0] = src[3][col3];
            dst11[1] = BayerToGreen(src[3][col2], src[2][col3], src[3][col4], src[4][col3], src[3][col1], src[1][col3], src[3][col5], src[5][col3]);
            dst11[2] = Average(src[2][col2], src[2][col4], src[4][col2], src[4][col4]);
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const uint8_t * src[6],
            size_t col0, size_t col1, size_t col2, size_t col3, size_t col4, size_t col5,
            uint8_t * dst00, uint8_t * dst01, uint8_t * dst10, uint8_t * dst11)
        {
            dst00[0] = src[2][col2];
            dst00[1] = BayerToGreen(src[2][col1], src[1][col2], src[2][col3], src[3][col2], src[2][col0], src[0][col2], src[2][col4], src[4][col2]);
            dst00[2] = Average(src[1][col1], src[1][col3], src[3][col1], src[3][col3]);

            dst01[0] = Average(src[2][col2], src[2][col4]);
            dst01[1] = src[2][col3];
            dst01[2] = Average(src[1][col3], src[3][col3]);

            dst10[0] = Average(src[2][col2], src[4][col2]);
            dst10[1] = src[3][col2];
            dst10[2] = Average(src[3][col1], src[3][col3]);

            dst11[0] = Average(src[2][col2], src[2][col4], src[4][col2], src[4][col4]);
            dst11[1] = BayerToGreen(src[3][col2], src[2][col3], src[3][col4], src[4][col3], src[3][col1], src[1][col3], src[3][col5], src[5][col3]);
            dst11[2] = src[3][col3];
        }
    }

#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        SIMD_INLINE void LoadBayerNose(const uint8_t * src, __m128i dst[3])
        {
            dst[2] = _mm_loadu_si128((__m128i*)(src + 1));
            dst[0] = _mm_or_si128(_mm_slli_si128(_mm_loadu_si128((__m128i*)src), 1), _mm_and_si128(dst[2], _mm_srli_si128(K_INV_ZERO, A - 1))); 
        }

        SIMD_INLINE void LoadBayerTail(const uint8_t * src, __m128i dst[3])
        {
            dst[0] = _mm_loadu_si128((__m128i*)(src - 1));
            dst[2] = _mm_or_si128(_mm_srli_si128(_mm_loadu_si128((__m128i*)src), 1), _mm_and_si128(dst[0], _mm_slli_si128(K_INV_ZERO, A - 1)));
        }

        template <bool align> SIMD_INLINE void LoadBayerNose(const uint8_t * src[3], size_t offset, size_t stride, __m128i dst[12])
        {
            dst[1] = Load<align>((__m128i*)(src[0] + offset));
            LoadBayerNose(src[0] + offset + stride, dst + 0);
            LoadNose3<align, 2>(src[1] + offset, dst + 3);
            LoadNose3<align, 2>(src[1] + offset + stride, dst + 6);
            LoadBayerNose(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m128i*)(src[2] + offset + stride));
        }

        template <bool align> SIMD_INLINE void LoadBayerBody(const uint8_t * src[3], size_t offset, size_t stride, __m128i dst[12])
        {
            dst[1] = Load<align>((__m128i*)(src[0] + offset));
            LoadBodyDx(src[0] + offset + stride, dst + 0);
            LoadBody3<align, 2>(src[1] + offset, dst + 3);
            LoadBody3<align, 2>(src[1] + offset + stride, dst + 6);
            LoadBodyDx(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m128i*)(src[2] + offset + stride));
        }

        template <bool align> SIMD_INLINE void LoadBayerTail(const uint8_t * src[3], size_t offset, size_t stride, __m128i dst[12])
        {
            dst[1] = Load<align>((__m128i*)(src[0] + offset));
            LoadBayerTail(src[0] + offset + stride, dst + 0);
            LoadTail3<align, 2>(src[1] + offset, dst + 3);
            LoadTail3<align, 2>(src[1] + offset + stride, dst + 6);
            LoadBayerTail(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m128i*)(src[2] + offset + stride));
        }

        template<int index, int part> SIMD_INLINE __m128i Get(const __m128i src[12])
        {
            return U8To16<part>(src[index]);
        }

        SIMD_INLINE __m128i BayerToGreen(const __m128i & greenLeft, const __m128i & greenTop, const __m128i & greenRight, const __m128i & greenBottom,
            const __m128i & blueOrRedLeft, const __m128i & blueOrRedTop, const __m128i & blueOrRedRight, const __m128i & blueOrRedBottom)
        {
            __m128i verticalAbsDifference = AbsDifferenceI16(blueOrRedTop, blueOrRedBottom);
            __m128i horizontalAbsDifference = AbsDifferenceI16(blueOrRedLeft, blueOrRedRight);
            __m128i green = Average16(greenLeft, greenTop, greenRight, greenBottom);
            green = Combine(_mm_cmplt_epi16(verticalAbsDifference, horizontalAbsDifference), _mm_avg_epu16(greenTop, greenBottom), green);
            return Combine(_mm_cmpgt_epi16(verticalAbsDifference, horizontalAbsDifference), _mm_avg_epu16(greenRight, greenLeft), green);
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const __m128i s[12], __m128i d[6]);

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const __m128i s[12], __m128i d[6])
        {
            d[0] = Merge16(_mm_avg_epu16(Get<0, 1>(s), Get<7, 0>(s)), Average16(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));
            d[1] = Merge16(Get<4, 0>(s), BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s), Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));
            d[2] = Merge16(_mm_avg_epu16(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
            d[3] = Merge16(Get<7, 0>(s), _mm_avg_epu16(Get<7, 0>(s), Get<8, 0>(s)));
            d[4] = Merge16(BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s), Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)), Get<7, 1>(s));
            d[5] = Merge16(Average16(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)), _mm_avg_epu16(Get<4, 1>(s), Get<11, 0>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const __m128i s[12], __m128i d[6])
        {
            d[0] = Merge16(_mm_avg_epu16(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
            d[1] = Merge16(Get<4, 0>(s), BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s), Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));
            d[2] = Merge16(_mm_avg_epu16(Get<0, 1>(s), Get<7, 0>(s)), Average16(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));
            d[3] = Merge16(Average16(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)), _mm_avg_epu16(Get<4, 1>(s), Get<11, 0>(s)));
            d[4] = Merge16(BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s), Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)), Get<7, 1>(s));
            d[5] = Merge16(Get<7, 0>(s), _mm_avg_epu16(Get<7, 0>(s), Get<8, 0>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const __m128i s[12], __m128i d[6])
        {
            d[0] = Merge16(Average16(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)), _mm_avg_epu16(Get<2, 0>(s), Get<7, 1>(s)));
            d[1] = Merge16(BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s), Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)), Get<4, 1>(s));
            d[2] = Merge16(Get<4, 0>(s), _mm_avg_epu16(Get<4, 0>(s), Get<5, 0>(s)));
            d[3] = Merge16(_mm_avg_epu16(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
            d[4] = Merge16(Get<7, 0>(s), BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s), Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
            d[5] = Merge16(_mm_avg_epu16(Get<4, 0>(s), Get<9, 1>(s)), Average16(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const __m128i s[12], __m128i d[6])
        {
            d[0] = Merge16(Get<4, 0>(s), _mm_avg_epu16(Get<4, 0>(s), Get<5, 0>(s)));
            d[1] = Merge16(BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s), Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)), Get<4, 1>(s));
            d[2] = Merge16(Average16(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)), _mm_avg_epu16(Get<2, 0>(s), Get<7, 1>(s)));
            d[3] = Merge16(_mm_avg_epu16(Get<4, 0>(s), Get<9, 1>(s)), Average16(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
            d[4] = Merge16(Get<7, 0>(s), BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s), Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
            d[5] = Merge16(_mm_avg_epu16(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
        }
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        SIMD_INLINE void LoadBayerNose(const uint8_t * src, __m256i dst[3])
        {
            dst[2] = _mm256_loadu_si256((__m256i*)(src + 1));
            __m128i lo = _mm_or_si128(_mm_slli_si128(_mm_loadu_si128((__m128i*)src), 1), 
                _mm_and_si128(_mm256_castsi256_si128(dst[2]), _mm_srli_si128(Sse2::K_INV_ZERO, HA - 1)));
            __m128i hi = _mm_loadu_si128((__m128i*)(src + HA - 1));
            dst[0] = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
        }

        SIMD_INLINE void LoadBayerTail(const uint8_t * src, __m256i dst[3])
        {
            dst[0] = _mm256_loadu_si256((__m256i*)(src - 1));
            __m128i lo = _mm_loadu_si128((__m128i*)(src + 1));
            __m128i hi = _mm_or_si128(_mm_srli_si128(_mm_loadu_si128((__m128i*)src + 1), 1), 
                _mm_and_si128(_mm256_extracti128_si256(dst[0], 1), _mm_slli_si128(Sse2::K_INV_ZERO, HA - 1)));
            dst[2] = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 0x1);
        }

        template <bool align> SIMD_INLINE void LoadBayerNose(const uint8_t * src[3], size_t offset, size_t stride, __m256i dst[12])
        {
            dst[1] = Load<align>((__m256i*)(src[0] + offset));
            LoadBayerNose(src[0] + offset + stride, dst + 0);
            LoadNose3<align, 2>(src[1] + offset, dst + 3);
            LoadNose3<align, 2>(src[1] + offset + stride, dst + 6);
            LoadBayerNose(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m256i*)(src[2] + offset + stride));
        }

        template <bool align> SIMD_INLINE void LoadBayerBody(const uint8_t * src[3], size_t offset, size_t stride, __m256i dst[12])
        {
            dst[1] = Load<align>((__m256i*)(src[0] + offset));
            LoadBodyDx(src[0] + offset + stride, dst + 0);
            LoadBody3<align, 2>(src[1] + offset, dst + 3);
            LoadBody3<align, 2>(src[1] + offset + stride, dst + 6);
            LoadBodyDx(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m256i*)(src[2] + offset + stride));
        }

        template <bool align> SIMD_INLINE void LoadBayerTail(const uint8_t * src[3], size_t offset, size_t stride, __m256i dst[12])
        {
            dst[1] = Load<align>((__m256i*)(src[0] + offset));
            LoadBayerTail(src[0] + offset + stride, dst + 0);
            LoadTail3<align, 2>(src[1] + offset, dst + 3);
            LoadTail3<align, 2>(src[1] + offset + stride, dst + 6);

            LoadBayerTail(src[2] + offset, dst + 9);
            dst[10] = Load<align>((__m256i*)(src[2] + offset + stride));
        }

        template<int index, int part> SIMD_INLINE __m256i Get(const __m256i src[12])
        {
            return U8To16<part>(src[index]);
        }

        SIMD_INLINE __m256i BayerToGreen(const __m256i & greenLeft, const __m256i & greenTop, const __m256i & greenRight, const __m256i & greenBottom,
            const __m256i & blueOrRedLeft, const __m256i & blueOrRedTop, const __m256i & blueOrRedRight, const __m256i & blueOrRedBottom)
        {
            __m256i verticalAbsDifference = AbsDifferenceI16(blueOrRedTop, blueOrRedBottom);
            __m256i horizontalAbsDifference = AbsDifferenceI16(blueOrRedLeft, blueOrRedRight);
            __m256i green = Average16(greenLeft, greenTop, greenRight, greenBottom);
            green = _mm256_blendv_epi8(green, _mm256_avg_epu16(greenTop, greenBottom), _mm256_cmpgt_epi16(horizontalAbsDifference, verticalAbsDifference));
            return _mm256_blendv_epi8(green, _mm256_avg_epu16(greenRight, greenLeft), _mm256_cmpgt_epi16(verticalAbsDifference, horizontalAbsDifference));
        }

        template <SimdPixelFormatType bayerFormat> void BayerToBgr(const __m256i s[12], __m256i d[6]);

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGrbg>(const __m256i s[12], __m256i d[6])
        {
            d[0] = Merge16(_mm256_avg_epu16(Get<0, 1>(s), Get<7, 0>(s)), Average16(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));
            d[1] = Merge16(Get<4, 0>(s), BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s), Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));
            d[2] = Merge16(_mm256_avg_epu16(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
            d[3] = Merge16(Get<7, 0>(s), _mm256_avg_epu16(Get<7, 0>(s), Get<8, 0>(s)));
            d[4] = Merge16(BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s), Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)), Get<7, 1>(s));
            d[5] = Merge16(Average16(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)), _mm256_avg_epu16(Get<4, 1>(s), Get<11, 0>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerGbrg>(const __m256i s[12], __m256i d[6])
        {
            d[0] = Merge16(_mm256_avg_epu16(Get<3, 1>(s), Get<4, 1>(s)), Get<4, 1>(s));
            d[1] = Merge16(Get<4, 0>(s), BayerToGreen(Get<4, 0>(s), Get<2, 0>(s), Get<5, 0>(s), Get<7, 1>(s), Get<3, 1>(s), Get<1, 1>(s), Get<5, 1>(s), Get<11, 0>(s)));
            d[2] = Merge16(_mm256_avg_epu16(Get<0, 1>(s), Get<7, 0>(s)), Average16(Get<0, 1>(s), Get<2, 1>(s), Get<7, 0>(s), Get<8, 0>(s)));
            d[3] = Merge16(Average16(Get<3, 1>(s), Get<4, 1>(s), Get<9, 0>(s), Get<11, 0>(s)), _mm256_avg_epu16(Get<4, 1>(s), Get<11, 0>(s)));
            d[4] = Merge16(BayerToGreen(Get<6, 1>(s), Get<4, 0>(s), Get<7, 1>(s), Get<9, 1>(s), Get<6, 0>(s), Get<0, 1>(s), Get<8, 0>(s), Get<10, 0>(s)), Get<7, 1>(s));
            d[5] = Merge16(Get<7, 0>(s), _mm256_avg_epu16(Get<7, 0>(s), Get<8, 0>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerRggb>(const __m256i s[12], __m256i d[6])
        {
            d[0] = Merge16(Average16(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)), _mm256_avg_epu16(Get<2, 0>(s), Get<7, 1>(s)));
            d[1] = Merge16(BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s), Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)), Get<4, 1>(s));
            d[2] = Merge16(Get<4, 0>(s), _mm256_avg_epu16(Get<4, 0>(s), Get<5, 0>(s)));
            d[3] = Merge16(_mm256_avg_epu16(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
            d[4] = Merge16(Get<7, 0>(s), BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s), Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
            d[5] = Merge16(_mm256_avg_epu16(Get<4, 0>(s), Get<9, 1>(s)), Average16(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
        }

        template <> SIMD_INLINE void BayerToBgr<SimdPixelFormatBayerBggr>(const __m256i s[12], __m256i d[6])
        {
            d[0] = Merge16(Get<4, 0>(s), _mm256_avg_epu16(Get<4, 0>(s), Get<5, 0>(s)));
            d[1] = Merge16(BayerToGreen(Get<3, 1>(s), Get<0, 1>(s), Get<4, 1>(s), Get<7, 0>(s), Get<3, 0>(s), Get<1, 0>(s), Get<5, 0>(s), Get<9, 1>(s)), Get<4, 1>(s));
            d[2] = Merge16(Average16(Get<0, 0>(s), Get<2, 0>(s), Get<6, 1>(s), Get<7, 1>(s)), _mm256_avg_epu16(Get<2, 0>(s), Get<7, 1>(s)));
            d[3] = Merge16(_mm256_avg_epu16(Get<4, 0>(s), Get<9, 1>(s)), Average16(Get<4, 0>(s), Get<5, 0>(s), Get<9, 1>(s), Get<11, 1>(s)));
            d[4] = Merge16(Get<7, 0>(s), BayerToGreen(Get<7, 0>(s), Get<4, 1>(s), Get<8, 0>(s), Get<11, 0>(s), Get<6, 1>(s), Get<2, 0>(s), Get<8, 1>(s), Get<10, 1>(s)));
            d[5] = Merge16(_mm256_avg_epu16(Get<6, 1>(s), Get<7, 1>(s)), Get<7, 1>(s));
        }
    }
#endif//SIMD_AVX2_ENABLE
}
#endif//__SimdBayer_h__

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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE __m128i Square8u(__m128i src)
        {
            const __m128i lo = _mm_unpacklo_epi8(src, _mm_setzero_si128());
            const __m128i hi = _mm_unpackhi_epi8(src, _mm_setzero_si128());
            return _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi));
        }

		void ValueSquareSums1(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSums, uint64_t * squareSums)
        {
            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i fullValueSum = _mm_setzero_si128();
			__m128i fullSquareSum = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
				__m128i rowSquareSum = _mm_setzero_si128();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i value = _mm_loadu_si128((__m128i*)(src + col));
                    fullValueSum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), fullValueSum);
                    rowSquareSum = _mm_add_epi32(rowSquareSum, Square8u(value));
                }
                if (width - bodyWidth)
                {
                    const __m128i value = _mm_and_si128(tailMask, _mm_loadu_si128((__m128i*)(src + width - A)));
                    fullValueSum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), fullValueSum);
                    rowSquareSum = _mm_add_epi32(rowSquareSum, Square8u(value));
                }
                fullSquareSum = _mm_add_epi64(fullSquareSum, HorizontalSum32(rowSquareSum));
                src += stride;
            }
            valueSums[0] = ExtractInt64Sum(fullValueSum);
			squareSums[0] = ExtractInt64Sum(fullSquareSum);
        }

        void ValueSquareSums2(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t size = width * 2;
            size_t sizeA = AlignLo(size, A);
            __m128i tail = ShiftLeft(K_INV_ZERO, A - size + sizeA);
            __m128i vSum0 = _mm_setzero_si128();
            __m128i vSum1 = _mm_setzero_si128();
            __m128i sSum0 = _mm_setzero_si128();
            __m128i sSum1 = _mm_setzero_si128();
            for (size_t y = 0; y < height; ++y)
            {
                __m128i rSum0 = _mm_setzero_si128();
                __m128i rSum1 = _mm_setzero_si128();
                for (size_t x = 0; x < sizeA; x += A)
                {
                    const __m128i val = _mm_loadu_si128((__m128i*)(src + x));
                    const __m128i v0 = _mm_and_si128(val, K16_00FF);
                    const __m128i v1 = _mm_and_si128(_mm_srli_si128(val, 1), K16_00FF);
                    vSum0 = _mm_add_epi64(_mm_sad_epu8(v0, K_ZERO), vSum0);
                    vSum1 = _mm_add_epi64(_mm_sad_epu8(v1, K_ZERO), vSum1);
                    rSum0 = _mm_add_epi32(rSum0, _mm_madd_epi16(v0, v0));
                    rSum1 = _mm_add_epi32(rSum1, _mm_madd_epi16(v1, v1));
                }
                if (size - sizeA)
                {
                    const __m128i val = _mm_and_si128(tail, _mm_loadu_si128((__m128i*)(src + size - A)));
                    const __m128i v0 = _mm_and_si128(val, K16_00FF);
                    const __m128i v1 = _mm_and_si128(_mm_srli_si128(val, 1), K16_00FF);
                    vSum0 = _mm_add_epi64(_mm_sad_epu8(v0, K_ZERO), vSum0);
                    vSum1 = _mm_add_epi64(_mm_sad_epu8(v1, K_ZERO), vSum1);
                    rSum0 = _mm_add_epi32(rSum0, _mm_madd_epi16(v0, v0));
                    rSum1 = _mm_add_epi32(rSum1, _mm_madd_epi16(v1, v1));
                }
                sSum0 = _mm_add_epi64(sSum0, HorizontalSum32(rSum0));
                sSum1 = _mm_add_epi64(sSum1, HorizontalSum32(rSum1));
                src += stride;
            }
            valueSums[0] = ExtractInt64Sum(vSum0);
            valueSums[1] = ExtractInt64Sum(vSum1);
            squareSums[0] = ExtractInt64Sum(sSum0);
            squareSums[1] = ExtractInt64Sum(sSum1);
        }

        void ValueSquareSums3(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t widthA = AlignLo(width, A);
            __m128i tail = ShiftLeft(K_INV_ZERO, A - width + widthA);
            __m128i vSum0 = _mm_setzero_si128();
            __m128i vSum1 = _mm_setzero_si128();
            __m128i vSum2 = _mm_setzero_si128();
            __m128i sSum0 = _mm_setzero_si128();
            __m128i sSum1 = _mm_setzero_si128();
            __m128i sSum2 = _mm_setzero_si128();
            __m128i bgr[3];
            for (size_t y = 0; y < height; ++y)
            {
                __m128i rSum0 = _mm_setzero_si128();
                __m128i rSum1 = _mm_setzero_si128();
                __m128i rSum2 = _mm_setzero_si128();
                for (size_t x = 0; x < widthA; x += A)
                {
                    const uint8_t* psrc = src + x * 3;
                    bgr[0] = _mm_loadu_si128((__m128i*)psrc + 0);
                    bgr[1] = _mm_loadu_si128((__m128i*)psrc + 1);
                    bgr[2] = _mm_loadu_si128((__m128i*)psrc + 2);
                    __m128i v0 = BgrToBlue(bgr);
                    vSum0 = _mm_add_epi64(_mm_sad_epu8(v0, K_ZERO), vSum0);
                    rSum0 = _mm_add_epi32(rSum0, Square8u(v0));
                    __m128i v1 = BgrToGreen(bgr);
                    vSum1 = _mm_add_epi64(_mm_sad_epu8(v1, K_ZERO), vSum1);
                    rSum1 = _mm_add_epi32(rSum1, Square8u(v1));
                    __m128i v2 = BgrToRed(bgr);
                    vSum2 = _mm_add_epi64(_mm_sad_epu8(v2, K_ZERO), vSum2);
                    rSum2 = _mm_add_epi32(rSum2, Square8u(v2));
                }
                if (width - widthA)
                {
                    const uint8_t* psrc = src + (width - A) * 3;
                    bgr[0] = _mm_loadu_si128((__m128i*)psrc + 0);
                    bgr[1] = _mm_loadu_si128((__m128i*)psrc + 1);
                    bgr[2] = _mm_loadu_si128((__m128i*)psrc + 2);
                    __m128i v0 = _mm_and_si128(tail, BgrToBlue(bgr));
                    vSum0 = _mm_add_epi64(_mm_sad_epu8(v0, K_ZERO), vSum0);
                    rSum0 = _mm_add_epi32(rSum0, Square8u(v0));
                    __m128i v1 = _mm_and_si128(tail, BgrToGreen(bgr));
                    vSum1 = _mm_add_epi64(_mm_sad_epu8(v1, K_ZERO), vSum1);
                    rSum1 = _mm_add_epi32(rSum1, Square8u(v1));
                    __m128i v2 = _mm_and_si128(tail, BgrToRed(bgr));
                    vSum2 = _mm_add_epi64(_mm_sad_epu8(v2, K_ZERO), vSum2);
                    rSum2 = _mm_add_epi32(rSum2, Square8u(v2));
                }
                sSum0 = _mm_add_epi64(sSum0, HorizontalSum32(rSum0));
                sSum1 = _mm_add_epi64(sSum1, HorizontalSum32(rSum1));
                sSum2 = _mm_add_epi64(sSum2, HorizontalSum32(rSum2));
                src += stride;
            }
            valueSums[0] = ExtractInt64Sum(vSum0);
            valueSums[1] = ExtractInt64Sum(vSum1);
            valueSums[2] = ExtractInt64Sum(vSum2);
            squareSums[0] = ExtractInt64Sum(sSum0);
            squareSums[1] = ExtractInt64Sum(sSum1);
            squareSums[2] = ExtractInt64Sum(sSum2);
        }

        const __m128i K8_SHFL_4_01 = SIMD_MM_SETR_EPI8(0x0, -1, 0x4, -1, 0x8, -1, 0xC, -1, 0x1, -1, 0x5, -1, 0x9, -1, 0xD, -1);
        const __m128i K8_SHFL_4_23 = SIMD_MM_SETR_EPI8(0x2, -1, 0x6, -1, 0xA, -1, 0xE, -1, 0x3, -1, 0x7, -1, 0xB, -1, 0xF, -1);

        SIMD_INLINE __m128i PairSum32(__m128i a)
        {
            return _mm_add_epi64(_mm_and_si128(a, K64_00000000FFFFFFFF), _mm_and_si128(_mm_srli_si128(a, 4), K64_00000000FFFFFFFF));
        }

        void ValueSquareSums4(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            __m128i tail = ShiftLeft(K_INV_ZERO, A - size + sizeA);
            __m128i vSum01 = _mm_setzero_si128();
            __m128i vSum23 = _mm_setzero_si128();
            __m128i sSum01 = _mm_setzero_si128();
            __m128i sSum23 = _mm_setzero_si128();
            for (size_t y = 0; y < height; ++y)
            {
                __m128i rSum01 = _mm_setzero_si128();
                __m128i rSum23 = _mm_setzero_si128();
                for (size_t x = 0; x < sizeA; x += A)
                {
                    const __m128i val = _mm_loadu_si128((__m128i*)(src + x));
                    const __m128i v01 = _mm_shuffle_epi8(val, K8_SHFL_4_01);
                    const __m128i v23 = _mm_shuffle_epi8(val, K8_SHFL_4_23);
                    vSum01 = _mm_add_epi64(_mm_sad_epu8(v01, K_ZERO), vSum01);
                    rSum01 = _mm_add_epi32(rSum01, _mm_madd_epi16(v01, v01));
                    vSum23 = _mm_add_epi64(_mm_sad_epu8(v23, K_ZERO), vSum23);
                    rSum23 = _mm_add_epi32(rSum23, _mm_madd_epi16(v23, v23));
                }
                if (size - sizeA)
                {
                    const __m128i val = _mm_and_si128(tail, _mm_loadu_si128((__m128i*)(src + size - A)));
                    const __m128i v01 = _mm_shuffle_epi8(val, K8_SHFL_4_01);
                    const __m128i v23 = _mm_shuffle_epi8(val, K8_SHFL_4_23);
                    vSum01 = _mm_add_epi64(_mm_sad_epu8(v01, K_ZERO), vSum01);
                    rSum01 = _mm_add_epi32(rSum01, _mm_madd_epi16(v01, v01));
                    vSum23 = _mm_add_epi64(_mm_sad_epu8(v23, K_ZERO), vSum23);
                    rSum23 = _mm_add_epi32(rSum23, _mm_madd_epi16(v23, v23));
                }
                sSum01 = _mm_add_epi64(sSum01, PairSum32(rSum01));
                sSum23 = _mm_add_epi64(sSum23, PairSum32(rSum23));
                src += stride;
            }
            valueSums[0] = ExtractInt64<0>(vSum01);
            valueSums[1] = ExtractInt64<1>(vSum01);
            valueSums[2] = ExtractInt64<0>(vSum23);
            valueSums[3] = ExtractInt64<1>(vSum23);
            squareSums[0] = ExtractInt64<0>(sSum01);
            squareSums[1] = ExtractInt64<1>(sSum01);
            squareSums[2] = ExtractInt64<0>(sSum23);
            squareSums[3] = ExtractInt64<1>(sSum23);
        }

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width >= A && width < 0x10000);

            switch (channels)
            {
            case 1: ValueSquareSums1(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums2(src, stride, width, height, valueSums, squareSums); break;
            case 3: ValueSquareSums3(src, stride, width, height, valueSums, squareSums); break;
            case 4: ValueSquareSums4(src, stride, width, height, valueSums, squareSums); break;
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_SSE41_ENABLE
}

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
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdConversion.h"

#if defined(__GNUC__)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wfree-nonheap-object"
#endif

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> void GetStatistic(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint8_t* min, uint8_t* max, uint8_t* average)
        {
            assert(width * height && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i sum = _mm_setzero_si128();
            __m128i min_ = K_INV_ZERO;
            __m128i max_ = K_ZERO;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i value = Load<align>((__m128i*)(src + col));
                    min_ = _mm_min_epu8(min_, value);
                    max_ = _mm_max_epu8(max_, value);
                    sum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), sum);
                }
                if (width - bodyWidth)
                {
                    const __m128i value = Load<false>((__m128i*)(src + width - A));
                    min_ = _mm_min_epu8(min_, value);
                    max_ = _mm_max_epu8(max_, value);
                    sum = _mm_add_epi64(_mm_sad_epu8(_mm_and_si128(tailMask, value), K_ZERO), sum);
                }
                src += stride;
            }

            uint8_t min_buffer[A], max_buffer[A];
            _mm_storeu_si128((__m128i*)min_buffer, min_);
            _mm_storeu_si128((__m128i*)max_buffer, max_);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractInt64Sum(sum) + width * height / 2) / (width * height));
        }

        void GetStatistic(const uint8_t* src, size_t stride, size_t width, size_t height,
            uint8_t* min, uint8_t* max, uint8_t* average)
        {
            if (Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void GetRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint32_t) * height);
            for (size_t row = 0; row < height; ++row)
            {
                __m128i sum = _mm_setzero_si128();
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _src = Load<align>((__m128i*)(src + col));
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src, K_ZERO));
                }
                if (alignedWidth != width)
                {
                    __m128i _src = _mm_and_si128(Load<false>((__m128i*)(src + width - A)), tailMask);
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src, K_ZERO));
                }
                sums[row] = ExtractInt32Sum(sum);
                src += stride;
            }
        }

        void GetRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void Sum16(__m128i src8, uint16_t* sums16)
        {
            Store<align>((__m128i*)sums16 + 0, _mm_add_epi16(Load<align>((__m128i*)sums16 + 0), _mm_unpacklo_epi8(src8, K_ZERO)));
            Store<align>((__m128i*)sums16 + 1, _mm_add_epi16(Load<align>((__m128i*)sums16 + 1), _mm_unpackhi_epi8(src8, K_ZERO)));
        }

        template <bool align> SIMD_INLINE void Sum32(__m128i src16, uint32_t* sums32)
        {
            Store<align>((__m128i*)sums32 + 0, _mm_add_epi32(Load<align>((__m128i*)sums32 + 0), _mm_unpacklo_epi16(src16, K_ZERO)));
            Store<align>((__m128i*)sums32 + 1, _mm_add_epi32(Load<align>((__m128i*)sums32 + 1), _mm_unpackhi_epi16(src16, K_ZERO)));
        }

        template <bool align> void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Array16u sums16(alignedHiWidth);
            Array32u sums32(alignedHiWidth, true);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step * stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                sums16.Clear();
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m128i src8 = Load<align>((__m128i*)(src + col));
                        Sum16<true>(src8, sums16.data + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        __m128i src8 = _mm_and_si128(Load<false>((__m128i*)(src + width - A)), tailMask);
                        Sum16<false>(src8, sums16.data + width - A);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    __m128i src16 = Load<true>((__m128i*)(sums16.data + col));
                    Sum32<true>(src16, sums32.data + col);
                }
            }
            memcpy(sums, sums32.data, sizeof(uint32_t) * width);
        }

        void GetColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedWidth);

            memset(sums, 0, sizeof(uint32_t) * height);
            const uint8_t* src0 = src;
            const uint8_t* src1 = src + stride;
            height--;
            for (size_t row = 0; row < height; ++row)
            {
                __m128i sum = _mm_setzero_si128();
                for (size_t col = 0; col < alignedWidth; col += A)
                {
                    __m128i _src0 = Load<align>((__m128i*)(src0 + col));
                    __m128i _src1 = Load<align>((__m128i*)(src1 + col));
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src0, _src1));
                }
                if (alignedWidth != width)
                {
                    __m128i _src0 = _mm_and_si128(Load<false>((__m128i*)(src0 + width - A)), tailMask);
                    __m128i _src1 = _mm_and_si128(Load<false>((__m128i*)(src1 + width - A)), tailMask);
                    sum = _mm_add_epi32(sum, _mm_sad_epu8(_src0, _src1));
                }
                sums[row] = ExtractInt32Sum(sum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            size_t alignedHiWidth = AlignHi(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + alignedLoWidth);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Array16u sums16(alignedHiWidth);
            Array32u sums32(alignedHiWidth);
            memset(sums32.data, 0, sizeof(uint32_t) * alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step * stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);

                memset(sums16.data, 0, sizeof(uint16_t) * width);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    for (size_t col = 0; col < alignedLoWidth; col += A)
                    {
                        __m128i _src0 = Load<align>((__m128i*)(src + col + 0));
                        __m128i _src1 = Load<false>((__m128i*)(src + col + 1));
                        Sum16<true>(AbsDifferenceU8(_src0, _src1), sums16.data + col);
                    }
                    if (alignedLoWidth != width)
                    {
                        __m128i _src0 = Load<false>((__m128i*)(src + width - A + 0));
                        __m128i _src1 = Load<false>((__m128i*)(src + width - A + 1));
                        Sum16<false>(_mm_and_si128(AbsDifferenceU8(_src0, _src1), tailMask), sums16.data + width - A);
                    }
                    src += stride;
                }

                for (size_t col = 0; col < alignedHiWidth; col += HA)
                {
                    __m128i src16 = Load<true>((__m128i*)(sums16.data + col));
                    Sum32<true>(src16, sums32.data + col);
                }
            }
            memcpy(sums, sums32.data, sizeof(uint32_t) * width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t* src, size_t stride, size_t width, size_t height, uint32_t* sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i fullSum = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i src_ = Load<align>((__m128i*)(src + col));
                    fullSum = _mm_add_epi64(_mm_sad_epu8(src_, K_ZERO), fullSum);
                }
                if (width - bodyWidth)
                {
                    const __m128i src_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(src + width - A)));
                    fullSum = _mm_add_epi64(_mm_sad_epu8(src_, K_ZERO), fullSum);
                }
                src += stride;
            }
            *sum = ExtractInt64Sum(fullSum);
        }

        void ValueSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Square(__m128i src)
        {
            const __m128i lo = _mm_unpacklo_epi8(src, _mm_setzero_si128());
            const __m128i hi = _mm_unpackhi_epi8(src, _mm_setzero_si128());
            return _mm_add_epi32(_mm_madd_epi16(lo, lo), _mm_madd_epi16(hi, hi));
        }

        template <bool align> void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i fullSum = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                __m128i rowSum = _mm_setzero_si128();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i src_ = Load<align>((__m128i*)(src + col));
                    rowSum = _mm_add_epi32(rowSum, Square(src_));
                }
                if (width - bodyWidth)
                {
                    const __m128i src_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(src + width - A)));
                    rowSum = _mm_add_epi32(rowSum, Square(src_));
                }
                fullSum = _mm_add_epi64(fullSum, HorizontalSum32(rowSum));
                src += stride;
            }
            *sum = ExtractInt64Sum(fullSum);
        }

        void SquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        //-----------------------------------------------------------------------------------------

        template <bool align> void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i fullValueSum = _mm_setzero_si128();
            __m128i fullSquareSum = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                __m128i rowSquareSum = _mm_setzero_si128();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i value = Load<align>((__m128i*)(src + col));
                    fullValueSum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), fullValueSum);
                    rowSquareSum = _mm_add_epi32(rowSquareSum, Square(value));
                }
                if (width - bodyWidth)
                {
                    const __m128i value = _mm_and_si128(tailMask, Load<false>((__m128i*)(src + width - A)));
                    fullValueSum = _mm_add_epi64(_mm_sad_epu8(value, K_ZERO), fullValueSum);
                    rowSquareSum = _mm_add_epi32(rowSquareSum, Square(value));
                }
                fullSquareSum = _mm_add_epi64(fullSquareSum, HorizontalSum32(rowSquareSum));
                src += stride;
            }
            *valueSum = ExtractInt64Sum(fullValueSum);
            *squareSum = ExtractInt64Sum(fullSquareSum);
        }

        void ValueSquareSum(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSum, uint64_t* squareSum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSum<true>(src, stride, width, height, valueSum, squareSum);
            else
                ValueSquareSum<false>(src, stride, width, height, valueSum, squareSum);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE __m128i Correlation(__m128i a, __m128i b)
        {
            const __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(a, _mm_setzero_si128()), _mm_unpacklo_epi8(b, _mm_setzero_si128()));
            const __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(a, _mm_setzero_si128()), _mm_unpackhi_epi8(b, _mm_setzero_si128()));
            return _mm_add_epi32(lo, hi);
        }

        template <bool align> void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t bodyWidth = AlignLo(width, A);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + bodyWidth);
            __m128i fullSum = _mm_setzero_si128();
            for (size_t row = 0; row < height; ++row)
            {
                __m128i rowSum = _mm_setzero_si128();
                for (size_t col = 0; col < bodyWidth; col += A)
                {
                    const __m128i a_ = Load<align>((__m128i*)(a + col));
                    const __m128i b_ = Load<align>((__m128i*)(b + col));
                    rowSum = _mm_add_epi32(rowSum, Correlation(a_, b_));
                }
                if (width - bodyWidth)
                {
                    const __m128i a_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(a + width - A)));
                    const __m128i b_ = _mm_and_si128(tailMask, Load<false>((__m128i*)(b + width - A)));
                    rowSum = _mm_add_epi32(rowSum, Correlation(a_, b_));
                }
                fullSum = _mm_add_epi64(fullSum, HorizontalSum32(rowSum));
                a += aStride;
                b += bStride;
            }
            *sum = ExtractInt64Sum(fullSum);
        }

        void CorrelationSum(const uint8_t* a, size_t aStride, const uint8_t* b, size_t bStride, size_t width, size_t height, uint64_t* sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }

        //-----------------------------------------------------------------------------------------

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
#endif
}

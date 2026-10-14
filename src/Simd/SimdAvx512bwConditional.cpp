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
#include "Simd/SimdMemory.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
#ifdef SIMD_X64_ENABLE
        template <bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u(const uint8_t * src, __m512i value, uint64_t * counts, __mmask64 tail = -1)
        {
            const __m512i _src = mask ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src);
            __mmask64 bits = Compare8u<compareType>(_src, value);
            counts[0] += _mm_popcnt_u64(bits&tail);
        }

        template <SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u4(const uint8_t * src, __m512i value, uint64_t * counts)
        {
            counts[0] += _mm_popcnt_u64(Compare8u<compareType>(_mm512_loadu_si512(src + 0 * A), value));
            counts[1] += _mm_popcnt_u64(Compare8u<compareType>(_mm512_loadu_si512(src + 1 * A), value));
            counts[2] += _mm_popcnt_u64(Compare8u<compareType>(_mm512_loadu_si512(src + 2 * A), value));
            counts[3] += _mm_popcnt_u64(Compare8u<compareType>(_mm512_loadu_si512(src + 3 * A), value));
        }

        template <SimdCompareType compareType> void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi8(value);
            uint64_t counts[4] = { 0, 0, 0, 0 };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    ConditionalCount8u4<compareType>(src + col, _value, counts);
                for (; col < alignedWidth; col += A)
                    ConditionalCount8u<false, compareType>(src + col, _value, counts);
                if (col < width)
                    ConditionalCount8u<true, compareType>(src + col, _value, counts, tailMask);
                src += stride;
            }
            *count = (uint32_t)(counts[0] + counts[1] + counts[2] + counts[3]);
        }
#else
        template <bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount8u(const uint8_t * src, __m512i value, uint32_t * counts, __mmask64 tail = -1)
        {
            const __m512i _src = mask ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src);
            union Mask
            {
                __mmask32 m32[2];
                __mmask64 m64[1];
            } bits;
            bits.m64[0] = Compare8u<compareType>(_src, value)&tail;
            counts[0] += _mm_popcnt_u32(bits.m32[0]) + _mm_popcnt_u32(bits.m32[1]);
        }

        template <SimdCompareType compareType> void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, uint32_t * count)
        {
            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi8(value);
            uint32_t counts[1] = { 0 };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    ConditionalCount8u<false, compareType>(src + col, _value, counts);
                if (col < width)
                    ConditionalCount8u<true, compareType>(src + col, _value, counts, tailMask);
                src += stride;
            }
            *count = counts[0];
        }
#endif//SIMD_X64_ENABLE

        void ConditionalCount8u(const uint8_t * src, size_t stride, size_t width, size_t height, uint8_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount8u<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount8u<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount8u<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount8u<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount8u<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount8u<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        template <bool mask, SimdCompareType compareType> SIMD_INLINE void ConditionalCount16i(const uint8_t * src, __m512i value, uint32_t * counts, __mmask32 tail = -1)
        {
            const __m512i _src = mask ? _mm512_maskz_loadu_epi16(tail, (int16_t*)src) : _mm512_loadu_si512((int16_t*)src);
            __mmask32 bits = Compare16i<compareType>(_src, value);
            counts[0] += _mm_popcnt_u32(bits&tail);
        }

        template <SimdCompareType compareType> SIMD_INLINE void ConditionalCount16i4(const uint8_t * src, __m512i value, uint32_t * counts)
        {
            counts[0] += _mm_popcnt_u32(Compare16i<compareType>(_mm512_loadu_si512(src + 0 * A), value));
            counts[1] += _mm_popcnt_u32(Compare16i<compareType>(_mm512_loadu_si512(src + 1 * A), value));
            counts[2] += _mm_popcnt_u32(Compare16i<compareType>(_mm512_loadu_si512(src + 2 * A), value));
            counts[3] += _mm_popcnt_u32(Compare16i<compareType>(_mm512_loadu_si512(src + 3 * A), value));
        }

        template <SimdCompareType compareType> void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, uint32_t * count)
        {
            width *= 2;
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask32 tailMask = TailMask32((width - alignedWidth) / 2);

            __m512i _value = _mm512_set1_epi16(value);
            uint32_t counts[4] = { 0, 0, 0, 0 };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    ConditionalCount16i4<compareType>(src + col, _value, counts);
                for (; col < alignedWidth; col += A)
                    ConditionalCount16i<false, compareType>(src + col, _value, counts);
                if (col < width)
                    ConditionalCount16i<true, compareType>(src + col, _value, counts, tailMask);
                src += stride;
            }
            *count = counts[0] + counts[1] + counts[2] + counts[3];
        }

        void ConditionalCount16i(const uint8_t * src, size_t stride, size_t width, size_t height, int16_t value, SimdCompareType compareType, uint32_t * count)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalCount16i<SimdCompareEqual>(src, stride, width, height, value, count);
            case SimdCompareNotEqual:
                return ConditionalCount16i<SimdCompareNotEqual>(src, stride, width, height, value, count);
            case SimdCompareGreater:
                return ConditionalCount16i<SimdCompareGreater>(src, stride, width, height, value, count);
            case SimdCompareGreaterOrEqual:
                return ConditionalCount16i<SimdCompareGreaterOrEqual>(src, stride, width, height, value, count);
            case SimdCompareLesser:
                return ConditionalCount16i<SimdCompareLesser>(src, stride, width, height, value, count);
            case SimdCompareLesserOrEqual:
                return ConditionalCount16i<SimdCompareLesserOrEqual>(src, stride, width, height, value, count);
            default:
                assert(0);
            }
        }

        template <SimdCompareType compareType> void ConditionalSum4(const uint8_t * src, const uint8_t * mask, const __m512i & value, __m512i * sums)
        {
            sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 0), value), src + A * 0), K_ZERO));
            sums[1] = _mm512_add_epi64(sums[1], _mm512_sad_epu8(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 1), value), src + A * 1), K_ZERO));
            sums[2] = _mm512_add_epi64(sums[2], _mm512_sad_epu8(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 2), value), src + A * 2), K_ZERO));
            sums[3] = _mm512_add_epi64(sums[3], _mm512_sad_epu8(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 3), value), src + A * 3), K_ZERO));
        }

        template <bool masked, SimdCompareType compareType> void ConditionalSum(const uint8_t * src, const uint8_t * mask, const __m512i & value, __m512i * sums, __mmask64 tail = -1)
        {
            const __m512i _mask = masked ? _mm512_maskz_loadu_epi8(tail, mask) : _mm512_loadu_si512(mask);
            __mmask64 mmask = Compare8u<compareType>(_mask, value)&tail;
            const __m512i _src = _mm512_maskz_loadu_epi8(mmask, src);
            sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(_src, K_ZERO));
        }

        template <SimdCompareType compareType> void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi8(value);
            __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    ConditionalSum4<compareType>(src + col, mask + col, _value, sums);
                for (; col < alignedWidth; col += A)
                    ConditionalSum<false, compareType>(src + col, mask + col, _value, sums);
                if (col < width)
                    ConditionalSum<true, compareType>(src + col, mask + col, _value, sums, tailMask);
                src += srcStride;
                mask += maskStride;
            }
            sums[0] = _mm512_add_epi64(_mm512_add_epi64(sums[0], sums[1]), _mm512_add_epi64(sums[2], sums[3]));
            *sum = ExtractSum<uint64_t>(sums[0]);
        }

        void ConditionalSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        SIMD_INLINE __m512i Square(__m512i value)
        {
            const __m512i lo = _mm512_unpacklo_epi8(value, K_ZERO);
            const __m512i hi = _mm512_unpackhi_epi8(value, K_ZERO);
            return _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi));
        }

        template <bool masked, SimdCompareType compareType> void ConditionalSquareSum(const uint8_t * src, const uint8_t * mask, const __m512i & value, __m512i * sums, __mmask64 tail = -1)
        {
            const __m512i _mask = masked ? _mm512_maskz_loadu_epi8(tail, mask) : _mm512_loadu_si512(mask);
            __mmask64 mmask = Compare8u<compareType>(_mask, value)&tail;
            const __m512i _src = _mm512_maskz_loadu_epi8(mmask, src);
            sums[0] = _mm512_add_epi32(sums[0], Square(_src));
        }

        template <SimdCompareType compareType> void ConditionalSquareSum4(const uint8_t * src, const uint8_t * mask, const __m512i & value, __m512i * sums)
        {
            sums[0] = _mm512_add_epi32(sums[0], Square(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 0), value), src + A * 0)));
            sums[1] = _mm512_add_epi32(sums[1], Square(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 1), value), src + A * 1)));
            sums[2] = _mm512_add_epi32(sums[2], Square(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 2), value), src + A * 2)));
            sums[3] = _mm512_add_epi32(sums[3], Square(_mm512_maskz_loadu_epi8(Compare8u<compareType>(_mm512_loadu_si512(mask + A * 3), value), src + A * 3)));
        }

        template <SimdCompareType compareType> void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width < 256 * 256 * F);

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi8(value);
            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                        ConditionalSquareSum4<compareType>(src + col, mask + col, _value, sums);
                    for (; col < alignedWidth; col += A)
                        ConditionalSquareSum<false, compareType>(src + col, mask + col, _value, sums);
                    if (col < width)
                        ConditionalSquareSum<true, compareType>(src + col, mask + col, _value, sums, tailMask);
                    src += srcStride;
                    mask += maskStride;
                }
                sums[0] = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]));
                _sum = _mm512_add_epi64(_sum, HorizontalSum32(sums[0]));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void ConditionalSquareSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        SIMD_INLINE __m512i SquaredDifference(const uint8_t * src, ptrdiff_t step, __mmask64 mask)
        {
            const __m512i a = _mm512_maskz_loadu_epi8(mask, src - step);
            const __m512i b = _mm512_maskz_loadu_epi8(mask, src + step);
            const __m512i lo = SubUnpackedU8<0>(a, b);
            const __m512i hi = SubUnpackedU8<1>(a, b);
            return _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi));
        }

        SIMD_INLINE __m512i SquareGradientSum(const uint8_t * src, ptrdiff_t stride, __mmask64 mask)
        {
            return _mm512_add_epi32(SquaredDifference(src, stride, mask), SquaredDifference(src, 1, mask));
        }

        template <bool masked, SimdCompareType compareType> void ConditionalSquareGradientSum(const uint8_t * src, ptrdiff_t stride, const uint8_t * pmask, const __m512i & value, __m512i * sums, __mmask64 tail = -1)
        {
            const __m512i _mask = masked ? _mm512_maskz_loadu_epi8(tail, pmask) : _mm512_loadu_si512(pmask);
            __mmask64 mask = Compare8u<compareType>(_mask, value)&tail;
            sums[0] = _mm512_add_epi32(sums[0], SquareGradientSum(src, stride, mask));
        }

        template <SimdCompareType compareType> void ConditionalSquareGradientSum4(const uint8_t * src, ptrdiff_t stride, const uint8_t * mask, const __m512i & value, __m512i * sums)
        {
            sums[0] = _mm512_add_epi32(sums[0], SquareGradientSum(src + A * 0, stride, Compare8u<compareType>(_mm512_loadu_si512(mask + A * 0), value)));
            sums[1] = _mm512_add_epi32(sums[1], SquareGradientSum(src + A * 1, stride, Compare8u<compareType>(_mm512_loadu_si512(mask + A * 1), value)));
            sums[2] = _mm512_add_epi32(sums[2], SquareGradientSum(src + A * 2, stride, Compare8u<compareType>(_mm512_loadu_si512(mask + A * 2), value)));
            sums[3] = _mm512_add_epi32(sums[3], SquareGradientSum(src + A * 3, stride, Compare8u<compareType>(_mm512_loadu_si512(mask + A * 3), value)));
        }

        template <SimdCompareType compareType> void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, uint64_t * sum)
        {
            assert(width >= 3 && height >= 3 && width < 256 * 256 * HF);

            src += srcStride;
            mask += maskStride;
            height -= 2;

            size_t alignedWidth = Simd::AlignLo(width - 1, A);
            size_t fullAlignedWidth = alignedWidth ? Simd::AlignLo(alignedWidth -  A, QA) + A : 0;
            __mmask64 noseMask = NoseMask64(A - 1);
            __mmask64 tailMask = TailMask64(width - 1 - alignedWidth);
            if (width <= A)
                noseMask = noseMask&tailMask;

            __m512i _value = _mm512_set1_epi8(value);
            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i _sum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block*blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    ConditionalSquareGradientSum<true, compareType>(src, srcStride, mask, _value, sums, noseMask);
                    size_t col = A;
                    for (; col < fullAlignedWidth; col += QA)
                        ConditionalSquareGradientSum4<compareType>(src + col, srcStride, mask + col, _value, sums);
                    for (; col < alignedWidth; col += A)
                        ConditionalSquareGradientSum<false, compareType>(src + col, srcStride, mask + col, _value, sums);
                    if (col < width)
                        ConditionalSquareGradientSum<true, compareType>(src + col, srcStride, mask + col, _value, sums, tailMask);
                    src += srcStride;
                    mask += maskStride;
                }
                sums[0] = _mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]));
                _sum = _mm512_add_epi64(_sum, _mm512_add_epi64(_mm512_unpacklo_epi32(sums[0], K_ZERO), _mm512_unpackhi_epi32(sums[0], K_ZERO)));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void ConditionalSquareGradientSum(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint64_t * sum)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalSquareGradientSum<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareNotEqual:
                return ConditionalSquareGradientSum<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreater:
                return ConditionalSquareGradientSum<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareGreaterOrEqual:
                return ConditionalSquareGradientSum<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesser:
                return ConditionalSquareGradientSum<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, sum);
            case SimdCompareLesserOrEqual:
                return ConditionalSquareGradientSum<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, sum);
            default:
                assert(0);
            }
        }

        template <bool masked, SimdCompareType compareType> SIMD_INLINE void ConditionalFill(const uint8_t * src, const __m512i & threshold, const __m512i & value, uint8_t * dst, __mmask64 tail = -1)
        {
            const __m512i _src = masked ? _mm512_maskz_loadu_epi8(tail, src) : _mm512_loadu_si512(src);
            _mm512_mask_storeu_epi8(dst, Compare8u<compareType>(_src, threshold)&tail, value);
        }

        template <SimdCompareType compareType> SIMD_INLINE void ConditionalFill4(const uint8_t * src, const __m512i & threshold, const __m512i & value, uint8_t * dst)
        {
            _mm512_mask_storeu_epi8(dst + 0 * A, Compare8u<compareType>(_mm512_loadu_si512(src + 0 * A), threshold), value);
            _mm512_mask_storeu_epi8(dst + 1 * A, Compare8u<compareType>(_mm512_loadu_si512(src + 1 * A), threshold), value);
            _mm512_mask_storeu_epi8(dst + 2 * A, Compare8u<compareType>(_mm512_loadu_si512(src + 2 * A), threshold), value);
            _mm512_mask_storeu_epi8(dst + 3 * A, Compare8u<compareType>(_mm512_loadu_si512(src + 3 * A), threshold), value);
        }

        template <SimdCompareType compareType>
        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            uint8_t threshold, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i _value = _mm512_set1_epi8(value);
            __m512i _threshold = _mm512_set1_epi8(threshold);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                    ConditionalFill4<compareType>(src + col, _threshold, _value, dst + col);
                for (; col < alignedWidth; col += A)
                    ConditionalFill<false, compareType>(src + col, _threshold, _value, dst + col);
                if (col < width)
                    ConditionalFill<true, compareType>(src + col, _threshold, _value, dst + col, tailMask);
                src += srcStride;
                dst += dstStride;
            }
        }

        void ConditionalFill(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t threshold, SimdCompareType compareType, uint8_t value, uint8_t * dst, size_t dstStride)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return ConditionalFill<SimdCompareEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareNotEqual:
                return ConditionalFill<SimdCompareNotEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreater:
                return ConditionalFill<SimdCompareGreater>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareGreaterOrEqual:
                return ConditionalFill<SimdCompareGreaterOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesser:
                return ConditionalFill<SimdCompareLesser>(src, srcStride, width, height, threshold, value, dst, dstStride);
            case SimdCompareLesserOrEqual:
                return ConditionalFill<SimdCompareLesserOrEqual>(src, srcStride, width, height, threshold, value, dst, dstStride);
            default:
                assert(0);
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

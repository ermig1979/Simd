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
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align> SIMD_INLINE void GetStatistic(const uint8_t * src, __m512i & min, __m512i & max, __m512i & sum)
        {
            const __m512i _src = Load<align>(src);
            min = _mm512_min_epu8(min, _src);
            max = _mm512_max_epu8(max, _src);
            sum = _mm512_add_epi64(_mm512_sad_epu8(_src, K_ZERO), sum);
        }

        template <bool align> SIMD_INLINE void GetStatistic(const uint8_t * src, __m512i & min, __m512i & max, __m512i & sum, __mmask64 tail)
        {
            const __m512i _src = Load<align, true>(src, tail);
            min = _mm512_mask_min_epu8(min, tail, min, _src);
            max = _mm512_mask_max_epu8(max, tail, max, _src);
            sum = _mm512_add_epi64(_mm512_sad_epu8(_src, K_ZERO), sum);
        }

        template <bool align> void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            assert(width*height && width >= A);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            __m512i sum = _mm512_setzero_si512();
            __m512i min512 = _mm512_set1_epi8(-1);
            __m512i max512 = _mm512_set1_epi8(0);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    GetStatistic<align>(src + col, min512, max512, sum);
                if (col < width)
                    GetStatistic<align>(src + col, min512, max512, sum, tailMask);
                src += stride;
            }

            __m128i min128 = _mm_min_epu8(_mm_min_epu8(_mm512_extracti32x4_epi32(min512, 0), _mm512_extracti32x4_epi32(min512, 1)),
                _mm_min_epu8(_mm512_extracti32x4_epi32(min512, 2), _mm512_extracti32x4_epi32(min512, 3)));
            __m128i max128 = _mm_max_epu8(_mm_max_epu8(_mm512_extracti32x4_epi32(max512, 0), _mm512_extracti32x4_epi32(max512, 1)),
                _mm_max_epu8(_mm512_extracti32x4_epi32(max512, 2), _mm512_extracti32x4_epi32(max512, 3)));

            uint8_t min_buffer[Sse41::A], max_buffer[Sse41::A];
            Sse41::Store<false>((__m128i*)min_buffer, min128);
            Sse41::Store<false>((__m128i*)max_buffer, max128);
            *min = UCHAR_MAX;
            *max = 0;
            for (size_t i = 0; i < Sse41::A; ++i)
            {
                *min = Base::MinU8(min_buffer[i], *min);
                *max = Base::MaxU8(max_buffer[i], *max);
            }
            *average = (uint8_t)((ExtractSum<uint64_t>(sum) + width*height / 2) / (width*height));
        }

        void GetStatistic(const uint8_t * src, size_t stride, size_t width, size_t height,
            uint8_t * min, uint8_t * max, uint8_t * average)
        {
            if (Aligned(src) && Aligned(stride))
                GetStatistic<true>(src, stride, width, height, min, max, average);
            else
                GetStatistic<false>(src, stride, width, height, min, max, average);
        }

        template <bool align> void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            memset(sums, 0, sizeof(uint32_t)*height);
            for (size_t row = 0; row < height; ++row)
            {
                __m512i sum = _mm512_setzero_si512();
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    sum = _mm512_add_epi32(sum, _mm512_sad_epu8(Load<align>(src + col), K_ZERO));
                if (col < width)
                    sum = _mm512_add_epi32(sum, _mm512_sad_epu8(Load<align, true>(src + col, tailMask), K_ZERO));
                sums[row] = ExtractSum<uint32_t>(sum);
                src += stride;
            }
        }

        void GetRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetRowSums<true>(src, stride, width, height, sums);
            else
                GetRowSums<false>(src, stride, width, height, sums);
        }

        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t)*width + sizeof(uint32_t)*width);
                    sums16 = (uint16_t*)_p;
                    sums32 = (uint32_t*)(sums16 + width);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t * sums16;
                uint32_t * sums32;
            private:
                void *_p;
            };
        }

        const __m512i K32_PERMUTE_FOR_COL_SUMS = SIMD_MM512_SETR_EPI32(0x0, 0x8, 0x4, 0xC, 0x1, 0x9, 0x5, 0xD, 0x2, 0xA, 0x6, 0xE, 0x3, 0xB, 0x7, 0xF);

        template<bool align, bool masked> SIMD_INLINE void GetColSum16(const uint8_t * src, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_COL_SUMS, (Load<align, masked>(src, tail)));
            Store<true>(dst + 00, _mm512_add_epi16(Load<true>(dst + 00), _mm512_unpacklo_epi8(_src, K_ZERO)));
            Store<true>(dst + HA, _mm512_add_epi16(Load<true>(dst + HA), _mm512_unpackhi_epi8(_src, K_ZERO)));
        }

        SIMD_INLINE void Sum16To32(const uint16_t * src, uint32_t * dst)
        {
            __m512i lo = Load<true>(src + 00);
            __m512i hi = Load<true>(src + HA);
            Store<true>(dst + 0 * F, _mm512_add_epi32(Load<true>(dst + 0 * F), _mm512_unpacklo_epi16(lo, K_ZERO)));
            Store<true>(dst + 1 * F, _mm512_add_epi32(Load<true>(dst + 1 * F), _mm512_unpacklo_epi16(hi, K_ZERO)));
            Store<true>(dst + 2 * F, _mm512_add_epi32(Load<true>(dst + 2 * F), _mm512_unpackhi_epi16(lo, K_ZERO)));
            Store<true>(dst + 3 * F, _mm512_add_epi32(Load<true>(dst + 3 * F), _mm512_unpackhi_epi16(hi, K_ZERO)));
        }

        template <bool align> void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedLoWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedLoWidth);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetColSum16<align, false>(src + col, buffer.sums16 + col);
                    if (col < width)
                        GetColSum16<align, true>(src + col, buffer.sums16 + col, tailMask);
                    src += stride;
                }
                for (size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
        }

        void GetColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetColSums<true>(src, stride, width, height, sums);
            else
                GetColSums<false>(src, stride, width, height, sums);
        }

        template <bool align, bool masked> void GetAbsDyRowSums(const uint8_t * src0, const uint8_t * src1, __m512i & sum, __mmask64 tail = -1)
        {
            __m512i _src0 = Load<align, masked>(src0, tail);
            __m512i _src1 = Load<align, masked>(src1, tail);
            sum = _mm512_add_epi32(sum, _mm512_sad_epu8(_src0, _src1));
        }

        template <bool align> void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedWidth);

            memset(sums, 0, sizeof(uint32_t)*height);
            const uint8_t * src0 = src;
            const uint8_t * src1 = src + stride;
            height--;
            for (size_t row = 0; row < height; ++row)
            {
                __m512i sum = _mm512_setzero_si512();
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    GetAbsDyRowSums<align, false>(src0 + col, src1 + col, sum);
                if (col < width)
                    GetAbsDyRowSums<align, true>(src0 + col, src1 + col, sum, tailMask);
                sums[row] = ExtractSum<uint32_t>(sum);
                src0 += stride;
                src1 += stride;
            }
        }

        void GetAbsDyRowSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDyRowSums<true>(src, stride, width, height, sums);
            else
                GetAbsDyRowSums<false>(src, stride, width, height, sums);
        }

        template<bool align, bool masked> SIMD_INLINE void GetAbsDxColSum16(const uint8_t * src, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i src0 = Load<align, masked>(src + 0, tail);
            __m512i src1 = Load<false, masked>(src + 1, tail);
            __m512i absDiff = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_COL_SUMS, AbsDifferenceU8(src0, src1));
            Store<true>(dst + 00, _mm512_add_epi16(Load<true>(dst + 00), _mm512_unpacklo_epi8(absDiff, K_ZERO)));
            Store<true>(dst + HA, _mm512_add_epi16(Load<true>(dst + HA), _mm512_unpackhi_epi8(absDiff, K_ZERO)));
        }

        template <bool align> void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            width--;
            size_t alignedLoWidth = AlignLo(width, A);
            __mmask64 tailMask = TailMask64(width - alignedLoWidth);
            size_t alignedHiWidth = AlignHi(width, A);
            size_t stepSize = SCHAR_MAX + 1;
            size_t stepCount = (height + SCHAR_MAX) / stepSize;

            Buffer buffer(alignedHiWidth);
            memset(buffer.sums32, 0, sizeof(uint32_t)*alignedHiWidth);
            for (size_t step = 0; step < stepCount; ++step)
            {
                size_t rowStart = step*stepSize;
                size_t rowEnd = Min(rowStart + stepSize, height);
                memset(buffer.sums16, 0, sizeof(uint16_t)*alignedHiWidth);
                for (size_t row = rowStart; row < rowEnd; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedLoWidth; col += A)
                        GetAbsDxColSum16<align, false>(src + col, buffer.sums16 + col);
                    if (col < width)
                        GetAbsDxColSum16<align, true>(src + col, buffer.sums16 + col, tailMask);
                    src += stride;
                }
                for (size_t col = 0; col < alignedHiWidth; col += A)
                    Sum16To32(buffer.sums16 + col, buffer.sums32 + col);
            }
            memcpy(sums, buffer.sums32, sizeof(uint32_t)*width);
            sums[width] = 0;
        }

        void GetAbsDxColSums(const uint8_t * src, size_t stride, size_t width, size_t height, uint32_t * sums)
        {
            if (Aligned(src) && Aligned(stride))
                GetAbsDxColSums<true>(src, stride, width, height, sums);
            else
                GetAbsDxColSums<false>(src, stride, width, height, sums);
        }

        template <bool align> void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            __m512i sums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += QA)
                {
                    sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(Load<align>(src + col + 0 * A), K_ZERO));
                    sums[1] = _mm512_add_epi64(sums[1], _mm512_sad_epu8(Load<align>(src + col + 1 * A), K_ZERO));
                    sums[2] = _mm512_add_epi64(sums[2], _mm512_sad_epu8(Load<align>(src + col + 2 * A), K_ZERO));
                    sums[3] = _mm512_add_epi64(sums[3], _mm512_sad_epu8(Load<align>(src + col + 3 * A), K_ZERO));
                }
                for (; col < alignedWidth; col += A)
                    sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(Load<align>(src + col), K_ZERO));
                if (col < width)
                    sums[0] = _mm512_add_epi64(sums[0], _mm512_sad_epu8(Load<align, true>(src + col, tailMask), K_ZERO));
                src += stride;
            }
            *sum = ExtractSum<uint64_t>(_mm512_add_epi64(_mm512_add_epi64(sums[0], sums[1]), _mm512_add_epi64(sums[2], sums[3])));
        }

        void ValueSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSum<true>(src, stride, width, height, sum);
            else
                ValueSum<false>(src, stride, width, height, sum);
        }

        SIMD_INLINE __m512i SquareSum(__m512i value)
        {
            const __m512i lo = _mm512_unpacklo_epi8(value, K_ZERO);
            const __m512i hi = _mm512_unpackhi_epi8(value, K_ZERO);
            return _mm512_add_epi32(_mm512_madd_epi16(lo, lo), _mm512_madd_epi16(hi, hi));
        }

        template <bool align, bool mask> void SquareSum(const uint8_t * src, __m512i * sums, __mmask64 tail = -1)
        {
            sums[0] = _mm512_add_epi32(sums[0], SquareSum(Load<align, mask>(src, tail)));
        }

        template <bool align> void SquareSum4(const uint8_t * src, __m512i * sums)
        {
            sums[0] = _mm512_add_epi32(sums[0], SquareSum(Load<align>(src + 0 * A)));
            sums[1] = _mm512_add_epi32(sums[1], SquareSum(Load<align>(src + 1 * A)));
            sums[2] = _mm512_add_epi32(sums[2], SquareSum(Load<align>(src + 2 * A)));
            sums[3] = _mm512_add_epi32(sums[3], SquareSum(Load<align>(src + 3 * A)));
        }

        template <bool align> void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 256 * 256 * F);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
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
                        SquareSum4<align>(src + col, sums);
                    for (; col < alignedWidth; col += A)
                        SquareSum<align, false>(src + col, sums);
                    if (col < width)
                        SquareSum<align, true>(src + col, sums, tailMask);
                    src += stride;
                }
                _sum = _mm512_add_epi64(_sum, HorizontalSum32(_mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]))));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void SquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(src) && Aligned(stride))
                SquareSum<true>(src, stride, width, height, sum);
            else
                SquareSum<false>(src, stride, width, height, sum);
        }

        template <int index> void ValueSquareSum(const __m512i & value, __m512i * valueSums, __m512i * squareSums)
        {
            valueSums[index] = _mm512_add_epi64(valueSums[index], _mm512_sad_epu8(value, K_ZERO));
            squareSums[index] = _mm512_add_epi32(squareSums[index], SquareSum(value));
        }

        template <bool align> void ValueSquareSum4(const uint8_t * src, __m512i * valueSums, __m512i * squareSums)
        {
            ValueSquareSum<0>(Load<align>(src + 0 * A), valueSums, squareSums);
            ValueSquareSum<1>(Load<align>(src + 1 * A), valueSums, squareSums);
            ValueSquareSum<2>(Load<align>(src + 2 * A), valueSums, squareSums);
            ValueSquareSum<3>(Load<align>(src + 3 * A), valueSums, squareSums);
        }

        template <bool align> void ValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
        {
            assert(width < 256 * 256 * F);
            if (align)
                assert(Aligned(src) && Aligned(stride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
            size_t blockSize = (256 * 256 * F) / width;
            size_t blockCount = height / blockSize + 1;
            __m512i valueSums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
            __m512i fullSquareSum = _mm512_setzero_si512();
            for (size_t block = 0; block < blockCount; ++block)
            {
                __m512i squareSums[4] = { _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512() };
                for (size_t row = block * blockSize, endRow = Simd::Min(row + blockSize, height); row < endRow; ++row)
                {
                    size_t col = 0;
                    for (; col < fullAlignedWidth; col += QA)
                        ValueSquareSum4<align>(src + col, valueSums, squareSums);
                    for (; col < alignedWidth; col += A)
                        ValueSquareSum<0>(Load<align>(src + col), valueSums, squareSums);
                    if (col < width)
                        ValueSquareSum<0>(Load<align, true>(src + col, tailMask), valueSums, squareSums);
                    src += stride;
                }
                fullSquareSum = _mm512_add_epi64(fullSquareSum, HorizontalSum32(
                    _mm512_add_epi32(_mm512_add_epi32(squareSums[0], squareSums[1]), _mm512_add_epi32(squareSums[2], squareSums[3]))));
            }
            *valueSum = ExtractSum<uint64_t>(_mm512_add_epi64(_mm512_add_epi64(valueSums[0], valueSums[1]), _mm512_add_epi64(valueSums[2], valueSums[3])));
            *squareSum = ExtractSum<uint64_t>(fullSquareSum);
        }

        void ValueSquareSum(const uint8_t * src, size_t stride, size_t width, size_t height, uint64_t * valueSum, uint64_t * squareSum)
        {
            if (Aligned(src) && Aligned(stride))
                ValueSquareSum<true>(src, stride, width, height, valueSum, squareSum);
            else
                ValueSquareSum<false>(src, stride, width, height, valueSum, squareSum);
        }

        void ValueSquareSums2(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t size = width * 2;
            size_t sizeA = AlignLo(size, A);
            __mmask64 tail = TailMask64(size - sizeA);
            __m512i vSum0 = _mm512_setzero_si512();
            __m512i vSum1 = _mm512_setzero_si512();
            __m512i sSum0 = _mm512_setzero_si512();
            __m512i sSum1 = _mm512_setzero_si512();
            for (size_t y = 0; y < height; ++y)
            {
                __m512i rSum0 = _mm512_setzero_si512();
                __m512i rSum1 = _mm512_setzero_si512();
                for (size_t x = 0; x < sizeA; x += A)
                {
                    const __m512i val = _mm512_loadu_si512(src + x);
                    const __m512i v0 = _mm512_and_si512(val, K16_00FF);
                    const __m512i v1 = _mm512_srli_epi16(val, 8);
                    vSum0 = _mm512_add_epi64(_mm512_sad_epu8(v0, K_ZERO), vSum0);
                    vSum1 = _mm512_add_epi64(_mm512_sad_epu8(v1, K_ZERO), vSum1);
                    rSum0 = _mm512_add_epi32(rSum0, _mm512_madd_epi16(v0, v0));
                    rSum1 = _mm512_add_epi32(rSum1, _mm512_madd_epi16(v1, v1));
                }
                if (size > sizeA)
                {
                    const __m512i val = _mm512_maskz_loadu_epi8(tail, src + sizeA);
                    const __m512i v0 = _mm512_and_si512(val, K16_00FF);
                    const __m512i v1 = _mm512_srli_epi16(val, 8);
                    vSum0 = _mm512_add_epi64(_mm512_sad_epu8(v0, K_ZERO), vSum0);
                    vSum1 = _mm512_add_epi64(_mm512_sad_epu8(v1, K_ZERO), vSum1);
                    rSum0 = _mm512_add_epi32(rSum0, _mm512_madd_epi16(v0, v0));
                    rSum1 = _mm512_add_epi32(rSum1, _mm512_madd_epi16(v1, v1));
                }
                sSum0 = _mm512_add_epi64(sSum0, HorizontalSum32(rSum0));
                sSum1 = _mm512_add_epi64(sSum1, HorizontalSum32(rSum1));
                src += stride;
            }
            valueSums[0] = ExtractSum<uint64_t>(vSum0);
            valueSums[1] = ExtractSum<uint64_t>(vSum1);
            squareSums[0] = ExtractSum<uint64_t>(sSum0);
            squareSums[1] = ExtractSum<uint64_t>(sSum1);
        }

        SIMD_INLINE __m256i Square8u(__m256i src)
        {
            const __m256i lo = _mm256_unpacklo_epi8(src, _mm256_setzero_si256());
            const __m256i hi = _mm256_unpackhi_epi8(src, _mm256_setzero_si256());
            return _mm256_add_epi32(_mm256_madd_epi16(lo, lo), _mm256_madd_epi16(hi, hi));
        }

        void ValueSquareSums3(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t widthA = AlignLo(width, Avx2::A);
            __m256i tail = Avx2::SetMask<uint8_t>(0, Avx2::A - width + widthA, 0xFF);
            __m256i vSum0 = _mm256_setzero_si256();
            __m256i vSum1 = _mm256_setzero_si256();
            __m256i vSum2 = _mm256_setzero_si256();
            __m256i sSum0 = _mm256_setzero_si256();
            __m256i sSum1 = _mm256_setzero_si256();
            __m256i sSum2 = _mm256_setzero_si256();
            __m256i bgr[3];
            for (size_t y = 0; y < height; ++y)
            {
                __m256i rSum0 = _mm256_setzero_si256();
                __m256i rSum1 = _mm256_setzero_si256();
                __m256i rSum2 = _mm256_setzero_si256();
                for (size_t x = 0; x < widthA; x += Avx2::A)
                {
                    const uint8_t* psrc = src + x * 3;
                    bgr[0] = _mm256_loadu_si256((__m256i*)psrc + 0);
                    bgr[1] = _mm256_loadu_si256((__m256i*)psrc + 1);
                    bgr[2] = _mm256_loadu_si256((__m256i*)psrc + 2);
                    __m256i v0 = Avx2::BgrToBlue(bgr);
                    vSum0 = _mm256_add_epi64(_mm256_sad_epu8(v0, Avx2::K_ZERO), vSum0);
                    rSum0 = _mm256_add_epi32(rSum0, Square8u(v0));
                    __m256i v1 = Avx2::BgrToGreen(bgr);
                    vSum1 = _mm256_add_epi64(_mm256_sad_epu8(v1, Avx2::K_ZERO), vSum1);
                    rSum1 = _mm256_add_epi32(rSum1, Square8u(v1));
                    __m256i v2 = Avx2::BgrToRed(bgr);
                    vSum2 = _mm256_add_epi64(_mm256_sad_epu8(v2, Avx2::K_ZERO), vSum2);
                    rSum2 = _mm256_add_epi32(rSum2, Square8u(v2));
                }
                if (width - widthA)
                {
                    const uint8_t* psrc = src + (width - Avx2::A) * 3;
                    bgr[0] = _mm256_loadu_si256((__m256i*)psrc + 0);
                    bgr[1] = _mm256_loadu_si256((__m256i*)psrc + 1);
                    bgr[2] = _mm256_loadu_si256((__m256i*)psrc + 2);
                    __m256i v0 = _mm256_and_si256(tail, Avx2::BgrToBlue(bgr));
                    vSum0 = _mm256_add_epi64(_mm256_sad_epu8(v0, Avx2::K_ZERO), vSum0);
                    rSum0 = _mm256_add_epi32(rSum0, Square8u(v0));
                    __m256i v1 = _mm256_and_si256(tail, Avx2::BgrToGreen(bgr));
                    vSum1 = _mm256_add_epi64(_mm256_sad_epu8(v1, Avx2::K_ZERO), vSum1);
                    rSum1 = _mm256_add_epi32(rSum1, Square8u(v1));
                    __m256i v2 = _mm256_and_si256(tail, Avx2::BgrToRed(bgr));
                    vSum2 = _mm256_add_epi64(_mm256_sad_epu8(v2, Avx2::K_ZERO), vSum2);
                    rSum2 = _mm256_add_epi32(rSum2, Square8u(v2));
                }
                sSum0 = _mm256_add_epi64(sSum0, Avx2::HorizontalSum32(rSum0));
                sSum1 = _mm256_add_epi64(sSum1, Avx2::HorizontalSum32(rSum1));
                sSum2 = _mm256_add_epi64(sSum2, Avx2::HorizontalSum32(rSum2));
                src += stride;
            }
            valueSums[0] = Avx2::ExtractSum<uint64_t>(vSum0);
            valueSums[1] = Avx2::ExtractSum<uint64_t>(vSum1);
            valueSums[2] = Avx2::ExtractSum<uint64_t>(vSum2);
            squareSums[0] = Avx2::ExtractSum<uint64_t>(sSum0);
            squareSums[1] = Avx2::ExtractSum<uint64_t>(sSum1);
            squareSums[2] = Avx2::ExtractSum<uint64_t>(sSum2);
        }

        const __m512i K8_SHFL_4_01 = SIMD_MM512_SETR_EPI8(
            0x0, -1, 0x4, -1, 0x8, -1, 0xC, -1, 0x1, -1, 0x5, -1, 0x9, -1, 0xD, -1,
            0x0, -1, 0x4, -1, 0x8, -1, 0xC, -1, 0x1, -1, 0x5, -1, 0x9, -1, 0xD, -1,
            0x0, -1, 0x4, -1, 0x8, -1, 0xC, -1, 0x1, -1, 0x5, -1, 0x9, -1, 0xD, -1,
            0x0, -1, 0x4, -1, 0x8, -1, 0xC, -1, 0x1, -1, 0x5, -1, 0x9, -1, 0xD, -1);
        const __m512i K8_SHFL_4_23 = SIMD_MM512_SETR_EPI8(
            0x2, -1, 0x6, -1, 0xA, -1, 0xE, -1, 0x3, -1, 0x7, -1, 0xB, -1, 0xF, -1,
            0x2, -1, 0x6, -1, 0xA, -1, 0xE, -1, 0x3, -1, 0x7, -1, 0xB, -1, 0xF, -1,
            0x2, -1, 0x6, -1, 0xA, -1, 0xE, -1, 0x3, -1, 0x7, -1, 0xB, -1, 0xF, -1,
            0x2, -1, 0x6, -1, 0xA, -1, 0xE, -1, 0x3, -1, 0x7, -1, 0xB, -1, 0xF, -1);

        SIMD_INLINE __m512i PairSum32(__m512i a)
        {
            return _mm512_add_epi64(_mm512_and_si512(a, K64_00000000FFFFFFFF), _mm512_srli_epi64(a, 32));
        }

        void ValueSquareSums4(const uint8_t* src, size_t stride, size_t width, size_t height, uint64_t* valueSums, uint64_t* squareSums)
        {
            size_t size = width * 4;
            size_t sizeA = AlignLo(size, A);
            __mmask64 tail = TailMask64(size - sizeA);
            __m512i vSum01 = _mm512_setzero_si512();
            __m512i vSum23 = _mm512_setzero_si512();
            __m512i sSum01 = _mm512_setzero_si512();
            __m512i sSum23 = _mm512_setzero_si512();
            for (size_t y = 0; y < height; ++y)
            {
                __m512i rSum01 = _mm512_setzero_si512();
                __m512i rSum23 = _mm512_setzero_si512();
                for (size_t x = 0; x < sizeA; x += A)
                {
                    const __m512i val = _mm512_loadu_si512((__m512i*)(src + x));
                    const __m512i v01 = _mm512_shuffle_epi8(val, K8_SHFL_4_01);
                    const __m512i v23 = _mm512_shuffle_epi8(val, K8_SHFL_4_23);
                    vSum01 = _mm512_add_epi64(_mm512_sad_epu8(v01, K_ZERO), vSum01);
                    rSum01 = _mm512_add_epi32(rSum01, _mm512_madd_epi16(v01, v01));
                    vSum23 = _mm512_add_epi64(_mm512_sad_epu8(v23, K_ZERO), vSum23);
                    rSum23 = _mm512_add_epi32(rSum23, _mm512_madd_epi16(v23, v23));
                }
                if (size > sizeA)
                {
                    const __m512i val = _mm512_maskz_loadu_epi8(tail, src + sizeA);
                    const __m512i v01 = _mm512_shuffle_epi8(val, K8_SHFL_4_01);
                    const __m512i v23 = _mm512_shuffle_epi8(val, K8_SHFL_4_23);
                    vSum01 = _mm512_add_epi64(_mm512_sad_epu8(v01, K_ZERO), vSum01);
                    rSum01 = _mm512_add_epi32(rSum01, _mm512_madd_epi16(v01, v01));
                    vSum23 = _mm512_add_epi64(_mm512_sad_epu8(v23, K_ZERO), vSum23);
                    rSum23 = _mm512_add_epi32(rSum23, _mm512_madd_epi16(v23, v23));
                }
                sSum01 = _mm512_add_epi64(sSum01, PairSum32(rSum01));
                sSum23 = _mm512_add_epi64(sSum23, PairSum32(rSum23));
                src += stride;
            }
            __m128i vSum01s = ExtractSum<__m128i>(vSum01);
            __m128i vSum23s = ExtractSum<__m128i>(vSum23);
            __m128i sSum01s = ExtractSum<__m128i>(sSum01);
            __m128i sSum23s = ExtractSum<__m128i>(sSum23);
            valueSums[0] = Sse41::ExtractInt64<0>(vSum01s);
            valueSums[1] = Sse41::ExtractInt64<1>(vSum01s);
            valueSums[2] = Sse41::ExtractInt64<0>(vSum23s);
            valueSums[3] = Sse41::ExtractInt64<1>(vSum23s);
            squareSums[0] = Sse41::ExtractInt64<0>(sSum01s);
            squareSums[1] = Sse41::ExtractInt64<1>(sSum01s);
            squareSums[2] = Sse41::ExtractInt64<0>(sSum23s);
            squareSums[3] = Sse41::ExtractInt64<1>(sSum23s);
        }

        void ValueSquareSums(const uint8_t* src, size_t stride, size_t width, size_t height, size_t channels, uint64_t* valueSums, uint64_t* squareSums)
        {
            assert(width >= Avx2::A && width < 0x10000);

            switch (channels)
            {
            case 1: ValueSquareSum<false>(src, stride, width, height, valueSums, squareSums); break;
            case 2: ValueSquareSums2(src, stride, width, height, valueSums, squareSums); break;
            case 3: ValueSquareSums3(src, stride, width, height, valueSums, squareSums); break;
            case 4: ValueSquareSums4(src, stride, width, height, valueSums, squareSums); break;
            default:
                assert(0);
            }
        }

        //---------------------------------------------------------------------

        SIMD_INLINE __m512i CorrelationSum(__m512i a, __m512i b)
        {
            const __m512i lo = _mm512_madd_epi16(_mm512_unpacklo_epi8(a, _mm512_setzero_si512()), _mm512_unpacklo_epi8(b, _mm512_setzero_si512()));
            const __m512i hi = _mm512_madd_epi16(_mm512_unpackhi_epi8(a, _mm512_setzero_si512()), _mm512_unpackhi_epi8(b, _mm512_setzero_si512()));
            return _mm512_add_epi32(lo, hi);
        }

        template <bool align, bool mask> void CorrelationSum(const uint8_t * a, const uint8_t * b, __m512i * sums, __mmask64 tail = -1)
        {
            sums[0] = _mm512_add_epi32(sums[0], CorrelationSum(Load<align, mask>(a, tail), Load<align, mask>(b, tail)));
        }

        template <bool align> void CorrelationSum4(const uint8_t * a, const uint8_t * b, __m512i * sums)
        {
            sums[0] = _mm512_add_epi32(sums[0], CorrelationSum(Load<align>(a + 0 * A), Load<align>(b + 0 * A)));
            sums[1] = _mm512_add_epi32(sums[1], CorrelationSum(Load<align>(a + 1 * A), Load<align>(b + 1 * A)));
            sums[2] = _mm512_add_epi32(sums[2], CorrelationSum(Load<align>(a + 2 * A), Load<align>(b + 2 * A)));
            sums[3] = _mm512_add_epi32(sums[3], CorrelationSum(Load<align>(a + 3 * A), Load<align>(b + 3 * A)));
        }

        template <bool align> void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            assert(width < 256 * 256 * F);
            if (align)
                assert(Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride));

            size_t alignedWidth = Simd::AlignLo(width, A);
            size_t fullAlignedWidth = Simd::AlignLo(width, QA);
            __mmask64 tailMask = TailMask64(width - alignedWidth);
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
                        CorrelationSum4<align>(a + col, b + col, sums);
                    for (; col < alignedWidth; col += A)
                        CorrelationSum<align, false>(a + col, b + col, sums);
                    if (col < width)
                        CorrelationSum<align, true>(a + col, b + col, sums, tailMask);
                    a += aStride;
                    b += bStride;
                }
                _sum = _mm512_add_epi64(_sum, HorizontalSum32(_mm512_add_epi32(_mm512_add_epi32(sums[0], sums[1]), _mm512_add_epi32(sums[2], sums[3]))));
            }
            *sum = ExtractSum<uint64_t>(_sum);
        }

        void CorrelationSum(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride, size_t width, size_t height, uint64_t * sum)
        {
            if (Aligned(a) && Aligned(aStride) && Aligned(b) && Aligned(bStride))
                CorrelationSum<true>(a, aStride, b, bStride, width, height, sum);
            else
                CorrelationSum<false>(a, aStride, b, bStride, width, height, sum);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

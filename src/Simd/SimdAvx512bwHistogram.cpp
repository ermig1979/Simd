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
#include "Simd/SimdCompare.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        namespace
        {
            template<class T> struct Buffer
            {
                Buffer(size_t rowSize, size_t histogramSize)
                {
                    _p = Allocate(sizeof(T)*rowSize + 4 * sizeof(uint32_t)*histogramSize);
                    v = (T*)_p;
                    h[0] = (uint32_t *)(v + rowSize);
                    h[1] = h[0] + histogramSize;
                    h[2] = h[1] + histogramSize;
                    h[3] = h[2] + histogramSize;
                    memset(h[0], 0, 4 * sizeof(uint32_t)*histogramSize);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                T * v;
                uint32_t * h[4];
            private:
                void *_p;
            };
        }

        SIMD_INLINE __m512i AbsSecondDerivative(const uint8_t * src, ptrdiff_t step)
        {
            const __m512i s0 = _mm512_loadu_si512(src - step);
            const __m512i s1 = _mm512_loadu_si512(src);
            const __m512i s2 = _mm512_loadu_si512(src + step);
            return AbsDifferenceU8(_mm512_avg_epu8(s0, s2), s1);
        }

        SIMD_INLINE void AbsSecondDerivative(const uint8_t * src, ptrdiff_t colStep, ptrdiff_t rowStep, uint8_t * dst)
        {
            const __m512i sdX = AbsSecondDerivative(src, colStep);
            const __m512i sdY = AbsSecondDerivative(src, rowStep);
            _mm512_storeu_si512(dst, _mm512_max_epu8(sdY, sdX));
        }

        SIMD_INLINE void SumHistograms(uint32_t * src, size_t start, uint32_t * dst)
        {
            uint32_t * src0 = src + start;
            uint32_t * src1 = src0 + start + HISTOGRAM_SIZE;
            uint32_t * src2 = src1 + start + HISTOGRAM_SIZE;
            uint32_t * src3 = src2 + start + HISTOGRAM_SIZE;
            for (size_t i = 0; i < HISTOGRAM_SIZE; i += F)
                _mm512_storeu_si512(dst + i, _mm512_add_epi32(_mm512_add_epi32(_mm512_loadu_si512(src0 + i), _mm512_loadu_si512(src1 + i)), _mm512_add_epi32(_mm512_loadu_si512(src2 + i), _mm512_loadu_si512(src3 + i))));
        }

#ifdef __GNUC__
        //#define SIMD_USE_GATHER_AND_SCATTER_FOR_HISTOGRAM // low performance
#endif

#if defined(SIMD_USE_GATHER_AND_SCATTER_FOR_HISTOGRAM)
        const __m512i K32_TO_HISTOGRAMS = SIMD_MM512_SETR_EPI32(0x000, 0x100, 0x200, 0x300, 0x000, 0x100, 0x200, 0x300, 0x000, 0x100, 0x200, 0x300, 0x000, 0x100, 0x200, 0x300);

        SIMD_INLINE void AddToHistogram(__m128i index, uint32_t * histogram)
        {
            __m128i hist = _mm_i32gather_epi32((int*)histogram, index, 4);
            hist = _mm_add_epi32(hist, Sse41::K32_00000001);
            _mm_i32scatter_epi32((int*)histogram, index, hist, 4);
        }
#endif

        void AbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t * histogram)
        {
            assert(width > 2 * indent && height > 2 * indent && indent >= step && width >= A + 2 * indent);

            Buffer<uint8_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE);
            buffer.v += indent;
            src += indent*(stride + 1);
            height -= 2 * indent;
            width -= 2 * indent;

            size_t rowStep = step*stride;
            size_t alignedWidth = Simd::AlignLo(width, 4);
            size_t fullAlignedWidth = Simd::AlignLo(width, Sse41::A);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                size_t colA = 0;
                for (; colA < widthAlignedA; colA += A)
                    AbsSecondDerivative(src + colA, step, rowStep, buffer.v + colA);
                if (width != widthAlignedA)
                    AbsSecondDerivative(src + width - A, step, rowStep, buffer.v + width - A);

                size_t col = 0;
#if defined(SIMD_USE_GATHER_AND_SCATTER_FOR_HISTOGRAM)
                for (; col < fullAlignedWidth; col += Sse41::A)
                {
                    __m512i index = _mm512_add_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(buffer.v + col))), K32_TO_HISTOGRAMS);
                    AddToHistogram(_mm512_extracti32x4_epi32(index, 0), buffer.h[0]);
                    AddToHistogram(_mm512_extracti32x4_epi32(index, 1), buffer.h[0]);
                    AddToHistogram(_mm512_extracti32x4_epi32(index, 2), buffer.h[0]);
                    AddToHistogram(_mm512_extracti32x4_epi32(index, 3), buffer.h[0]);
                }
#endif
                for (; col < alignedWidth; col += 4)
                {
                    ++buffer.h[0][buffer.v[col + 0]];
                    ++buffer.h[1][buffer.v[col + 1]];
                    ++buffer.h[2][buffer.v[col + 2]];
                    ++buffer.h[3][buffer.v[col + 3]];
                }
                for (; col < width; ++col)
                    ++buffer.h[0][buffer.v[col + 0]];
                src += stride;
            }

            SumHistograms(buffer.h[0], 0, histogram);
        }

        template <bool masked> SIMD_INLINE void MaskSrc(const uint8_t * src, const uint8_t * mask, const __m512i & index, ptrdiff_t offset, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, masked ? _mm512_maskz_loadu_epi8(tail, src + offset) : _mm512_loadu_si512(src + offset));
            __mmask64 mmask = _mm512_cmpeq_epi8_mask(masked ? _mm512_maskz_loadu_epi8(tail, mask + offset) : _mm512_loadu_si512(mask + offset), index);
            _mm512_storeu_si512(dst + offset + 00, _mm512_maskz_add_epi16(__mmask32(mmask >> 00), UnpackU8<0>(_src), K16_0010));
            _mm512_storeu_si512(dst + offset + HA, _mm512_maskz_add_epi16(__mmask32(mmask >> 32), UnpackU8<1>(_src), K16_0010));
        }

        void HistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
        {
            Buffer<uint16_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE + F);
            size_t widthAligned4 = Simd::AlignLo(width, 4);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            size_t widthAlignedDA = Simd::AlignLo(width, DA);
            __m512i _index = _mm512_set1_epi8(index);
            __mmask64 tailMask = TailMask64(width - widthAlignedA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthAlignedDA; col += DA)
                {
                    MaskSrc<false>(src, mask, _index, col + 0, buffer.v);
                    MaskSrc<false>(src, mask, _index, col + A, buffer.v);
                }
                for (; col < widthAlignedA; col += A)
                    MaskSrc<false>(src, mask, _index, col, buffer.v);
                if (col < width)
                    MaskSrc<true>(src, mask, _index, col, buffer.v, tailMask);

                for (col = 0; col < widthAligned4; col += 4)
                {
                    ++buffer.h[0][buffer.v[col + 0]];
                    ++buffer.h[1][buffer.v[col + 1]];
                    ++buffer.h[2][buffer.v[col + 2]];
                    ++buffer.h[3][buffer.v[col + 3]];
                }
                for (; col < width; ++col)
                    ++buffer.h[0][buffer.v[col]];

                src += srcStride;
                mask += maskStride;
            }

            SumHistograms(buffer.h[0], F, histogram);
        }

        template <SimdCompareType compareType, bool masked>
        SIMD_INLINE void ConditionalSrc(const uint8_t * src, const uint8_t * mask, const __m512i & value, ptrdiff_t offset, uint16_t * dst, __mmask64 tail = -1)
        {
            __m512i _src = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, masked ? _mm512_maskz_loadu_epi8(tail, src + offset) : _mm512_loadu_si512(src + offset));
            __mmask64 mmask = Compare8u<compareType>(masked ? _mm512_maskz_loadu_epi8(tail, mask + offset) : _mm512_loadu_si512(mask + offset), value) & tail;
            _mm512_storeu_si512(dst + offset + 00, _mm512_maskz_add_epi16(__mmask32(mmask >> 00), UnpackU8<0>(_src), K16_0010));
            _mm512_storeu_si512(dst + offset + HA, _mm512_maskz_add_epi16(__mmask32(mmask >> 32), UnpackU8<1>(_src), K16_0010));
        }

        template<SimdCompareType compareType> void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint32_t * histogram)
        {
            Buffer<uint16_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE + F);
            size_t widthAligned4 = Simd::AlignLo(width, 4);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            size_t widthAlignedDA = Simd::AlignLo(width, DA);
            __m512i _value = _mm512_set1_epi8(value);
            __mmask64 tailMask = TailMask64(width - widthAlignedA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthAlignedDA; col += DA)
                {
                    ConditionalSrc<compareType, false>(src, mask, _value, col, buffer.v);
                    ConditionalSrc<compareType, false>(src, mask, _value, col + A, buffer.v);
                }
                for (; col < widthAlignedA; col += A)
                    ConditionalSrc<compareType, false>(src, mask, _value, col, buffer.v);
                if (col < width)
                    ConditionalSrc<compareType, true>(src, mask, _value, col, buffer.v, tailMask);

                for (col = 0; col < widthAligned4; col += 4)
                {
                    ++buffer.h[0][buffer.v[col + 0]];
                    ++buffer.h[1][buffer.v[col + 1]];
                    ++buffer.h[2][buffer.v[col + 2]];
                    ++buffer.h[3][buffer.v[col + 3]];
                }
                for (; col < width; ++col)
                    ++buffer.h[0][buffer.v[col]];

                src += srcStride;
                mask += maskStride;
            }

            SumHistograms(buffer.h[0], F, histogram);
        }

        void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t * histogram)
        {
            switch (compareType)
            {
            case SimdCompareEqual:
                return HistogramConditional<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareNotEqual:
                return HistogramConditional<SimdCompareNotEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreater:
                return HistogramConditional<SimdCompareGreater>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareGreaterOrEqual:
                return HistogramConditional<SimdCompareGreaterOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesser:
                return HistogramConditional<SimdCompareLesser>(src, srcStride, width, height, mask, maskStride, value, histogram);
            case SimdCompareLesserOrEqual:
                return HistogramConditional<SimdCompareLesserOrEqual>(src, srcStride, width, height, mask, maskStride, value, histogram);
            default:
                assert(0);
            }
        }

        SIMD_INLINE __m512i ChangeColors(const __m512i & src, const __m512i colors[4])
        {
            __mmask32 blend = _mm512_cmpge_epi16_mask(src, K16_0080);
            __m512i permute = _mm512_srli_epi16(src, 1);
            __m512i shift = _mm512_slli_epi16(_mm512_and_si512(src, K16_0001), 3);
            __m512i permute0 = _mm512_permutex2var_epi16(colors[0], permute, colors[1]);
            __m512i permute1 = _mm512_permutex2var_epi16(colors[2], permute, colors[3]);
            __m512i blended = _mm512_mask_blend_epi16(blend, permute0, permute1);
            return _mm512_and_si512(_mm512_srlv_epi16(blended, shift), K16_00FF);
        }

        SIMD_INLINE void ChangeColors(const uint8_t * src, const __m512i colors[4], uint8_t * dst)
        {
            __m512i _src = _mm512_cvtepu8_epi16(_mm256_loadu_si256((__m256i*)src));
            __m512i _dst = ChangeColors(_src, colors);
            _mm256_storeu_si256((__m256i*)dst, _mm512_cvtepi16_epi8(_dst));
        }

        SIMD_INLINE void ChangeColors(const uint8_t * src, const __m512i colors[4], uint8_t * dst, __mmask64 tail)
        {
            __m512i _src = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(_mm512_maskz_loadu_epi8(tail, src)));
            __m512i _dst = ChangeColors(_src, colors);
            _mm512_mask_storeu_epi8(dst, tail, _mm512_castsi256_si512(_mm512_cvtepi16_epi8(_dst)));
        }

        void ChangeColors(const uint8_t * src, size_t srcStride, size_t width, size_t height, const uint8_t * colors, uint8_t * dst, size_t dstStride)
        {
            assert(width >= Avx512bw::HA);

            __m512i _colors[4];
            _colors[0] = _mm512_loadu_si512(colors + 0 * A);
            _colors[1] = _mm512_loadu_si512(colors + 1 * A);
            _colors[2] = _mm512_loadu_si512(colors + 2 * A);
            _colors[3] = _mm512_loadu_si512(colors + 3 * A);

            size_t widthHA = Simd::AlignLo(width, HA);
            __mmask64 tail = TailMask64(width - widthHA);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthHA; col += HA)
                    ChangeColors(src + col, _colors, dst + col);
                if(col < width)
                    ChangeColors(src + col, _colors, dst + col, tail);
                src += srcStride;
                dst += dstStride;
            }
        }

        void NormalizeHistogram(const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            uint32_t histogram[HISTOGRAM_SIZE];
            Base::Histogram(src, width, height, srcStride, histogram);

            uint8_t colors[HISTOGRAM_SIZE];
            Base::NormalizedColors(histogram, colors);

            ChangeColors(src, srcStride, width, height, colors, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

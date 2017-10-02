/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdCompare.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
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

        template <bool srcAlign, bool stepAlign>
        SIMD_INLINE uint8x16_t AbsSecondDerivative(const uint8_t * src, ptrdiff_t step)
        {
            const uint8x16_t s0 = Load<srcAlign && stepAlign>(src - step);
            const uint8x16_t s1 = Load<srcAlign>(src);
            const uint8x16_t s2 = Load<srcAlign && stepAlign>(src + step);
            return vabdq_u8(vrhaddq_u8(s0, s2), s1);
        }

        template <bool align>
        SIMD_INLINE void AbsSecondDerivative(const uint8_t * src, ptrdiff_t colStep, ptrdiff_t rowStep, uint8_t * dst)
        {
            const uint8x16_t sdX = AbsSecondDerivative<align, false>(src, colStep);
            const uint8x16_t sdY = AbsSecondDerivative<align, true>(src, rowStep);
            Store<align>(dst, vmaxq_u8(sdY, sdX));
        }

        SIMD_INLINE void SumHistograms(uint32_t * src, size_t start, uint32_t * dst)
        {
            uint32_t * src0 = src + start;
            uint32_t * src1 = src0 + start + HISTOGRAM_SIZE;
            uint32_t * src2 = src1 + start + HISTOGRAM_SIZE;
            uint32_t * src3 = src2 + start + HISTOGRAM_SIZE;
            for (size_t i = 0; i < HISTOGRAM_SIZE; i += 4)
                Store<false>(dst + i, vaddq_u32(
                    vaddq_u32(Load<true>(src0 + i), Load<true>(src1 + i)),
                    vaddq_u32(Load<true>(src2 + i), Load<true>(src3 + i))));
        }

        template<bool align> void AbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t * histogram)
        {
            Buffer<uint8_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE);
            buffer.v += indent;
            src += indent*(stride + 1);
            height -= 2 * indent;
            width -= 2 * indent;

            ptrdiff_t bodyStart = (uint8_t*)AlignHi(buffer.v, A) - buffer.v;
            ptrdiff_t bodyEnd = bodyStart + AlignLo(width - bodyStart, A);
            size_t rowStep = step*stride;
            size_t alignedWidth = Simd::AlignLo(width, 4);
            for (size_t row = 0; row < height; ++row)
            {
                if (bodyStart)
                    AbsSecondDerivative<false>(src, step, rowStep, buffer.v);
                for (ptrdiff_t col = bodyStart; col < bodyEnd; col += A)
                    AbsSecondDerivative<align>(src + col, step, rowStep, buffer.v + col);
                if (width != (size_t)bodyEnd)
                    AbsSecondDerivative<false>(src + width - A, step, rowStep, buffer.v + width - A);

                size_t col = 0;
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

        void AbsSecondDerivativeHistogram(const uint8_t *src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t * histogram)
        {
            assert(width > 2 * indent && height > 2 * indent && indent >= step && width >= A + 2 * indent);

            if (Aligned(src) && Aligned(stride))
                AbsSecondDerivativeHistogram<true>(src, width, height, stride, step, indent, histogram);
            else
                AbsSecondDerivativeHistogram<false>(src, width, height, stride, step, indent, histogram);
        }

        template <bool srcAlign, bool dstAlign>
        SIMD_INLINE void MaskSrc(const uint8_t * src, const uint8_t * mask, const uint8x16_t & index, ptrdiff_t offset, uint16_t * dst)
        {
            const uint8x16_t _src = Load<srcAlign>(src + offset);
            const uint8x16_t _mask = vandq_u8(vceqq_u8(Load<srcAlign>(mask + offset), index), K8_01);
            Store<dstAlign>(dst + offset + 0, vmulq_u16(vaddw_u8(K16_0004, Half<0>(_src)), UnpackU8<0>(_mask)));
            Store<dstAlign>(dst + offset + HA, vmulq_u16(vaddw_u8(K16_0004, Half<1>(_src)), UnpackU8<1>(_mask)));
        }

        template<bool align> void HistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
        {
            Buffer<uint16_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE + 4);
            size_t widthAligned4 = Simd::AlignLo(width, 4);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            size_t widthAlignedDA = Simd::AlignLo(width, DA);
            uint8x16_t _index = vdupq_n_u8(index);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthAlignedDA; col += DA)
                {
                    MaskSrc<align, true>(src, mask, _index, col, buffer.v);
                    MaskSrc<align, true>(src, mask, _index, col + A, buffer.v);
                }
                for (; col < widthAlignedA; col += A)
                    MaskSrc<align, true>(src, mask, _index, col, buffer.v);
                if (width != widthAlignedA)
                    MaskSrc<false, false>(src, mask, _index, width - A, buffer.v);

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

            SumHistograms(buffer.h[0], 4, histogram);
        }

        void HistogramMasked(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t index, uint32_t * histogram)
        {
            assert(width >= A);

            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                HistogramMasked<true>(src, srcStride, width, height, mask, maskStride, index, histogram);
            else
                HistogramMasked<false>(src, srcStride, width, height, mask, maskStride, index, histogram);
        }

        template <SimdCompareType compareType, bool srcAlign, bool dstAlign>
        SIMD_INLINE void ConditionalSrc(const uint8_t * src, const uint8_t * mask, const uint8x16_t & value, ptrdiff_t offset, uint16_t * dst)
        {
            const uint8x16_t _src = Load<srcAlign>(src + offset);
            const uint8x16_t _mask = vandq_u8(Compare8u<compareType>(Load<srcAlign>(mask + offset), value), K8_01);
            Store<dstAlign>(dst + offset + 0, vmulq_u16(vaddw_u8(K16_0004, Half<0>(_src)), UnpackU8<0>(_mask)));
            Store<dstAlign>(dst + offset + HA, vmulq_u16(vaddw_u8(K16_0004, Half<1>(_src)), UnpackU8<1>(_mask)));
        }

        template<SimdCompareType compareType, bool align> void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint32_t * histogram)
        {
            Buffer<uint16_t> buffer(AlignHi(width, A), HISTOGRAM_SIZE + 4);
            size_t widthAligned4 = Simd::AlignLo(width, 4);
            size_t widthAlignedA = Simd::AlignLo(width, A);
            size_t widthAlignedDA = Simd::AlignLo(width, DA);
            uint8x16_t _value = vdupq_n_u8(value);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthAlignedDA; col += DA)
                {
                    ConditionalSrc<compareType, align, true>(src, mask, _value, col, buffer.v);
                    ConditionalSrc<compareType, align, true>(src, mask, _value, col + A, buffer.v);
                }
                for (; col < widthAlignedA; col += A)
                    ConditionalSrc<compareType, align, true>(src, mask, _value, col, buffer.v);
                if (width != widthAlignedA)
                    ConditionalSrc<compareType, false, false>(src, mask, _value, width - A, buffer.v);

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

            SumHistograms(buffer.h[0], 4, histogram);
        }

        template <SimdCompareType compareType>
        void HistogramConditional(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * mask, size_t maskStride, uint8_t value, uint32_t * histogram)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(mask) && Aligned(maskStride))
                return HistogramConditional<compareType, true>(src, srcStride, width, height, mask, maskStride, value, histogram);
            else
                return HistogramConditional<compareType, false>(src, srcStride, width, height, mask, maskStride, value, histogram);
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
    }
#endif// SIMD_NEON_ENABLE
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        namespace
        {
            template<class T> struct Buffer
            {
                Buffer(size_t rowSize, size_t histogramSize)
                {
                    _p = Allocate(sizeof(T) * rowSize + 4 * sizeof(uint32_t) * histogramSize);
                    v = (T*)_p;
                    h[0] = (uint32_t*)(v + rowSize);
                    h[1] = h[0] + histogramSize;
                    h[2] = h[1] + histogramSize;
                    h[3] = h[2] + histogramSize;
                    memset(h[0], 0, 4 * sizeof(uint32_t) * histogramSize);
                }

                ~Buffer()
                {
                    Free(_p);
                }

                T* v;
                uint32_t* h[4];
            private:
                void* _p;
            };
        }

        SIMD_INLINE svuint8_t AbsSecondDerivative(const uint8_t* src, ptrdiff_t step, const svbool_t& mask)
        {
            const svuint8_t s0 = svld1_u8(mask, src - step);
            const svuint8_t s1 = svld1_u8(mask, src);
            const svuint8_t s2 = svld1_u8(mask, src + step);
            return svabd_u8_x(mask, svrhadd_u8_x(mask, s0, s2), s1);
        }

        SIMD_INLINE void AbsSecondDerivative(const uint8_t* src, ptrdiff_t colStep, ptrdiff_t rowStep, uint8_t* dst, const svbool_t& mask)
        {
            const svuint8_t sdX = AbsSecondDerivative(src, colStep, mask);
            const svuint8_t sdY = AbsSecondDerivative(src, rowStep, mask);
            svst1_u8(mask, dst, svmax_u8_x(mask, sdY, sdX));
        }

        SIMD_INLINE void SumHistograms(uint32_t* src, size_t start, uint32_t* dst)
        {
            uint32_t* src0 = src + start;
            uint32_t* src1 = src0 + start + HISTOGRAM_SIZE;
            uint32_t* src2 = src1 + start + HISTOGRAM_SIZE;
            uint32_t* src3 = src2 + start + HISTOGRAM_SIZE;
            for (size_t i = 0; i < HISTOGRAM_SIZE; ++i)
                dst[i] = src0[i] + src1[i] + src2[i] + src3[i];
        }

        void AbsSecondDerivativeHistogram(const uint8_t* src, size_t width, size_t height, size_t stride,
            size_t step, size_t indent, uint32_t* histogram)
        {
            assert(width > 2 * indent && height > 2 * indent && indent >= step);

            size_t A = svlen(svuint8_t());
            src += indent * (stride + 1);
            height -= 2 * indent;
            width -= 2 * indent;

            Buffer<uint8_t> buffer(AlignHiAny(width, A), HISTOGRAM_SIZE);
            ptrdiff_t rowStep = step * stride;
            size_t widthA = AlignLoAny(width, A);
            size_t alignedWidth4 = Simd::AlignLo(width, 4);
            const svbool_t body = svptrue_b8();
            const svbool_t tail = svwhilelt_b8(widthA, width);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    AbsSecondDerivative(src + col, step, rowStep, buffer.v + col, body);
                if (widthA < width)
                    AbsSecondDerivative(src + col, step, rowStep, buffer.v + col, tail);

                for (col = 0; col < alignedWidth4; col += 4)
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

        template <SimdCompareType compareType> SIMD_INLINE
        void ConditionalSrc(const uint8_t* src, const uint8_t* mask, const svbool_t& mask8,
            const svuint8_t& value, uint16_t* dst, const svbool_t& lo16, const svbool_t& hi16)
        {
            svuint8_t _src = svld1_u8(mask8, src);
            svuint8_t _mask = svld1_u8(mask8, mask);
            svuint8_t enable = svand_u8_z(Compare8u<compareType>(mask8, _mask, value), svdup_n_u8(1), svdup_n_u8(1));
            svst1_u16(lo16, dst, svmul_u16_x(lo16, svadd_n_u16_x(lo16, svunpklo_u16(_src), 4), svunpklo_u16(enable)));
            svst1_u16(hi16, dst + svlen(svuint16_t()), svmul_u16_x(hi16, svadd_n_u16_x(hi16, svunpkhi_u16(_src), 4), svunpkhi_u16(enable)));
        }

        template <SimdCompareType compareType>
        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, uint32_t* histogram)
        {
            size_t A = svlen(svuint8_t()), HA = svlen(svuint16_t());
            Buffer<uint16_t> buffer(AlignHiAny(width, A), HISTOGRAM_SIZE + 4);
            size_t widthA = AlignLoAny(width, A);
            size_t alignedWidth4 = Simd::AlignLo(width, 4);
            const svbool_t body8 = svptrue_b8(), body16 = svptrue_b16();
            const svbool_t tail8 = svwhilelt_b8(widthA, width);
            const svbool_t tailLo16 = svwhilelt_b16(widthA, width);
            const svbool_t tailHi16 = svwhilelt_b16(widthA + HA, width);
            const svuint8_t _value = svdup_n_u8(value);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    ConditionalSrc<compareType>(src + col, mask + col, body8, _value, buffer.v + col, body16, body16);
                if (widthA < width)
                    ConditionalSrc<compareType>(src + col, mask + col, tail8, _value, buffer.v + col, tailLo16, tailHi16);

                for (col = 0; col < alignedWidth4; col += 4)
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

        void HistogramMasked(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t index, uint32_t* histogram)
        {
            HistogramConditional<SimdCompareEqual>(src, srcStride, width, height, mask, maskStride, index, histogram);
        }

        void HistogramConditional(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            const uint8_t* mask, size_t maskStride, uint8_t value, SimdCompareType compareType, uint32_t* histogram)
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
#endif
}

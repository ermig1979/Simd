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

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t width)
                {
                    _p = Allocate(sizeof(uint16_t) * 3 * width);
                    src0 = (uint16_t*)_p;
                    src1 = src0 + width;
                    src2 = src1 + width;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint16_t* src0;
                uint16_t* src1;
                uint16_t* src2;
            private:
                void* _p;
            };
        }

        SIMD_INLINE svuint16_t BinomialSumLo(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpklo_u16(left), svlsl_n_u16_x(mask, svunpklo_u16(center), 1)), svunpklo_u16(right));
        }

        SIMD_INLINE svuint16_t BinomialSumHi(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right)
        {
            const svbool_t mask = svptrue_b16();
            return svadd_u16_x(mask, svadd_u16_x(mask, svunpkhi_u16(left), svlsl_n_u16_x(mask, svunpkhi_u16(center), 1)), svunpkhi_u16(right));
        }

        SIMD_INLINE void BlurRow(const uint8_t* src, size_t step, uint16_t* dst, size_t half,
            const svbool_t& mask8, const svbool_t& maskLo, const svbool_t& maskHi)
        {
            svuint8_t left = svld1_u8(mask8, src - step);
            svuint8_t center = svld1_u8(mask8, src);
            svuint8_t right = svld1_u8(mask8, src + step);
            svst1_u16(maskLo, dst, BinomialSumLo(left, center, right));
            svst1_u16(maskHi, dst + half, BinomialSumHi(left, center, right));
        }

        template <size_t step> void BlurRow(const uint8_t* src, size_t width, uint16_t* dst)
        {
            size_t size = width * step;
            if (width == 1)
            {
                for (size_t col = 0; col < size; ++col)
                    dst[col] = uint16_t(src[col]) << 2;
                return;
            }

            for (size_t col = 0; col < step; ++col)
                dst[col] = uint16_t(src[col]) * 3 + src[col + step];

            const size_t A = svcntb(), HA = svcnth(), end = size - step;
            const svbool_t body8 = svptrue_b8(), body16 = svptrue_b16();
            size_t col = step;
            for (; col + A <= end; col += A)
                BlurRow(src + col, step, dst + col, HA, body8, body16, body16);
            if (col < end)
                BlurRow(src + col, step, dst + col, HA, svwhilelt_b8(col, end), svwhilelt_b16(col, end), svwhilelt_b16(col + HA, end));

            for (size_t col = end; col < size; ++col)
                dst[col] = src[col - step] + uint16_t(src[col]) * 3;
        }

        SIMD_INLINE svuint16_t DivideBy16(const svuint16_t& value)
        {
            const svbool_t mask = svptrue_b16();
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 8), 4);
        }

        SIMD_INLINE svuint16_t BlurCol(const uint16_t* src0, const uint16_t* src1, const uint16_t* src2, const svbool_t& mask)
        {
            return DivideBy16(svadd_u16_x(mask, svadd_u16_x(mask, svld1_u16(mask, src0), svlsl_n_u16_x(mask, svld1_u16(mask, src1), 1)), svld1_u16(mask, src2)));
        }

        SIMD_INLINE svuint8_t PackU16ToU8(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svqxtnb_u16(lo), svqxtnb_u16(hi));
        }

        SIMD_INLINE void BlurCol(const Buffer& buffer, size_t offset, size_t half, uint8_t* dst,
            const svbool_t& mask8, const svbool_t& maskLo, const svbool_t& maskHi)
        {
            svuint16_t lo = BlurCol(buffer.src0 + offset, buffer.src1 + offset, buffer.src2 + offset, maskLo);
            svuint16_t hi = BlurCol(buffer.src0 + offset + half, buffer.src1 + offset + half, buffer.src2 + offset + half, maskHi);
            svst1_u8(mask8, dst + offset, PackU16ToU8(lo, hi));
        }

        SIMD_INLINE void BlurCols(const Buffer& buffer, size_t size, uint8_t* dst)
        {
            const size_t A = svcntb(), HA = svcnth();
            const svbool_t body8 = svptrue_b8(), body16 = svptrue_b16();
            size_t col = 0;
            for (; col + A <= size; col += A)
                BlurCol(buffer, col, HA, dst, body8, body16, body16);
            if (col < size)
                BlurCol(buffer, col, HA, dst, svwhilelt_b8(col, size), svwhilelt_b16(col, size), svwhilelt_b16(col + HA, size));
        }

        template <size_t step> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * step;
            Buffer buffer(size);

            BlurRow<step>(src, width, buffer.src0);
            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t) * size);

            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* src2 = src + srcStride * (row + 1 < height ? row + 1 : height - 1);
                BlurRow<step>(src2, width, buffer.src2);

                BlurCols(buffer, size, dst);

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
                dst += dstStride;
            }
        }

        void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}

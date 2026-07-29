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

        template <size_t step> SIMD_INLINE svuint8_t LoadBeforeFirst(const svuint8_t& first)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t iota = svindex_u8(0, 1);
            svuint8_t idx = svsel_u8(svcmplt_n_u8(mask, iota, step), iota, svsub_n_u8_x(mask, iota, step));
            return svtbl_u8(first, idx);
        }

        template <size_t step> SIMD_INLINE svuint8_t LoadAfterLast(const svuint8_t& last)
        {
            const svbool_t mask = svptrue_b8();
            const size_t A = svcntb();
            svuint8_t iota = svindex_u8(0, 1);
            svuint8_t idx = svsel_u8(svcmplt_n_u8(mask, iota, A - step), svadd_n_u8_x(mask, iota, step), iota);
            return svtbl_u8(last, idx);
        }

        SIMD_INLINE svuint16_t BinomialSum16(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c, const svbool_t& mask)
        {
            return svadd_u16_x(mask, svadd_u16_x(mask, a, c), svadd_u16_x(mask, b, b));
        }

        SIMD_INLINE void BlurCol(const svuint8_t& left, const svuint8_t& center, const svuint8_t& right, uint16_t* dst)
        {
            const svbool_t mask = svptrue_b16();
            const size_t HA = svcnth();
            svst1_u16(mask, dst + 0, BinomialSum16(svunpklo_u16(left), svunpklo_u16(center), svunpklo_u16(right), mask));
            svst1_u16(mask, dst + HA, BinomialSum16(svunpkhi_u16(left), svunpkhi_u16(center), svunpkhi_u16(right), mask));
        }

        template <size_t step> SIMD_INLINE void BlurColNose(const uint8_t* p, uint16_t* dst)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t center = svld1_u8(mask, p);
            BlurCol(LoadBeforeFirst<step>(center), center, svld1_u8(mask, p + step), dst);
        }

        template <size_t step> SIMD_INLINE void BlurColBody(const uint8_t* p, uint16_t* dst)
        {
            const svbool_t mask = svptrue_b8();
            BlurCol(svld1_u8(mask, p - step), svld1_u8(mask, p), svld1_u8(mask, p + step), dst);
        }

        template <size_t step> SIMD_INLINE void BlurColTail(const uint8_t* p, uint16_t* dst)
        {
            const svbool_t mask = svptrue_b8();
            svuint8_t center = svld1_u8(mask, p);
            BlurCol(svld1_u8(mask, p - step), center, LoadAfterLast<step>(center), dst);
        }

        SIMD_INLINE svuint16_t DivideBy16(const svuint16_t& value, const svbool_t& mask)
        {
            return svlsr_n_u16_x(mask, svadd_n_u16_x(mask, value, 8), 4);
        }

        SIMD_INLINE void BlurRow(const Buffer& buffer, size_t offset, uint8_t* dst)
        {
            const svbool_t mask = svptrue_b16();
            const size_t HA = svcnth();
            svuint16_t lo = DivideBy16(BinomialSum16(
                svld1_u16(mask, buffer.src0 + offset),
                svld1_u16(mask, buffer.src1 + offset),
                svld1_u16(mask, buffer.src2 + offset), mask), mask);
            svuint16_t hi = DivideBy16(BinomialSum16(
                svld1_u16(mask, buffer.src0 + offset + HA),
                svld1_u16(mask, buffer.src1 + offset + HA),
                svld1_u16(mask, buffer.src2 + offset + HA), mask), mask);
            svst1b_u16(mask, dst + offset, lo);
            svst1b_u16(mask, dst + offset + HA, hi);
        }

        template <size_t step> void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            const size_t A = svcntb();
            assert(step * (width - 1) >= A);

            size_t size = step * width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));

            BlurColNose<step>(src + 0, buffer.src0 + 0);
            for (size_t col = A; col < bodySize; col += A)
                BlurColBody<step>(src + col, buffer.src0 + col);
            BlurColTail<step>(src + size - A, buffer.src0 + size - A);

            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t) * size);

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t* src2 = src + srcStride * (row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride * (height - 1);

                BlurColNose<step>(src2 + 0, buffer.src2 + 0);
                for (size_t col = A; col < bodySize; col += A)
                    BlurColBody<step>(src2 + col, buffer.src2 + col);
                BlurColTail<step>(src2 + size - A, buffer.src2 + size - A);

                for (size_t col = 0; col < bodySize; col += A)
                    BlurRow(buffer, col, dst);
                BlurRow(buffer, size - A, dst);

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <size_t step> void BlurRowSmall(const uint8_t* src, size_t width, uint16_t* dst)
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
            for (size_t col = step; col < size - step; ++col)
                dst[col] = src[col - step] + (uint16_t(src[col]) << 1) + src[col + step];
            for (size_t col = size - step; col < size; ++col)
                dst[col] = src[col - step] + uint16_t(src[col]) * 3;
        }

        template <size_t step> void GaussianBlur3x3Small(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            size_t size = width * step;
            Buffer buffer(size);

            BlurRowSmall<step>(src, width, buffer.src0);
            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t) * size);

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t* src2 = src + srcStride * (row + 1 < height ? row + 1 : height - 1);
                BlurRowSmall<step>(src2, width, buffer.src2);
                for (size_t col = 0; col < size; ++col)
                    dst[col] = uint8_t((buffer.src0[col] + 2 * buffer.src1[col] + buffer.src2[col] + 8) >> 4);
                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <size_t step> void GaussianBlur3x3Auto(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            if (step * (width - 1) >= svcntb())
                GaussianBlur3x3<step>(src, srcStride, width, height, dst, dstStride);
            else
                GaussianBlur3x3Small<step>(src, srcStride, width, height, dst, dstStride);
        }

        void GaussianBlur3x3(const uint8_t* src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3Auto<1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3Auto<2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3Auto<3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3Auto<4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }
    }
#endif
}

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
#include "Simd/SimdLoadBlock.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
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

                uint16_t * src0;
                uint16_t * src1;
                uint16_t * src2;
            private:
                void * _p;
            };
        }

        SIMD_INLINE v128_u16 DivideBy16(v128_u16 value)
        {
            return vec_sr(vec_add(value, K16_0008), K16_0004);
        }

        template<bool align> SIMD_INLINE void BlurCol(v128_u8 a[3], uint16_t * b)
        {
            Store<align>(b, BinomialSum(UnpackLoU8(a[0]), UnpackLoU8(a[1]), UnpackLoU8(a[2])));
            Store<align>(b + HA, BinomialSum(UnpackHiU8(a[0]), UnpackHiU8(a[1]), UnpackHiU8(a[2])));
        }

        template<bool align> SIMD_INLINE v128_u8 BlurRow(const Buffer & buffer, size_t offset)
        {
            v128_u16 lo = DivideBy16(BinomialSum(Load<align>(buffer.src0 + offset), Load<align>(buffer.src1 + offset), Load<align>(buffer.src2 + offset)));
            offset += HA;
            v128_u16 hi = DivideBy16(BinomialSum(Load<align>(buffer.src0 + offset), Load<align>(buffer.src1 + offset), Load<align>(buffer.src2 + offset)));
            return vec_pack(lo, hi);
        }

        template <bool align, size_t step> void GaussianBlur3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

            v128_u8 a[3];

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));

            LoadNose3<align, step>(src + 0, a);
            BlurCol<true>(a, buffer.src0 + 0);
            for (size_t col = A; col < bodySize; col += A)
            {
                LoadBody3<align, step>(src + col, a);
                BlurCol<true>(a, buffer.src0 + col);
            }
            LoadTail3<align, step>(src + size - A, a);
            BlurCol<align>(a, buffer.src0 + size - A);

            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*size);

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride*(height - 1);

                LoadNose3<align, step>(src2 + 0, a);
                BlurCol<true>(a, buffer.src2 + 0);
                for (size_t col = A; col < bodySize; col += A)
                {
                    LoadBody3<align, step>(src2 + col, a);
                    BlurCol<true>(a, buffer.src2 + col);
                }
                LoadTail3<align, step>(src2 + size - A, a);
                BlurCol<align>(a, buffer.src2 + size - A);

                Storer<align> _dst(dst);
                _dst.First(BlurRow<true>(buffer, 0));
                for (size_t col = A; col < bodySize; col += A)
                    _dst.Next(BlurRow<true>(buffer, col));
                Flush(_dst);
                Store<align>(dst + size - A, BlurRow<align>(buffer, size - A));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <bool align> void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: GaussianBlur3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: GaussianBlur3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: GaussianBlur3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: GaussianBlur3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(channelCount*width) && Aligned(dst) && Aligned(dstStride))
                GaussianBlur3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                GaussianBlur3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}

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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
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

        template<int part> SIMD_INLINE __m512i SumCol(__m512i a[3])
        {
            return _mm512_add_epi16(_mm512_maddubs_epi16(UnpackU8<part>(a[0], a[1]), K8_01), UnpackU8<part>(a[2]));
        }

        template<bool align> SIMD_INLINE void SumCol(__m512i a[3], uint16_t * b)
        {
            Store<align>(b + 00, SumCol<0>(a));
            Store<align>(b + HA, SumCol<1>(a));
        }

        template <bool align, size_t step> void SumCol(const uint8_t * src, size_t aligned, size_t full, uint16_t * dst)
        {
            __m512i a[3];
            LoadNose3<align, step>(src, a);
            SumCol<true>(a, dst);
            for (size_t col = A; col < aligned; col += A)
            {
                LoadBody3<align, step>(src + col, a);
                SumCol<true>(a, dst + col);
            }
            LoadTail3<false, step>(src + full - A, a);
            SumCol<true>(a, dst + aligned);
        }

        template<bool align> SIMD_INLINE __m512i AverageRow16(const Buffer & buffer, size_t offset)
        {
            return _mm512_mulhi_epu16(K16_DIVISION_BY_9_FACTOR, _mm512_add_epi16(
                _mm512_add_epi16(K16_0005, Load<align>(buffer.src0 + offset)),
                _mm512_add_epi16(Load<align>(buffer.src1 + offset), Load<align>(buffer.src2 + offset))));
        }

        template<bool align> SIMD_INLINE __m512i AverageRow(const Buffer & buffer, size_t offset)
        {
            return _mm512_packus_epi16(AverageRow16<align>(buffer, offset), AverageRow16<align>(buffer, offset + HA));
        }

        template <bool align, size_t step> void MeanFilter3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*(width - 1) >= A);
            if (align)
                assert(Aligned(src) && Aligned(srcStride) && Aligned(step*width) && Aligned(dst) && Aligned(dstStride));

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));

            SumCol<align, step>(src, bodySize, size, buffer.src0);
            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*(bodySize + A));

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride*(height - 1);

                SumCol<align, step>(src2, bodySize, size, buffer.src2);

                for (size_t col = 0; col < bodySize; col += A)
                    Store<align>(dst + col, AverageRow<true>(buffer, col));
                Store<align>(dst + size - A, AverageRow<true>(buffer, bodySize));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        template <bool align> void MeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount > 0 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: MeanFilter3x3<align, 1>(src, srcStride, width, height, dst, dstStride); break;
            case 2: MeanFilter3x3<align, 2>(src, srcStride, width, height, dst, dstStride); break;
            case 3: MeanFilter3x3<align, 3>(src, srcStride, width, height, dst, dstStride); break;
            case 4: MeanFilter3x3<align, 4>(src, srcStride, width, height, dst, dstStride); break;
            }
        }

        void MeanFilter3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(channelCount*width) && Aligned(dst) && Aligned(dstStride))
                MeanFilter3x3<true>(src, srcStride, width, height, channelCount, dst, dstStride);
            else
                MeanFilter3x3<false>(src, srcStride, width, height, channelCount, dst, dstStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

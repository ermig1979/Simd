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
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
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

        const __m128i K8_01_02 = SIMD_MM_SET2_EPI8(0x01, 0x02);

        template<int part> SIMD_INLINE __m128i BinomialSumUnpackedU8(__m128i a[3])
        {
            return _mm_add_epi16(_mm_maddubs_epi16(UnpackU8<part>(a[0], a[1]), K8_01_02), UnpackU8<part>(a[2]));
        }

        SIMD_INLINE void BlurCol(__m128i a[3], uint16_t * b)
        {
            _mm_storeu_si128((__m128i*)b + 0, BinomialSumUnpackedU8<0>(a));
            _mm_storeu_si128((__m128i*)b + 1, BinomialSumUnpackedU8<1>(a));
        }

        SIMD_INLINE __m128i BlurRow16(const Buffer & buffer, size_t offset)
        {
            return DivideBy16(BinomialSum16(
                _mm_loadu_si128((__m128i*)(buffer.src0 + offset)),
                _mm_loadu_si128((__m128i*)(buffer.src1 + offset)),
                _mm_loadu_si128((__m128i*)(buffer.src2 + offset))));
        }

        SIMD_INLINE __m128i BlurRow(const Buffer & buffer, size_t offset)
        {
            return _mm_packus_epi16(BlurRow16(buffer, offset), BlurRow16(buffer, offset + HA));
        }

        template <size_t step> void GaussianBlur3x3(
            const uint8_t * src, size_t srcStride, size_t width, size_t height, uint8_t * dst, size_t dstStride)
        {
            assert(step*width >= A);

            __m128i a[3];

            size_t size = step*width;
            size_t bodySize = Simd::AlignHi(size, A) - A;

            Buffer buffer(Simd::AlignHi(size, A));

            a[1] = _mm_loadu_si128((__m128i*)(src + 0));
            a[0] = LoadBeforeFirst<step>(a[1]);
            a[2] = _mm_loadu_si128((__m128i*)(src + step));
            BlurCol(a, buffer.src0 + 0);
            for (size_t col = A; col < bodySize; col += A)
            {
                a[0] = _mm_loadu_si128((__m128i*)(src + col - step));
                a[1] = _mm_loadu_si128((__m128i*)(src + col));
                a[2] = _mm_loadu_si128((__m128i*)(src + col + step));
                BlurCol(a, buffer.src0 + col);
            }
            a[0] = _mm_loadu_si128((__m128i*)(src + size - A - step));
            a[1] = _mm_loadu_si128((__m128i*)(src + size - A));
            a[2] = LoadAfterLast<step>(a[1]);
            BlurCol(a, buffer.src0 + size - A);

            memcpy(buffer.src1, buffer.src0, sizeof(uint16_t)*size);

            for (size_t row = 0; row < height; ++row, dst += dstStride)
            {
                const uint8_t *src2 = src + srcStride*(row + 1);
                if (row >= height - 2)
                    src2 = src + srcStride*(height - 1);

                a[1] = _mm_loadu_si128((__m128i*)(src2 + 0));
                a[0] = LoadBeforeFirst<step>(a[1]);
                a[2] = _mm_loadu_si128((__m128i*)(src2 + step));
                BlurCol(a, buffer.src2 + 0);
                for (size_t col = A; col < bodySize; col += A)
                {
                    a[0] = _mm_loadu_si128((__m128i*)(src2 + col - step));
                    a[1] = _mm_loadu_si128((__m128i*)(src2 + col));
                    a[2] = _mm_loadu_si128((__m128i*)(src2 + col + step));
                    BlurCol(a, buffer.src2 + col);
                }
                a[0] = _mm_loadu_si128((__m128i*)(src2 + size - A - step));
                a[1] = _mm_loadu_si128((__m128i*)(src2 + size - A));
                a[2] = LoadAfterLast<step>(a[1]);
                BlurCol(a, buffer.src2 + size - A);

                for (size_t col = 0; col < bodySize; col += A)
                    _mm_storeu_si128((__m128i*)(dst + col), BlurRow(buffer, col));
                _mm_storeu_si128((__m128i*)(dst + size - A), BlurRow(buffer, size - A));

                Swap(buffer.src0, buffer.src2);
                Swap(buffer.src0, buffer.src1);
            }
        }

        void GaussianBlur3x3(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            size_t channelCount, uint8_t * dst, size_t dstStride)
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

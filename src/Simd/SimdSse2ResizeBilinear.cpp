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
#include "Simd/SimdBase.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE
    namespace Sse2
    {
        namespace
        {
            struct Buffer
            {
                Buffer(size_t size, size_t width, size_t height)
                {
                    _p = Allocate(2 * size + sizeof(int16_t) * 2 * width + sizeof(int)*(2 * height + width));
                    bx[0] = (uint8_t*)_p;
                    bx[1] = bx[0] + size;
                    ax = (int16_t*)(bx[1] + size);
                    ix = (int*)(ax + 2 * width);
                    iy = ix + width;
                    ay = iy + height;
                }

                ~Buffer()
                {
                    Free(_p);
                }

                uint8_t * bx[2];
                int16_t * ax;
                int * ix;
                int * ay;
                int * iy;
            private:
                void *_p;
            };
        }

        void EstimateAlphaIndexX(size_t srcSize, size_t dstSize, int * indexes, int16_t * alphas)
        {
            float scale = (float)srcSize / dstSize;

            for (size_t i = 0; i < dstSize; ++i)
            {
                float alpha = (float)((i + 0.5)*scale - 0.5);
                ptrdiff_t index = (ptrdiff_t)::floor(alpha);
                alpha -= index;

                if (index < 0)
                {
                    index = 0;
                    alpha = 0;
                }

                if (index > (ptrdiff_t)srcSize - 2)
                {
                    index = srcSize - 2;
                    alpha = 1;
                }

                indexes[i] = (int)index;
                alphas[1] = (int16_t)(alpha * Base::FRACTION_RANGE + 0.5);
                alphas[0] = (int16_t)(Base::FRACTION_RANGE - alphas[1]);
                alphas += 2;
            }
        }

        template <size_t channelCount> void InterpolateX(const __m128i * alpha, __m128i * buffer);

        SIMD_INLINE void InterpolateX1(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i lo = _mm_madd_epi16(_mm_unpacklo_epi8(src, K_ZERO), _mm_load_si128(alpha + 0));
            __m128i hi = _mm_madd_epi16(_mm_unpackhi_epi8(src, K_ZERO), _mm_load_si128(alpha + 1));
            _mm_store_si128(buffer, _mm_packs_epi32(lo, hi));
        }

        template <> SIMD_INLINE void InterpolateX<1>(const __m128i * alpha, __m128i * buffer)
        {
            InterpolateX1(alpha + 0, buffer + 0);
            InterpolateX1(alpha + 2, buffer + 1);
        }

        SIMD_INLINE void InterpolateX2(const __m128i * alpha, __m128i * buffer)
        {
            __m128i src = _mm_load_si128(buffer);
            __m128i a = _mm_load_si128(alpha);
            __m128i u = _mm_madd_epi16(_mm_and_si128(src, K16_00FF), a);
            __m128i v = _mm_madd_epi16(_mm_and_si128(_mm_srli_si128(src, 1), K16_00FF), a);
            _mm_store_si128(buffer, _mm_or_si128(u, _mm_slli_si128(v, 2)));
        }

        template <> SIMD_INLINE void InterpolateX<2>(const __m128i * alpha, __m128i * buffer)
        {
            InterpolateX2(alpha + 0, buffer + 0);
            InterpolateX2(alpha + 1, buffer + 1);
        }

        const __m128i K16_FRACTION_ROUND_TERM = SIMD_MM_SET1_EPI16(Base::BILINEAR_ROUND_TERM);

        template<bool align> SIMD_INLINE __m128i InterpolateY(const __m128i * pbx0, const __m128i * pbx1, __m128i alpha[2])
        {
            __m128i sum = _mm_add_epi16(_mm_mullo_epi16(Load<align>(pbx0), alpha[0]), _mm_mullo_epi16(Load<align>(pbx1), alpha[1]));
            return _mm_srli_epi16(_mm_add_epi16(sum, K16_FRACTION_ROUND_TERM), Base::BILINEAR_SHIFT);
        }

        template<bool align> SIMD_INLINE void InterpolateY(const uint8_t * bx0, const uint8_t * bx1, __m128i alpha[2], uint8_t * dst)
        {
            __m128i lo = InterpolateY<align>((__m128i*)bx0 + 0, (__m128i*)bx1 + 0, alpha);
            __m128i hi = InterpolateY<align>((__m128i*)bx0 + 1, (__m128i*)bx1 + 1, alpha);
            Store<false>((__m128i*)dst, _mm_packus_epi16(lo, hi));
        }

        template <size_t channelCount> void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride)
        {
            assert(dstWidth >= A);

            struct One { uint8_t channels[channelCount]; };
            struct Two { uint8_t channels[channelCount * 2]; };

            size_t size = 2 * dstWidth*channelCount;
            size_t bufferSize = AlignHi(dstWidth, A)*channelCount * 2;
            size_t alignedSize = AlignHi(size, DA) - DA;
            Buffer buffer(bufferSize, dstWidth, dstHeight);

            const size_t stepB = A / channelCount;
            const size_t stepA = DA / channelCount;
            size_t bufferWidth = AlignHi(dstWidth, stepB);

            Base::EstimateAlphaIndex(srcHeight, dstHeight, buffer.iy, buffer.ay, 1);

            EstimateAlphaIndexX(srcWidth, dstWidth, buffer.ix, buffer.ax);

            ptrdiff_t previous = -2;

            __m128i a[2];

            for (size_t yDst = 0; yDst < dstHeight; yDst++, dst += dstStride)
            {
                a[0] = _mm_set1_epi16(int16_t(Base::FRACTION_RANGE - buffer.ay[yDst]));
                a[1] = _mm_set1_epi16(int16_t(buffer.ay[yDst]));

                ptrdiff_t sy = buffer.iy[yDst];
                int k = 0;

                if (sy == previous)
                    k = 2;
                else if (sy == previous + 1)
                {
                    Swap(buffer.bx[0], buffer.bx[1]);
                    k = 1;
                }

                previous = sy;

                for (; k < 2; k++)
                {
                    Two * pb = (Two *)buffer.bx[k];
                    const One * ps = (const One *)(src + (sy + k)*srcStride);
                    for (size_t x = 0; x < dstWidth; x++)
                        pb[x] = *(Two *)(ps + buffer.ix[x]);

                    for (size_t ib = 0, ia = 0; ib < bufferWidth; ib += stepB, ia += stepA)
                        InterpolateX<channelCount>((__m128i*)(buffer.ax + ia), (__m128i*)(pb + ib));
                }

                for (size_t ib = 0, id = 0; ib < alignedSize; ib += DA, id += A)
                    InterpolateY<true>(buffer.bx[0] + ib, buffer.bx[1] + ib, a, dst + id);
                size_t i = size - DA;
                InterpolateY<false>(buffer.bx[0] + i, buffer.bx[1] + i, a, dst + i / 2);
            }
        }

        void ResizeBilinear(
            const uint8_t *src, size_t srcWidth, size_t srcHeight, size_t srcStride,
            uint8_t *dst, size_t dstWidth, size_t dstHeight, size_t dstStride, size_t channelCount)
        {
            switch (channelCount)
            {
            case 1:
                ResizeBilinear<1>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
                break;
            case 2:
                ResizeBilinear<2>(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride);
                break;
            default:
                Base::ResizeBilinear(src, srcWidth, srcHeight, srcStride, dst, dstWidth, dstHeight, dstStride, channelCount);
            }
        }
    }
#endif//SIMD_SSE2_ENABLE
}


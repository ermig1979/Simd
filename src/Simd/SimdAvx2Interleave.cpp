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
#include "Simd/SimdConversion.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> SIMD_INLINE void InterleaveUv(const uint8_t * u, const uint8_t * v, uint8_t * uv)
        {
            __m256i _u = LoadPermuted<align>((__m256i*)u);
            __m256i _v = LoadPermuted<align>((__m256i*)v);
            Store<align>((__m256i*)uv + 0, UnpackU8<0>(_u, _v));
            Store<align>((__m256i*)uv + 1, UnpackU8<1>(_u, _v));
        }

        template <bool align> void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += DA)
                    InterleaveUv<align>(u + col, v + col, uv + offset);
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 2 * col;
                    InterleaveUv<false>(u + col, v + col, uv + offset);
                }
                u += uStride;
                v += vStride;
                uv += uvStride;
            }
        }

        void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                InterleaveUv<true>(u, uStride, v, vStride, width, height, uv, uvStride);
            else
                InterleaveUv<false>(u, uStride, v, vStride, width, height, uv, uvStride);
        }

        template <bool align> SIMD_INLINE void InterleaveBgr(const uint8_t * b, const uint8_t * g, const uint8_t * r, size_t offset, uint8_t * bgr)
        {
            __m256i _b = Load<align>((__m256i*)(b + offset));
            __m256i _g = Load<align>((__m256i*)(g + offset));
            __m256i _r = Load<align>((__m256i*)(r + offset));
            Store<align>((__m256i*)bgr + 0, InterleaveBgr<0>(_b, _g, _r));
            Store<align>((__m256i*)bgr + 1, InterleaveBgr<1>(_b, _g, _r));
            Store<align>((__m256i*)bgr + 2, InterleaveBgr<2>(_b, _g, _r));
        }

        template <bool align> void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride));
                assert(Aligned(r) && Aligned(rStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            size_t A3 = A * 3;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += A3)
                    InterleaveBgr<align>(b, g, r, col, bgr + offset);
                if (tail)
                    InterleaveBgr<false>(b, g, r, width - A, bgr + 3 * (width - A));
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }

        void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride)
                && Aligned(r) && Aligned(rStride) && Aligned(bgr) && Aligned(bgrStride))
                InterleaveBgr<true>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
            else
                InterleaveBgr<false>(b, bStride, g, gStride, r, rStride, width, height, bgr, bgrStride);
        }

        template <bool align> SIMD_INLINE void InterleaveBgra(const uint8_t * b, const uint8_t * g, const uint8_t * r, const uint8_t * a, size_t offset, uint8_t * bgra)
        {
            __m256i _b = Load<align>((__m256i*)(b + offset));
            __m256i _g = Load<align>((__m256i*)(g + offset));
            __m256i _r = Load<align>((__m256i*)(r + offset));
            __m256i _a = Load<align>((__m256i*)(a + offset));
            __m256i bg0 = PermutedUnpackLoU8(_b, _g);
            __m256i bg1 = PermutedUnpackHiU8(_b, _g);
            __m256i ra0 = PermutedUnpackLoU8(_r, _a);
            __m256i ra1 = PermutedUnpackHiU8(_r, _a);
            Store<align>((__m256i*)bgra + 0, UnpackU16<0>(bg0, ra0));
            Store<align>((__m256i*)bgra + 1, UnpackU16<0>(bg1, ra1));
            Store<align>((__m256i*)bgra + 2, UnpackU16<1>(bg0, ra0));
            Store<align>((__m256i*)bgra + 3, UnpackU16<1>(bg1, ra1));
        }

        template <bool align> void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride));
                assert(Aligned(r) && Aligned(rStride) && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < alignedWidth; col += A, offset += QA)
                    InterleaveBgra<align>(b, g, r, a, col, bgra + offset);
                if (tail)
                    InterleaveBgra<false>(b, g, r, a, width - A, bgra + 4 * (width - A));
                b += bStride;
                g += gStride;
                r += rStride;
                a += aStride;
                bgra += bgraStride;
            }
        }

        void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            if (Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride)
                && Aligned(r) && Aligned(rStride) && Aligned(bgra) && Aligned(bgraStride))
                InterleaveBgra<true>(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
            else
                InterleaveBgra<false>(b, bStride, g, gStride, r, rStride, a, aStride, width, height, bgra, bgraStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

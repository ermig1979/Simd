/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
        SIMD_INLINE __m256i DeinterleavedU(__m256i uv0, __m256i uv1)
        {
            return PackU16ToU8(_mm256_and_si256(uv0, K16_00FF), _mm256_and_si256(uv1, K16_00FF));
        }

        SIMD_INLINE __m256i DeinterleavedV(__m256i uv0, __m256i uv1)
        {
            return DeinterleavedU(_mm256_srli_si256(uv0, 1), _mm256_srli_si256(uv1, 1));
        }

        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, 
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if(align)
            {
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0, offset = 0; col < bodyWidth; col += A, offset += DA)
                {
                    __m256i uv0 = Load<align>((__m256i*)(uv + offset));
                    __m256i uv1 = Load<align>((__m256i*)(uv + offset + A));
                    Store<align>((__m256i*)(u + col), DeinterleavedU(uv0, uv1));
                    Store<align>((__m256i*)(v + col), DeinterleavedV(uv0, uv1));
                }
                if(tail)
                {
                    size_t col = width - A;
                    size_t offset = 2*col;
                    __m256i uv0 = Load<false>((__m256i*)(uv + offset));
                    __m256i uv1 = Load<false>((__m256i*)(uv + offset + A));
                    Store<false>((__m256i*)(u + col), DeinterleavedU(uv0, uv1));
                    Store<false>((__m256i*)(v + col), DeinterleavedV(uv0, uv1));
                }
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, 
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                DeinterleaveUv<true>(uv, uvStride, width, height, u, uStride, v, vStride);
            else
                DeinterleaveUv<false>(uv, uvStride, width, height, u, uStride, v, vStride);
        }

        template <bool align> SIMD_INLINE void DeinterleaveBgr(const uint8_t * bgr, uint8_t * b, uint8_t * g, uint8_t * r, size_t offset)
        {
            __m256i _bgr[3] = { Load<align>((__m256i*)bgr + 0), Load<align>((__m256i*)bgr + 1), Load<align>((__m256i*)bgr + 2) };
            Store<align>((__m256i*)(b + offset), BgrToBlue(_bgr));
            Store<align>((__m256i*)(g + offset), BgrToGreen(_bgr));
            Store<align>((__m256i*)(r + offset), BgrToRed(_bgr));
        }

        template <bool align> void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            assert(width >= A);
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride));

            size_t alignedWidth = AlignLo(width, A);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < alignedWidth; col += A)
                    DeinterleaveBgr<align>(bgr + col * 3, b, g, r, col);
                if (width != alignedWidth)
                    DeinterleaveBgr<false>(bgr + 3 * (width - A), b, g, r, width - A);
                bgr += bgrStride;
                b += bStride;
                g += gStride;
                r += rStride;
            }
        }

        void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            if (Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride))
                DeinterleaveBgr<true>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
            else
                DeinterleaveBgr<false>(bgr, bgrStride, width, height, b, bStride, g, gStride, r, rStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}
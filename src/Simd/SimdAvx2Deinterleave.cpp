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
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
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
                {
                    __m256i uv0 = Deinterleave8To64(Load<align>((__m256i*)(uv + offset)));
                    __m256i uv1 = Deinterleave8To64(Load<align>((__m256i*)(uv + offset + A)));
                    Store<align>((__m256i*)(u + col), Deinterleave64<0>(uv0, uv1));
                    Store<align>((__m256i*)(v + col), Deinterleave64<1>(uv0, uv1));
                }
                if (tail)
                {
                    size_t col = width - A;
                    size_t offset = 2 * col;
                    __m256i uv0 = Deinterleave8To64(Load<false>((__m256i*)(uv + offset)));
                    __m256i uv1 = Deinterleave8To64(Load<false>((__m256i*)(uv + offset + A)));
                    Store<false>((__m256i*)(u + col), Deinterleave64<0>(uv0, uv1));
                    Store<false>((__m256i*)(v + col), Deinterleave64<1>(uv0, uv1));
                }
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                DeinterleaveUv<true>(uv, uvStride, width, height, u, uStride, v, vStride);
            else
                DeinterleaveUv<false>(uv, uvStride, width, height, u, uStride, v, vStride);
        }

        //---------------------------------------------------------------------

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

        //---------------------------------------------------------------------

        const __m256i K8_SHUFFLE_BGRA = SIMD_MM256_SETR_EPI8(
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);

        const __m256i K32_PERMUTE_BGRA = SIMD_MM256_SETR_EPI32(0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7);

        template <bool align, bool alpha> SIMD_INLINE void DeinterleaveBgra(const uint8_t * bgra, uint8_t * b, uint8_t * g, uint8_t * r, uint8_t *a, size_t offset)
        {
            __m256i _bgra[4];
            _bgra[0] = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 0), K8_SHUFFLE_BGRA);
            _bgra[1] = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 1), K8_SHUFFLE_BGRA);
            _bgra[2] = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 2), K8_SHUFFLE_BGRA);
            _bgra[3] = _mm256_shuffle_epi8(Load<align>((__m256i*)bgra + 3), K8_SHUFFLE_BGRA);

            __m256i bbgg0 = _mm256_unpacklo_epi32(_bgra[0], _bgra[1]);
            __m256i bbgg1 = _mm256_unpacklo_epi32(_bgra[2], _bgra[3]);

            Store<align>((__m256i*)(b + offset), _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi64(bbgg0, bbgg1), K32_PERMUTE_BGRA));
            Store<align>((__m256i*)(g + offset), _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi64(bbgg0, bbgg1), K32_PERMUTE_BGRA));

            __m256i rraa0 = _mm256_unpackhi_epi32(_bgra[0], _bgra[1]);
            __m256i rraa1 = _mm256_unpackhi_epi32(_bgra[2], _bgra[3]);

            Store<align>((__m256i*)(r + offset), _mm256_permutevar8x32_epi32(_mm256_unpacklo_epi64(rraa0, rraa1), K32_PERMUTE_BGRA));
            if(alpha)
                Store<align>((__m256i*)(a + offset), _mm256_permutevar8x32_epi32(_mm256_unpackhi_epi64(rraa0, rraa1), K32_PERMUTE_BGRA));
        }

        template <bool align> void DeinterleaveBgra(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride, uint8_t* a, size_t aStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && (Aligned(aStride) || a == NULL));
            }

            size_t alignedWidth = AlignLo(width, A);

            if (a)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                        DeinterleaveBgra<align, true>(bgra + col * 4, b, g, r, a, col);
                    if (width != alignedWidth)
                        DeinterleaveBgra<false, true>(bgra + 4 * (width - A), b, g, r, a, width - A);
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                    a += aStride;
                }
            }
            else
            {
                for (size_t row = 0; row < height; ++row)
                {
                    for (size_t col = 0; col < alignedWidth; col += A)
                        DeinterleaveBgra<align, false>(bgra + col * 4, b, g, r, NULL, col);
                    if (width != alignedWidth)
                        DeinterleaveBgra<false, false>(bgra + 4 * (width - A), b, g, r, NULL, width - A);
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                }
            }
        }

        void DeinterleaveBgra(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride, uint8_t* a, size_t aStride)
        {
            if (Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride) &&
                Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && (Aligned(aStride) || a == NULL))
                DeinterleaveBgra<true>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
            else
                DeinterleaveBgra<false>(bgra, bgraStride, width, height, b, bStride, g, gStride, r, rStride, a, aStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

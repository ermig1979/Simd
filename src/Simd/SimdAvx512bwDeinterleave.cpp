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

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K8_SHUFFLE_DEINTERLEAVE_UV = SIMD_MM512_SETR_EPI8(
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF,
            0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE, 0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        const __m512i K64_PERMUTE_UV_U = SIMD_MM512_SETR_EPI64(0x0, 0x2, 0x4, 0x6, 0x8, 0xA, 0xC, 0xE);
        const __m512i K64_PERMUTE_UV_V = SIMD_MM512_SETR_EPI64(0x1, 0x3, 0x5, 0x7, 0x9, 0xB, 0xD, 0xF);

        template <bool align, bool mask> SIMD_INLINE void DeinterleaveUv(const uint8_t * uv, uint8_t * u, uint8_t * v, const __mmask64 * tailMasks)
        {
            const __m512i uv0 = Load<align, mask>(uv + 0, tailMasks[0]);
            const __m512i uv1 = Load<align, mask>(uv + A, tailMasks[1]);
            const __m512i shuffledUV0 = _mm512_shuffle_epi8(uv0, K8_SHUFFLE_DEINTERLEAVE_UV);
            const __m512i shuffledUV1 = _mm512_shuffle_epi8(uv1, K8_SHUFFLE_DEINTERLEAVE_UV);
            Store<align, mask>(u, _mm512_permutex2var_epi64(shuffledUV0, K64_PERMUTE_UV_U, shuffledUV1), tailMasks[2]);
            Store<align, mask>(v, _mm512_permutex2var_epi64(shuffledUV0, K64_PERMUTE_UV_V, shuffledUV1), tailMasks[2]);
        }

        template <bool align> SIMD_INLINE void DeinterleaveUv2(const uint8_t * uv, uint8_t * u, uint8_t * v)
        {
            const __m512i uv0 = _mm512_shuffle_epi8(Load<align>(uv + 0 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
            const __m512i uv1 = _mm512_shuffle_epi8(Load<align>(uv + 1 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
            Store<align>(u + 0, _mm512_permutex2var_epi64(uv0, K64_PERMUTE_UV_U, uv1));
            Store<align>(v + 0, _mm512_permutex2var_epi64(uv0, K64_PERMUTE_UV_V, uv1));
            const __m512i uv2 = _mm512_shuffle_epi8(Load<align>(uv + 2 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
            const __m512i uv3 = _mm512_shuffle_epi8(Load<align>(uv + 3 * A), K8_SHUFFLE_DEINTERLEAVE_UV);
            Store<align>(u + A, _mm512_permutex2var_epi64(uv2, K64_PERMUTE_UV_U, uv3));
            Store<align>(v + A, _mm512_permutex2var_epi64(uv2, K64_PERMUTE_UV_V, uv3));
        }

        template <bool align> void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (align)
                assert(Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride));

            size_t alignedWidth = AlignLo(width, A);
            size_t fullAlignedWidth = AlignLo(width, DA);
            __mmask64 tailMasks[3];
            for (size_t c = 0; c < 2; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 2 - A*c);
            tailMasks[2] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < fullAlignedWidth; col += DA)
                    DeinterleaveUv2<align>(uv + col * 2, u + col, v + col);
                for (; col < alignedWidth; col += A)
                    DeinterleaveUv<align, false>(uv + col * 2, u + col, v + col, tailMasks);
                if (col < width)
                    DeinterleaveUv<align, true>(uv + col * 2, u + col, v + col, tailMasks);
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

        const __m512i K8_SHUFFLE_DEINTERLEAVE_BGR = SIMD_MM512_SETR_EPI8(
            0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
            0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
            0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1,
            0x0, 0x3, 0x6, 0x9, 0x1, 0x4, 0x7, 0xA, 0x2, 0x5, 0x8, 0xB, -1, -1, -1, -1);

        const __m512i K32_PERMUTE_BGR_B0 = SIMD_MM512_SETR_EPI32(0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K32_PERMUTE_BGR_B1 = SIMD_MM512_SETR_EPI32(-1, -1, -1, -1, -1, -1, -1, -1, 0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C);
        const __m512i K32_PERMUTE_BGR_G0 = SIMD_MM512_SETR_EPI32(0x01, 0x05, 0x09, 0x0D, 0x11, 0x15, 0x19, 0x1D, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K32_PERMUTE_BGR_G1 = SIMD_MM512_SETR_EPI32(-1, -1, -1, -1, -1, -1, -1, -1, 0x01, 0x05, 0x09, 0x0D, 0x11, 0x15, 0x19, 0x1D);
        const __m512i K32_PERMUTE_BGR_R0 = SIMD_MM512_SETR_EPI32(0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m512i K32_PERMUTE_BGR_R1 = SIMD_MM512_SETR_EPI32(-1, -1, -1, -1, -1, -1, -1, -1, 0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E);

        template <bool align, bool mask> SIMD_INLINE void DeinterleaveBgr(const uint8_t * bgr, uint8_t * b, uint8_t * g, uint8_t * r, const __mmask64 * tailMasks)
        {
            const __m512i bgr0 = Load<align, mask>(bgr + 0 * A, tailMasks[0]);
            const __m512i bgr1 = Load<align, mask>(bgr + 1 * A, tailMasks[1]);
            const __m512i bgr2 = Load<align, mask>(bgr + 2 * A, tailMasks[2]);

            const __m512i sp0 = _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_0, bgr0), K8_SHUFFLE_DEINTERLEAVE_BGR);
            const __m512i sp1 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(bgr0, K32_PERMUTE_BGR_TO_BGRA_1, bgr1), K8_SHUFFLE_DEINTERLEAVE_BGR);
            const __m512i sp2 = _mm512_shuffle_epi8(_mm512_permutex2var_epi32(bgr1, K32_PERMUTE_BGR_TO_BGRA_2, bgr2), K8_SHUFFLE_DEINTERLEAVE_BGR);
            const __m512i sp3 = _mm512_shuffle_epi8(_mm512_permutexvar_epi32(K32_PERMUTE_BGR_TO_BGRA_3, bgr2), K8_SHUFFLE_DEINTERLEAVE_BGR);

            Store<align, mask>(b, _mm512_or_si512(_mm512_permutex2var_epi32(sp0, K32_PERMUTE_BGR_B0, sp1), _mm512_permutex2var_epi32(sp2, K32_PERMUTE_BGR_B1, sp3)), tailMasks[3]);
            Store<align, mask>(g, _mm512_or_si512(_mm512_permutex2var_epi32(sp0, K32_PERMUTE_BGR_G0, sp1), _mm512_permutex2var_epi32(sp2, K32_PERMUTE_BGR_G1, sp3)), tailMasks[3]);
            Store<align, mask>(r, _mm512_or_si512(_mm512_permutex2var_epi32(sp0, K32_PERMUTE_BGR_R0, sp1), _mm512_permutex2var_epi32(sp2, K32_PERMUTE_BGR_R1, sp3)), tailMasks[3]);
        }

        template <bool align> void DeinterleaveBgr(const uint8_t * bgr, size_t bgrStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride)
        {
            if (align)
                assert(Aligned(bgr) && Aligned(bgrStride) && Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride));

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            tailMasks[3] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    DeinterleaveBgr<align, false>(bgr + col * 3, b + col, g + col, r + col, tailMasks);
                if (col < width)
                    DeinterleaveBgr<align, true>(bgr + col * 3, b + col, g + col, r + col, tailMasks);
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

        const __m512i K8_SHUFFLE_BGRA = SIMD_MM512_SETR_EPI8(
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF,
            0x0, 0x4, 0x8, 0xC, 0x1, 0x5, 0x9, 0xD, 0x2, 0x6, 0xA, 0xE, 0x3, 0x7, 0xB, 0xF);

        const __m512i K32_PERMUTE_BGRA_BG = SIMD_MM512_SETR_EPI32(0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x01, 0x05, 0x09, 0x0D, 0x11, 0x15, 0x19, 0x1D);
        const __m512i K32_PERMUTE_BGRA_RA = SIMD_MM512_SETR_EPI32(0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E, 0x03, 0x07, 0x0B, 0x0F, 0x13, 0x17, 0x1B, 0x1F);

        template <bool align, bool mask, bool alpha> SIMD_INLINE void DeinterleaveBgra(const uint8_t * bgra, uint8_t * b, uint8_t * g, uint8_t * r, uint8_t * a, const __mmask64 * tailMasks)
        {
            const __m512i bgra0 = _mm512_shuffle_epi8((Load<align, mask>(bgra + 0 * A, tailMasks[0])), K8_SHUFFLE_BGRA);
            const __m512i bgra1 = _mm512_shuffle_epi8((Load<align, mask>(bgra + 1 * A, tailMasks[1])), K8_SHUFFLE_BGRA);
            const __m512i bgra2 = _mm512_shuffle_epi8((Load<align, mask>(bgra + 2 * A, tailMasks[2])), K8_SHUFFLE_BGRA);
            const __m512i bgra3 = _mm512_shuffle_epi8((Load<align, mask>(bgra + 3 * A, tailMasks[3])), K8_SHUFFLE_BGRA);

            const __m512i bg0 = _mm512_permutex2var_epi32(bgra0, K32_PERMUTE_BGRA_BG, bgra1);
            const __m512i ra0 = _mm512_permutex2var_epi32(bgra0, K32_PERMUTE_BGRA_RA, bgra1);
            const __m512i bg1 = _mm512_permutex2var_epi32(bgra2, K32_PERMUTE_BGRA_BG, bgra3);
            const __m512i ra1 = _mm512_permutex2var_epi32(bgra2, K32_PERMUTE_BGRA_RA, bgra3);

            Store<align, mask>(b, _mm512_shuffle_i64x2(bg0, bg1, 0x44), tailMasks[4]);
            Store<align, mask>(g, _mm512_shuffle_i64x2(bg0, bg1, 0xEE), tailMasks[4]);
            Store<align, mask>(r, _mm512_shuffle_i64x2(ra0, ra1, 0x44), tailMasks[4]);
            if(alpha)
                Store<align, mask>(a, _mm512_shuffle_i64x2(ra0, ra1, 0xEE), tailMasks[4]);
        }

        template <bool align> void DeinterleaveBgra(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t * b, size_t bStride, uint8_t * g, size_t gStride, uint8_t * r, size_t rStride, uint8_t * a, size_t aStride)
        {
            if (align)
            {
                assert(Aligned(bgra) && Aligned(bgraStride) && Aligned(b) && Aligned(bStride));
                assert(Aligned(g) && Aligned(gStride) && Aligned(r) && Aligned(rStride) && Aligned(a) && (a || Aligned(aStride)));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A*c);
            tailMasks[4] = TailMask64(width - alignedWidth);
            if (a)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < alignedWidth; col += A)
                        DeinterleaveBgra<align, false, true>(bgra + col * 4, b + col, g + col, r + col, a + col, tailMasks);
                    if (col < width)
                        DeinterleaveBgra<align, true, true>(bgra + col * 4, b + col, g + col, r + col, a + col, tailMasks);
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
                    size_t col = 0;
                    for (; col < alignedWidth; col += A)
                        DeinterleaveBgra<align, false, false>(bgra + col * 4, b + col, g + col, r + col, NULL, tailMasks);
                    if (col < width)
                        DeinterleaveBgra<align, true, false>(bgra + col * 4, b + col, g + col, r + col, NULL, tailMasks);
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
#endif// SIMD_AVX512BW_ENABLE
}

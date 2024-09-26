/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        const __m512i K8_UV_TO_U = SIMD_MM512_SETR_EPI8(
            0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,
            0x20, 0x22, 0x24, 0x26, 0x28, 0x2A, 0x2C, 0x2E, 0x30, 0x32, 0x34, 0x36, 0x38, 0x3A, 0x3C, 0x3E,
            0x40, 0x42, 0x44, 0x46, 0x48, 0x4A, 0x4C, 0x4E, 0x50, 0x52, 0x54, 0x56, 0x58, 0x5A, 0x5C, 0x5E,
            0x60, 0x62, 0x64, 0x66, 0x68, 0x6A, 0x6C, 0x6E, 0x70, 0x72, 0x74, 0x76, 0x78, 0x7A, 0x7C, 0x7E);

        const __m512i K8_UV_TO_V = SIMD_MM512_SETR_EPI8(
            0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F,
            0x21, 0x23, 0x25, 0x27, 0x29, 0x2B, 0x2D, 0x2F, 0x31, 0x33, 0x35, 0x37, 0x39, 0x3B, 0x3D, 0x3F,
            0x41, 0x43, 0x45, 0x47, 0x49, 0x4B, 0x4D, 0x4F, 0x51, 0x53, 0x55, 0x57, 0x59, 0x5B, 0x5D, 0x5F,
            0x61, 0x63, 0x65, 0x67, 0x69, 0x6B, 0x6D, 0x6F, 0x71, 0x73, 0x75, 0x77, 0x79, 0x7B, 0x7D, 0x7F);

        SIMD_INLINE void DeinterleaveUv(const uint8_t * uv, uint8_t * u, uint8_t * v)
        {
            const __m512i uv0 = _mm512_loadu_si512(uv + 0);
            const __m512i uv1 = _mm512_loadu_si512(uv + A);
            _mm512_storeu_si512(u, _mm512_permutex2var_epi8(uv0, K8_UV_TO_U, uv1));
            _mm512_storeu_si512(v, _mm512_permutex2var_epi8(uv0, K8_UV_TO_V, uv1));
        }

        SIMD_INLINE void DeinterleaveUv(const uint8_t* uv, uint8_t* u, uint8_t* v, __mmask64 tail)
        {
            const __m512i uv0 = _mm512_maskz_loadu_epi16(__mmask32(tail >> 00), (uint16_t*)(uv + 0));
            const __m512i uv1 = _mm512_maskz_loadu_epi16(__mmask32(tail >> 32), (uint16_t*)(uv + A));
            _mm512_mask_storeu_epi8(u, tail, _mm512_permutex2var_epi8(uv0, K8_UV_TO_U, uv1));
            _mm512_mask_storeu_epi8(v, tail, _mm512_permutex2var_epi8(uv0, K8_UV_TO_V, uv1));
        }

        void DeinterleaveUv(const uint8_t * uv, size_t uvStride, size_t width, size_t height, uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            size_t widthA = AlignLo(width, A);
            __mmask64 tail = TailMask64(width - widthA);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    DeinterleaveUv(uv + col * 2, u + col, v + col);
                if (col < width)
                    DeinterleaveUv(uv + col * 2, u + col, v + col, tail);
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        const __m512i K8_BGR_TO_BG0 = SIMD_MM512_SETR_EPI8(
            0x00, 0x03, 0x06, 0x09, 0x0C, 0x0F, 0x12, 0x15, 0x18, 0x1B, 0x1E, 0x21, 0x24, 0x27, 0x2A, 0x2D,
            0x30, 0x33, 0x36, 0x39, 0x3C, 0x3F, 0x42, 0x45, 0x48, 0x4B, 0x4E, 0x51, 0x54, 0x57, 0x5A, 0x5D,
            0x01, 0x04, 0x07, 0x0A, 0x0D, 0x10, 0x13, 0x16, 0x19, 0x1C, 0x1F, 0x22, 0x25, 0x28, 0x2B, 0x2E,
            0x31, 0x34, 0x37, 0x3A, 0x3D, 0x40, 0x43, 0x46, 0x49, 0x4C, 0x4F, 0x52, 0x55, 0x58, 0x5B, 0x5E);

        const __m512i K8_BGR_TO_BG1 = SIMD_MM512_SETR_EPI8(
            0x20, 0x23, 0x26, 0x29, 0x2C, 0x2F, 0x32, 0x35, 0x38, 0x3B, 0x3E, 0x41, 0x44, 0x47, 0x4A, 0x4D,
            0x50, 0x53, 0x56, 0x59, 0x5C, 0x5F, 0x62, 0x65, 0x68, 0x6B, 0x6E, 0x71, 0x74, 0x77, 0x7A, 0x7D,
            0x21, 0x24, 0x27, 0x2A, 0x2D, 0x30, 0x33, 0x36, 0x39, 0x3C, 0x3F, 0x42, 0x45, 0x48, 0x4B, 0x4E,
            0x51, 0x54, 0x57, 0x5A, 0x5D, 0x60, 0x63, 0x66, 0x69, 0x6C, 0x6F, 0x72, 0x75, 0x78, 0x7B, 0x7E);

        const __m512i K8_BGR_TO_R0 = SIMD_MM512_SETR_EPI8(
            0x02, 0x05, 0x08, 0x0B, 0x0E, 0x11, 0x14, 0x17, 0x1A, 0x1D, 0x20, 0x23, 0x26, 0x29, 0x2C, 0x2F,
            0x32, 0x35, 0x38, 0x3B, 0x3E, 0x41, 0x44, 0x47, 0x4A, 0x4D, 0x50, 0x53, 0x56, 0x59, 0x5C, 0x5F,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);

        const __m512i K8_BGR_TO_R1 = SIMD_MM512_SETR_EPI8(
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x22, 0x25, 0x28, 0x2B, 0x2E, 0x31, 0x34, 0x37, 0x3A, 0x3D, 0x40, 0x43, 0x46, 0x49, 0x4C, 0x4F,
            0x52, 0x55, 0x58, 0x5B, 0x5E, 0x61, 0x64, 0x67, 0x6A, 0x6D, 0x70, 0x73, 0x76, 0x79, 0x7C, 0x7F);

        SIMD_INLINE void DeinterleaveBgr(const uint8_t* bgr, __mmask64 tail0, __mmask64 tail1, __mmask64 tail2, uint8_t* b, uint8_t* g, uint8_t* r, __mmask64 tail)
        {
            const __m512i bgr0 = _mm512_maskz_loadu_epi8(tail0, bgr + 0 * A);
            const __m512i bgr1 = _mm512_maskz_loadu_epi8(tail1, bgr + 1 * A);
            const __m512i bgr2 = _mm512_maskz_loadu_epi8(tail2, bgr + 2 * A);
            const __m512i bg0 = _mm512_permutex2var_epi8(bgr0, K8_BGR_TO_BG0, bgr1);
            const __m512i bg1 = _mm512_permutex2var_epi8(bgr1, K8_BGR_TO_BG1, bgr2);
            _mm512_mask_storeu_epi8(b, tail, _mm512_shuffle_i64x2(bg0, bg1, 0x44));
            _mm512_mask_storeu_epi8(g, tail, _mm512_shuffle_i64x2(bg0, bg1, 0xEE));
            _mm512_mask_storeu_epi8(r, tail, _mm512_mask_blend_epi64(0xF0, _mm512_permutex2var_epi8(bgr0, K8_BGR_TO_R0, bgr1), _mm512_permutex2var_epi8(bgr1, K8_BGR_TO_R1, bgr2)));
        }

        void DeinterleaveBgr(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride)
        {
            size_t widthA = AlignLo(width, A);
            __mmask64 body = __mmask64(-1), tail = TailMask64(width - widthA);
            __mmask64 tail0 = TailMask64((width - widthA) * 3 - 0 * A);
            __mmask64 tail1 = TailMask64((width - widthA) * 3 - 1 * A);
            __mmask64 tail2 = TailMask64((width - widthA) * 3 - 2 * A);

            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < widthA; col += A)
                    DeinterleaveBgr(bgr + col * 3, body, body, body, b + col, g + col, r + col, body);
                if (col < width)
                    DeinterleaveBgr(bgr + col * 3, tail0, tail1, tail2, b + col, g + col, r + col, tail);
                bgr += bgrStride;
                b += bStride;
                g += gStride;
                r += rStride;
            }
        }

        //-------------------------------------------------------------------------------------------------

        const __m512i K8_BGRA_TO_BG = SIMD_MM512_SETR_EPI8(
            0x00, 0x04, 0x08, 0x0C, 0x10, 0x14, 0x18, 0x1C, 0x20, 0x24, 0x28, 0x2C, 0x30, 0x34, 0x38, 0x3C,
            0x40, 0x44, 0x48, 0x4C, 0x50, 0x54, 0x58, 0x5C, 0x60, 0x64, 0x68, 0x6C, 0x70, 0x74, 0x78, 0x7C,
            0x01, 0x05, 0x09, 0x0D, 0x11, 0x15, 0x19, 0x1D, 0x21, 0x25, 0x29, 0x2D, 0x31, 0x35, 0x39, 0x3D,
            0x41, 0x45, 0x49, 0x4D, 0x51, 0x55, 0x59, 0x5D, 0x61, 0x65, 0x69, 0x6D, 0x71, 0x75, 0x79, 0x7D);

        const __m512i K8_BGRA_TO_RA = SIMD_MM512_SETR_EPI8(
            0x02, 0x06, 0x0A, 0x0E, 0x12, 0x16, 0x1A, 0x1E, 0x22, 0x26, 0x2A, 0x2E, 0x32, 0x36, 0x3A, 0x3E,
            0x42, 0x46, 0x4A, 0x4E, 0x52, 0x56, 0x5A, 0x5E, 0x62, 0x66, 0x6A, 0x6E, 0x72, 0x76, 0x7A, 0x7E,
            0x03, 0x07, 0x0B, 0x0F, 0x13, 0x17, 0x1B, 0x1F, 0x23, 0x27, 0x2B, 0x2F, 0x33, 0x37, 0x3B, 0x3F,
            0x43, 0x47, 0x4B, 0x4F, 0x53, 0x57, 0x5B, 0x5F, 0x63, 0x67, 0x6B, 0x6F, 0x73, 0x77, 0x7B, 0x7F);

        template <bool alpha> SIMD_INLINE void DeinterleaveBgra(const uint8_t* bgra, __mmask64 t0, __mmask64 t1, __mmask64 t2, __mmask64 t3,
            uint8_t* b, uint8_t* g, uint8_t* r, uint8_t* a, const __mmask64 tail)
        {
            const __m512i bgra0 = _mm512_maskz_loadu_epi8(t0, bgra + 0 * A);
            const __m512i bgra1 = _mm512_maskz_loadu_epi8(t1, bgra + 1 * A);
            const __m512i bgra2 = _mm512_maskz_loadu_epi8(t2, bgra + 2 * A);
            const __m512i bgra3 = _mm512_maskz_loadu_epi8(t3, bgra + 3 * A);

            const __m512i bg01 = _mm512_permutex2var_epi8(bgra0, K8_BGRA_TO_BG, bgra1);
            const __m512i bg23 = _mm512_permutex2var_epi8(bgra2, K8_BGRA_TO_BG, bgra3);
            const __m512i ra01 = _mm512_permutex2var_epi8(bgra0, K8_BGRA_TO_RA, bgra1);
            const __m512i ra23 = _mm512_permutex2var_epi8(bgra2, K8_BGRA_TO_RA, bgra3);

            _mm512_mask_storeu_epi8(b, tail, _mm512_shuffle_i64x2(bg01, bg23, 0x44));
            _mm512_mask_storeu_epi8(g, tail, _mm512_shuffle_i64x2(bg01, bg23, 0xEE));
            _mm512_mask_storeu_epi8(r, tail, _mm512_shuffle_i64x2(ra01, ra23, 0x44));
            if (alpha)
                _mm512_mask_storeu_epi8(a, tail, _mm512_shuffle_i64x2(ra01, ra23, 0xEE));
        }

        void DeinterleaveBgra(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* b, size_t bStride, uint8_t* g, size_t gStride, uint8_t* r, size_t rStride, uint8_t* a, size_t aStride)
        {
            size_t widthA = AlignLo(width, A);
            __mmask64 body = __mmask64(-1), tail = TailMask64(width - widthA);
            __mmask64 tail0 = TailMask64((width - widthA) * 4 - 0 * A);
            __mmask64 tail1 = TailMask64((width - widthA) * 4 - 1 * A);
            __mmask64 tail2 = TailMask64((width - widthA) * 4 - 2 * A);
            __mmask64 tail3 = TailMask64((width - widthA) * 4 - 3 * A);
            if (a)
            {
                for (size_t row = 0; row < height; ++row)
                {
                    size_t col = 0;
                    for (; col < widthA; col += A)
                        DeinterleaveBgra<true>(bgra + col * 4, body, body, body, body, b + col, g + col, r + col, a + col, body);
                    if (col < width)
                        DeinterleaveBgra<true>(bgra + col * 4, tail0, tail1, tail2, tail3, b + col, g + col, r + col, a + col, tail);
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
                    for (; col < widthA; col += A)
                        DeinterleaveBgra<false>(bgra + col * 4, body, body, body, body, b + col, g + col, r + col, NULL, body);
                    if (col < width)
                        DeinterleaveBgra<false>(bgra + col * 4, tail0, tail1, tail2, tail3, b + col, g + col, r + col, NULL, tail);
                    bgra += bgraStride;
                    b += bStride;
                    g += gStride;
                    r += rStride;
                }
            }
        }
    }
#endif
}

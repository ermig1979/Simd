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
#include "Simd/SimdStore.h"
#include "Simd/SimdConversion.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE void InterleaveUv(const uint8_t * u, const uint8_t * v, uint8_t * uv, const __mmask64 * tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[2], u) : _mm512_loadu_si512(u));
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[2], v) : _mm512_loadu_si512(v));
            if (mask)
            {
                _mm512_mask_storeu_epi8(uv + 0, tails[0], UnpackU8<0>(_u, _v));
                _mm512_mask_storeu_epi8(uv + A, tails[1], UnpackU8<1>(_u, _v));
            }
            else
            {
                _mm512_storeu_si512(uv + 0, UnpackU8<0>(_u, _v));
                _mm512_storeu_si512(uv + A, UnpackU8<1>(_u, _v));
            }
        }

        SIMD_INLINE void InterleaveUv2(const uint8_t * u, const uint8_t * v, uint8_t * uv)
        {
            __m512i u0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512(u + 0));
            __m512i v0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512(v + 0));
            _mm512_storeu_si512(uv + 0 * A, UnpackU8<0>(u0, v0));
            _mm512_storeu_si512(uv + 1 * A, UnpackU8<1>(u0, v0));
            __m512i u1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512(u + A));
            __m512i v1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, _mm512_loadu_si512(v + A));
            _mm512_storeu_si512(uv + 2 * A, UnpackU8<0>(u1, v1));
            _mm512_storeu_si512(uv + 3 * A, UnpackU8<1>(u1, v1));
        }

        void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
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
                    InterleaveUv2(u + col, v + col, uv + col * 2);
                for (; col < alignedWidth; col += A)
                    InterleaveUv<false>(u + col, v + col, uv + col * 2, tailMasks);
                if (col < width)
                    InterleaveUv<true>(u + col, v + col, uv + col * 2, tailMasks);
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        template <bool mask> SIMD_INLINE void InterleaveBgr(const uint8_t * b, const uint8_t * g, const uint8_t * r, uint8_t * bgr, const __mmask64 * tails)
        {
            __m512i _b = mask ? _mm512_maskz_loadu_epi8(tails[3], b) : _mm512_loadu_si512(b);
            __m512i _g = mask ? _mm512_maskz_loadu_epi8(tails[3], g) : _mm512_loadu_si512(g);
            __m512i _r = mask ? _mm512_maskz_loadu_epi8(tails[3], r) : _mm512_loadu_si512(r);
            if (mask)
            {
                _mm512_mask_storeu_epi8(bgr + 0 * A, tails[0], InterleaveBgr<0>(_b, _g, _r));
                _mm512_mask_storeu_epi8(bgr + 1 * A, tails[1], InterleaveBgr<1>(_b, _g, _r));
                _mm512_mask_storeu_epi8(bgr + 2 * A, tails[2], InterleaveBgr<2>(_b, _g, _r));
            }
            else
            {
                _mm512_storeu_si512(bgr + 0 * A, InterleaveBgr<0>(_b, _g, _r));
                _mm512_storeu_si512(bgr + 1 * A, InterleaveBgr<1>(_b, _g, _r));
                _mm512_storeu_si512(bgr + 2 * A, InterleaveBgr<2>(_b, _g, _r));
            }
        }

        void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            tailMasks[3] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    InterleaveBgr<false>(b + col, g + col, r + col, bgr + col * 3, tailMasks);
                if (col < width)
                    InterleaveBgr<true>(b + col, g + col, r + col, bgr + col * 3, tailMasks);
                b += bStride;
                g += gStride;
                r += rStride;
                bgr += bgrStride;
            }
        }

        template <bool mask> SIMD_INLINE void InterleaveBgra(const uint8_t * b, const uint8_t * g, const uint8_t * r, const uint8_t * a, uint8_t * bgra, const __mmask64 * tails)
        {
            __m512i _b = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[4], b) : _mm512_loadu_si512(b));
            __m512i _g = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[4], g) : _mm512_loadu_si512(g));
            __m512i _r = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[4], r) : _mm512_loadu_si512(r));
            __m512i _a = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, mask ? _mm512_maskz_loadu_epi8(tails[4], a) : _mm512_loadu_si512(a));
            __m512i bg0 = UnpackU8<0>(_b, _g);
            __m512i bg1 = UnpackU8<1>(_b, _g);
            __m512i ra0 = UnpackU8<0>(_r, _a);
            __m512i ra1 = UnpackU8<1>(_r, _a);
            if (mask)
            {
                _mm512_mask_storeu_epi8(bgra + 0 * A, tails[0], UnpackU16<0>(bg0, ra0));
                _mm512_mask_storeu_epi8(bgra + 1 * A, tails[1], UnpackU16<1>(bg0, ra0));
                _mm512_mask_storeu_epi8(bgra + 2 * A, tails[2], UnpackU16<0>(bg1, ra1));
                _mm512_mask_storeu_epi8(bgra + 3 * A, tails[3], UnpackU16<1>(bg1, ra1));
            }
            else
            {
                _mm512_storeu_si512(bgra + 0 * A, UnpackU16<0>(bg0, ra0));
                _mm512_storeu_si512(bgra + 1 * A, UnpackU16<1>(bg0, ra0));
                _mm512_storeu_si512(bgra + 2 * A, UnpackU16<0>(bg1, ra1));
                _mm512_storeu_si512(bgra + 3 * A, UnpackU16<1>(bg1, ra1));
            }
        }

        void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A*c);
            tailMasks[4] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    InterleaveBgra<false>(b + col, g + col, r + col, a + col, bgra + col * 4, tailMasks);
                if (col < width)
                    InterleaveBgra<true>(b + col, g + col, r + col, a + col, bgra + col * 4, tailMasks);
                b += bStride;
                g += gStride;
                r += rStride;
                a += aStride;
                bgra += bgraStride;
            }
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

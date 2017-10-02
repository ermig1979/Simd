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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void InterleaveUv(const uint8_t * u, const uint8_t * v, uint8_t * uv, const __mmask64 * tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[2])));
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[2])));
            Store<align, mask>(uv + 0, UnpackU8<0>(_u, _v), tails[0]);
            Store<align, mask>(uv + A, UnpackU8<1>(_u, _v), tails[1]);
        }

        template <bool align> SIMD_INLINE void InterleaveUv2(const uint8_t * u, const uint8_t * v, uint8_t * uv)
        {
            __m512i u0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(u + 0));
            __m512i v0 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(v + 0));
            Store<align>(uv + 0 * A, UnpackU8<0>(u0, v0));
            Store<align>(uv + 1 * A, UnpackU8<1>(u0, v0));
            __m512i u1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(u + A));
            __m512i v1 = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, Load<align>(v + A));
            Store<align>(uv + 2 * A, UnpackU8<0>(u1, v1));
            Store<align>(uv + 3 * A, UnpackU8<1>(u1, v1));
        }

        template <bool align> void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * uv, size_t uvStride)
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
                    InterleaveUv2<align>(u + col, v + col, uv + col * 2);
                for (; col < alignedWidth; col += A)
                    InterleaveUv<align, false>(u + col, v + col, uv + col * 2, tailMasks);
                if (col < width)
                    InterleaveUv<align, true>(u + col, v + col, uv + col * 2, tailMasks);
                uv += uvStride;
                u += uStride;
                v += vStride;
            }
        }

        void InterleaveUv(const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride, size_t width, size_t height, uint8_t * uv, size_t uvStride)
        {
            if (Aligned(uv) && Aligned(uvStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride))
                InterleaveUv<true>(u, uStride, v, vStride, width, height, uv, uvStride);
            else
                InterleaveUv<false>(u, uStride, v, vStride, width, height, uv, uvStride);
        }

        template <bool align, bool mask> SIMD_INLINE void InterleaveBgr(const uint8_t * b, const uint8_t * g, const uint8_t * r, uint8_t * bgr, const __mmask64 * tails)
        {
            __m512i _b = Load<align, mask>(b, tails[3]);
            __m512i _g = Load<align, mask>(g, tails[3]);
            __m512i _r = Load<align, mask>(r, tails[3]);
            Store<align, mask>(bgr + 0 * A, InterleaveBgr<0>(_b, _g, _r), tails[0]);
            Store<align, mask>(bgr + 1 * A, InterleaveBgr<1>(_b, _g, _r), tails[1]);
            Store<align, mask>(bgr + 2 * A, InterleaveBgr<2>(_b, _g, _r), tails[2]);
        }

        template <bool align> void InterleaveBgr(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (align)
            {
                assert(Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride));
                assert(Aligned(r) && Aligned(rStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[4];
            for (size_t c = 0; c < 3; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 3 - A*c);
            tailMasks[3] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    InterleaveBgr<align, false>(b + col, g + col, r + col, bgr + col * 3, tailMasks);
                if (col < width)
                    InterleaveBgr<align, true>(b + col, g + col, r + col, bgr + col * 3, tailMasks);
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

        template <bool align, bool mask> SIMD_INLINE void InterleaveBgra(const uint8_t * b, const uint8_t * g, const uint8_t * r, const uint8_t * a, uint8_t * bgra, const __mmask64 * tails)
        {
            __m512i _b = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<align, mask>(b, tails[4])));
            __m512i _g = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<align, mask>(g, tails[4])));
            __m512i _r = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<align, mask>(r, tails[4])));
            __m512i _a = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, (Load<align, mask>(a, tails[4])));
            __m512i bg0 = UnpackU8<0>(_b, _g);
            __m512i bg1 = UnpackU8<1>(_b, _g);
            __m512i ra0 = UnpackU8<0>(_r, _a);
            __m512i ra1 = UnpackU8<1>(_r, _a);
            Store<align, mask>(bgra + 0 * A, UnpackU16<0>(bg0, ra0), tails[0]);
            Store<align, mask>(bgra + 1 * A, UnpackU16<1>(bg0, ra0), tails[1]);
            Store<align, mask>(bgra + 2 * A, UnpackU16<0>(bg1, ra1), tails[2]);
            Store<align, mask>(bgra + 3 * A, UnpackU16<1>(bg1, ra1), tails[3]);
        }

        template <bool align> void InterleaveBgra(const uint8_t * b, size_t bStride, const uint8_t * g, size_t gStride, const uint8_t * r, size_t rStride, const uint8_t * a, size_t aStride, size_t width, size_t height, uint8_t * bgra, size_t bgraStride)
        {
            if (align)
            {
                assert(Aligned(b) && Aligned(bStride) && Aligned(g) && Aligned(gStride));
                assert(Aligned(r) && Aligned(rStride) && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            __mmask64 tailMasks[5];
            for (size_t c = 0; c < 4; ++c)
                tailMasks[c] = TailMask64((width - alignedWidth) * 4 - A*c);
            tailMasks[4] = TailMask64(width - alignedWidth);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    InterleaveBgra<align, false>(b + col, g + col, r + col, a + col, bgra + col * 4, tailMasks);
                if (col < width)
                    InterleaveBgra<align, true>(b + col, g + col, r + col, a + col, bgra + col * 4, tailMasks);
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
#endif// SIMD_AVX512BW_ENABLE
}

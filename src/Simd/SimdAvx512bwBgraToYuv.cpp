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
        template <bool align, bool mask> SIMD_INLINE void LoadPreparedBgra16(const uint8_t * bgra, __m512i & b16_r16, __m512i & g16_1, const __mmask64 * ms)
        {
            __m512i _bgra = Load<align, mask>(bgra, ms[0]);
            b16_r16 = _mm512_and_si512(_bgra, K16_00FF);
            g16_1 = _mm512_or_si512(_mm512_shuffle_epi8(_bgra, K8_SUFFLE_BGRA_TO_G000), K32_00010000);
        }

        template <bool align, bool mask> SIMD_INLINE __m512i LoadAndConvertBgraToY16(const uint8_t * bgra, __m512i & b16_r16, __m512i & g16_1, const __mmask64 * ms)
        {
            __m512i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<align, mask>(bgra + 0, _b16_r16[0], _g16_1[0], ms + 0);
            LoadPreparedBgra16<align, mask>(bgra + A, _b16_r16[1], _g16_1[1], ms + 1);
            b16_r16 = Hadd32(_b16_r16[0], _b16_r16[1]);
            g16_1 = Hadd32(_g16_1[0], _g16_1[1]);
            return Saturate16iTo8u(_mm512_add_epi16(K16_Y_ADJUST, _mm512_packs_epi32(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
        }

        template <bool align, bool mask> SIMD_INLINE __m512i LoadAndConvertBgraToY8(const uint8_t * bgra, __m512i b16_r16[2], __m512i g16_1[2], const __mmask64 * ms)
        {
            __m512i lo = LoadAndConvertBgraToY16<align, mask>(bgra + 0 * A, b16_r16[0], g16_1[0], ms + 0);
            __m512i hi = LoadAndConvertBgraToY16<align, mask>(bgra + 2 * A, b16_r16[1], g16_1[1], ms + 2);
            return Permuted2Pack16iTo8u(lo, hi);
        }

        SIMD_INLINE void Average16(__m512i & a, const __m512i & b)
        {
            a = _mm512_srli_epi16(_mm512_add_epi16(_mm512_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE __m512i ConvertU16(__m512i b16_r16[2], __m512i g16_1[2])
        {
            return Saturate16iTo8u(_mm512_add_epi16(K16_UV_ADJUST, _mm512_packs_epi32(BgrToU32(b16_r16[0], g16_1[0]), BgrToU32(b16_r16[1], g16_1[1]))));
        }

        SIMD_INLINE __m512i ConvertV16(__m512i b16_r16[2], __m512i g16_1[2])
        {
            return Saturate16iTo8u(_mm512_add_epi16(K16_UV_ADJUST, _mm512_packs_epi32(BgrToV32(b16_r16[0], g16_1[0]), BgrToV32(b16_r16[1], g16_1[1]))));
        }

        template <bool align, bool mask> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v, const __mmask64 * ms)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            __m512i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<align, mask>(y0 + 0, LoadAndConvertBgraToY8<align, mask>(bgra0 + 0 * A, _b16_r16[0][0], _g16_1[0][0], ms + 0), ms[8]);
            Store<align, mask>(y0 + A, LoadAndConvertBgraToY8<align, mask>(bgra0 + 4 * A, _b16_r16[0][1], _g16_1[0][1], ms + 4), ms[9]);
            Store<align, mask>(y1 + 0, LoadAndConvertBgraToY8<align, mask>(bgra1 + 0 * A, _b16_r16[1][0], _g16_1[1][0], ms + 0), ms[8]);
            Store<align, mask>(y1 + A, LoadAndConvertBgraToY8<align, mask>(bgra1 + 4 * A, _b16_r16[1][1], _g16_1[1][1], ms + 4), ms[9]);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align, mask>(u, Permuted2Pack16iTo8u(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])), ms[10]);
            Store<align, mask>(v, Permuted2Pack16iTo8u(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])), ms[10]);
        }

        template <bool align> void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[11];
            for (size_t i = 0; i < 8; ++i)
                tailMasks[i] = TailMask64(tail * 8 - A*i);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[8 + i] = TailMask64(tail * 2 - A*i);
            tailMasks[10] = TailMask64(tail);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgraToYuv420p<align, false>(bgra + col * 8, bgraStride, y + col * 2, yStride, u + col, v + col, tailMasks);
                if (col < width)
                    BgraToYuv420p<align, true>(bgra + col * 8, bgraStride, y + col * 2, yStride, u + col, v + col, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv420p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv420p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE void Average16(__m512i a[2][2])
        {
            a[0][0] = _mm512_srli_epi16(_mm512_add_epi16(a[0][0], K16_0001), 1);
            a[0][1] = _mm512_srli_epi16(_mm512_add_epi16(a[0][1], K16_0001), 1);
            a[1][0] = _mm512_srli_epi16(_mm512_add_epi16(a[1][0], K16_0001), 1);
            a[1][1] = _mm512_srli_epi16(_mm512_add_epi16(a[1][1], K16_0001), 1);
        }

        template <bool align, bool mask> SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v, const __mmask64 * ms)
        {
            __m512i _b16_r16[2][2], _g16_1[2][2];
            Store<align, mask>(y + 0, LoadAndConvertBgraToY8<align, mask>(bgra + 0 * A, _b16_r16[0], _g16_1[0], ms + 0), ms[8]);
            Store<align, mask>(y + A, LoadAndConvertBgraToY8<align, mask>(bgra + 4 * A, _b16_r16[1], _g16_1[1], ms + 4), ms[9]);

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<align, mask>(u, Permuted2Pack16iTo8u(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])), ms[10]);
            Store<align, mask>(v, Permuted2Pack16iTo8u(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])), ms[10]);
        }

        template <bool align> void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width % 2 == 0);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[11];
            for (size_t i = 0; i < 8; ++i)
                tailMasks[i] = TailMask64(tail * 8 - A*i);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[8 + i] = TailMask64(tail * 2 - A*i);
            tailMasks[10] = TailMask64(tail);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgraToYuv422p<align, false>(bgra + col * 8, y + col * 2, u + col, v + col, tailMasks);
                if (col < width)
                    BgraToYuv422p<align, true>(bgra + col * 8, y + col * 2, u + col, v + col, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv422p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv422p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        SIMD_INLINE __m512i ConvertY16(__m512i b16_r16[2], __m512i g16_1[2])
        {
            return Saturate16iTo8u(_mm512_add_epi16(K16_Y_ADJUST, _mm512_packs_epi32(BgrToY32(b16_r16[0], g16_1[0]), BgrToY32(b16_r16[1], g16_1[1]))));
        }

        template <bool align, bool mask> SIMD_INLINE void BgraToYuv444p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v, const __mmask64 * ms)
        {
            __m512i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgra16<align, mask>(bgra + 0 * A, _b16_r16[0][0], _g16_1[0][0], ms + 0);
            LoadPreparedBgra16<align, mask>(bgra + 1 * A, _b16_r16[0][1], _g16_1[0][1], ms + 1);
            LoadPreparedBgra16<align, mask>(bgra + 2 * A, _b16_r16[1][0], _g16_1[1][0], ms + 2);
            LoadPreparedBgra16<align, mask>(bgra + 3 * A, _b16_r16[1][1], _g16_1[1][1], ms + 3);

            Store<align, mask>(y, Permuted2Pack16iTo8u(ConvertY16(_b16_r16[0], _g16_1[0]), ConvertY16(_b16_r16[1], _g16_1[1])), ms[4]);
            Store<align, mask>(u, Permuted2Pack16iTo8u(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])), ms[4]);
            Store<align, mask>(v, Permuted2Pack16iTo8u(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])), ms[4]);
        }

        template <bool align> void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[5];
            for (size_t i = 0; i < 4; ++i)
                tailMasks[i] = TailMask64(tail * 4 - A*i);
            tailMasks[4] = TailMask64(tail);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    BgraToYuv444p<align, false>(bgra + col * 4, y + col, u + col, v + col, tailMasks);
                if (col < width)
                    BgraToYuv444p<align, true>(bgra + col * 4, y + col, u + col, v + col, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv444p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv444p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

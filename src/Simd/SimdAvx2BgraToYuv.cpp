/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
        template <bool align> SIMD_INLINE void LoadPreparedBgra16(const __m256i * bgra, __m256i & b16_r16, __m256i & g16_1)
        {
            __m256i _bgra = Load<align>(bgra);
            b16_r16 = _mm256_and_si256(_bgra, K16_00FF);
            g16_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(_bgra, 1), K32_000000FF), K32_00010000);
        }

        template <bool align> SIMD_INLINE __m256i LoadAndConvertY16(const __m256i * bgra, __m256i & b16_r16, __m256i & g16_1)
        {
            __m256i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<align>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<align>(bgra + 1, _b16_r16[1], _g16_1[1]);
            b16_r16 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_b16_r16[0], _b16_r16[1]), 0xD8);
            g16_1 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_g16_1[0], _g16_1[1]), 0xD8);
            return SaturateI16ToU8(_mm256_add_epi16(K16_Y_ADJUST, PackI32ToI16(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
        }

        template <bool align> SIMD_INLINE __m256i LoadAndConvertY8(const __m256i * bgra, __m256i b16_r16[2], __m256i g16_1[2])
        {
            return PackI16ToU8(LoadAndConvertY16<align>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndConvertY16<align>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        SIMD_INLINE void Average16(__m256i & a, const __m256i & b)
        {
            a = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE __m256i ConvertU16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_UV_ADJUST, PackI32ToI16(BgrToU32(b16_r16[0], g16_1[0]), BgrToU32(b16_r16[1], g16_1[1]))));
        }

        SIMD_INLINE __m256i ConvertV16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_UV_ADJUST, PackI32ToI16(BgrToV32(b16_r16[0], g16_1[0]), BgrToV32(b16_r16[1], g16_1[1]))));
        }

        template <bool align> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            __m256i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<align>((__m256i*)y0 + 0, LoadAndConvertY8<align>((__m256i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0]));
            Store<align>((__m256i*)y0 + 1, LoadAndConvertY8<align>((__m256i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1]));
            Store<align>((__m256i*)y1 + 0, LoadAndConvertY8<align>((__m256i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0]));
            Store<align>((__m256i*)y1 + 1, LoadAndConvertY8<align>((__m256i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1]));

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align>((__m256i*)u, PackI16ToU8(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])));
            Store<align>((__m256i*)v, PackI16ToU8(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <bool align> void BgraToYuv420p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < alignedWidth; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv420p<align>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuv420p<false>(bgra + offset * 4, bgraStride, y + offset, yStride, u + offset / 2, v + offset / 2);
                }
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

        SIMD_INLINE void Average16(__m256i a[2][2])
        {
            a[0][0] = _mm256_srli_epi16(_mm256_add_epi16(a[0][0], K16_0001), 1);
            a[0][1] = _mm256_srli_epi16(_mm256_add_epi16(a[0][1], K16_0001), 1);
            a[1][0] = _mm256_srli_epi16(_mm256_add_epi16(a[1][0], K16_0001), 1);
            a[1][1] = _mm256_srli_epi16(_mm256_add_epi16(a[1][1], K16_0001), 1);
        }

        template <bool align> SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            __m256i _b16_r16[2][2], _g16_1[2][2];
            Store<align>((__m256i*)y + 0, LoadAndConvertY8<align>((__m256i*)bgra + 0, _b16_r16[0], _g16_1[0]));
            Store<align>((__m256i*)y + 1, LoadAndConvertY8<align>((__m256i*)bgra + 4, _b16_r16[1], _g16_1[1]));

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<align>((__m256i*)u, PackI16ToU8(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m256i*)v, PackI16ToU8(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])));
        }

        template <bool align> void BgraToYuv422p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < alignedWidth; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv422p<align>(bgra + colBgra, y + colY, u + colUV, v + colUV);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuv422p<false>(bgra + offset * 4, y + offset, u + offset / 2, v + offset / 2);
                }
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

        SIMD_INLINE __m256i ConvertY16(__m256i b16_r16[2], __m256i g16_1[2])
        {
            return SaturateI16ToU8(_mm256_add_epi16(K16_Y_ADJUST, PackI32ToI16(BgrToY32(b16_r16[0], g16_1[0]), BgrToY32(b16_r16[1], g16_1[1]))));
        }

        template <bool align> SIMD_INLINE void BgraToYuv444p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            __m256i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgra16<align>((__m256i*)bgra + 0, _b16_r16[0][0], _g16_1[0][0]);
            LoadPreparedBgra16<align>((__m256i*)bgra + 1, _b16_r16[0][1], _g16_1[0][1]);
            LoadPreparedBgra16<align>((__m256i*)bgra + 2, _b16_r16[1][0], _g16_1[1][0]);
            LoadPreparedBgra16<align>((__m256i*)bgra + 3, _b16_r16[1][1], _g16_1[1][1]);

            Store<align>((__m256i*)y, PackI16ToU8(ConvertY16(_b16_r16[0], _g16_1[0]), ConvertY16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m256i*)u, PackI16ToU8(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m256i*)v, PackI16ToU8(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])));
        }

        template <bool align> void BgraToYuv444p(const uint8_t * bgra, size_t width, size_t height, size_t bgraStride, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgra = 0; col < alignedWidth; col += A, colBgra += QA)
                    BgraToYuv444p<align>(bgra + colBgra, y + col, u + col, v + col);
                if (width != alignedWidth)
                {
                    size_t offset = width - A;
                    BgraToYuv444p<false>(bgra + offset * 4, y + offset, u + offset, v + offset);
                }
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

        template <bool align> SIMD_INLINE void LoadPreparedBgra16(const __m256i * bgra, __m256i & b16_r16, __m256i & g16_1, __m256i & a32)
        {
            __m256i _bgra = Load<align>(bgra);
            b16_r16 = _mm256_and_si256(_bgra, K16_00FF);
            g16_1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_si256(_bgra, 1), K32_000000FF), K32_00010000);
            a32 = _mm256_and_si256(_mm256_srli_si256(_bgra, 3), K32_000000FF);
        }

        template <bool align> SIMD_INLINE void LoadAndConvertYA16(const __m256i * bgra, __m256i & b16_r16, __m256i & g16_1, __m256i & y16, __m256i & a16)
        {
            __m256i _b16_r16[2], _g16_1[2], a32[2];
            LoadPreparedBgra16<align>(bgra + 0, _b16_r16[0], _g16_1[0], a32[0]);
            LoadPreparedBgra16<align>(bgra + 1, _b16_r16[1], _g16_1[1], a32[1]);
            b16_r16 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_b16_r16[0], _b16_r16[1]), 0xD8);
            g16_1 = _mm256_permute4x64_epi64(_mm256_hadd_epi32(_g16_1[0], _g16_1[1]), 0xD8);
            y16 = SaturateI16ToU8(_mm256_add_epi16(K16_Y_ADJUST, PackI32ToI16(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
            a16 = PackI32ToI16(a32[0], a32[1]);
        }

        template <bool align> SIMD_INLINE void LoadAndStoreYA(const __m256i * bgra, __m256i b16_r16[2], __m256i g16_1[2], __m256i * y, __m256i * a)
        {
            __m256i y16[2], a16[2];
            LoadAndConvertYA16<align>(bgra + 0, b16_r16[0], g16_1[0], y16[0], a16[0]);
            LoadAndConvertYA16<align>(bgra + 2, b16_r16[1], g16_1[1], y16[1], a16[1]);
            Store<align>(y, PackI16ToU8(y16[0], y16[1]));
            Store<align>(a, PackI16ToU8(a16[0], a16[1]));
        }

        template <bool align> SIMD_INLINE void BgraToYuva420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v, uint8_t * a0, size_t aStride)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;
            uint8_t * a1 = a0 + aStride;

            __m256i _b16_r16[2][2][2], _g16_1[2][2][2];
            LoadAndStoreYA<align>((__m256i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0], (__m256i*)y0 + 0, (__m256i*)a0 + 0);
            LoadAndStoreYA<align>((__m256i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1], (__m256i*)y0 + 1, (__m256i*)a0 + 1);
            LoadAndStoreYA<align>((__m256i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0], (__m256i*)y1 + 0, (__m256i*)a1 + 0);
            LoadAndStoreYA<align>((__m256i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1], (__m256i*)y1 + 1, (__m256i*)a1 + 1);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align>((__m256i*)u, PackI16ToU8(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])));
            Store<align>((__m256i*)v, PackI16ToU8(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <bool align> void BgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride));
            }

            size_t alignedWidth = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colYA = 0, colBgra = 0; colYA < alignedWidth; colYA += DA, colUV += A, colBgra += A8)
                    BgraToYuva420p<align>(bgra + colBgra, bgraStride, y + colYA, yStride, u + colUV, v + colUV, a + colYA, aStride);
                if (width != alignedWidth)
                {
                    size_t offset = width - DA;
                    BgraToYuva420p<false>(bgra + offset * 4, bgraStride, y + offset, yStride, u + offset / 2, v + offset / 2, a + offset, aStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuva420p(const uint8_t * bgra, size_t bgraStride, size_t width, size_t height, uint8_t * y, size_t yStride,
            uint8_t * u, size_t uStride, uint8_t * v, size_t vStride, uint8_t * a, size_t aStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride) && Aligned(v) && Aligned(vStride)
                && Aligned(a) && Aligned(aStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuva420p<true>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
            else
                BgraToYuva420p<false>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

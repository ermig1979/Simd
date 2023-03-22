/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <bool align> SIMD_INLINE __m128i LoadAndConvertY16(const __m128i * bgra, __m128i & b16_r16, __m128i & g16_1)
        {
            __m128i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<align>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<align>(bgra + 1, _b16_r16[1], _g16_1[1]);
            b16_r16 = _mm_hadd_epi32(_b16_r16[0], _b16_r16[1]);
            g16_1 = _mm_hadd_epi32(_g16_1[0], _g16_1[1]);
            return SaturateI16ToU8(_mm_add_epi16(K16_Y_ADJUST, _mm_packs_epi32(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
        }

        template <bool align> SIMD_INLINE __m128i LoadAndConvertY8(const __m128i * bgra, __m128i b16_r16[2], __m128i g16_1[2])
        {
            return _mm_packus_epi16(LoadAndConvertY16<align>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndConvertY16<align>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        SIMD_INLINE void Average16(__m128i & a, const __m128i & b)
        {
            a = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(a, b), K16_0002), 2);
        }

        SIMD_INLINE __m128i ConvertU16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(BgrToU32(b16_r16[0], g16_1[0]), BgrToU32(b16_r16[1], g16_1[1]))));
        }

        SIMD_INLINE __m128i ConvertV16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_UV_ADJUST, _mm_packs_epi32(BgrToV32(b16_r16[0], g16_1[0]), BgrToV32(b16_r16[1], g16_1[1]))));
        }

        template <bool align> SIMD_INLINE void BgraToYuv420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;

            __m128i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<align>((__m128i*)y0 + 0, LoadAndConvertY8<align>((__m128i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0]));
            Store<align>((__m128i*)y0 + 1, LoadAndConvertY8<align>((__m128i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1]));
            Store<align>((__m128i*)y1 + 0, LoadAndConvertY8<align>((__m128i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0]));
            Store<align>((__m128i*)y1 + 1, LoadAndConvertY8<align>((__m128i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1]));

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align>((__m128i*)u, _mm_packus_epi16(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])));
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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void Average16(__m128i a[2][2])
        {
            a[0][0] = _mm_srli_epi16(_mm_add_epi16(a[0][0], K16_0001), 1);
            a[0][1] = _mm_srli_epi16(_mm_add_epi16(a[0][1], K16_0001), 1);
            a[1][0] = _mm_srli_epi16(_mm_add_epi16(a[1][0], K16_0001), 1);
            a[1][1] = _mm_srli_epi16(_mm_add_epi16(a[1][1], K16_0001), 1);
        }

        template <bool align> SIMD_INLINE void BgraToYuv422p(const uint8_t * bgra, uint8_t * y, uint8_t * u, uint8_t * v)
        {
            __m128i _b16_r16[2][2], _g16_1[2][2];
            Store<align>((__m128i*)y + 0, LoadAndConvertY8<align>((__m128i*)bgra + 0, _b16_r16[0], _g16_1[0]));
            Store<align>((__m128i*)y + 1, LoadAndConvertY8<align>((__m128i*)bgra + 4, _b16_r16[1], _g16_1[1]));

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<align>((__m128i*)u, _mm_packus_epi16(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])));
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

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE __m128i ConvertY16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            return SaturateI16ToU8(_mm_add_epi16(K16_Y_ADJUST, _mm_packs_epi32(BgrToY32(b16_r16[0], g16_1[0]), BgrToY32(b16_r16[1], g16_1[1]))));
        }

        template <bool align> SIMD_INLINE void BgraToYuv444p(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            __m128i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgra16<align>((__m128i*)bgra + 0, _b16_r16[0][0], _g16_1[0][0]);
            LoadPreparedBgra16<align>((__m128i*)bgra + 1, _b16_r16[0][1], _g16_1[0][1]);
            LoadPreparedBgra16<align>((__m128i*)bgra + 2, _b16_r16[1][0], _g16_1[1][0]);
            LoadPreparedBgra16<align>((__m128i*)bgra + 3, _b16_r16[1][1], _g16_1[1][1]);

            Store<align>((__m128i*)y, _mm_packus_epi16(ConvertY16(_b16_r16[0], _g16_1[0]), ConvertY16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m128i*)u, _mm_packus_epi16(ConvertU16(_b16_r16[0], _g16_1[0]), ConvertU16(_b16_r16[1], _g16_1[1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(ConvertV16(_b16_r16[0], _g16_1[0]), ConvertV16(_b16_r16[1], _g16_1[1])));
        }

        template <bool align> void BgraToYuv444p(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
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

        void BgraToYuv444p(const uint8_t* bgra, size_t width, size_t height, size_t bgraStride, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgra) && Aligned(bgraStride))
                BgraToYuv444p<true>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
            else
                BgraToYuv444p<false>(bgra, width, height, bgraStride, y, yStride, u, uStride, v, vStride);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void LoadAndConvertYA16(const __m128i * bgra, __m128i & b16_r16, __m128i & g16_1, __m128i & y16, __m128i & a16)
        {
            __m128i _b16_r16[2], _g16_1[2], a32[2];
            LoadPreparedBgra16<align>(bgra + 0, _b16_r16[0], _g16_1[0], a32[0]);
            LoadPreparedBgra16<align>(bgra + 1, _b16_r16[1], _g16_1[1], a32[1]);
            b16_r16 = _mm_hadd_epi32(_b16_r16[0], _b16_r16[1]);
            g16_1 = _mm_hadd_epi32(_g16_1[0], _g16_1[1]);
            y16 = SaturateI16ToU8(_mm_add_epi16(K16_Y_ADJUST, _mm_packs_epi32(BgrToY32(_b16_r16[0], _g16_1[0]), BgrToY32(_b16_r16[1], _g16_1[1]))));
            a16 = _mm_packs_epi32(a32[0], a32[1]);
        }

        template <bool align> SIMD_INLINE void LoadAndStoreYA(const __m128i * bgra, __m128i b16_r16[2], __m128i g16_1[2], __m128i * y, __m128i * a)
        {
            __m128i y16[2], a16[2];
            LoadAndConvertYA16<align>(bgra + 0, b16_r16[0], g16_1[0], y16[0], a16[0]);
            LoadAndConvertYA16<align>(bgra + 2, b16_r16[1], g16_1[1], y16[1], a16[1]);
            Store<align>(y, _mm_packus_epi16(y16[0], y16[1]));
            Store<align>(a, _mm_packus_epi16(a16[0], a16[1]));
        }

        template <bool align> SIMD_INLINE void BgraToYuva420p(const uint8_t * bgra0, size_t bgraStride, uint8_t * y0, size_t yStride, uint8_t * u, uint8_t * v, uint8_t * a0, size_t aStride)
        {
            const uint8_t * bgra1 = bgra0 + bgraStride;
            uint8_t * y1 = y0 + yStride;
            uint8_t * a1 = a0 + aStride;

            __m128i _b16_r16[2][2][2], _g16_1[2][2][2];
            LoadAndStoreYA<align>((__m128i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0], (__m128i*)y0 + 0, (__m128i*)a0 + 0);
            LoadAndStoreYA<align>((__m128i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1], (__m128i*)y0 + 1, (__m128i*)a0 + 1);
            LoadAndStoreYA<align>((__m128i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0], (__m128i*)y1 + 0, (__m128i*)a1 + 0);
            LoadAndStoreYA<align>((__m128i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1], (__m128i*)y1 + 1, (__m128i*)a1 + 1);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<align>((__m128i*)u, _mm_packus_epi16(ConvertU16(_b16_r16[0][0], _g16_1[0][0]), ConvertU16(_b16_r16[0][1], _g16_1[0][1])));
            Store<align>((__m128i*)v, _mm_packus_epi16(ConvertV16(_b16_r16[0][0], _g16_1[0][0]), ConvertV16(_b16_r16[0][1], _g16_1[0][1])));
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

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m128i BgrToY16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            static const __m128i Y_LO = SIMD_MM_SET1_EPI16(T::Y_LO);
            return SaturateI16ToU8(_mm_add_epi16(Y_LO, _mm_packs_epi32(BgrToY32<T>(b16_r16[0], g16_1[0]), BgrToY32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE __m128i BgrToU16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            static const __m128i UV_Z = SIMD_MM_SET1_EPI16(T::UV_Z);
            return SaturateI16ToU8(_mm_add_epi16(UV_Z, _mm_packs_epi32(BgrToU32<T>(b16_r16[0], g16_1[0]), BgrToU32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE __m128i BgrToV16(__m128i b16_r16[2], __m128i g16_1[2])
        {
            static const __m128i UV_Z = SIMD_MM_SET1_EPI16(T::UV_Z);
            return SaturateI16ToU8(_mm_add_epi16(UV_Z, _mm_packs_epi32(BgrToV32<T>(b16_r16[0], g16_1[0]), BgrToV32<T>(b16_r16[1], g16_1[1]))));
        }

        template <class T> SIMD_INLINE void BgraToYuv444pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            __m128i _b16_r16[2][2], _g16_1[2][2];
            LoadPreparedBgra16<false>((__m128i*)bgra + 0, _b16_r16[0][0], _g16_1[0][0]);
            LoadPreparedBgra16<false>((__m128i*)bgra + 1, _b16_r16[0][1], _g16_1[0][1]);
            LoadPreparedBgra16<false>((__m128i*)bgra + 2, _b16_r16[1][0], _g16_1[1][0]);
            LoadPreparedBgra16<false>((__m128i*)bgra + 3, _b16_r16[1][1], _g16_1[1][1]);

            Store<false>((__m128i*)y, _mm_packus_epi16(BgrToY16<T>(_b16_r16[0], _g16_1[0]), BgrToY16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m128i*)u, _mm_packus_epi16(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m128i*)v, _mm_packus_epi16(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])));
        }

        template <class T> void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert(width >= A);

            size_t widthA = AlignLo(width, A);
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0; col < widthA; col += A)
                    BgraToYuv444pV2<T>(bgra + col * 4, y + col, u + col, v + col);
                if (width != widthA)
                {
                    size_t col = width - A;
                    BgraToYuv444pV2<T>(bgra + col * 4, y + col, u + col, v + col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv444pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv444pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv444pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv444pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv444pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv444pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE __m128i LoadAndBgrToY16(const __m128i* bgra, __m128i& b16_r16, __m128i& g16_1)
        {
            static const __m128i Y_LO = SIMD_MM_SET1_EPI16(T::Y_LO);
            __m128i _b16_r16[2], _g16_1[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1]);
            b16_r16 = _mm_hadd_epi32(_b16_r16[0], _b16_r16[1]);
            g16_1 = _mm_hadd_epi32(_g16_1[0], _g16_1[1]);
            return BgrToY16<T>(_b16_r16, _g16_1);
        }

        template <class T> SIMD_INLINE __m128i LoadAndBgrToY8(const __m128i* bgra, __m128i b16_r16[2], __m128i g16_1[2])
        {
            return _mm_packus_epi16(LoadAndBgrToY16<T>(bgra + 0, b16_r16[0], g16_1[0]), LoadAndBgrToY16<T>(bgra + 2, b16_r16[1], g16_1[1]));
        }

        template <class T> SIMD_INLINE void BgraToYuv422pV2(const uint8_t* bgra, uint8_t* y, uint8_t* u, uint8_t* v)
        {
            __m128i _b16_r16[2][2], _g16_1[2][2];
            Store<false>((__m128i*)y + 0, LoadAndBgrToY8<T>((__m128i*)bgra + 0, _b16_r16[0], _g16_1[0]));
            Store<false>((__m128i*)y + 1, LoadAndBgrToY8<T>((__m128i*)bgra + 4, _b16_r16[1], _g16_1[1]));

            Average16(_b16_r16);
            Average16(_g16_1);

            Store<false>((__m128i*)u, _mm_packus_epi16(BgrToU16<T>(_b16_r16[0], _g16_1[0]), BgrToU16<T>(_b16_r16[1], _g16_1[1])));
            Store<false>((__m128i*)v, _mm_packus_epi16(BgrToV16<T>(_b16_r16[0], _g16_1[0]), BgrToV16<T>(_b16_r16[1], _g16_1[1])));
        }

        template <class T>  void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (width >= DA));

            size_t widthDA = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv422pV2<T>(bgra + colBgra, y + colY, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuv422pV2<T>(bgra + colY * 4, y + colY, u + colY / 2, v + colY / 2);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgra += bgraStride;
            }
        }

        void BgraToYuv422pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv422pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv422pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv422pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv422pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv422pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void BgraToYuv420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            __m128i _b16_r16[2][2][2], _g16_1[2][2][2];
            Store<false>((__m128i*)y0 + 0, LoadAndBgrToY8<T>((__m128i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0]));
            Store<false>((__m128i*)y0 + 1, LoadAndBgrToY8<T>((__m128i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1]));
            Store<false>((__m128i*)y1 + 0, LoadAndBgrToY8<T>((__m128i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0]));
            Store<false>((__m128i*)y1 + 1, LoadAndBgrToY8<T>((__m128i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1]));

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            Store<false>((__m128i*)u, _mm_packus_epi16(BgrToU16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToU16<T>(_b16_r16[0][1], _g16_1[0][1])));
            Store<false>((__m128i*)v, _mm_packus_epi16(BgrToV16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToV16<T>(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <class T>  void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));

            size_t widthDA = AlignLo(width, DA);
            const size_t A8 = A * 8;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < widthDA; colY += DA, colUV += A, colBgra += A8)
                    BgraToYuv420pV2<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuv420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colY / 2, v + colY / 2);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgra += 2 * bgraStride;
            }
        }

        void BgraToYuv420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuv420pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuv420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: BgraToYuv420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: BgraToYuv420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: BgraToYuv420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
#endif
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void LoadAndConvertYA16(const __m128i* bgra, __m128i& b16_r16, __m128i& g16_1, __m128i& y16, __m128i& a16)
        {
            __m128i _b16_r16[2], _g16_1[2], a32[2];
            LoadPreparedBgra16<false>(bgra + 0, _b16_r16[0], _g16_1[0], a32[0]);
            LoadPreparedBgra16<false>(bgra + 1, _b16_r16[1], _g16_1[1], a32[1]);
            b16_r16 = _mm_hadd_epi32(_b16_r16[0], _b16_r16[1]);
            g16_1 = _mm_hadd_epi32(_g16_1[0], _g16_1[1]);
            static const __m128i Y_LO = SIMD_MM_SET1_EPI16(T::Y_LO);
            y16 = SaturateI16ToU8(_mm_add_epi16(Y_LO, _mm_packs_epi32(BgrToY32<T>(_b16_r16[0], _g16_1[0]), BgrToY32<T>(_b16_r16[1], _g16_1[1]))));
            a16 = _mm_packs_epi32(a32[0], a32[1]);
        }

        template <class T> SIMD_INLINE void LoadAndStoreYA(const __m128i* bgra, __m128i b16_r16[2], __m128i g16_1[2], __m128i* y, __m128i* a)
        {
            __m128i y16[2], a16[2];
            LoadAndConvertYA16<T>(bgra + 0, b16_r16[0], g16_1[0], y16[0], a16[0]);
            LoadAndConvertYA16<T>(bgra + 2, b16_r16[1], g16_1[1], y16[1], a16[1]);
            _mm_storeu_si128(y, _mm_packus_epi16(y16[0], y16[1]));
            _mm_storeu_si128(a, _mm_packus_epi16(a16[0], a16[1]));
        }

        template <class T> SIMD_INLINE void BgraToYuva420pV2(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v, uint8_t* a0, size_t aStride)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;
            uint8_t* a1 = a0 + aStride;

            __m128i _b16_r16[2][2][2], _g16_1[2][2][2];
            LoadAndStoreYA<T>((__m128i*)bgra0 + 0, _b16_r16[0][0], _g16_1[0][0], (__m128i*)y0 + 0, (__m128i*)a0 + 0);
            LoadAndStoreYA<T>((__m128i*)bgra0 + 4, _b16_r16[0][1], _g16_1[0][1], (__m128i*)y0 + 1, (__m128i*)a0 + 1);
            LoadAndStoreYA<T>((__m128i*)bgra1 + 0, _b16_r16[1][0], _g16_1[1][0], (__m128i*)y1 + 0, (__m128i*)a1 + 0);
            LoadAndStoreYA<T>((__m128i*)bgra1 + 4, _b16_r16[1][1], _g16_1[1][1], (__m128i*)y1 + 1, (__m128i*)a1 + 1);

            Average16(_b16_r16[0][0][0], _b16_r16[1][0][0]);
            Average16(_b16_r16[0][0][1], _b16_r16[1][0][1]);
            Average16(_b16_r16[0][1][0], _b16_r16[1][1][0]);
            Average16(_b16_r16[0][1][1], _b16_r16[1][1][1]);

            Average16(_g16_1[0][0][0], _g16_1[1][0][0]);
            Average16(_g16_1[0][0][1], _g16_1[1][0][1]);
            Average16(_g16_1[0][1][0], _g16_1[1][1][0]);
            Average16(_g16_1[0][1][1], _g16_1[1][1][1]);

            _mm_storeu_si128((__m128i*)u, _mm_packus_epi16(BgrToU16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToU16<T>(_b16_r16[0][1], _g16_1[0][1])));
            _mm_storeu_si128((__m128i*)v, _mm_packus_epi16(BgrToV16<T>(_b16_r16[0][0], _g16_1[0][0]), BgrToV16<T>(_b16_r16[0][1], _g16_1[0][1])));
        }

        template <class T>  void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA));

            size_t widthDA = AlignLo(width, DA);
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0; colY < widthDA; colY += DA, colUV += A)
                    BgraToYuva420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colUV, v + colUV, a + colY, aStride);
                if (width != widthDA)
                {
                    size_t colY = width - DA;
                    BgraToYuva420pV2<T>(bgra + colY * 4, bgraStride, y + colY, yStride, u + colY / 2, v + colY / 2, a + colY, aStride);
                }
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                a += 2 * aStride;
            }
        }

        void BgraToYuva420pV2(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height, uint8_t* y, size_t yStride,
            uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, uint8_t* a, size_t aStride, SimdYuvType yuvType)
        {
#if defined(SIMD_X86_ENABLE) && defined(NDEBUG) && defined(_MSC_VER) && _MSC_VER <= 1900
            Base::BgraToYuva420pV2(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride, yuvType);
#else
            switch (yuvType)
            {
            case SimdYuvBt601: BgraToYuva420pV2<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt709: BgraToYuva420pV2<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvBt2020: BgraToYuva420pV2<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            case SimdYuvTrect871: BgraToYuva420pV2<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride, a, aStride); break;
            default:
                assert(0);
            }
#endif
        }
    }
#endif
}

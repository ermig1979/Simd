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
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool align, bool mask> SIMD_INLINE void YuvToBgr(const __m512i & y, const __m512i & u, const __m512i & v, uint8_t * bgr, const __mmask64 * tails)
        {
            __m512i blue = YuvToBlue(y, u);
            __m512i green = YuvToGreen(y, u, v);
            __m512i red = YuvToRed(y, v);
            Store<align, mask>(bgr + 0 * A, InterleaveBgr<0>(blue, green, red), tails[0]);
            Store<align, mask>(bgr + 1 * A, InterleaveBgr<1>(blue, green, red), tails[1]);
            Store<align, mask>(bgr + 2 * A, InterleaveBgr<2>(blue, green, red), tails[2]);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv420pToBgr(const uint8_t * y0, const uint8_t * y1, const uint8_t * u, const uint8_t * v, uint8_t * bgr0, uint8_t * bgr1, const __mmask64 * tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgr<align, mask>(Load<align, mask>(y0 + 0, tails[1]), u0, v0, bgr0 + 0 * A, tails + 3);
            YuvToBgr<align, mask>(Load<align, mask>(y0 + A, tails[2]), u1, v1, bgr0 + 3 * A, tails + 6);
            YuvToBgr<align, mask>(Load<align, mask>(y1 + 0, tails[1]), u0, v0, bgr1 + 0 * A, tails + 3);
            YuvToBgr<align, mask>(Load<align, mask>(y1 + A, tails[2]), u1, v1, bgr1 + 3 * A, tails + 6);
        }

        template <bool align> void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv420pToBgr<align, false>(y + col * 2, y + yStride + col * 2, u + col, v + col, bgr + col * 6, bgr + bgrStride + col * 6, tailMasks);
                if (col < width)
                    Yuv420pToBgr<align, true>(y + col * 2, y + yStride + col * 2, u + col, v + col, bgr + col * 6, bgr + bgrStride + col * 6, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void Yuv420pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv420pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv420pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv422pToBgr(const uint8_t * y, const uint8_t * u, const uint8_t * v, uint8_t * bgr, const __mmask64 * tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[0])));
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[0])));
            YuvToBgr<align, mask>(Load<align, mask>(y + 0, tails[1]), _mm512_unpacklo_epi8(_u, _u), _mm512_unpacklo_epi8(_v, _v), bgr + 0 * A, tails + 3);
            YuvToBgr<align, mask>(Load<align, mask>(y + A, tails[2]), _mm512_unpackhi_epi8(_u, _u), _mm512_unpackhi_epi8(_v, _v), bgr + 3 * A, tails + 6);
        }

        template <bool align> void Yuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            assert((width % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv422pToBgr<align, false>(y + col * 2, u + col, v + col, bgr + col * 6, tailMasks);
                if (col < width)
                    Yuv422pToBgr<align, true>(y + col * 2, u + col, v + col, bgr + col * 6, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv422pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv422pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv422pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv444pToBgr(const uint8_t * y, const uint8_t * u, const uint8_t * v, uint8_t * bgr, const __mmask64 * tails)
        {
            YuvToBgr<align, mask>(Load<align, mask>(y, tails[0]), Load<align, mask>(u, tails[0]), Load<align, mask>(v, tails[0]), bgr, tails + 1);
        }

        template <bool align> void Yuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[4];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 3; ++i)
                tailMasks[1 + i] = TailMask64(tail * 3 - A * i);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToBgr<align, false>(y + col, u + col, v + col, bgr + col * 3, tailMasks);
                if (col < width)
                    Yuv444pToBgr<align, true>(y + col, u + col, v + col, bgr + col * 3, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgr(const uint8_t * y, size_t yStride, const uint8_t * u, size_t uStride, const uint8_t * v, size_t vStride,
            size_t width, size_t height, uint8_t * bgr, size_t bgrStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv444pToBgr<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
            else
                Yuv444pToBgr<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride);
        }

        //---------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void YuvToRgb(const __m512i& y, const __m512i& u, const __m512i& v, uint8_t* rgb, const __mmask64* tails)
        {
            __m512i blue = YuvToBlue(y, u);
            __m512i green = YuvToGreen(y, u, v);
            __m512i red = YuvToRed(y, v);
            Store<align, mask>(rgb + 0 * A, InterleaveBgr<0>(red, green, blue), tails[0]);
            Store<align, mask>(rgb + 1 * A, InterleaveBgr<1>(red, green, blue), tails[1]);
            Store<align, mask>(rgb + 2 * A, InterleaveBgr<2>(red, green, blue), tails[2]);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv420pToRgb(const uint8_t* y0, const uint8_t* y1, const uint8_t* u, const uint8_t* v, uint8_t* rgb0, uint8_t* rgb1, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[0])));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[0])));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToRgb<align, mask>(Load<align, mask>(y0 + 0, tails[1]), u0, v0, rgb0 + 0 * A, tails + 3);
            YuvToRgb<align, mask>(Load<align, mask>(y0 + A, tails[2]), u1, v1, rgb0 + 3 * A, tails + 6);
            YuvToRgb<align, mask>(Load<align, mask>(y1 + 0, tails[1]), u0, v0, rgb1 + 0 * A, tails + 3);
            YuvToRgb<align, mask>(Load<align, mask>(y1 + A, tails[2]), u1, v1, rgb1 + 3 * A, tails + 6);
        }

        template <bool align> void Yuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; row += 2)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv420pToRgb<align, false>(y + col * 2, y + yStride + col * 2, u + col, v + col, rgb + col * 6, rgb + rgbStride + col * 6, tailMasks);
                if (col < width)
                    Yuv420pToRgb<align, true>(y + col * 2, y + yStride + col * 2, u + col, v + col, rgb + col * 6, rgb + rgbStride + col * 6, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                rgb += 2 * rgbStride;
            }
        }

        void Yuv420pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv420pToRgb<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
            else
                Yuv420pToRgb<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv422pToRgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* rgb, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(u, tails[0])));
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, (Load<align, mask>(v, tails[0])));
            YuvToRgb<align, mask>(Load<align, mask>(y + 0, tails[1]), _mm512_unpacklo_epi8(_u, _u), _mm512_unpacklo_epi8(_v, _v), rgb + 0 * A, tails + 3);
            YuvToRgb<align, mask>(Load<align, mask>(y + A, tails[2]), _mm512_unpackhi_epi8(_u, _u), _mm512_unpackhi_epi8(_v, _v), rgb + 3 * A, tails + 6);
        }

        template <bool align> void Yuv422pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv422pToRgb<align, false>(y + col * 2, u + col, v + col, rgb + col * 6, tailMasks);
                if (col < width)
                    Yuv422pToRgb<align, true>(y + col * 2, u + col, v + col, rgb + col * 6, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv422pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv422pToRgb<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
            else
                Yuv422pToRgb<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
        }

        template <bool align, bool mask> SIMD_INLINE void Yuv444pToRgb(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* rgb, const __mmask64* tails)
        {
            YuvToRgb<align, mask>(Load<align, mask>(y, tails[0]), Load<align, mask>(u, tails[0]), Load<align, mask>(v, tails[0]), rgb, tails + 1);
        }

        template <bool align> void Yuv444pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[4];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 3; ++i)
                tailMasks[1 + i] = TailMask64(tail * 3 - A * i);
            for (size_t row = 0; row < height; ++row)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToRgb<align, false>(y + col, u + col, v + col, rgb + col * 3, tailMasks);
                if (col < width)
                    Yuv444pToRgb<align, true>(y + col, u + col, v + col, rgb + col * 3, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv444pToRgb(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv444pToRgb<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
            else
                Yuv444pToRgb<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

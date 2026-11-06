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
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdInterleave.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask, class T> SIMD_INLINE void YuvToBgr(const __m512i& y, const __m512i& u, const __m512i& v, uint8_t* bgr, const __mmask64* tails)
        {
            __m512i b = YuvToBlue<T>(y, u);
            __m512i g = YuvToGreen<T>(y, u, v);
            __m512i r = YuvToRed<T>(y, v);
            if (mask)
                _mm512_mask_storeu_epi8(bgr + 0 * A, tails[0], InterleaveBgr<0>(b, g, r));
            else
                _mm512_storeu_si512(bgr + 0 * A, InterleaveBgr<0>(b, g, r));
            if (mask)
                _mm512_mask_storeu_epi8(bgr + 1 * A, tails[1], InterleaveBgr<1>(b, g, r));
            else
                _mm512_storeu_si512(bgr + 1 * A, InterleaveBgr<1>(b, g, r));
            if (mask)
                _mm512_mask_storeu_epi8(bgr + 2 * A, tails[2], InterleaveBgr<2>(b, g, r));
            else
                _mm512_storeu_si512(bgr + 2 * A, InterleaveBgr<2>(b, g, r));
        }

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv420pToBgrV2(const uint8_t* y0, const uint8_t* y1, 
            const uint8_t* u, const uint8_t* v, uint8_t* bgr0, uint8_t* bgr1, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u))));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v))));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y0 + 0) : _mm512_loadu_si512(y0 + 0)), u0, v0, bgr0 + 0 * A, tails + 3);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y0 + A) : _mm512_loadu_si512(y0 + A)), u1, v1, bgr0 + 3 * A, tails + 6);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y1 + 0) : _mm512_loadu_si512(y1 + 0)), u0, v0, bgr1 + 0 * A, tails + 3);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y1 + A) : _mm512_loadu_si512(y1 + A)), u1, v1, bgr1 + 3 * A, tails + 6);
        }

        template <class T> void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
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
                    Yuv420pToBgrV2<false, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, bgr + col * 6, bgr + bgrStride + col * 6, tailMasks);
                if (col < width)
                    Yuv420pToBgrV2<true, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, bgr + col * 6, bgr + bgrStride + col * 6, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv420pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv420pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv420pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv422pToBgrV2(const uint8_t* y, 
            const uint8_t* u, const uint8_t* v, uint8_t* bgr, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u))));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v))));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y + 0) : _mm512_loadu_si512(y + 0)), u0, v0, bgr + 0 * A, tails + 3);
            YuvToBgr<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y + A) : _mm512_loadu_si512(y + A)), u1, v1, bgr + 3 * A, tails + 6);
        }

        template <class T> void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert(width % 2 == 0);
            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv422pToBgrV2<false, T>(y + col * 2, u + col, v + col, bgr + col * 6, tailMasks);
                if (col < width)
                    Yuv422pToBgrV2<true, T>(y + col * 2, u + col, v + col, bgr + col * 6, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv422pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv422pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv422pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv444pToBgrV2(const uint8_t* y,
            const uint8_t* u, const uint8_t* v, uint8_t* bgr, const __mmask64* tails)
        {
            __m512i _y = (mask ? _mm512_maskz_loadu_epi8(tails[0], y) : _mm512_loadu_si512(y));
            __m512i _u = (mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u));
            __m512i _v = (mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v));
            YuvToBgr<mask, T>(_y, _u, _v, bgr, tails + 1);
        }

        template <class T> void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[4];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 3; ++i)
                tailMasks[1 + i] = TailMask64(tail * 3 - A * i);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToBgrV2<false, T>(y + col, u + col, v + col, bgr + col * 3, tailMasks);
                if (col < width)
                    Yuv444pToBgrV2<true, T>(y + col, u + col, v + col, bgr + col * 3, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgrV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv444pToBgrV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv444pToBgrV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv444pToBgrV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_INLINE void YuvToRgb(const __m512i& y, const __m512i& u, const __m512i& v, uint8_t* rgb, const __mmask64* tails)
        {
            __m512i b = YuvToBlue<T>(y, u);
            __m512i g = YuvToGreen<T>(y, u, v);
            __m512i r = YuvToRed<T>(y, v);
            if (mask)
                _mm512_mask_storeu_epi8(rgb + 0 * A, tails[0], InterleaveBgr<0>(r, g, b));
            else
                _mm512_storeu_si512(rgb + 0 * A, InterleaveBgr<0>(r, g, b));
            if (mask)
                _mm512_mask_storeu_epi8(rgb + 1 * A, tails[1], InterleaveBgr<1>(r, g, b));
            else
                _mm512_storeu_si512(rgb + 1 * A, InterleaveBgr<1>(r, g, b));
            if (mask)
                _mm512_mask_storeu_epi8(rgb + 2 * A, tails[2], InterleaveBgr<2>(r, g, b));
            else
                _mm512_storeu_si512(rgb + 2 * A, InterleaveBgr<2>(r, g, b));
        }

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv420pToRgbV2(const uint8_t* y0, const uint8_t* y1,
            const uint8_t* u, const uint8_t* v, uint8_t* rgb0, uint8_t* rgb1, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u))));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v))));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y0 + 0) : _mm512_loadu_si512(y0 + 0)), u0, v0, rgb0 + 0 * A, tails + 3);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y0 + A) : _mm512_loadu_si512(y0 + A)), u1, v1, rgb0 + 3 * A, tails + 6);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y1 + 0) : _mm512_loadu_si512(y1 + 0)), u0, v0, rgb1 + 0 * A, tails + 3);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y1 + A) : _mm512_loadu_si512(y1 + A)), u1, v1, rgb1 + 3 * A, tails + 6);
        }

        template <class T> void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0));
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
                    Yuv420pToRgbV2<false, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, rgb + col * 6, rgb + rgbStride + col * 6, tailMasks);
                if (col < width)
                    Yuv420pToRgbV2<true, T>(y + col * 2, y + yStride + col * 2, u + col, v + col, rgb + col * 6, rgb + rgbStride + col * 6, tailMasks);
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                rgb += 2 * rgbStride;
            }
        }

        void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToRgbV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv420pToRgbV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv420pToRgbV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv420pToRgbV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv422pToRgbV2(const uint8_t* y,
            const uint8_t* u, const uint8_t* v, uint8_t* rgb, const __mmask64* tails)
        {
            __m512i _u = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u))));
            __m512i u0 = UnpackU8<0>(_u, _u);
            __m512i u1 = UnpackU8<1>(_u, _u);
            __m512i _v = _mm512_permutexvar_epi64(K64_PERMUTE_FOR_UNPACK, ((mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v))));
            __m512i v0 = UnpackU8<0>(_v, _v);
            __m512i v1 = UnpackU8<1>(_v, _v);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[1], y + 0) : _mm512_loadu_si512(y + 0)), u0, v0, rgb + 0 * A, tails + 3);
            YuvToRgb<mask, T>((mask ? _mm512_maskz_loadu_epi8(tails[2], y + A) : _mm512_loadu_si512(y + A)), u1, v1, rgb + 3 * A, tails + 6);
        }

        template <class T> void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert(width % 2 == 0);
            width /= 2;
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[9];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 2; ++i)
                tailMasks[1 + i] = TailMask64(tail * 2 - A * i);
            for (size_t i = 0; i < 6; ++i)
                tailMasks[3 + i] = TailMask64(tail * 6 - A * i);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv422pToRgbV2<false, T>(y + col * 2, u + col, v + col, rgb + col * 6, tailMasks);
                if (col < width)
                    Yuv422pToRgbV2<true, T>(y + col * 2, u + col, v + col, rgb + col * 6, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToRgbV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv422pToRgbV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv422pToRgbV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv422pToRgbV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask, class T> SIMD_YUV_TO_BGR_INLINE void Yuv444pToRgbV2(const uint8_t* y,
            const uint8_t* u, const uint8_t* v, uint8_t* rgb, const __mmask64* tails)
        {
            __m512i _y = (mask ? _mm512_maskz_loadu_epi8(tails[0], y) : _mm512_loadu_si512(y));
            __m512i _u = (mask ? _mm512_maskz_loadu_epi8(tails[0], u) : _mm512_loadu_si512(u));
            __m512i _v = (mask ? _mm512_maskz_loadu_epi8(tails[0], v) : _mm512_loadu_si512(v));
            YuvToRgb<mask, T>(_y, _u, _v, rgb, tails + 1);
        }

        template <class T> void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            size_t tail = width - alignedWidth;
            __mmask64 tailMasks[4];
            tailMasks[0] = TailMask64(tail);
            for (size_t i = 0; i < 3; ++i)
                tailMasks[1 + i] = TailMask64(tail * 3 - A * i);
            for (size_t row = 0; row < height; row += 1)
            {
                size_t col = 0;
                for (; col < alignedWidth; col += A)
                    Yuv444pToRgbV2<false, T>(y + col, u + col, v + col, rgb + col * 3, tailMasks);
                if (col < width)
                    Yuv444pToRgbV2<true, T>(y + col, u + col, v + col, rgb + col * 3, tailMasks);
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToRgbV2<Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv444pToRgbV2<Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv444pToRgbV2<Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv444pToRgbV2<Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }
    }
#endif
}

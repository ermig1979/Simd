/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdInterleave.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        template <class T> SIMD_YUV_TO_BGR_INLINE void YuvToBgrV2(__m128i y8, __m128i u8, __m128i v8, __m128i* bgr)
        {
            __m128i blue = YuvToBlue<T>(y8, u8);
            __m128i green = YuvToGreen<T>(y8, u8, v8);
            __m128i red = YuvToRed<T>(y8, v8);
            _mm_storeu_si128(bgr + 0, InterleaveBgr<0>(blue, green, red));
            _mm_storeu_si128(bgr + 1, InterleaveBgr<1>(blue, green, red));
            _mm_storeu_si128(bgr + 2, InterleaveBgr<2>(blue, green, red));
        }

        template <class T> SIMD_INLINE void Yuv422pToBgrV2(const uint8_t* y, const __m128i& u, const __m128i& v,
            uint8_t* bgr)
        {
            YuvToBgrV2<T>(_mm_loadu_si128((__m128i*)y + 0), _mm_unpacklo_epi8(u, u), _mm_unpacklo_epi8(v, v), (__m128i*)bgr + 0);
            YuvToBgrV2<T>(_mm_loadu_si128((__m128i*)y + 1), _mm_unpackhi_epi8(u, u), _mm_unpackhi_epi8(v, v), (__m128i*)bgr + 3);
        }

        template <class T> void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < bodyWidth; colY += DA, colUV += A, colBgr += A * 6)
                {
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + colUV));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + colUV));
                    Yuv422pToBgrV2<T>(y + colY, u_, v_, bgr + colBgr);
                    Yuv422pToBgrV2<T>(y + colY + yStride, u_, v_, bgr + colBgr + bgrStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + offset / 2));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + offset / 2));
                    Yuv422pToBgrV2<T>(y + offset, u_, v_, bgr + 3 * offset);
                    Yuv422pToBgrV2<T>(y + offset + yStride, u_, v_, bgr + 3 * offset + bgrStride);
                }
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

        template <class T> void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= DA));

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < bodyWidth; colY += DA, colUV += A, colBgr += A * 6)
                {
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + colUV));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + colUV));
                    Yuv422pToBgrV2<T>(y + colY, u_, v_, bgr + colBgr);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + offset / 2));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + offset / 2));
                    Yuv422pToBgrV2<T>(y + offset, u_, v_, bgr + 3 * offset);
                }
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

        template <class T> SIMD_INLINE void Yuv444pToBgrV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* bgr)
        {
            YuvToBgrV2<T>(_mm_loadu_si128((__m128i*)y), _mm_loadu_si128((__m128i*)u), _mm_loadu_si128((__m128i*)v), (__m128i*)bgr);
        }

        template <class T> void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert(width >= A);

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t col = 0, colBgr = 0; col < bodyWidth; col += A, colBgr += A * 3)
                    Yuv444pToBgrV2<T>(y + col, u + col, v + col, bgr + colBgr);
                if (tail)
                {
                    size_t offset = width - A;
                    Yuv444pToBgrV2<T>(y + offset, u + offset, v + offset, bgr + 3 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (width < A)
            {
                Base::Yuv444pToBgrV2(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
                return;
            }

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

        template <class T> SIMD_YUV_TO_BGR_INLINE void YuvToRgbV2(__m128i y8, __m128i u8, __m128i v8, __m128i* rgb)
        {
            __m128i blue = YuvToBlue<T>(y8, u8);
            __m128i green = YuvToGreen<T>(y8, u8, v8);
            __m128i red = YuvToRed<T>(y8, v8);
            _mm_storeu_si128(rgb + 0, InterleaveBgr<0>(red, green, blue));
            _mm_storeu_si128(rgb + 1, InterleaveBgr<1>(red, green, blue));
            _mm_storeu_si128(rgb + 2, InterleaveBgr<2>(red, green, blue));
        }

        template <class T> SIMD_INLINE void Yuv422pToRgbV2(const uint8_t* y, const __m128i& u, const __m128i& v,
            uint8_t* rgb)
        {
            YuvToRgbV2<T>(_mm_loadu_si128((__m128i*)y + 0), _mm_unpacklo_epi8(u, u), _mm_unpacklo_epi8(v, v), (__m128i*)rgb + 0);
            YuvToRgbV2<T>(_mm_loadu_si128((__m128i*)y + 1), _mm_unpackhi_epi8(u, u), _mm_unpackhi_epi8(v, v), (__m128i*)rgb + 3);
        }

        template <class T> void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < bodyWidth; colY += DA, colUV += A, colRgb += A * 6)
                {
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + colUV));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + colUV));
                    Yuv422pToRgbV2<T>(y + colY, u_, v_, rgb + colRgb);
                    Yuv422pToRgbV2<T>(y + colY + yStride, u_, v_, rgb + colRgb + rgbStride);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + offset / 2));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + offset / 2));
                    Yuv422pToRgbV2<T>(y + offset, u_, v_, rgb + 3 * offset);
                    Yuv422pToRgbV2<T>(y + offset + yStride, u_, v_, rgb + 3 * offset + rgbStride);
                }
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

        template <class T> void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (width >= DA));

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < bodyWidth; colY += DA, colUV += A, colRgb += A * 6)
                {
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + colUV));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + colUV));
                    Yuv422pToRgbV2<T>(y + colY, u_, v_, rgb + colRgb);
                }
                if (tail)
                {
                    size_t offset = width - DA;
                    __m128i u_ = _mm_loadu_si128((__m128i*)(u + offset / 2));
                    __m128i v_ = _mm_loadu_si128((__m128i*)(v + offset / 2));
                    Yuv422pToRgbV2<T>(y + offset, u_, v_, rgb + 3 * offset);
                }
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

        template <class T> SIMD_INLINE void Yuv444pToRgbV2(const uint8_t* y, const uint8_t* u, const uint8_t* v, uint8_t* rgb)
        {
            YuvToRgbV2<T>(_mm_loadu_si128((__m128i*)y), _mm_loadu_si128((__m128i*)u), _mm_loadu_si128((__m128i*)v), (__m128i*)rgb);
        }

        template <class T> void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; row += 1)
            {
                for (size_t col = 0, colRgb = 0; col < bodyWidth; col += A, colRgb += A * 3)
                    Yuv444pToRgbV2<T>(y + col, u + col, v + col, rgb + colRgb);
                if (tail)
                {
                    size_t offset = width - A;
                    Yuv444pToRgbV2<T>(y + offset, u + offset, v + offset, rgb + 3 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            if (width < A)
            {
                Base::Yuv444pToRgbV2(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
                return;
            }

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

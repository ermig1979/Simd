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

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        const size_t A3 = A * 3;
        const size_t A6 = A * 6;

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void YuvToBgr(const uint8x16_t& y, const uint8x16_t& u, const uint8x16_t& v, uint8_t* bgr)
        {
            uint8x16x3_t _bgr;
            YuvToBgr<T>(y, u, v, _bgr);
            Store3<align>(bgr, _bgr);
        }

        template <bool align, class T> void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colBgr = 0; col < bodyWidth; col += A, colBgr += A3)
                {
                    uint8x16_t _y = Load<align>(y + col);
                    uint8x16_t _u = Load<align>(u + col);
                    uint8x16_t _v = Load<align>(v + col);
                    YuvToBgr<T, align>(_y, _u, _v, bgr + colBgr);
                }
                if (tail)
                {
                    size_t col = width - A;
                    uint8x16_t _y = Load<false>(y + col);
                    uint8x16_t _u = Load<false>(u + col);
                    uint8x16_t _v = Load<false>(v + col);
                    YuvToBgr<T, false>(_y, _u, _v, bgr + 3 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        template <bool align> void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToBgrV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv444pToBgrV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv444pToBgrV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv444pToBgrV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv444pToBgrV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
            else
                Yuv444pToBgrV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void Yuv422pToBgr(const uint8_t* y, const uint8x16x2_t& u, const uint8x16x2_t& v, uint8_t* bgr)
        {
            YuvToBgr<T, align>(Load<align>(y + 0), u.val[0], v.val[0], bgr + 0 * A);
            YuvToBgr<T, align>(Load<align>(y + A), u.val[1], v.val[1], bgr + 3 * A);
        }

        template <bool align, class T> void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < bodyWidth; colY += DA, colUV += A, colBgr += A6)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgr<T, align>(y + colY, _u, _v, bgr + colBgr);

                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgr<T, false>(y + offset, _u, _v, bgr + 3 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                bgr += bgrStride;
            }
        }

        template <bool align> void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToBgrV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv422pToBgrV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv422pToBgrV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv422pToBgrV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Yuv422pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv422pToBgrV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
            else
                Yuv422pToBgrV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, class T> void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgr = 0; colY < bodyWidth; colY += DA, colUV += A, colBgr += A6)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgr<T, align>(y + colY, _u, _v, bgr + colBgr);
                    Yuv422pToBgr<T, align>(y + colY + yStride, _u, _v, bgr + colBgr + bgrStride);

                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToBgr<T, false>(y + offset, _u, _v, bgr + 3 * offset);
                    Yuv422pToBgr<T, false>(y + offset + yStride, _u, _v, bgr + 3 * offset + bgrStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                bgr += 2 * bgrStride;
            }
        }

        template <bool align> void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToBgrV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt709: Yuv420pToBgrV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvBt2020: Yuv420pToBgrV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            case SimdYuvTrect871: Yuv420pToBgrV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride); break;
            default:
                assert(0);
            }
        }

        void Yuv420pToBgrV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* bgr, size_t bgrStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(bgr) && Aligned(bgrStride))
                Yuv420pToBgrV2<true>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
            else
                Yuv420pToBgrV2<false>(y, yStride, u, uStride, v, vStride, width, height, bgr, bgrStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void YuvToRgb(const uint8x16_t& y, const uint8x16_t& u, const uint8x16_t& v, uint8_t* rgb)
        {
            uint8x16x3_t _rgb;
            YuvToRgb<T>(y, u, v, _rgb);
            Store3<align>(rgb, _rgb);
        }

        template <bool align, class T> void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            size_t bodyWidth = AlignLo(width, A);
            size_t tail = width - bodyWidth;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, colRgb = 0; col < bodyWidth; col += A, colRgb += A3)
                {
                    uint8x16_t _y = Load<align>(y + col);
                    uint8x16_t _u = Load<align>(u + col);
                    uint8x16_t _v = Load<align>(v + col);
                    YuvToRgb<T, align>(_y, _u, _v, rgb + colRgb);
                }
                if (tail)
                {
                    size_t col = width - A;
                    uint8x16_t _y = Load<false>(y + col);
                    uint8x16_t _u = Load<false>(u + col);
                    uint8x16_t _v = Load<false>(v + col);
                    YuvToRgb<T, false>(_y, _u, _v, rgb + 3 * col);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        template <bool align> void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv444pToRgbV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv444pToRgbV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv444pToRgbV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv444pToRgbV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        void Yuv444pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv444pToRgbV2<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
            else
                Yuv444pToRgbV2<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <class T, bool align> SIMD_INLINE void Yuv422pToRgb(const uint8_t* y, const uint8x16x2_t& u, const uint8x16x2_t& v, uint8_t* rgb)
        {
            YuvToRgb<T, align>(Load<align>(y + 0), u.val[0], v.val[0], rgb + 0 * A);
            YuvToRgb<T, align>(Load<align>(y + A), u.val[1], v.val[1], rgb + 3 * A);
        }

        template <bool align, class T> void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (width >= DA));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < bodyWidth; colY += DA, colUV += A, colRgb += A6)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToRgb<T, align>(y + colY, _u, _v, rgb + colRgb);

                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToRgb<T, false>(y + offset, _u, _v, rgb + 3 * offset);
                }
                y += yStride;
                u += uStride;
                v += vStride;
                rgb += rgbStride;
            }
        }

        template <bool align> void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv422pToRgbV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv422pToRgbV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv422pToRgbV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv422pToRgbV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        void Yuv422pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv422pToRgbV2<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
            else
                Yuv422pToRgbV2<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, class T> void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= DA) && (height >= 2));
            if (align)
            {
                assert(Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride));
                assert(Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride));
            }

            size_t bodyWidth = AlignLo(width, DA);
            size_t tail = width - bodyWidth;
            uint8x16x2_t _u, _v;
            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colRgb = 0; colY < bodyWidth; colY += DA, colUV += A, colRgb += A6)
                {
                    _u.val[1] = _u.val[0] = Load<align>(u + colUV);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<align>(v + colUV);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToRgb<T, align>(y + colY, _u, _v, rgb + colRgb);
                    Yuv422pToRgb<T, align>(y + colY + yStride, _u, _v, rgb + colRgb + rgbStride);

                }
                if (tail)
                {
                    size_t offset = width - DA;
                    _u.val[1] = _u.val[0] = Load<false>(u + offset / 2);
                    _u = vzipq_u8(_u.val[0], _u.val[1]);
                    _v.val[1] = _v.val[0] = Load<false>(v + offset / 2);
                    _v = vzipq_u8(_v.val[0], _v.val[1]);
                    Yuv422pToRgb<T, false>(y + offset, _u, _v, rgb + 3 * offset);
                    Yuv422pToRgb<T, false>(y + offset + yStride, _u, _v, rgb + 3 * offset + rgbStride);
                }
                y += 2 * yStride;
                u += uStride;
                v += vStride;
                rgb += 2 * rgbStride;
            }
        }

        template <bool align> void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: Yuv420pToRgbV2<align, Base::Bt601>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt709: Yuv420pToRgbV2<align, Base::Bt709>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvBt2020: Yuv420pToRgbV2<align, Base::Bt2020>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            case SimdYuvTrect871: Yuv420pToRgbV2<align, Base::Trect871>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride); break;
            default:
                assert(0);
            }
        }

        void Yuv420pToRgbV2(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
            size_t width, size_t height, uint8_t* rgb, size_t rgbStride, SimdYuvType yuvType)
        {
            if (Aligned(y) && Aligned(yStride) && Aligned(u) && Aligned(uStride)
                && Aligned(v) && Aligned(vStride) && Aligned(rgb) && Aligned(rgbStride))
                Yuv420pToRgbV2<true>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
            else
                Yuv420pToRgbV2<false>(y, yStride, u, uStride, v, vStride, width, height, rgb, rgbStride, yuvType);
        }
    }
#endif// SIMD_NEON_ENABLE
}
